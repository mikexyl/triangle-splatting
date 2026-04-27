import os
import time
import uuid
from argparse import Namespace
from dataclasses import dataclass
from itertools import count
from random import randint

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scene import Scene, TriangleModel
from scene.dataset_readers import fetchPly, fetchTriangleSoup
from triangle_renderer import render
from utils.image_utils import psnr
from utils.loss_utils import equilateral_regularizer, l1_loss, l2_loss, ssim
from utils.rerun_utils import RerunLogger, create_rerun_config

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


RERUN_ARG_NAMES = (
    "rerun",
    "rerun_spawn",
    "rerun_save",
    "rerun_image_every",
    "rerun_mesh_every",
    "rerun_max_triangles",
)

ONLINE_TRAIN_ARG_NAMES = (
    "online_train_initial_cameras",
    "online_train_camera_growth_interval",
    "online_train_camera_growth_count",
    "online_train_window_size",
    "online_train_min_prune_cameras",
    "online_train_unbounded",
    "online_train_stop_when_frames_exhausted",
)

SEED_INIT_ARG_NAMES = (
    "seed_init_mode",
)


@dataclass(frozen=True)
class TrainingRunConfig:
    rerun_app_id: str
    online_train: bool = False
    online_train_initial_cameras: int = 1
    online_train_camera_growth_interval: int = 0
    online_train_camera_growth_count: int = 1
    online_train_window_size: int = 10
    online_train_min_prune_cameras: int = 250
    online_train_unbounded: bool = False
    online_train_stop_when_frames_exhausted: bool = False


class _PyramidTrainingView:
    def __init__(self, base_view, gt_image: torch.Tensor):
        self.uid = getattr(base_view, "uid", None)
        self.colmap_id = getattr(base_view, "colmap_id", None)
        self.R = getattr(base_view, "R", None)
        self.T = getattr(base_view, "T", None)
        self.FoVx = base_view.FoVx
        self.FoVy = base_view.FoVy
        self.image_name = base_view.image_name
        self.original_image = gt_image
        self.image_width = gt_image.shape[2]
        self.image_height = gt_image.shape[1]
        self.zfar = base_view.zfar
        self.znear = base_view.znear
        self.trans = getattr(base_view, "trans", None)
        self.scale = getattr(base_view, "scale", None)
        self.world_view_transform = base_view.world_view_transform
        self.projection_matrix = base_view.projection_matrix
        self.full_proj_transform = base_view.full_proj_transform
        self.camera_center = base_view.camera_center
        self.gt_alpha_mask = None


def _gaussian_downsample(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[0]
    kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=image.dtype, device=image.device) / 16.0
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 5, 5)
    kernel = kernel_2d.expand(channels, 1, 5, 5)
    image_batch = F.pad(image.unsqueeze(0), (2, 2, 2, 2), mode="replicate")
    return F.conv2d(image_batch, kernel, stride=2, groups=channels).squeeze(0).clamp(0.0, 1.0)


def _build_gaussian_pyramid(image: torch.Tensor, level_count: int) -> list[torch.Tensor]:
    pyramid = [image]
    for _ in range(1, max(1, level_count)):
        previous = pyramid[-1]
        if min(previous.shape[1], previous.shape[2]) <= 1:
            break
        pyramid.append(_gaussian_downsample(previous))
    return pyramid


def _get_gaussian_pyramid(view, level_count: int) -> list[torch.Tensor]:
    requested_levels = max(1, int(level_count))
    cached = getattr(view, "_training_gaussian_pyramid", None)
    if cached is None or len(cached) < requested_levels:
        with torch.no_grad():
            cached = _build_gaussian_pyramid(view.original_image.cuda(), requested_levels)
        setattr(view, "_training_gaussian_pyramid", cached)
    return cached


def _pyramid_schedule_until(opt) -> int:
    configured_until = int(getattr(opt, "pyramid_schedule_until_iter", 0))
    if configured_until > 0:
        return configured_until
    densify_until = int(getattr(opt, "densify_until_iter", 0))
    if densify_until > 0:
        return densify_until
    return int(opt.iterations)


def _pyramid_level_for_iteration(opt, iteration: int) -> int:
    if not getattr(opt, "pyramid_training", False):
        return 0

    level_count = int(getattr(opt, "pyramid_levels", 1))
    if level_count < 1:
        raise ValueError(f"pyramid_levels must be >= 1, got {level_count}")
    max_level = level_count - 1
    if max_level == 0:
        return 0

    schedule_until = max(1, _pyramid_schedule_until(opt))
    level = max_level - ((max(1, iteration) - 1) * level_count // schedule_until)
    return max(0, min(max_level, level))


def _training_view_for_iteration(view, opt, iteration: int):
    pyramid_level = _pyramid_level_for_iteration(opt, iteration)
    if pyramid_level == 0:
        return view, view.original_image.cuda(), 0

    pyramid = _get_gaussian_pyramid(view, int(getattr(opt, "pyramid_levels", 1)))
    pyramid_level = min(pyramid_level, len(pyramid) - 1)
    if pyramid_level == 0:
        return view, view.original_image.cuda(), 0

    gt_image = pyramid[pyramid_level]
    return _PyramidTrainingView(view, gt_image), gt_image, pyramid_level


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _online_frame_exhaustion_iteration(scene, run_config, growth_interval: int) -> int:
    growth_count = int(run_config.online_train_camera_growth_count)
    total_count = scene.getTotalTrainCameraCount()
    initial_count = min(int(run_config.online_train_initial_cameras), total_count)
    remaining_count = max(0, total_count - initial_count)
    reveal_steps = _ceil_div(remaining_count, growth_count) if remaining_count > 0 else 0
    last_reveal_iteration = 1 + reveal_steps * growth_interval
    return last_reveal_iteration + growth_interval - 1


def _resolve_online_growth_schedule(scene, run_config, opt):
    requested_interval = int(run_config.online_train_camera_growth_interval)
    growth_count = int(run_config.online_train_camera_growth_count)
    if requested_interval < 0:
        raise ValueError(f"online_train_camera_growth_interval must be >= 0, got {requested_interval}")
    if growth_count <= 0:
        raise ValueError(f"online_train_camera_growth_count must be > 0, got {growth_count}")

    total_count = scene.getTotalTrainCameraCount()
    requested_initial_count = int(run_config.online_train_initial_cameras)
    if requested_initial_count <= 0:
        raise ValueError(f"online_train_initial_cameras must be > 0, got {requested_initial_count}")

    initial_count = min(requested_initial_count, total_count)
    iterations = int(opt.iterations)
    unbounded_online_train = bool(getattr(run_config, "online_train_unbounded", False)) or iterations <= 0

    if requested_interval > 0:
        if unbounded_online_train:
            return requested_interval, None, False
        steps = max(iterations - 1, 0) // requested_interval
        expected_final_count = min(total_count, initial_count + steps * growth_count)
        return requested_interval, expected_final_count, False

    remaining_count = max(0, total_count - initial_count)
    if remaining_count == 0:
        return 1, total_count, True
    if unbounded_online_train:
        return 1, total_count, True

    updates_needed = _ceil_div(remaining_count, growth_count)
    max_update_steps = max(1, iterations - 1)
    growth_interval = max(1, max_update_steps // updates_needed)
    steps = max(iterations - 1, 0) // growth_interval
    expected_final_count = min(total_count, initial_count + steps * growth_count)
    return growth_interval, expected_final_count, True


def add_common_training_args(parser) -> None:
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--no_dome", action="store_true", default=False)
    parser.add_argument("--outdoor", action="store_true", default=False)
    parser.add_argument(
        "--seed_init_mode",
        choices=("point", "mesh_triangle"),
        default="point",
        help="Initial geometry seed mode. 'point' keeps the existing point-cloud path; 'mesh_triangle' uses Kimera mesh triangle-soup seeds.",
    )


def add_rerun_args(parser) -> None:
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rerun_spawn", action="store_true")
    parser.add_argument("--rerun_save", type=str, default=None)
    parser.add_argument("--rerun_image_every", type=int, default=25)
    parser.add_argument("--rerun_mesh_every", type=int, default=100)
    parser.add_argument(
        "--rerun_max_triangles",
        type=int,
        default=5000,
        help="Maximum number of triangles to log to Rerun. Use 0 to log all triangles.",
    )


def add_online_training_args(parser) -> None:
    parser.add_argument(
        "--online_train_initial_cameras",
        type=int,
        default=1,
        help="Initial number of train cameras made visible in the primitive online replay.",
    )
    parser.add_argument(
        "--online_train_camera_growth_interval",
        type=int,
        default=10,
        help=(
            "Reveal more train cameras every N iterations during the primitive online replay. "
            "The default 10 gives 10 optimization iterations per newly revealed frame. Use 0 to auto-select "
            "a cadence from the bounded run length, or one new frame per iteration in open-ended replay."
        ),
    )
    parser.add_argument(
        "--online_train_camera_growth_count",
        type=int,
        default=1,
        help="Number of additional train cameras to reveal at each replay step.",
    )
    parser.add_argument(
        "--online_train_window_size",
        type=int,
        default=10,
        help="Sliding local optimization window over the revealed train cameras; 0 uses the full revealed prefix.",
    )
    parser.add_argument(
        "--online_train_min_prune_cameras",
        type=int,
        default=250,
        help="Minimum number of revealed train cameras before densification/pruning is allowed.",
    )
    parser.add_argument(
        "--online_train_unbounded",
        action="store_true",
        help="Run primitive online replay without a user-specified final iteration cap.",
    )
    parser.add_argument(
        "--online_train_stop_when_frames_exhausted",
        action="store_true",
        help="For unbounded primitive online replay, stop after every train frame has been revealed and optimized for one growth interval.",
    )


def copy_named_args(target, source, names) -> None:
    for name in names:
        setattr(target, name, getattr(source, name))


def _merge_point_clouds(point_clouds):
    valid_clouds = [pcd for pcd in point_clouds if pcd is not None and len(pcd.points) > 0]
    if not valid_clouds:
        return None

    return valid_clouds[0].__class__(
        points=np.concatenate([np.asarray(pcd.points) for pcd in valid_clouds], axis=0),
        colors=np.concatenate([np.asarray(pcd.colors) for pcd in valid_clouds], axis=0),
        normals=np.concatenate([np.asarray(pcd.normals) for pcd in valid_clouds], axis=0),
    )


def _merge_triangle_soups(triangle_soups):
    valid_soups = [soup for soup in triangle_soups if soup is not None and len(soup[0]) > 0]
    if not valid_soups:
        return None

    return (
        np.concatenate([soup[0] for soup in valid_soups], axis=0),
        np.concatenate([soup[1] for soup in valid_soups], axis=0),
    )


def _get_view_seed_path(view, seed_init_mode):
    if seed_init_mode == "mesh_triangle":
        return getattr(view, "seed_triangles_path", None)
    return getattr(view, "seed_points_path", None)


def _append_online_seed_geometry(triangles, views, seen_seed_paths, opt, seed_init_mode):
    seed_paths = []
    for view in views:
        seed_path = _get_view_seed_path(view, seed_init_mode)
        if not seed_path or seed_path in seen_seed_paths:
            continue
        seen_seed_paths.add(seed_path)
        seed_paths.append(seed_path)

    if not seed_paths:
        return 0

    if seed_init_mode == "mesh_triangle":
        seed_triangle_soup = _merge_triangle_soups([fetchTriangleSoup(seed_path) for seed_path in seed_paths])
        if seed_triangle_soup is None:
            return 0

        return triangles.append_from_triangle_soup(
            seed_triangle_soup[0],
            seed_triangle_soup[1],
            opacity=opt.set_opacity,
            set_sigma=opt.set_sigma,
        )

    seed_point_cloud = _merge_point_clouds([fetchPly(seed_path) for seed_path in seed_paths])
    if seed_point_cloud is None:
        return 0

    return triangles.append_from_pcd(
        seed_point_cloud,
        init_size=opt.triangle_size,
        opacity=opt.set_opacity,
        nb_points=opt.nb_points,
        set_sigma=opt.set_sigma,
    )


def run_training(
    dataset,
    opt,
    pipe,
    no_dome,
    outdoor,
    testing_iterations,
    save_iterations,
    checkpoint,
    debug_from,
    run_config: TrainingRunConfig,
):
    lpips_fn = lpips.LPIPS(net="vgg").to(device="cuda")

    first_iter = 0
    last_iteration = 0
    tb_writer = prepare_output_and_logger(dataset)
    rerun_logger = RerunLogger(run_config.rerun_app_id, create_rerun_config(dataset))

    try:
        unbounded_online_train = run_config.online_train and (
            bool(getattr(run_config, "online_train_unbounded", False)) or int(opt.iterations) <= 0
        )
        stop_when_frames_exhausted = unbounded_online_train and bool(
            getattr(run_config, "online_train_stop_when_frames_exhausted", False)
        )
        if not unbounded_online_train and int(opt.iterations) <= 0:
            raise ValueError("--iterations must be > 0 unless primitive online replay is unbounded")

        seed_init_mode = getattr(dataset, "seed_init_mode", "point")
        if seed_init_mode == "mesh_triangle" and opt.nb_points != 3:
            raise ValueError("seed_init_mode=mesh_triangle requires --nb_points 3")

        triangles = TriangleModel(dataset.sh_degree)
        scene = Scene(
            dataset,
            triangles,
            opt.set_opacity,
            opt.triangle_size,
            opt.nb_points,
            opt.set_sigma,
            no_dome,
            shuffle=not run_config.online_train,
        )
        if run_config.online_train:
            online_growth_interval, expected_final_count, auto_growth_interval = _resolve_online_growth_schedule(
                scene,
                run_config,
                opt,
            )
            scene.enable_online_train_schedule(
                initial_count=run_config.online_train_initial_cameras,
                growth_interval=online_growth_interval,
                growth_count=run_config.online_train_camera_growth_count,
                window_size=run_config.online_train_window_size,
            )
            if auto_growth_interval:
                if unbounded_online_train:
                    print(
                        "Online train camera growth interval auto-selected: "
                        f"every {online_growth_interval} iteration(s). No final iteration cap is set; "
                        f"the replay will reveal up to {scene.getTotalTrainCameraCount()} train cameras as it runs."
                    )
                else:
                    print(
                        "Online train camera growth interval auto-selected: "
                        f"every {online_growth_interval} iteration(s), expected to reveal "
                        f"{expected_final_count}/{scene.getTotalTrainCameraCount()} train cameras by iteration "
                        f"{int(opt.iterations)}."
                    )
            elif expected_final_count is not None and expected_final_count < scene.getTotalTrainCameraCount():
                print(
                    "Warning: configured online train camera growth interval will reveal only "
                    f"{expected_final_count}/{scene.getTotalTrainCameraCount()} train cameras by iteration "
                    f"{int(opt.iterations)}. Use --online_train_camera_growth_interval 0 for automatic full-run coverage."
                )
            frame_exhaustion_iteration = None
            if stop_when_frames_exhausted:
                frame_exhaustion_iteration = _online_frame_exhaustion_iteration(
                    scene,
                    run_config,
                    online_growth_interval,
                )
                testing_iterations = sorted(set(testing_iterations) | {frame_exhaustion_iteration})
                save_iterations = sorted(set(save_iterations) | {frame_exhaustion_iteration})
                print(
                    "Online replay will stop when train frames are exhausted: "
                    f"{scene.getTotalTrainCameraCount()} train frames, "
                    f"{online_growth_interval} optimization iteration(s) per reveal step, "
                    f"final iteration {frame_exhaustion_iteration}."
                )
            print(
                "Primitive online train replay enabled: "
                "this mode only reveals frames incrementally and optionally restricts optimization "
                "to a sliding local window. It does not yet implement a robust online mapping pipeline. "
                f"{scene.getActiveTrainCameraCount()}/{scene.getTotalTrainCameraCount()} "
                f"train cameras revealed, optimizing window of {scene.getActiveTrainWindowCount()} "
                f"frames starting at index {scene.getActiveTrainWindowStart()}"
            )
            rerun_logger.log_online_setup(
                train_views=scene.getAllTrainCameras(),
                test_views=scene.getTestCameras(),
            )
        seen_online_seed_paths = set()
        if run_config.online_train:
            seen_online_seed_paths = {
                _get_view_seed_path(view, seed_init_mode)
                for view in scene.getRevealedTrainCameras()
                if _get_view_seed_path(view, seed_init_mode)
            }
        active_train_count = scene.getActiveTrainCameraCount()

        triangles.training_setup(
            opt,
            opt.lr_mask,
            opt.feature_lr,
            opt.opacity_lr,
            opt.lr_sigma,
            opt.lr_triangles_points_init,
        )

        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            triangles.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if getattr(opt, "pyramid_training", False):
            if int(getattr(opt, "pyramid_levels", 1)) < 1:
                raise ValueError(f"pyramid_levels must be >= 1, got {opt.pyramid_levels}")
            if int(getattr(opt, "pyramid_schedule_until_iter", 0)) < 0:
                raise ValueError(
                    f"pyramid_schedule_until_iter must be >= 0, got {opt.pyramid_schedule_until_iter}"
                )
            schedule_until = _pyramid_schedule_until(opt)
            print(
                "Gaussian pyramid supervision enabled: "
                f"{int(opt.pyramid_levels)} levels including the original image; "
                f"scheduled to reach full resolution by iteration {schedule_until}."
            )

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = scene.getTrainCameras().copy()
        number_of_views = len(viewpoint_stack)
        if number_of_views == 0:
            raise RuntimeError("Training requires at least one active train camera.")

        ema_loss_for_log = 0.0
        effective_final_iteration = None
        if stop_when_frames_exhausted:
            effective_final_iteration = frame_exhaustion_iteration
        elif not unbounded_online_train:
            effective_final_iteration = int(opt.iterations)
        open_ended_online_train = unbounded_online_train and effective_final_iteration is None
        progress_bar = (
            tqdm(total=None, initial=first_iter, desc="Training progress")
            if open_ended_online_train
            else tqdm(range(first_iter, opt.iterations), desc="Training progress")
            if effective_final_iteration is None
            else tqdm(range(first_iter, effective_final_iteration), desc="Training progress")
        )
        start_iteration = first_iter + 1
        iteration_range = (
            count(start_iteration)
            if open_ended_online_train
            else range(start_iteration, effective_final_iteration + 1)
        )

        total_dead = 0
        opacity_now = True
        new_round = False
        removed_them = False

        large_scene = triangles.large
        loss_fn = l2_loss if large_scene and outdoor else l1_loss

        for iteration in iteration_range:
            last_iteration = iteration
            iter_start.record()
            triangles.update_learning_rate(iteration)

            online_schedule_changed = False
            if run_config.online_train and scene.update_online_train_set(iteration):
                online_schedule_changed = True
                newly_revealed_views = scene.getNewlyRevealedTrainCameras(active_train_count)
                added_triangles = _append_online_seed_geometry(
                    triangles,
                    newly_revealed_views,
                    seen_online_seed_paths,
                    opt,
                    seed_init_mode,
                )
                active_train_count = scene.getActiveTrainCameraCount()
                viewpoint_stack = scene.getTrainCameras().copy()
                print(
                    f"[ITER {iteration}] Online train replay: "
                    f"{scene.getActiveTrainCameraCount()}/{scene.getTotalTrainCameraCount()} "
                    f"train cameras revealed, optimizing window of {scene.getActiveTrainWindowCount()} "
                    f"frames starting at index {scene.getActiveTrainWindowStart()}"
                )
                if added_triangles > 0:
                    print(f"[ITER {iteration}] Added {added_triangles} triangles from newly revealed frame meshes")

            if iteration % 1000 == 0:
                triangles.oneupSHdegree()

            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                if not new_round and removed_them:
                    new_round = True
                    removed_them = False
                else:
                    new_round = False

            number_of_views = len(scene.getTrainCameras())
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            training_view, gt_image, pyramid_level = _training_view_for_iteration(viewpoint_cam, opt, iteration)
            render_pkg = render(training_view, triangles, pipe, bg)
            image = render_pkg["render"]
            triangle_area = render_pkg["density_factor"].detach()
            image_size = render_pkg["scaling"].detach()
            importance_score = render_pkg["max_blending"].detach()

            if new_round:
                mask = triangle_area > 1
                triangles.triangle_area[mask] += 1

            mask = image_size > triangles.image_size
            triangles.image_size[mask] = image_size[mask]
            mask = importance_score > triangles.importance_score
            triangles.importance_score[mask] = importance_score[mask]

            pixel_loss = loss_fn(image, gt_image)
            loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss_opacity = torch.abs(triangles.get_opacity).mean() * opt.lambda_opacity

            rend_normal = render_pkg["rend_normal"]
            surf_normal = render_pkg["surf_normal"]
            lambda_dist = opt.lambda_dist if iteration > opt.iteration_mesh else 0
            lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
            rend_dist = render_pkg["rend_dist"]
            dist_loss = lambda_dist * (rend_dist).mean()
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()

            loss_size = 1 / equilateral_regularizer(triangles.get_triangles_points).mean()
            loss_size = loss_size * opt.lambda_size

            if iteration < opt.densify_until_iter:
                loss = loss_image + loss_opacity + normal_loss + dist_loss + loss_size
            else:
                loss = loss_image + loss_opacity + normal_loss + dist_loss

            loss.backward()

            iter_end.record()

            with torch.no_grad():
                torch.cuda.synchronize()
                elapsed_ms = iter_start.elapsed_time(iter_end)

                if run_config.online_train:
                    visualization_start = time.perf_counter()
                    online_live_view = training_view
                    online_live_image = image
                    online_live_gt_image = gt_image
                    if rerun_logger.should_log_online_live(iteration, online_schedule_changed):
                        active_window = scene.getActiveTrainWindow()
                        latest_active_view = active_window[-1] if active_window else viewpoint_cam
                        if latest_active_view is not viewpoint_cam:
                            online_live_view, online_live_gt_image, _ = _training_view_for_iteration(
                                latest_active_view,
                                opt,
                                iteration,
                            )
                            online_live_image = render(online_live_view, triangles, pipe, bg)["render"]

                    rerun_logger.log_online_iteration(
                        iteration=iteration,
                        scene=scene,
                        current_view=online_live_view,
                        total_loss=loss.item(),
                        pixel_loss=pixel_loss.item(),
                        elapsed_ms=elapsed_ms,
                        total_triangles=scene.triangles.get_triangles_points.shape[0],
                        triangles_points=scene.triangles.get_triangles_points,
                        features_dc=scene.triangles._features_dc,
                        opacity=scene.triangles.get_opacity,
                        render_image=online_live_image,
                        gt_image=online_live_gt_image,
                        schedule_changed=online_schedule_changed,
                    )
                    if rerun_logger.enabled:
                        visualization_ms = (time.perf_counter() - visualization_start) * 1000.0
                        rerun_logger.log_scalar("online/visualization_ms", "iteration", iteration, visualization_ms)
                else:
                    visualization_start = time.perf_counter()
                    rerun_logger.log_training_iteration(
                        iteration=iteration,
                        total_loss=loss.item(),
                        pixel_loss=pixel_loss.item(),
                        elapsed_ms=elapsed_ms,
                        total_triangles=scene.triangles.get_triangles_points.shape[0],
                        triangles_points=scene.triangles.get_triangles_points,
                        features_dc=scene.triangles._features_dc,
                        opacity=scene.triangles.get_opacity,
                        render_image=image,
                        gt_image=gt_image,
                    )
                    if rerun_logger.enabled:
                        visualization_ms = (time.perf_counter() - visualization_start) * 1000.0
                        rerun_logger.log_scalar("training/visualization_ms", "iteration", iteration, visualization_ms)

                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    postfix = {"Loss": f"{ema_loss_for_log:.5f}"}
                    if getattr(opt, "pyramid_training", False):
                        postfix["Pyramid"] = pyramid_level
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if effective_final_iteration is not None and iteration == effective_final_iteration:
                    progress_bar.close()
                if tb_writer and getattr(opt, "pyramid_training", False):
                    tb_writer.add_scalar("training/pyramid_level", pyramid_level, iteration)

                training_report(
                    tb_writer,
                    iteration,
                    pixel_loss,
                    loss,
                    loss_fn,
                    elapsed_ms,
                    testing_iterations,
                    scene,
                    render,
                    (pipe, background),
                    lpips_fn,
                    rerun_logger,
                )
                if iteration in save_iterations:
                    print(f"\n[ITER {iteration}] Saving Triangles")
                    scene.save(iteration)
                if iteration % 1000 == 0:
                    total_dead = 0

                active_train_views = scene.getActiveTrainCameraCount() if run_config.online_train else number_of_views
                allow_online_pruning = (not run_config.online_train) or (
                    active_train_views >= run_config.online_train_min_prune_cameras
                )

                if (
                    iteration < opt.densify_until_iter
                    and iteration % opt.densification_interval == 0
                    and iteration > opt.densify_from_iter
                ):
                    if not allow_online_pruning:
                        print(
                            f"[ITER {iteration}] Skipping densification/pruning until "
                            f"{run_config.online_train_min_prune_cameras} train cameras are active "
                            f"(currently {active_train_views})"
                        )
                        continue

                    if number_of_views < 250:
                        dead_mask = torch.logical_or(
                            (triangles.importance_score < opt.importance_threshold).squeeze(),
                            (triangles.get_opacity <= opt.opacity_dead).squeeze(),
                        )
                    else:
                        if not new_round:
                            dead_mask = torch.logical_or(
                                (triangles.importance_score < opt.importance_threshold).squeeze(),
                                (triangles.get_opacity <= opt.opacity_dead).squeeze(),
                            )
                        else:
                            dead_mask = (triangles.get_opacity <= opt.opacity_dead).squeeze()

                    if iteration > 1000 and not new_round:
                        mask_test = triangles.triangle_area < 2
                        dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())

                        if not outdoor:
                            mask_test = triangles.image_size > 1400
                            dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())

                    total_dead += dead_mask.sum()

                    if opt.proba_distr == 0:
                        odd_group = True
                    elif opt.proba_distr == 1:
                        odd_group = False
                    else:
                        if opacity_now:
                            odd_group = opacity_now
                            opacity_now = False
                        else:
                            odd_group = opacity_now
                            opacity_now = True

                    removed_them = True
                    new_round = False

                    triangles.add_new_gs(cap_max=opt.max_shapes, oddGroup=odd_group, dead_mask=dead_mask)

                if iteration > opt.densify_until_iter and iteration % opt.densification_interval == 0:
                    if not allow_online_pruning:
                        print(
                            f"[ITER {iteration}] Skipping final pruning until "
                            f"{run_config.online_train_min_prune_cameras} train cameras are active "
                            f"(currently {active_train_views})"
                        )
                        continue

                    if number_of_views < 250:
                        dead_mask = torch.logical_or(
                            (triangles.importance_score < opt.importance_threshold).squeeze(),
                            (triangles.get_opacity <= opt.opacity_dead).squeeze(),
                        )
                    else:
                        if not new_round:
                            dead_mask = torch.logical_or(
                                (triangles.importance_score < opt.importance_threshold).squeeze(),
                                (triangles.get_opacity <= opt.opacity_dead).squeeze(),
                            )
                        else:
                            dead_mask = (triangles.get_opacity <= opt.opacity_dead).squeeze()

                    if not new_round:
                        mask_test = triangles.triangle_area < 2
                        dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                    triangles.remove_final_points(dead_mask)
                    removed_them = True
                    new_round = False

                if effective_final_iteration is None or iteration < effective_final_iteration:
                    triangles.optimizer.step()
                    triangles.optimizer.zero_grad(set_to_none=True)
    except KeyboardInterrupt:
        if not run_config.online_train:
            raise
        print("\nOnline training interrupted by user.")
        if run_config.online_train and last_iteration > 0:
            print(f"[ITER {last_iteration}] Saving Triangles before exit")
            scene.save(last_iteration)
    finally:
        if "progress_bar" in locals():
            progress_bar.close()
        rerun_logger.close()

    print("Training is done")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)

    print("Tensorboard not available: not logging progress")
    return None


def training_report(
    tb_writer,
    iteration,
    pixel_loss,
    loss,
    loss_fn,
    elapsed,
    testing_iterations,
    scene,
    render_func,
    render_args,
    lpips_fn,
    rerun_logger=None,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/pixel_loss", pixel_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                total_time = 0.0
                image = None
                gt_image = None

                for idx, viewpoint in enumerate(config["cameras"]):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    image = torch.clamp(render_func(viewpoint, scene.triangles, *render_args)["render"], 0.0, 1.0)
                    end_event.record()
                    torch.cuda.synchronize()
                    total_time += start_event.elapsed_time(end_event)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and idx < 5:
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}/render",
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                                gt_image[None],
                                global_step=iteration,
                            )
                    pixel_loss_test += loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()

                psnr_test /= len(config["cameras"])
                pixel_loss_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                lpips_test /= len(config["cameras"])
                total_time /= len(config["cameras"])
                print(
                    f"\n[ITER {iteration}] Evaluating {config['name']}: "
                    f"L1 {pixel_loss_test} PSNR {psnr_test} SSIM {ssim_test} LPIPS {lpips_test}"
                )
                if rerun_logger is not None:
                    rerun_logger.log_validation_iteration(
                        iteration=iteration,
                        split_name=config["name"],
                        l1_loss=float(pixel_loss_test),
                        psnr=float(psnr_test),
                        ssim=float(ssim_test),
                        lpips=float(lpips_test),
                        render_image=image,
                        gt_image=gt_image,
                    )

                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", pixel_loss_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.triangles.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.triangles.get_triangles_points.shape[0], iteration)
        torch.cuda.empty_cache()
