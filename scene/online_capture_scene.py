from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.kimera_capture import (
    TriangleScaleOptimizationConfig,
    _TriangleFrameScaleOptimizer,
    _adaptive_mesh_triangles,
    _camera_rt_from_capture_pose,
    _iter_capture_rows,
    _load_euroc_calibration,
    _load_mesh_geometry,
    _load_texture_image,
    _load_undistorted_rgb_image,
    _load_frame_rows,
    _scale_triangles_about_centroids,
    _subdivide_triangles_by_max_edge,
    _triangle_normals,
    _triangle_seed_colors,
    _training_frame_rows,
)


@dataclass(frozen=True)
class OnlineMeshSeedConfig:
    triangle_max_edge: float
    triangle_max_count: int
    triangle_color_source: str
    triangle_merge_mode: str
    triangle_merge_voxel_size: float
    triangle_merge_normal_bins: int
    triangle_scale: float
    triangle_sizing_mode: str
    adaptive_min_edge: float
    adaptive_max_edge: float
    adaptive_color_threshold: float
    adaptive_normal_threshold_deg: float
    adaptive_plane_threshold: float
    adaptive_max_patch_diameter: float


@dataclass(frozen=True)
class OnlineMeshSeed:
    time_ns: int
    obj_filename: str
    triangles: np.ndarray
    colors: np.ndarray
    input_triangle_count: int
    output_triangle_count: int
    accepted_triangle_count: int
    scale_optimization: dict[str, object] | None = None


def _capture_pose_radius(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 1.0
    centers = np.stack([row["_camera_pose_world"][:3, 3] for row in rows], axis=0).astype(np.float64)
    center = centers.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(centers - center, axis=1).max() * 1.1)
    return max(radius, 1.0)


def _resolve_resolution(width: int, height: int, resolution_arg: int | float) -> tuple[tuple[int, int], float, float]:
    if resolution_arg in (1, 2, 4, 8):
        scale = float(resolution_arg)
    elif resolution_arg == -1:
        scale = width / 1600.0 if width > 1600 else 1.0
    else:
        scale = width / float(resolution_arg)
    out_width = max(int(width / scale), 1)
    out_height = max(int(height / scale), 1)
    return (out_width, out_height), width / out_width, height / out_height


class OnlineMeshSeedGenerator:
    def __init__(
        self,
        capture_dir: Path,
        config: OnlineMeshSeedConfig,
        scale_optimizer: _TriangleFrameScaleOptimizer | None,
    ) -> None:
        self.capture_dir = capture_dir
        self.config = config
        self.scale_optimizer = scale_optimizer
        self._seen_voxel_keys: set[tuple[int, ...]] = set()

    def _filter_incremental_triangles(
        self,
        triangles: np.ndarray,
        colors: np.ndarray,
        texture_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            self.config.triangle_merge_mode != "voxel"
            or self.config.triangle_merge_voxel_size <= 0.0
            or len(triangles) == 0
        ):
            return triangles, colors, texture_mask

        centroids = triangles.mean(axis=1)
        keys = [np.floor(centroids / self.config.triangle_merge_voxel_size).astype(np.int64)]
        if self.config.triangle_merge_normal_bins > 0:
            normals = np.clip(_triangle_normals(triangles), -1.0, 1.0)
            normal_keys = np.floor((normals + 1.0) * self.config.triangle_merge_normal_bins).astype(np.int64)
            normal_keys = np.clip(normal_keys, 0, self.config.triangle_merge_normal_bins * 2 - 1)
            keys.append(normal_keys)
        voxel_keys = np.concatenate(keys, axis=1)

        keep = np.zeros((len(triangles),), dtype=bool)
        for idx, key in enumerate(voxel_keys):
            key_tuple = tuple(int(value) for value in key)
            if key_tuple in self._seen_voxel_keys:
                continue
            self._seen_voxel_keys.add(key_tuple)
            keep[idx] = True
        return triangles[keep], colors[keep], texture_mask[keep]

    def generate(self, row: dict[str, str]) -> OnlineMeshSeed | None:
        obj_filename = row.get("obj_filename", "").strip()
        if not obj_filename:
            return None
        mesh_path = self.capture_dir / "meshes" / obj_filename
        if not mesh_path.exists():
            return None

        mesh = _load_mesh_geometry(mesh_path, row.get("texture_filename", "").strip() or None)
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None

        mesh_triangles = mesh.vertices[mesh.faces]
        input_triangle_count = int(len(mesh_triangles))
        uv_triangles = None
        if len(mesh.texcoords) > 0 and mesh.face_texcoords.shape == mesh.faces.shape:
            uv_triangles = np.full((len(mesh.faces), 3, 2), np.nan, dtype=np.float32)
            valid_uv_mask = mesh.face_texcoords >= 0
            if np.any(valid_uv_mask):
                uv_triangles[valid_uv_mask] = mesh.texcoords[mesh.face_texcoords[valid_uv_mask]]

        vertex_color_triangles = None
        if self.config.triangle_color_source in ("texture", "vertex") and len(mesh.vertex_colors) == len(mesh.vertices):
            vertex_color_triangles = mesh.vertex_colors[mesh.faces]

        texture = _load_texture_image(mesh.texture_path) if self.config.triangle_color_source == "texture" else None
        if self.config.triangle_sizing_mode == "uniform":
            mesh_triangles, subdivided_attributes, _cap_reached = _subdivide_triangles_by_max_edge(
                mesh_triangles,
                max_edge=self.config.triangle_max_edge,
                max_count=self.config.triangle_max_count,
                vertex_attributes=[uv_triangles, vertex_color_triangles],
            )
            uv_triangles, vertex_color_triangles = subdivided_attributes
            if self.config.triangle_color_source == "gray":
                vertex_color_triangles = None
            colors, texture_mask, _color_source = _triangle_seed_colors(
                len(mesh_triangles),
                uv_triangles,
                texture,
                vertex_color_triangles,
            )
        else:
            mesh_triangles, colors, texture_mask, _adaptive_stats, _cap_reached = _adaptive_mesh_triangles(
                mesh_triangles,
                mesh.faces,
                uv_triangles,
                vertex_color_triangles,
                texture,
                self.config.triangle_color_source,
                min_edge=self.config.adaptive_min_edge,
                max_edge=self.config.adaptive_max_edge,
                color_threshold=self.config.adaptive_color_threshold,
                normal_threshold_deg=self.config.adaptive_normal_threshold_deg,
                plane_threshold=self.config.adaptive_plane_threshold,
                max_patch_diameter=self.config.adaptive_max_patch_diameter,
                max_count=self.config.triangle_max_count,
            )

        mesh_timestamp = row.get("mesh_timestamp_ns", "").strip() or row.get("pose_timestamp_ns", "").strip()
        mesh_time_ns = int(mesh_timestamp) if mesh_timestamp else 0
        scale_result = None
        if self.scale_optimizer is not None:
            mesh_triangles, scale_result = self.scale_optimizer.optimize(
                mesh_triangles,
                colors,
                mesh_time_ns,
                obj_filename,
            )
        else:
            mesh_triangles = _scale_triangles_about_centroids(mesh_triangles, self.config.triangle_scale)

        output_triangle_count = int(len(mesh_triangles))
        mesh_triangles, colors, texture_mask = self._filter_incremental_triangles(
            mesh_triangles,
            colors,
            texture_mask,
        )
        if len(mesh_triangles) == 0:
            return None

        return OnlineMeshSeed(
            time_ns=mesh_time_ns,
            obj_filename=obj_filename,
            triangles=mesh_triangles.astype(np.float32, copy=False),
            colors=np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False),
            input_triangle_count=input_triangle_count,
            output_triangle_count=output_triangle_count,
            accepted_triangle_count=int(len(mesh_triangles)),
            scale_optimization=scale_result,
        )


class OnlineKimeraCaptureScene:
    def __init__(
        self,
        args,
        triangles,
        init_opacity: float,
        init_size: float,
        nb_points: int,
        set_sigma: float,
        no_dome: bool,
    ) -> None:
        self.model_path = args.model_path
        self.source_path = getattr(args, "source_path", "")
        self.triangles = triangles
        self.loaded_iter = None

        self.capture_dir = Path(args.online_capture_dir or getattr(args, "capture_dir", "")).expanduser()
        self.mav0_dir = Path(args.online_mav0_dir or getattr(args, "mav0_dir", "")).expanduser()
        self.camera_name = args.online_camera
        self._args_resolution = getattr(args, "resolution", -1)
        self._args_data_device = getattr(args, "data_device", "cuda")
        self.online_train_enabled = False
        self.online_train_initial_count = 0
        self.online_train_growth_interval = 0
        self.online_train_growth_count = 1
        self.online_train_window_size = 0
        self.online_train_count = 0
        self._pending_seed_soups: list[OnlineMeshSeed] = []
        self._last_seed_stats: list[OnlineMeshSeed] = []

        self.calibration = _load_euroc_calibration(self.mav0_dir, self.camera_name)
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.calibration.camera_matrix,
            self.calibration.distortion,
            (self.calibration.width, self.calibration.height),
            0.0,
            (self.calibration.width, self.calibration.height),
            centerPrincipalPoint=True,
        )
        self.fx = float(self.new_camera_matrix[0, 0])
        self.fy = float(self.new_camera_matrix[1, 1])

        frame_rows = _load_frame_rows(self.capture_dir, self.camera_name)
        if int(args.online_max_frames) > 0:
            frame_rows = frame_rows[: int(args.online_max_frames)]
        if not frame_rows:
            raise RuntimeError(f"No online capture frames found in {self.capture_dir}")
        self._all_frame_rows = frame_rows
        hold = int(args.online_eval_hold)
        self._train_rows = _training_frame_rows(frame_rows, hold)
        self._test_rows = [row for idx, row in enumerate(frame_rows) if hold > 0 and idx % hold == 0]
        if not self._train_rows:
            raise RuntimeError("Online capture mode requires at least one training frame after eval holdout split.")

        meshes_csv = self.capture_dir / "meshes.csv"
        self._mesh_rows = sorted(
            [row for row in _iter_capture_rows(meshes_csv) if row.get("obj_filename", "").strip()],
            key=lambda row: int(row.get("mesh_timestamp_ns", "").strip() or row.get("pose_timestamp_ns", "0").strip() or "0"),
        )
        self._mesh_index = 0
        scale_optimizer = None
        if bool(args.online_mesh_triangle_scale_optimize):
            scale_optimizer = _TriangleFrameScaleOptimizer(
                TriangleScaleOptimizationConfig(
                    enabled=True,
                    iterations=int(args.online_mesh_triangle_scale_opt_iterations),
                    lr=float(args.online_mesh_triangle_scale_opt_lr),
                    min_scale=float(args.online_mesh_triangle_scale_opt_min),
                    max_scale=float(args.online_mesh_triangle_scale_opt_max),
                    resolution=int(args.online_mesh_triangle_scale_opt_resolution),
                    initial_scale=float(args.online_mesh_triangle_scale),
                    opacity=float(init_opacity),
                    sigma=float(set_sigma),
                    sh_degree=int(args.sh_degree),
                ),
                self._train_rows,
                self.calibration,
                self.new_camera_matrix,
                self.fx,
                self.fy,
            )
            scale_optimizer._load_imports()
        self._seed_generator = OnlineMeshSeedGenerator(
            self.capture_dir,
            OnlineMeshSeedConfig(
                triangle_max_edge=float(args.online_mesh_triangle_max_edge),
                triangle_max_count=int(args.online_mesh_triangle_max_count),
                triangle_color_source=args.online_mesh_triangle_color_source,
                triangle_merge_mode=args.online_mesh_triangle_merge_mode,
                triangle_merge_voxel_size=float(args.online_mesh_triangle_merge_voxel_size),
                triangle_merge_normal_bins=int(args.online_mesh_triangle_merge_normal_bins),
                triangle_scale=float(args.online_mesh_triangle_scale),
                triangle_sizing_mode=args.online_mesh_triangle_sizing_mode,
                adaptive_min_edge=float(args.online_mesh_triangle_adaptive_min_edge),
                adaptive_max_edge=float(args.online_mesh_triangle_adaptive_max_edge),
                adaptive_color_threshold=float(args.online_mesh_triangle_adaptive_color_threshold),
                adaptive_normal_threshold_deg=float(args.online_mesh_triangle_adaptive_normal_threshold_deg),
                adaptive_plane_threshold=float(args.online_mesh_triangle_adaptive_plane_threshold),
                adaptive_max_patch_diameter=float(args.online_mesh_triangle_adaptive_max_patch_diameter),
            ),
            scale_optimizer,
        )

        self.cameras_extent = _capture_pose_radius(self._train_rows)
        self.train_cameras = {1.0: []}
        self.test_cameras = {1.0: [self._build_camera(row, idx) for idx, row in enumerate(self._test_rows)]}

        initial_count = min(max(int(args.online_train_initial_cameras), 1), len(self._train_rows))
        self._reveal_to_count(initial_count)
        initial_soup = self.consume_pending_seed_soup()
        if initial_soup is None:
            raise RuntimeError(
                "Online capture mode could not generate initial Kimera mesh triangles from the first revealed frame."
            )
        triangles.create_from_triangle_soup(
            initial_soup[0],
            initial_soup[1],
            self.cameras_extent,
            init_opacity,
            init_size,
            nb_points,
            set_sigma,
            no_dome,
        )

    def _build_camera(self, row: dict[str, str], uid: int) -> Camera:
        image = _load_undistorted_rgb_image(
            Path(row["_image_path"]),
            self.calibration,
            self.new_camera_matrix,
        )
        if image is None:
            raise RuntimeError(f"Failed to load online capture image {row['_image_path']}")
        pil_image = Image.fromarray(image)
        resolution, scale_x, scale_y = _resolve_resolution(image.shape[1], image.shape[0], self._resolution_arg)
        image_tensor = PILtoTorch(pil_image, resolution)
        fx = self.fx / scale_x
        fy = self.fy / scale_y
        fovx = 2.0 * math.atan(resolution[0] / (2.0 * fx))
        fovy = 2.0 * math.atan(resolution[1] / (2.0 * fy))
        rotation, translation = _camera_rt_from_capture_pose(row["_camera_pose_world"])
        camera = Camera(
            colmap_id=int(row["image_timestamp_ns"]),
            R=rotation,
            T=translation,
            FoVx=fovx,
            FoVy=fovy,
            image=image_tensor,
            gt_alpha_mask=None,
            image_name=Path(row["_image_path"]).stem,
            uid=uid,
            data_device=self._data_device,
        )
        camera.time_ns = int(row["image_timestamp_ns"])
        return camera

    @property
    def _resolution_arg(self):
        return getattr(self, "_args_resolution", -1)

    @property
    def _data_device(self):
        return getattr(self, "_args_data_device", "cuda")

    def enable_online_train_schedule(self, initial_count, growth_interval, growth_count, window_size=0):
        if initial_count <= 0:
            raise ValueError(f"initial_count must be > 0, got {initial_count}")
        if growth_interval <= 0:
            raise ValueError(f"growth_interval must be > 0, got {growth_interval}")
        if growth_count <= 0:
            raise ValueError(f"growth_count must be > 0, got {growth_count}")
        if window_size < 0:
            raise ValueError(f"window_size must be >= 0, got {window_size}")
        self.online_train_enabled = True
        self.online_train_initial_count = int(initial_count)
        self.online_train_growth_interval = int(growth_interval)
        self.online_train_growth_count = int(growth_count)
        self.online_train_window_size = int(window_size)
        target = min(self.online_train_initial_count, len(self._train_rows))
        if target > self.online_train_count:
            self._reveal_to_count(target)

    def _reveal_to_count(self, target_count: int) -> None:
        target_count = min(max(int(target_count), 0), len(self._train_rows))
        while self.online_train_count < target_count:
            row = self._train_rows[self.online_train_count]
            camera = self._build_camera(row, self.online_train_count)
            self.train_cameras[1.0].append(camera)
            self.online_train_count += 1
            self._generate_meshes_until(int(row["image_timestamp_ns"]))

    def _generate_meshes_until(self, time_ns: int) -> None:
        while self._mesh_index < len(self._mesh_rows):
            row = self._mesh_rows[self._mesh_index]
            mesh_time = int(row.get("mesh_timestamp_ns", "").strip() or row.get("pose_timestamp_ns", "0").strip() or "0")
            if mesh_time > time_ns:
                break
            self._mesh_index += 1
            seed = self._seed_generator.generate(row)
            if seed is not None:
                self._pending_seed_soups.append(seed)

    def consume_pending_seed_soup(self):
        self._last_seed_stats = list(self._pending_seed_soups)
        if not self._pending_seed_soups:
            return None
        triangles = np.concatenate([seed.triangles for seed in self._pending_seed_soups], axis=0)
        colors = np.concatenate([seed.colors for seed in self._pending_seed_soups], axis=0)
        self._pending_seed_soups.clear()
        return triangles, colors

    def getLastOnlineSeedStats(self) -> list[OnlineMeshSeed]:
        return list(self._last_seed_stats)

    def update_online_train_set(self, iteration):
        if not self.online_train_enabled:
            return False
        steps = max(iteration - 1, 0) // self.online_train_growth_interval
        target_count = min(
            len(self._train_rows),
            self.online_train_initial_count + steps * self.online_train_growth_count,
        )
        if target_count == self.online_train_count:
            return False
        self._reveal_to_count(target_count)
        return True

    def getActiveTrainCameraCount(self, scale=1.0):
        return self.online_train_count

    def getActiveTrainWindowCount(self, scale=1.0):
        return len(self.getActiveTrainWindow(scale))

    def getTotalTrainCameraCount(self, scale=1.0):
        return len(self._train_rows)

    def getActiveTrainWindowStart(self, scale=1.0):
        if not self.online_train_enabled or self.online_train_window_size <= 0:
            return 0
        return max(0, self.online_train_count - self.online_train_window_size)

    def getAllTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getRevealedTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getNewlyRevealedTrainCameras(self, previous_count, scale=1.0):
        if previous_count >= self.online_train_count:
            return []
        return self.train_cameras[scale][previous_count:self.online_train_count]

    def getActiveTrainWindow(self, scale=1.0):
        window_end = self.online_train_count
        window_start = self.getActiveTrainWindowStart(scale)
        return self.train_cameras[scale][window_start:window_end]

    def getTrainCameras(self, scale=1.0):
        return self.getActiveTrainWindow(scale)

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def save(self, iteration):
        point_cloud_path = Path(self.model_path) / f"point_cloud/iteration_{iteration}"
        self.triangles.save(str(point_cloud_path))
