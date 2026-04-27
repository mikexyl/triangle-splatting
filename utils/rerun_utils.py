from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.sh_utils import SH2RGB


def _torch_image_to_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().clamp(0.0, 1.0)
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3:
            image = image.permute(1, 2, 0)
        image_np = image.cpu().numpy()
    else:
        image_np = np.asarray(image)

    if image_np.dtype != np.uint8:
        if np.issubdtype(image_np.dtype, np.floating):
            image_np = (255.0 * np.clip(image_np, 0.0, 1.0)).round().astype(np.uint8)
        else:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    return image_np


def _sanitize_entity_name(name: str) -> str:
    sanitized = []
    for char in name:
        if char.isalnum() or char in ("-", "_"):
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_") or "camera"


def _view_world_from_camera(view: Any) -> tuple[np.ndarray, np.ndarray]:
    # Camera.R is stored as the camera-to-world rotation in the repo's COLMAP-style
    # camera convention: +X right, +Y down, +Z forward. That matches Rerun's RDF.
    if hasattr(view, "R") and hasattr(view, "camera_center"):
        rotation = np.asarray(view.R, dtype=np.float32)
        translation = np.asarray(view.camera_center.detach().cpu().numpy(), dtype=np.float32)
        return rotation, translation

    world_from_camera = torch.linalg.inv(view.world_view_transform.T).detach().cpu().numpy().astype(np.float32)
    return world_from_camera[:3, :3], world_from_camera[:3, 3]


def _camera_positions(views: list[Any] | tuple[Any, ...]) -> np.ndarray:
    if not views:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack([_view_world_from_camera(view)[1] for view in views], axis=0).astype(np.float32)


@dataclass
class RerunConfig:
    enabled: bool
    spawn: bool
    save_path: str | None
    max_triangles: int | None
    mesh_every: int
    image_every: int


def create_rerun_config(args: Any) -> RerunConfig:
    max_triangles = int(getattr(args, "rerun_max_triangles", 5000))
    return RerunConfig(
        enabled=bool(getattr(args, "rerun", False)),
        spawn=bool(getattr(args, "rerun_spawn", False)),
        save_path=getattr(args, "rerun_save", None),
        max_triangles=None if max_triangles <= 0 else max_triangles,
        mesh_every=max(1, int(getattr(args, "rerun_mesh_every", 100))),
        image_every=max(1, int(getattr(args, "rerun_image_every", 25))),
    )


class RerunLogger:
    def __init__(self, app_id: str, config: RerunConfig):
        self.app_id = app_id
        self.config = config
        self.enabled = config.enabled
        self._rr = None
        self._scalar_ctor = None
        self._default_blueprint = None

        if not self.enabled:
            return

        try:
            import rerun as rr
        except ImportError as exc:
            raise RuntimeError(
                "Rerun logging was requested, but the rerun Python SDK is not installed."
            ) from exc

        self._rr = rr
        self._scalar_ctor = getattr(rr, "Scalars", None) or getattr(rr, "Scalar")
        rr.init(app_id, spawn=config.spawn)
        self._default_blueprint = self._make_default_blueprint()
        if self._default_blueprint is not None and hasattr(rr, "send_blueprint"):
            rr.send_blueprint(self._default_blueprint, make_active=True, make_default=True)

    def close(self) -> None:
        if not self.enabled or not self.config.save_path:
            return

        save_path = Path(self.config.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._rr.save(str(save_path), default_blueprint=self._default_blueprint)

    def _make_default_blueprint(self) -> Any | None:
        try:
            import rerun.blueprint as rrb
        except ImportError:
            return None

        if self.app_id.endswith(".train"):
            return rrb.Blueprint(
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial3DView(
                            name="Triangles",
                            origin="/training/live/triangles",
                        ),
                        rrb.Vertical(
                            rrb.Spatial2DView(
                                name="Render",
                                origin="/training/live/render",
                            ),
                            rrb.Spatial2DView(
                                name="Ground Truth",
                                origin="/training/live/ground_truth",
                            ),
                        ),
                    ),
                    rrb.Horizontal(
                        rrb.TimeSeriesView(
                            name="Loss",
                            origin="/training/loss",
                        ),
                        rrb.TimeSeriesView(
                            name="Runtime",
                            origin="/training",
                            contents=[
                                "/training/runtime_ms",
                                "/training/visualization_ms",
                                "/training/state/triangle_count",
                            ],
                        ),
                        rrb.TimeSeriesView(
                            name="Validation",
                            origin="/validation",
                        ),
                    ),
                    rrb.Tabs(
                        rrb.Horizontal(
                            rrb.Spatial2DView(
                                name="Test Render",
                                origin="/validation/test/render",
                            ),
                            rrb.Spatial2DView(
                                name="Test Ground Truth",
                                origin="/validation/test/ground_truth",
                            ),
                            name="Validation Test",
                        ),
                        rrb.Horizontal(
                            rrb.Spatial2DView(
                                name="Train Render",
                                origin="/validation/train/render",
                            ),
                            rrb.Spatial2DView(
                                name="Train Ground Truth",
                                origin="/validation/train/ground_truth",
                            ),
                            name="Validation Train",
                        ),
                        active_tab=0,
                    ),
                ),
                collapse_panels=True,
            )

        if self.app_id.endswith(".online"):
            return rrb.Blueprint(
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial3DView(
                            name="Online Replay",
                            origin="/online",
                            contents=[
                                "/online/live/triangles",
                                "/online/scene/**",
                                "/online/state/revealed/**",
                                "/online/state/active_window/**",
                                "/online/live/current_camera",
                            ],
                        ),
                        rrb.Vertical(
                            rrb.Spatial2DView(
                                name="Current Render",
                                origin="/online/live/render",
                            ),
                            rrb.Spatial2DView(
                                name="Current Ground Truth",
                                origin="/online/live/ground_truth",
                            ),
                        ),
                    ),
                    rrb.Horizontal(
                        rrb.TimeSeriesView(
                            name="Loss",
                            origin="/online/loss",
                        ),
                        rrb.TimeSeriesView(
                            name="Online State",
                            origin="/online/state",
                            contents=[
                                "/online/runtime_ms",
                                "/online/visualization_ms",
                                "/online/state/triangle_count",
                                "/online/state/revealed_count",
                                "/online/state/window_count",
                                "/online/state/window_start",
                            ],
                        ),
                        rrb.TimeSeriesView(
                            name="Validation",
                            origin="/validation",
                        ),
                    ),
                    rrb.Tabs(
                        rrb.Horizontal(
                            rrb.Spatial2DView(
                                name="Test Render",
                                origin="/validation/test/render",
                            ),
                            rrb.Spatial2DView(
                                name="Test Ground Truth",
                                origin="/validation/test/ground_truth",
                            ),
                            name="Validation Test",
                        ),
                        rrb.Horizontal(
                            rrb.Spatial2DView(
                                name="Train Render",
                                origin="/validation/train/render",
                            ),
                            rrb.Spatial2DView(
                                name="Train Ground Truth",
                                origin="/validation/train/ground_truth",
                            ),
                            name="Validation Train",
                        ),
                        active_tab=0,
                    ),
                ),
                collapse_panels=True,
            )

        if self.app_id.endswith(".render"):
            return rrb.Blueprint(
                rrb.Tabs(
                    rrb.Horizontal(
                        rrb.Spatial3DView(
                            name="Train Triangles",
                            origin="/render/train",
                            contents=[
                                "/render/train/triangles",
                                "/render/train/cameras/**",
                            ],
                        ),
                        rrb.Vertical(
                            rrb.Spatial2DView(
                                name="Train Render",
                                origin="/render/train/render",
                            ),
                            rrb.Spatial2DView(
                                name="Train Ground Truth",
                                origin="/render/train/ground_truth",
                            ),
                        ),
                        name="Train",
                    ),
                    rrb.Horizontal(
                        rrb.Spatial3DView(
                            name="Test Triangles",
                            origin="/render/test",
                            contents=[
                                "/render/test/triangles",
                                "/render/test/cameras/**",
                            ],
                        ),
                        rrb.Vertical(
                            rrb.Spatial2DView(
                                name="Test Render",
                                origin="/render/test/render",
                            ),
                            rrb.Spatial2DView(
                                name="Test Ground Truth",
                                origin="/render/test/ground_truth",
                            ),
                        ),
                        name="Test",
                    ),
                    active_tab=1,
                ),
                collapse_panels=True,
            )

        return None

    def _set_time(self, timeline: str, value: int) -> None:
        if hasattr(self._rr, "set_time_sequence"):
            self._rr.set_time_sequence(timeline, value)
        else:
            self._rr.set_time(timeline, sequence=value)

    def log_scalar(self, path: str, timeline: str, step: int, value: float) -> None:
        if not self.enabled:
            return
        self._set_time(timeline, step)
        self._rr.log(path, self._scalar_ctor(float(value)))

    def log_image(self, path: str, timeline: str, step: int, image: torch.Tensor | np.ndarray) -> None:
        if not self.enabled:
            return
        self._set_time(timeline, step)
        self._rr.log(path, self._rr.Image(_torch_image_to_uint8(image)))

    def _log_pinhole_camera(
        self,
        path: str,
        view: Any,
        color: list[int] | tuple[int, ...],
        *,
        static: bool = False,
        timeline: str | None = None,
        step: int | None = None,
    ) -> None:
        if not self.enabled:
            return

        if timeline is not None and step is not None:
            self._set_time(timeline, step)

        rotation, translation = _view_world_from_camera(view)
        width = float(view.image_width)
        height = float(view.image_height)
        focal_x = width / (2.0 * np.tan(float(view.FoVx) / 2.0))
        focal_y = height / (2.0 * np.tan(float(view.FoVy) / 2.0))
        principal_point = [width / 2.0, height / 2.0]
        image_from_camera = [
            [focal_x, 0.0, principal_point[0]],
            [0.0, focal_y, principal_point[1]],
            [0.0, 0.0, 1.0],
        ]

        self._rr.log(
            path,
            self._rr.Transform3D(
                translation=translation,
                mat3x3=rotation,
                relation=self._rr.TransformRelation.ParentFromChild,
            ),
            self._rr.Pinhole(
                image_from_camera=image_from_camera,
                resolution=[width, height],
                camera_xyz=self._rr.ViewCoordinates.RDF,
                image_plane_distance=1.0,
                color=list(color),
                line_width=0.01,
            ),
            static=static,
        )

    def _log_camera_points(
        self,
        path: str,
        views: list[Any] | tuple[Any, ...],
        color: list[int] | tuple[int, int, int],
        *,
        static: bool = False,
        timeline: str | None = None,
        step: int | None = None,
        radius: float = 0.03,
    ) -> None:
        if not self.enabled or not views:
            return

        if timeline is not None and step is not None:
            self._set_time(timeline, step)

        positions = _camera_positions(views)
        colors = np.tile(np.asarray(color, dtype=np.uint8), (positions.shape[0], 1))
        radii = np.full((positions.shape[0],), radius, dtype=np.float32)
        self._rr.log(
            path,
            self._rr.Points3D(positions=positions, colors=colors, radii=radii),
            static=static,
        )

    def _log_camera_path(
        self,
        path: str,
        views: list[Any] | tuple[Any, ...],
        color: list[int] | tuple[int, int, int],
        *,
        static: bool = False,
        timeline: str | None = None,
        step: int | None = None,
        radius: float = 0.01,
    ) -> None:
        if not self.enabled or len(views) < 2:
            return

        if timeline is not None and step is not None:
            self._set_time(timeline, step)

        positions = _camera_positions(views)
        colors = np.asarray([color], dtype=np.uint8)
        radii = np.asarray([radius], dtype=np.float32)
        self._rr.log(
            path,
            self._rr.LineStrips3D([positions], colors=colors, radii=radii),
            static=static,
        )

    def log_render_cameras(self, split_name: str, views: list[Any] | tuple[Any, ...]) -> None:
        if not self.enabled:
            return

        self._rr.log(f"render/{split_name}", self._rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        for idx, view in enumerate(views):
            image_name = getattr(view, "image_name", f"camera_{idx:03d}")
            camera_path = f"render/{split_name}/cameras/{idx:03d}_{_sanitize_entity_name(image_name)}"
            image_path = f"{camera_path}/image"
            self._log_pinhole_camera(camera_path, view, [255, 170, 40, 255], static=True)
            self._rr.log(
                image_path,
                self._rr.Image(_torch_image_to_uint8(view.original_image[0:3])),
                static=True,
            )

    def _log_mesh(
        self,
        path: str,
        timeline: str,
        step: int,
        triangles_points: torch.Tensor,
        features_dc: torch.Tensor,
        opacity: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return

        triangles = triangles_points.detach().cpu().numpy()
        triangle_colors = SH2RGB(features_dc.detach().squeeze(1).cpu().numpy())
        triangle_colors = np.clip(255.0 * triangle_colors, 0.0, 255.0).astype(np.uint8)
        triangle_alpha = opacity.detach().reshape(-1, 1).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
        triangle_rgba = np.concatenate([triangle_colors, triangle_alpha], axis=1)

        if self.config.max_triangles is not None and triangles.shape[0] > self.config.max_triangles:
            keep = np.linspace(0, triangles.shape[0] - 1, self.config.max_triangles, dtype=np.int64)
            triangles = triangles[keep]
            triangle_rgba = triangle_rgba[keep]

        vertex_positions = triangles.reshape(-1, 3)
        vertex_colors = np.repeat(triangle_rgba, 3, axis=0)
        triangle_indices = np.arange(vertex_positions.shape[0], dtype=np.uint32).reshape(-1, 3)

        self._set_time(timeline, step)
        try:
            mesh = self._rr.Mesh3D(
                vertex_positions=vertex_positions,
                triangle_indices=triangle_indices,
                vertex_colors=vertex_colors,
            )
        except TypeError:
            mesh = self._rr.Mesh3D(
                vertex_positions=vertex_positions,
                indices=triangle_indices,
                vertex_colors=vertex_colors,
            )
        self._rr.log(path, mesh)

    def log_training_iteration(
        self,
        iteration: int,
        total_loss: float,
        pixel_loss: float,
        elapsed_ms: float,
        total_triangles: int,
        triangles_points: torch.Tensor,
        features_dc: torch.Tensor,
        opacity: torch.Tensor,
        render_image: torch.Tensor,
        gt_image: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return

        self.log_scalar("training/loss/total", "iteration", iteration, total_loss)
        self.log_scalar("training/loss/pixel", "iteration", iteration, pixel_loss)
        self.log_scalar("training/runtime_ms", "iteration", iteration, elapsed_ms)
        self.log_scalar("training/state/triangle_count", "iteration", iteration, float(total_triangles))

        if iteration == 1 or iteration % self.config.image_every == 0:
            self.log_image("training/live/render", "iteration", iteration, render_image)
            self.log_image("training/live/ground_truth", "iteration", iteration, gt_image)

        if iteration == 1 or iteration % self.config.mesh_every == 0:
            self._log_mesh("training/live/triangles", "iteration", iteration, triangles_points, features_dc, opacity)

    def log_online_setup(
        self,
        train_views: list[Any] | tuple[Any, ...],
        test_views: list[Any] | tuple[Any, ...],
    ) -> None:
        if not self.enabled:
            return

        self._rr.log("online", self._rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._log_camera_points("online/scene/train_points", train_views, [90, 150, 255], static=True)
        self._log_camera_path("online/scene/train_path", train_views, [90, 150, 255], static=True)
        if test_views:
            self._log_camera_points("online/scene/test_points", test_views, [140, 140, 140], static=True)
            self._log_camera_path("online/scene/test_path", test_views, [140, 140, 140], static=True)

    def should_log_online_live(self, iteration: int, schedule_changed: bool = False) -> bool:
        if not self.enabled:
            return False
        return iteration == 1 or schedule_changed or iteration % self.config.image_every == 0

    def log_online_iteration(
        self,
        iteration: int,
        scene: Any,
        current_view: Any,
        total_loss: float,
        pixel_loss: float,
        elapsed_ms: float,
        total_triangles: int,
        triangles_points: torch.Tensor,
        features_dc: torch.Tensor,
        opacity: torch.Tensor,
        render_image: torch.Tensor,
        gt_image: torch.Tensor,
        schedule_changed: bool = False,
    ) -> None:
        if not self.enabled:
            return

        self.log_scalar("online/loss/total", "iteration", iteration, total_loss)
        self.log_scalar("online/loss/pixel", "iteration", iteration, pixel_loss)
        self.log_scalar("online/runtime_ms", "iteration", iteration, elapsed_ms)
        self.log_scalar("online/state/triangle_count", "iteration", iteration, float(total_triangles))
        self.log_scalar("online/state/revealed_count", "iteration", iteration, float(scene.getActiveTrainCameraCount()))
        self.log_scalar("online/state/window_count", "iteration", iteration, float(scene.getActiveTrainWindowCount()))
        self.log_scalar("online/state/window_start", "iteration", iteration, float(scene.getActiveTrainWindowStart()))

        if iteration == 1 or schedule_changed:
            revealed_views = scene.getRevealedTrainCameras()
            active_window_views = scene.getActiveTrainWindow()
            self._log_camera_points(
                "online/state/revealed/points",
                revealed_views,
                [70, 200, 120],
                timeline="iteration",
                step=iteration,
            )
            self._log_camera_path(
                "online/state/revealed/path",
                revealed_views,
                [70, 200, 120],
                timeline="iteration",
                step=iteration,
            )
            self._log_camera_points(
                "online/state/active_window/points",
                active_window_views,
                [255, 170, 40],
                timeline="iteration",
                step=iteration,
                radius=0.04,
            )
            self._log_camera_path(
                "online/state/active_window/path",
                active_window_views,
                [255, 170, 40],
                timeline="iteration",
                step=iteration,
                radius=0.015,
            )

        if self.should_log_online_live(iteration, schedule_changed):
            self.log_image("online/live/render", "iteration", iteration, render_image)
            self.log_image("online/live/ground_truth", "iteration", iteration, gt_image)
            self._log_pinhole_camera(
                "online/live/current_camera",
                current_view,
                [255, 220, 80, 255],
                timeline="iteration",
                step=iteration,
            )

        if iteration == 1 or schedule_changed or iteration % self.config.mesh_every == 0:
            self._log_mesh("online/live/triangles", "iteration", iteration, triangles_points, features_dc, opacity)

    def log_validation_iteration(
        self,
        iteration: int,
        split_name: str,
        l1_loss: float,
        psnr: float,
        ssim: float,
        lpips: float,
        render_image: torch.Tensor | None = None,
        gt_image: torch.Tensor | None = None,
    ) -> None:
        if not self.enabled:
            return

        base = f"validation/{split_name}"
        self.log_scalar(f"{base}/l1_loss", "iteration", iteration, l1_loss)
        self.log_scalar(f"{base}/psnr", "iteration", iteration, psnr)
        self.log_scalar(f"{base}/ssim", "iteration", iteration, ssim)
        self.log_scalar(f"{base}/lpips", "iteration", iteration, lpips)

        if render_image is not None and gt_image is not None:
            self.log_image(f"{base}/render", "iteration", iteration, render_image)
            self.log_image(f"{base}/ground_truth", "iteration", iteration, gt_image)

    def log_render_frame(
        self,
        split_name: str,
        frame_idx: int,
        render_image: torch.Tensor,
        gt_image: torch.Tensor,
        triangles_points: torch.Tensor | None = None,
        features_dc: torch.Tensor | None = None,
        opacity: torch.Tensor | None = None,
    ) -> None:
        if not self.enabled:
            return

        timeline = f"{split_name}_frame"
        self.log_image(f"render/{split_name}/render", timeline, frame_idx, render_image)
        self.log_image(f"render/{split_name}/ground_truth", timeline, frame_idx, gt_image)

        if frame_idx == 0 and triangles_points is not None and features_dc is not None and opacity is not None:
            self._log_mesh(f"render/{split_name}/triangles", timeline, frame_idx, triangles_points, features_dc, opacity)
