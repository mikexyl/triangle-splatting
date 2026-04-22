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


@dataclass
class RerunConfig:
    enabled: bool
    spawn: bool
    save_path: str | None
    max_triangles: int
    mesh_every: int
    image_every: int


def create_rerun_config(args: Any) -> RerunConfig:
    return RerunConfig(
        enabled=bool(getattr(args, "rerun", False)),
        spawn=bool(getattr(args, "rerun_spawn", False)),
        save_path=getattr(args, "rerun_save", None),
        max_triangles=int(getattr(args, "rerun_max_triangles", 5000)),
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

    def log_render_cameras(self, split_name: str, views: list[Any] | tuple[Any, ...]) -> None:
        if not self.enabled:
            return

        self._rr.log(f"render/{split_name}", self._rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        for idx, view in enumerate(views):
            image_name = getattr(view, "image_name", f"camera_{idx:03d}")
            camera_path = f"render/{split_name}/cameras/{idx:03d}_{_sanitize_entity_name(image_name)}"
            image_path = f"{camera_path}/image"
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
                camera_path,
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
                    color=[255, 170, 40, 255],
                    line_width=0.01,
                ),
                static=True,
            )
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

        if triangles.shape[0] > self.config.max_triangles:
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
