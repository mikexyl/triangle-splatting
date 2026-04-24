#!/usr/bin/env python3
"""Save a Rerun visualization of a COLMAP sparse reconstruction."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_colmap_loader():
    module_path = REPO_ROOT / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("_triangle_splatting_colmap_loader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load COLMAP helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COLMAP = _load_colmap_loader()


def _read_sparse_model(sparse_dir: Path):
    if (sparse_dir / "images.bin").exists():
        images = COLMAP.read_extrinsics_binary(str(sparse_dir / "images.bin"))
    else:
        images = COLMAP.read_extrinsics_text(str(sparse_dir / "images.txt"))

    if (sparse_dir / "cameras.bin").exists():
        cameras = COLMAP.read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
    else:
        cameras = COLMAP.read_intrinsics_text(str(sparse_dir / "cameras.txt"))

    if (sparse_dir / "points3D.bin").exists():
        points, colors, errors = COLMAP.read_points3D_binary(str(sparse_dir / "points3D.bin"))
    else:
        points, colors, errors = COLMAP.read_points3D_text(str(sparse_dir / "points3D.txt"))

    return images, cameras, points, colors, errors


def _camera_intrinsics(camera) -> tuple[float, float, float, float]:
    if camera.model == "SIMPLE_PINHOLE":
        focal, cx, cy = camera.params[:3]
        return float(focal), float(focal), float(cx), float(cy)
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    if camera.model == "OPENCV":
        fx, fy, cx, cy = camera.params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    raise ValueError(f"Unsupported camera model for visualization: {camera.model}")


def _camera_center_and_frustum(image, camera, scale: float) -> tuple[np.ndarray, list[np.ndarray]]:
    rotation_world_to_camera = COLMAP.qvec2rotmat(image.qvec)
    rotation_camera_to_world = rotation_world_to_camera.T
    center = -rotation_camera_to_world @ image.tvec
    fx, fy, cx, cy = _camera_intrinsics(camera)

    pixel_corners = np.array(
        [
            [0.0, 0.0],
            [float(camera.width), 0.0],
            [float(camera.width), float(camera.height)],
            [0.0, float(camera.height)],
        ],
        dtype=np.float64,
    )
    camera_corners = np.column_stack(
        [
            (pixel_corners[:, 0] - cx) / fx * scale,
            (pixel_corners[:, 1] - cy) / fy * scale,
            np.full(4, scale, dtype=np.float64),
        ]
    )
    world_corners = center[None, :] + camera_corners @ rotation_camera_to_world.T
    strips = [
        np.vstack([center, world_corners[0], world_corners[1], world_corners[2], world_corners[3], world_corners[0]]),
        np.vstack([center, world_corners[1]]),
        np.vstack([center, world_corners[2]]),
        np.vstack([center, world_corners[3]]),
    ]
    return center, strips


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a Rerun visualization of COLMAP sparse points and cameras")
    parser.add_argument("--sparse-dir", required=True, help="COLMAP sparse model directory, usually dataset/sparse/0")
    parser.add_argument("--output", required=True, help="Path to write the .rrd recording")
    parser.add_argument("--frustum-scale", type=float, default=0.15, help="Depth of camera frustums in COLMAP units")
    parser.add_argument("--max-points", type=int, default=250000, help="Maximum sparse points to log; 0 logs all")
    args = parser.parse_args()

    sparse_dir = Path(args.sparse_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images, cameras, points, colors, errors = _read_sparse_model(sparse_dir)
    if args.max_points > 0 and len(points) > args.max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    centers = []
    frustum_strips = []
    for image in sorted(images.values(), key=lambda item: item.name):
        camera = cameras[image.camera_id]
        center, strips = _camera_center_and_frustum(image, camera, args.frustum_scale)
        centers.append(center)
        frustum_strips.extend(strips)

    rr.init("triangle_splatting.colmap_sfm", spawn=False)
    rr.save(output_path)
    rr.log("world/sparse_points", rr.Points3D(points, colors=colors, radii=0.01))
    if centers:
        rr.log("world/camera_centers", rr.Points3D(np.asarray(centers), colors=[255, 80, 40], radii=0.03))
        rr.log("world/camera_frustums", rr.LineStrips3D(frustum_strips, colors=[255, 180, 40], radii=0.004))

    finite_errors = errors[np.isfinite(errors)] if len(errors) else np.empty((0,), dtype=np.float64)
    robust_errors = finite_errors[finite_errors < 1e6]
    summary = {
        "sparse_dir": str(sparse_dir),
        "output": str(output_path),
        "point_count_logged": int(len(points)),
        "camera_count": int(len(images)),
        "median_reprojection_error": float(np.median(finite_errors)) if len(finite_errors) else None,
        "mean_reprojection_error_robust": float(np.mean(robust_errors)) if len(robust_errors) else None,
        "large_error_count": int(len(finite_errors) - len(robust_errors)),
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
