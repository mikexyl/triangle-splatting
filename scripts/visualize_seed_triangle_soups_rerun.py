#!/usr/bin/env python3
"""Save a Rerun comparison of mesh-seed and SfM-seed initial triangle soups."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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


def _read_colmap_points(sparse_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (sparse_dir / "points3D.bin").exists():
        points, colors, errors = COLMAP.read_points3D_binary(str(sparse_dir / "points3D.bin"))
    elif (sparse_dir / "points3D.txt").exists():
        points, colors, errors = COLMAP.read_points3D_text(str(sparse_dir / "points3D.txt"))
    else:
        raise FileNotFoundError(f"No points3D.bin or points3D.txt found in {sparse_dir}")

    return points.astype(np.float32), colors.astype(np.float32), errors.reshape(-1).astype(np.float32)


def _load_mesh_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    triangles = data["triangles"].astype(np.float32)
    colors = data["colors"].astype(np.float32)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(f"Expected triangles with shape (N, 3, 3), got {triangles.shape}")
    if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] != triangles.shape[0]:
        raise ValueError(f"Expected colors with shape ({triangles.shape[0]}, 3), got {colors.shape}")
    return triangles, colors


def _to_uint8_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.size == 0:
        return colors.astype(np.uint8)
    if np.nanmax(colors) <= 1.0:
        colors = colors * 255.0
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def _rgba(colors_rgb: np.ndarray) -> np.ndarray:
    colors_rgb = _to_uint8_colors(colors_rgb)
    alpha = np.full((colors_rgb.shape[0], 1), 255, dtype=np.uint8)
    return np.concatenate([colors_rgb, alpha], axis=1)


def _fibonacci_directions(nb_points: int) -> np.ndarray:
    if nb_points < 2:
        raise ValueError("--nb-points must be at least 2")

    directions = []
    for i in range(nb_points):
        z_coord = 1.0 - (2.0 * i / (nb_points - 1))
        radius_xy = math.sqrt(max(1.0 - z_coord * z_coord, 0.0))
        theta = math.pi * (3.0 - math.sqrt(5.0)) * i
        directions.append([radius_xy * math.cos(theta), radius_xy * math.sin(theta), z_coord])
    return np.asarray(directions, dtype=np.float32)


def _random_rotation_matrices(count: int, rng: np.random.Generator) -> np.ndarray:
    axes = rng.normal(size=(count, 3)).astype(np.float32)
    axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
    angles = (2.0 * math.pi * rng.random(count)).astype(np.float32)

    sin_t = np.sin(angles)[:, None, None]
    cos_t = np.cos(angles)[:, None, None]
    ux, uy, uz = axes[:, 0], axes[:, 1], axes[:, 2]

    skew = np.zeros((count, 3, 3), dtype=np.float32)
    skew[:, 0, 1] = -uz
    skew[:, 0, 2] = uy
    skew[:, 1, 0] = uz
    skew[:, 1, 2] = -ux
    skew[:, 2, 0] = -uy
    skew[:, 2, 1] = ux

    eye = np.eye(3, dtype=np.float32)[None, :, :]
    return eye + sin_t * skew + (1.0 - cos_t) * np.matmul(skew, skew)


def _nearest_neighbor_dist2(points: np.ndarray) -> tuple[np.ndarray, str]:
    if len(points) < 2:
        return np.full((len(points),), 1e-7, dtype=np.float32), "constant"

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        try:
            distances, _indices = tree.query(points, k=2, workers=-1)
        except TypeError:
            distances, _indices = tree.query(points, k=2)
        return np.maximum(distances[:, 1] ** 2, 1e-7).astype(np.float32), "scipy.cKDTree"
    except ImportError:
        import open3d as o3d

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        tree = o3d.geometry.KDTreeFlann(point_cloud)
        dist2 = np.empty((len(points),), dtype=np.float32)
        for idx, point in enumerate(points):
            count, _indices, distances = tree.search_knn_vector_3d(point.astype(np.float64), 2)
            dist2[idx] = distances[1] if count > 1 else 1e-7
        return np.maximum(dist2, 1e-7), "open3d.KDTreeFlann"


def _points_to_triangles(
    points: np.ndarray,
    nearest_dist2: np.ndarray,
    triangle_size: float,
    nb_points: int,
    seed: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    base_dirs = _fibonacci_directions(nb_points)
    radii = triangle_size * np.sqrt(np.maximum(nearest_dist2, 1e-7)).astype(np.float32)
    triangles = np.empty((len(points), nb_points, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)

    for start_idx in range(0, len(points), chunk_size):
        end_idx = min(start_idx + chunk_size, len(points))
        rotations = _random_rotation_matrices(end_idx - start_idx, rng)
        rotated = np.einsum("nij,pj->npi", rotations, base_dirs, optimize=True)
        triangles[start_idx:end_idx] = points[start_idx:end_idx, None, :] + rotated * radii[start_idx:end_idx, None, None]

    return triangles, radii


def _sample_rows(
    triangles: np.ndarray,
    colors: np.ndarray,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if max_rows <= 0 or len(triangles) <= max_rows:
        return triangles, colors, None

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(triangles), size=max_rows, replace=False))
    return triangles[indices], colors[indices], indices


def _bbox(vertices: np.ndarray) -> dict[str, list[float]]:
    if len(vertices) == 0:
        return {"min": [], "max": [], "extent": [], "center": []}
    min_xyz = vertices.min(axis=0)
    max_xyz = vertices.max(axis=0)
    return {
        "min": min_xyz.tolist(),
        "max": max_xyz.tolist(),
        "extent": (max_xyz - min_xyz).tolist(),
        "center": ((min_xyz + max_xyz) * 0.5).tolist(),
    }


def _side_by_side(
    mesh_triangles: np.ndarray,
    sfm_triangles: np.ndarray,
    gap: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    mesh_bbox = _bbox(mesh_triangles.reshape(-1, 3))
    sfm_bbox = _bbox(sfm_triangles.reshape(-1, 3))
    mesh_extent = np.asarray(mesh_bbox["extent"], dtype=np.float32)
    sfm_extent = np.asarray(sfm_bbox["extent"], dtype=np.float32)
    max_extent = float(max(np.max(mesh_extent), np.max(sfm_extent), 1e-6))
    separation = max(float(mesh_extent[0]), float(sfm_extent[0])) + (gap if gap is not None else 0.25 * max_extent)

    mesh_center = np.asarray(mesh_bbox["center"], dtype=np.float32)
    sfm_center = np.asarray(sfm_bbox["center"], dtype=np.float32)
    mesh_target = np.array([-0.5 * separation, 0.0, 0.0], dtype=np.float32)
    sfm_target = np.array([0.5 * separation, 0.0, 0.0], dtype=np.float32)
    return mesh_triangles - mesh_center + mesh_target, sfm_triangles - sfm_center + sfm_target, separation


def _log_triangle_soup(entity_path: str, triangles: np.ndarray, colors: np.ndarray, point_radius: float) -> None:
    vertices = triangles.reshape(-1, 3)
    triangle_indices = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
    vertex_colors = np.repeat(_rgba(colors), repeats=3, axis=0)
    rr.log(
        f"{entity_path}/triangles",
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=triangle_indices,
            vertex_colors=vertex_colors,
        ),
    )
    centers = triangles.mean(axis=1)
    rr.log(f"{entity_path}/centers", rr.Points3D(centers, colors=_to_uint8_colors(colors), radii=point_radius))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mesh and SfM initial triangle soups in Rerun")
    parser.add_argument("--mesh-triangles", required=True, help="Path to Kimera mesh_seed_triangles/all.npz")
    parser.add_argument("--sfm-sparse-dir", required=True, help="COLMAP sparse model directory for SfM point seed")
    parser.add_argument("--output", required=True, help="Path to write the .rrd recording")
    parser.add_argument("--triangle-size", type=float, default=2.23, help="Point-seed triangle scale; matches train.py --triangle_size")
    parser.add_argument("--nb-points", type=int, default=3, help="Vertices per point-seed triangle; matches train.py --nb_points")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for SfM triangle orientations and sampling")
    parser.add_argument("--chunk-size", type=int, default=65536, help="Chunk size for SfM triangle generation")
    parser.add_argument("--max-mesh-triangles", type=int, default=0, help="Maximum mesh triangles to log; 0 logs all")
    parser.add_argument("--max-sfm-triangles", type=int, default=0, help="Maximum SfM triangles to log; 0 logs all")
    parser.add_argument("--side-by-side", action="store_true", help="Center the soups next to each other for comparison")
    parser.add_argument("--side-by-side-gap", type=float, default=None, help="Override the gap used by --side-by-side")
    parser.add_argument("--center-radius", type=float, default=0.006, help="Rerun point radius for triangle centers")
    args = parser.parse_args()

    mesh_path = Path(args.mesh_triangles).expanduser().resolve()
    sfm_sparse_dir = Path(args.sfm_sparse_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh_triangles, mesh_colors = _load_mesh_triangles(mesh_path)
    sfm_points, sfm_colors, sfm_errors = _read_colmap_points(sfm_sparse_dir)
    sfm_dist2, knn_backend = _nearest_neighbor_dist2(sfm_points)
    sfm_triangles, sfm_radii = _points_to_triangles(
        sfm_points,
        sfm_dist2,
        triangle_size=args.triangle_size,
        nb_points=args.nb_points,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    mesh_triangles_log, mesh_colors_log, mesh_sample_indices = _sample_rows(
        mesh_triangles,
        mesh_colors,
        max_rows=args.max_mesh_triangles,
        seed=args.seed,
    )
    sfm_triangles_log, sfm_colors_log, sfm_sample_indices = _sample_rows(
        sfm_triangles,
        sfm_colors,
        max_rows=args.max_sfm_triangles,
        seed=args.seed + 1,
    )

    separation = None
    if args.side_by_side:
        mesh_triangles_log, sfm_triangles_log, separation = _side_by_side(
            mesh_triangles_log,
            sfm_triangles_log,
            gap=args.side_by_side_gap,
        )

    rr.init("triangle_splatting.seed_triangle_soups", spawn=False)
    rr.save(output_path)
    _log_triangle_soup("world/mesh_seed", mesh_triangles_log, mesh_colors_log, point_radius=args.center_radius)
    _log_triangle_soup("world/sfm_seed", sfm_triangles_log, sfm_colors_log, point_radius=args.center_radius)

    finite_errors = sfm_errors[np.isfinite(sfm_errors)]
    summary = {
        "mesh_triangles": str(mesh_path),
        "sfm_sparse_dir": str(sfm_sparse_dir),
        "output": str(output_path),
        "side_by_side": bool(args.side_by_side),
        "side_by_side_separation": separation,
        "triangle_size": args.triangle_size,
        "nb_points": args.nb_points,
        "seed": args.seed,
        "sfm_knn_backend": knn_backend,
        "mesh_triangle_count_total": int(len(mesh_triangles)),
        "mesh_triangle_count_logged": int(len(mesh_triangles_log)),
        "mesh_sampled": mesh_sample_indices is not None,
        "sfm_point_count_total": int(len(sfm_points)),
        "sfm_triangle_count_logged": int(len(sfm_triangles_log)),
        "sfm_sampled": sfm_sample_indices is not None,
        "sfm_radius_median": float(np.median(sfm_radii)) if len(sfm_radii) else None,
        "sfm_radius_mean": float(np.mean(sfm_radii)) if len(sfm_radii) else None,
        "sfm_reprojection_error_median": float(np.median(finite_errors)) if len(finite_errors) else None,
        "mesh_bbox_original": _bbox(mesh_triangles.reshape(-1, 3)),
        "sfm_bbox_original": _bbox(sfm_triangles.reshape(-1, 3)),
        "mesh_bbox_logged": _bbox(mesh_triangles_log.reshape(-1, 3)),
        "sfm_bbox_logged": _bbox(sfm_triangles_log.reshape(-1, 3)),
        "note": "SfM triangle orientations are deterministic for visualization; training uses random rotations with the same nearest-neighbor size rule.",
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
