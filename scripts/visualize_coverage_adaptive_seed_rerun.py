#!/usr/bin/env python3
"""Visualize coverage-aware adaptive mesh seed triangle soups in Rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr


def _load_triangle_soup(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        triangles = data["triangles"].astype(np.float32)
        colors = data["colors"].astype(np.float32)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(f"Expected {path} triangles to have shape [N, 3, 3], got {triangles.shape}")
    if colors.shape != (len(triangles), 3):
        raise ValueError(f"Expected {path} colors to have shape [{len(triangles)}, 3], got {colors.shape}")
    return triangles, colors


def _to_uint8_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.size == 0:
        return colors.astype(np.uint8)
    if float(np.nanmax(colors)) <= 1.0:
        colors = colors * 255.0
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def _rgba(colors_rgb: np.ndarray, alpha: int = 255) -> np.ndarray:
    colors_rgb = _to_uint8_colors(colors_rgb)
    alpha_values = np.full((colors_rgb.shape[0], 1), alpha, dtype=np.uint8)
    return np.concatenate([colors_rgb, alpha_values], axis=1)


def _sample_rows(
    triangles: np.ndarray,
    colors: np.ndarray,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if max_rows <= 0 or len(triangles) <= max_rows:
        return triangles, colors, False
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(triangles), size=max_rows, replace=False))
    return triangles[indices], colors[indices], True


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


def _log_triangle_soup(
    entity_path: str,
    triangles: np.ndarray,
    colors: np.ndarray,
    center_radius: float,
    alpha: int = 255,
) -> None:
    if len(triangles) == 0:
        return
    vertices = triangles.reshape(-1, 3)
    triangle_indices = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
    vertex_colors = np.repeat(_rgba(colors, alpha=alpha), repeats=3, axis=0)
    rr.log(
        f"{entity_path}/triangles",
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=triangle_indices,
            vertex_colors=vertex_colors,
        ),
    )
    centers = triangles.mean(axis=1)
    rr.log(
        f"{entity_path}/centers",
        rr.Points3D(centers, colors=_to_uint8_colors(colors), radii=center_radius),
    )


def _side_by_side_offsets(reference_triangles: np.ndarray, adaptive_triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    reference_bbox = _bbox(reference_triangles.reshape(-1, 3))
    adaptive_bbox = _bbox(adaptive_triangles.reshape(-1, 3))
    reference_extent = np.asarray(reference_bbox["extent"], dtype=np.float32)
    adaptive_extent = np.asarray(adaptive_bbox["extent"], dtype=np.float32)
    max_extent = float(max(np.max(reference_extent), np.max(adaptive_extent), 1e-6))
    separation = max(float(reference_extent[0]), float(adaptive_extent[0])) + 0.25 * max_extent
    reference_center = np.asarray(reference_bbox["center"], dtype=np.float32)
    adaptive_center = np.asarray(adaptive_bbox["center"], dtype=np.float32)
    reference_offset = np.array([-0.5 * separation, 0.0, 0.0], dtype=np.float32) - reference_center
    adaptive_offset = np.array([0.5 * separation, 0.0, 0.0], dtype=np.float32) - adaptive_center
    return reference_offset, adaptive_offset, separation


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize coverage-aware adaptive mesh seed triangle soups")
    parser.add_argument("--dataset", required=True, help="Dataset directory containing mesh_seed_triangles/")
    parser.add_argument("--output", required=True, help="Path to write the .rrd recording")
    parser.add_argument(
        "--reference-mesh-triangles",
        default=None,
        help="Optional baseline mesh_seed_triangles/all.npz to show next to the adaptive seed.",
    )
    parser.add_argument("--max-mesh-triangles", type=int, default=0, help="Maximum adaptive mesh triangles to log")
    parser.add_argument("--max-fallback-triangles", type=int, default=0, help="Maximum fallback triangles to log")
    parser.add_argument("--max-reference-triangles", type=int, default=0, help="Maximum reference triangles to log")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for visualization-only downsampling")
    parser.add_argument("--center-radius", type=float, default=0.006, help="Rerun point radius for triangle centers")
    parser.add_argument("--side-by-side", action="store_true", help="Shift reference and adaptive seeds apart")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    triangle_dir = dataset_dir / "mesh_seed_triangles"
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adaptive_triangles, adaptive_colors = _load_triangle_soup(triangle_dir / "adaptive_mesh.npz")
    fallback_path = triangle_dir / "uncovered_image_plane_fallback.npz"
    fallback_kind = "image_plane"
    if not fallback_path.exists():
        fallback_path = triangle_dir / "uncovered_sfm_fallback.npz"
        fallback_kind = "sfm"
    if fallback_path.exists():
        fallback_triangles, fallback_colors = _load_triangle_soup(fallback_path)
    else:
        fallback_triangles = np.empty((0, 3, 3), dtype=np.float32)
        fallback_colors = np.empty((0, 3), dtype=np.float32)
        fallback_kind = "none"

    reference_triangles = np.empty((0, 3, 3), dtype=np.float32)
    reference_colors = np.empty((0, 3), dtype=np.float32)
    if args.reference_mesh_triangles:
        reference_triangles, reference_colors = _load_triangle_soup(Path(args.reference_mesh_triangles).expanduser().resolve())

    adaptive_triangles_log, adaptive_colors_log, adaptive_sampled = _sample_rows(
        adaptive_triangles,
        adaptive_colors,
        args.max_mesh_triangles,
        args.seed,
    )
    fallback_triangles_log, fallback_colors_log, fallback_sampled = _sample_rows(
        fallback_triangles,
        fallback_colors,
        args.max_fallback_triangles,
        args.seed + 1,
    )
    reference_triangles_log, reference_colors_log, reference_sampled = _sample_rows(
        reference_triangles,
        reference_colors,
        args.max_reference_triangles,
        args.seed + 2,
    )

    separation = None
    if args.side_by_side and len(reference_triangles_log) and len(adaptive_triangles_log):
        adaptive_combined = (
            np.concatenate([adaptive_triangles_log, fallback_triangles_log], axis=0)
            if len(fallback_triangles_log)
            else adaptive_triangles_log
        )
        reference_offset, adaptive_offset, separation = _side_by_side_offsets(reference_triangles_log, adaptive_combined)
        reference_triangles_log = reference_triangles_log + reference_offset[None, None, :]
        adaptive_triangles_log = adaptive_triangles_log + adaptive_offset[None, None, :]
        fallback_triangles_log = fallback_triangles_log + adaptive_offset[None, None, :]

    rr.init("triangle_splatting.coverage_adaptive_seed", spawn=False)
    rr.save(output_path)
    _log_triangle_soup("world/adaptive_mesh_seed", adaptive_triangles_log, adaptive_colors_log, args.center_radius)
    _log_triangle_soup("world/uncovered_fallback", fallback_triangles_log, fallback_colors_log, args.center_radius)
    _log_triangle_soup("world/reference_mesh_seed", reference_triangles_log, reference_colors_log, args.center_radius)

    summary = {
        "dataset": str(dataset_dir),
        "output": str(output_path),
        "reference_mesh_triangles": args.reference_mesh_triangles,
        "side_by_side": bool(args.side_by_side),
        "side_by_side_separation": separation,
        "adaptive_triangle_count_total": int(len(adaptive_triangles)),
        "adaptive_triangle_count_logged": int(len(adaptive_triangles_log)),
        "adaptive_sampled": bool(adaptive_sampled),
        "fallback_kind": fallback_kind,
        "fallback_triangle_count_total": int(len(fallback_triangles)),
        "fallback_triangle_count_logged": int(len(fallback_triangles_log)),
        "fallback_sampled": bool(fallback_sampled),
        "reference_triangle_count_total": int(len(reference_triangles)),
        "reference_triangle_count_logged": int(len(reference_triangles_log)),
        "reference_sampled": bool(reference_sampled),
        "adaptive_bbox": _bbox(adaptive_triangles.reshape(-1, 3)),
        "fallback_bbox": _bbox(fallback_triangles.reshape(-1, 3)),
        "reference_bbox": _bbox(reference_triangles.reshape(-1, 3)),
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
