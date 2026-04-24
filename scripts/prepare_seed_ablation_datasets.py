#!/usr/bin/env python3
"""Prepare hybrid Kimera/COLMAP datasets for seed-initialization ablations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData

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


def _read_colmap_images(sparse_dir: Path):
    if (sparse_dir / "images.bin").exists():
        return COLMAP.read_extrinsics_binary(str(sparse_dir / "images.bin"))
    return COLMAP.read_extrinsics_text(str(sparse_dir / "images.txt"))


def _read_colmap_points(sparse_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if (sparse_dir / "points3D.bin").exists():
        points, colors, _errors = COLMAP.read_points3D_binary(str(sparse_dir / "points3D.bin"))
    else:
        points, colors, _errors = COLMAP.read_points3D_text(str(sparse_dir / "points3D.txt"))
    return points.astype(np.float64), colors.astype(np.float64)


def _read_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vertices = ply["vertex"]
    points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float64)
    return points, colors


def _colmap_camera_centers(sparse_dir: Path) -> dict[str, np.ndarray]:
    centers = {}
    for image in _read_colmap_images(sparse_dir).values():
        rotation_world_to_camera = COLMAP.qvec2rotmat(image.qvec)
        center = -rotation_world_to_camera.T @ image.tvec
        centers[Path(image.name).stem] = center.astype(np.float64)
    return centers


def _kimera_camera_centers(dataset_dir: Path) -> dict[str, np.ndarray]:
    centers = {}
    for transform_name in ("transforms_train.json", "transforms_test.json"):
        transform_path = dataset_dir / transform_name
        if not transform_path.exists():
            continue
        contents = json.loads(transform_path.read_text(encoding="utf-8"))
        for frame in contents.get("frames", []):
            stem = Path(str(frame["file_path"])).stem
            transform = np.asarray(frame["transform_matrix"], dtype=np.float64)
            centers[stem] = transform[:3, 3].copy()
    return centers


def _estimate_sim3(source_points: np.ndarray, target_points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source_points.shape != target_points.shape or source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("Sim(3) alignment expects source and target arrays with shape [N, 3]")
    if len(source_points) < 3:
        raise ValueError("At least three matched camera centers are required for Sim(3) alignment")

    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    covariance = target_centered.T @ source_centered / len(source_points)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    source_variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    scale = float(np.sum(singular_values * np.diag(correction)) / max(source_variance, 1e-12))
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def _apply_sim3(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    original_shape = points.shape
    flat_points = points.reshape(-1, 3).astype(np.float64)
    transformed = scale * (flat_points @ rotation.T) + translation[None, :]
    return transformed.reshape(original_shape)


def _apply_inverse_sim3(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    original_shape = points.shape
    flat_points = points.reshape(-1, 3).astype(np.float64)
    transformed = (flat_points - translation[None, :]) @ rotation / scale
    return transformed.reshape(original_shape)


def _link_dir(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    if destination.exists():
        return
    destination.symlink_to(source, target_is_directory=True)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_colmap_sparse(source_dataset: Path, destination_dataset: Path) -> None:
    source_sparse = source_dataset / "sparse"
    destination_sparse = destination_dataset / "sparse"
    destination_sparse.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_sparse, destination_sparse, dirs_exist_ok=True)


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.asarray(colors, dtype=np.float64)
    if colors.size and colors.max() <= 1.0:
        colors = colors * 255.0
    colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    normals = np.zeros_like(points, dtype=np.float32)

    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for point, normal, color in zip(points, normals, colors):
            handle.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _alignment_summary(
    kimera_dataset: Path,
    sfm_dataset: Path,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    matched_stems: list[str],
    kimera_centers: np.ndarray,
    colmap_centers: np.ndarray,
) -> dict[str, object]:
    transformed = _apply_sim3(kimera_centers, scale, rotation, translation)
    errors = np.linalg.norm(transformed - colmap_centers, axis=1)
    return {
        "kimera_dataset": str(kimera_dataset),
        "sfm_dataset": str(sfm_dataset),
        "matched_camera_count": int(len(matched_stems)),
        "matched_camera_stems_preview": matched_stems[:10],
        "kimera_to_colmap": {
            "scale": float(scale),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "camera_center_rmse": float(np.sqrt(np.mean(errors * errors))) if len(errors) else None,
        "camera_center_median_error": float(np.median(errors)) if len(errors) else None,
        "camera_center_max_error": float(np.max(errors)) if len(errors) else None,
    }


def prepare_hybrid_datasets(kimera_dataset: Path, sfm_dataset: Path, output_dir: Path) -> dict[str, object]:
    kimera_dataset = kimera_dataset.expanduser().resolve()
    sfm_dataset = sfm_dataset.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    kimera_centers_by_stem = _kimera_camera_centers(kimera_dataset)
    colmap_centers_by_stem = _colmap_camera_centers(sfm_dataset / "sparse" / "0")
    matched_stems = sorted(set(kimera_centers_by_stem) & set(colmap_centers_by_stem))
    if len(matched_stems) < 3:
        raise RuntimeError(
            f"Only {len(matched_stems)} matching Kimera/COLMAP camera names found; need at least 3"
        )

    kimera_centers = np.stack([kimera_centers_by_stem[stem] for stem in matched_stems], axis=0)
    colmap_centers = np.stack([colmap_centers_by_stem[stem] for stem in matched_stems], axis=0)
    scale, rotation, translation = _estimate_sim3(kimera_centers, colmap_centers)
    summary = _alignment_summary(
        kimera_dataset,
        sfm_dataset,
        scale,
        rotation,
        translation,
        matched_stems,
        kimera_centers,
        colmap_centers,
    )

    mesh_data = np.load(kimera_dataset / "mesh_seed_triangles" / "all.npz")
    mesh_triangles = mesh_data["triangles"].astype(np.float64)
    mesh_colors = mesh_data["colors"].astype(np.float32)
    mesh_triangles_colmap = _apply_sim3(mesh_triangles, scale, rotation, translation).astype(np.float32)

    mesh_points, mesh_point_colors = _read_ply_points(kimera_dataset / "points3d.ply")
    mesh_points_colmap = _apply_sim3(mesh_points, scale, rotation, translation).astype(np.float32)

    sfm_points, sfm_colors = _read_colmap_points(sfm_dataset / "sparse" / "0")
    sfm_points_kimera = _apply_inverse_sim3(sfm_points, scale, rotation, translation).astype(np.float32)

    colmap_mesh_dir = output_dir / "colmap_mesh_triangle"
    colmap_mesh_dir.mkdir(parents=True, exist_ok=True)
    _link_dir(sfm_dataset / "images", colmap_mesh_dir / "images")
    _copy_colmap_sparse(sfm_dataset, colmap_mesh_dir)
    (colmap_mesh_dir / "mesh_seed_triangles").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        colmap_mesh_dir / "mesh_seed_triangles" / "all.npz",
        triangles=mesh_triangles_colmap,
        colors=mesh_colors,
    )

    colmap_mesh_point_dir = output_dir / "colmap_mesh_point"
    colmap_mesh_point_dir.mkdir(parents=True, exist_ok=True)
    _link_dir(sfm_dataset / "images", colmap_mesh_point_dir / "images")
    _copy_colmap_sparse(sfm_dataset, colmap_mesh_point_dir)
    _write_ply(colmap_mesh_point_dir / "sparse" / "0" / "points3D.ply", mesh_points_colmap, mesh_point_colors)

    kimera_sfm_dir = output_dir / "kimera_sfm_point"
    kimera_sfm_dir.mkdir(parents=True, exist_ok=True)
    _link_dir(kimera_dataset / "images", kimera_sfm_dir / "images")
    _copy_file(kimera_dataset / "transforms_train.json", kimera_sfm_dir / "transforms_train.json")
    _copy_file(kimera_dataset / "transforms_test.json", kimera_sfm_dir / "transforms_test.json")
    _write_ply(kimera_sfm_dir / "points3d.ply", sfm_points_kimera, sfm_colors)

    summary.update(
        {
            "datasets": {
                "colmap_mesh_triangle": str(colmap_mesh_dir),
                "colmap_mesh_point": str(colmap_mesh_point_dir),
                "kimera_sfm_point": str(kimera_sfm_dir),
            },
            "mesh_triangle_count": int(len(mesh_triangles)),
            "mesh_point_count": int(len(mesh_points)),
            "sfm_point_count": int(len(sfm_points)),
        }
    )
    (output_dir / "alignment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare hybrid datasets for seed-initialization ablations")
    parser.add_argument("--kimera-dataset", required=True, help="Reduced Kimera mesh-triangle dataset")
    parser.add_argument("--sfm-dataset", required=True, help="COLMAP SfM dataset")
    parser.add_argument("--output-dir", required=True, help="Where to write hybrid datasets")
    args = parser.parse_args()

    summary = prepare_hybrid_datasets(
        Path(args.kimera_dataset),
        Path(args.sfm_dataset),
        Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
