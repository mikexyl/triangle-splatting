#!/usr/bin/env python3
"""Build a COLMAP SfM baseline initialized from Kimera camera poses."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_kimera_capture_dataset import _load_euroc_calibration, _load_frame_rows


def _load_colmap_loader():
    module_path = REPO_ROOT / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("_triangle_splatting_colmap_loader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load COLMAP helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_COLMAP_LOADER = _load_colmap_loader()
read_points3D_binary = _COLMAP_LOADER.read_points3D_binary
read_points3D_text = _COLMAP_LOADER.read_points3D_text
rotmat2qvec = _COLMAP_LOADER.rotmat2qvec


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _resolve_colmap_executable(requested: str | None) -> Path:
    candidates = []
    if requested:
        candidates.append(Path(requested).expanduser())
    path_colmap = shutil.which("colmap")
    if path_colmap:
        candidates.append(Path(path_colmap))

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find a COLMAP executable. Install it with `pixi install` "
        "and run via `pixi run`, or pass `--colmap-executable`."
    )


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _stage_raw_images(rows: list[dict[str, object]], image_dir: Path, copy_images: bool) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = Path(str(row["_image_path"]))
        destination = image_dir / source.name
        if destination.exists() or destination.is_symlink():
            continue
        if copy_images:
            shutil.copy2(source, destination)
        else:
            os.symlink(source, destination)


def _run_command(command: list[str], log_path: Path, dry_run: bool) -> dict[str, object]:
    printable = shlex.join(command)
    print(f"+ {printable}")
    start = time.time()
    if dry_run:
        return {"command": command, "elapsed_sec": 0.0, "dry_run": True}

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ {printable}\n")
        log_file.flush()
        subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=True)

    return {"command": command, "elapsed_sec": time.time() - start, "dry_run": False}


def _read_database_images(database_path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY name"
        ).fetchall()
    return [{"image_id": int(row[0]), "name": str(row[1]), "camera_id": int(row[2])} for row in rows]


def _write_cameras_txt(path: Path, camera_ids: set[int], calibration) -> None:
    params = [
        calibration.fx,
        calibration.fy,
        calibration.cx,
        calibration.cy,
        float(calibration.distortion[0]),
        float(calibration.distortion[1]),
        float(calibration.distortion[2]),
        float(calibration.distortion[3]),
    ]
    params_text = " ".join(f"{value:.17g}" for value in params)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(f"# Number of cameras: {len(camera_ids)}\n")
        for camera_id in sorted(camera_ids):
            handle.write(f"{camera_id} OPENCV {calibration.width} {calibration.height} {params_text}\n")


def _colmap_world_to_camera(camera_pose_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation_world_from_camera = camera_pose_world[:3, :3]
    translation_world_from_camera = camera_pose_world[:3, 3]
    rotation_camera_from_world = rotation_world_from_camera.T
    translation_camera_from_world = -rotation_camera_from_world @ translation_world_from_camera
    return rotmat2qvec(rotation_camera_from_world), translation_camera_from_world


def _write_images_txt(
    path: Path,
    database_images: list[dict[str, object]],
    rows_by_name: dict[str, dict[str, object]],
) -> int:
    matched = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image in database_images:
            name = str(image["name"])
            row = rows_by_name.get(name) or rows_by_name.get(Path(name).name)
            if row is None:
                continue
            qvec, tvec = _colmap_world_to_camera(np.asarray(row["_camera_pose_world"], dtype=np.float64))
            pose_values = [*qvec.tolist(), *tvec.tolist()]
            pose_text = " ".join(f"{value:.17g}" for value in pose_values)
            handle.write(f"{image['image_id']} {pose_text} {image['camera_id']} {name}\n")
            handle.write("\n")
            matched += 1
    return matched


def _write_empty_points3d_txt(path: Path) -> None:
    path.write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0, mean track length: 0\n",
        encoding="utf-8",
    )


def _write_seed_sparse_model(
    sparse_dir: Path,
    database_images: list[dict[str, object]],
    rows: list[dict[str, object]],
    calibration,
) -> int:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    rows_by_name = {Path(str(row["_image_path"])).name: row for row in rows}
    camera_ids = {int(image["camera_id"]) for image in database_images}
    _write_cameras_txt(sparse_dir / "cameras.txt", camera_ids, calibration)
    matched = _write_images_txt(sparse_dir / "images.txt", database_images, rows_by_name)
    _write_empty_points3d_txt(sparse_dir / "points3D.txt")
    if matched < 2:
        raise RuntimeError(f"Need at least two database images with Kimera poses; matched {matched}.")
    return matched


def _normalize_colmap_dataset_sparse_layout(dataset_dir: Path) -> None:
    sparse_dir = dataset_dir / "sparse"
    sparse_zero_dir = sparse_dir / "0"
    if sparse_zero_dir.exists():
        return

    model_files = [
        path
        for path in sparse_dir.iterdir()
        if path.is_file() and path.name in {"cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"}
    ]
    if not model_files:
        raise RuntimeError(f"No COLMAP sparse model files found in {sparse_dir}")

    sparse_zero_dir.mkdir(parents=True, exist_ok=True)
    for model_file in model_files:
        shutil.move(str(model_file), sparse_zero_dir / model_file.name)


def _count_points3d(model_dir: Path) -> int | None:
    binary_path = model_dir / "points3D.bin"
    text_path = model_dir / "points3D.txt"
    try:
        if binary_path.exists():
            xyz, _, _ = read_points3D_binary(str(binary_path))
            return int(len(xyz))
        if text_path.exists():
            xyz, _, _ = read_points3D_text(str(text_path))
            return int(len(xyz))
    except Exception:
        return None
    return None


def _write_train_command(output_dir: Path, dataset_dir: Path) -> str:
    model_dir = output_dir / "train"
    command = [
        "python",
        "train.py",
        "-s",
        str(dataset_dir),
        "-m",
        str(model_dir),
        "--eval",
        "--seed_init_mode",
        "point",
    ]
    text = shlex.join(command)
    (output_dir / "train_sfm_baseline.sh").write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{text}\n", encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COLMAP SfM initialized from Kimera camera poses")
    parser.add_argument(
        "--capture-dir",
        default="/home/mikexyl/workspaces/kimera_ros2_ws/artifacts/vicon_room_1",
        help="Path to the recorded Kimera capture with images and cam*_frames.csv",
    )
    parser.add_argument("--mav0-dir", required=True, help="EuRoC mav0 directory containing cam*/sensor.yaml")
    parser.add_argument("--camera", default="cam0", choices=["cam0", "cam1"], help="Camera stream to reconstruct")
    parser.add_argument("--output-dir", required=True, help="Output directory under output/ for the SfM ablation")
    parser.add_argument("--colmap-executable", default=None, help="Path to a COLMAP executable")
    parser.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential"], help="COLMAP matcher")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU SIFT extraction and matching")
    parser.add_argument(
        "--guided-matching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable COLMAP guided feature matching",
    )
    parser.add_argument(
        "--run-bundle-adjustment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refine the Kimera-initialized COLMAP model after triangulation while keeping intrinsics fixed",
    )
    parser.add_argument(
        "--run-point-filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter high-error sparse points before exporting the trainable COLMAP dataset",
    )
    parser.add_argument("--point-filter-max-reproj-error", type=float, default=4.0)
    parser.add_argument("--point-filter-min-track-len", type=int, default=2)
    parser.add_argument("--point-filter-min-tri-angle", type=float, default=1.5)
    parser.add_argument("--copy-images", action="store_true", help="Copy raw images instead of symlinking them")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    parser.add_argument("--dry-run", action="store_true", help="Write staged files and print COLMAP commands without running them")
    args = parser.parse_args()

    capture_dir = _path(args.capture_dir)
    mav0_dir = _path(args.mav0_dir)
    output_dir = _path(args.output_dir)
    colmap = _resolve_colmap_executable(args.colmap_executable)
    calibration = _load_euroc_calibration(mav0_dir, args.camera)
    rows = _load_frame_rows(capture_dir, args.camera)
    if not rows:
        raise RuntimeError(f"No posed {args.camera} frames found in {capture_dir}")

    _prepare_output_dir(output_dir, args.overwrite)
    raw_images_dir = output_dir / "raw_images"
    database_path = output_dir / "database.db"
    seed_sparse_dir = output_dir / "sparse_kimera_pose_seed"
    triangulated_sparse_dir = output_dir / "sparse_triangulated"
    refined_sparse_dir = output_dir / "sparse_refined"
    filtered_sparse_dir = output_dir / "sparse_filtered"
    dataset_dir = output_dir / "dataset"
    log_path = output_dir / "colmap.log"

    _stage_raw_images(rows, raw_images_dir, args.copy_images)

    use_gpu = "0" if args.no_gpu else "1"
    camera_params = ",".join(
        f"{value:.17g}"
        for value in [
            calibration.fx,
            calibration.fy,
            calibration.cx,
            calibration.cy,
            float(calibration.distortion[0]),
            float(calibration.distortion[1]),
            float(calibration.distortion[2]),
            float(calibration.distortion[3]),
        ]
    )

    commands = []
    commands.append(
        _run_command(
            [
                str(colmap),
                "feature_extractor",
                "--database_path",
                str(database_path),
                "--image_path",
                str(raw_images_dir),
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_model",
                "OPENCV",
                "--ImageReader.camera_params",
                camera_params,
                "--FeatureExtraction.use_gpu",
                use_gpu,
            ],
            log_path,
            args.dry_run,
        )
    )

    matcher_command = "exhaustive_matcher" if args.matcher == "exhaustive" else "sequential_matcher"
    commands.append(
        _run_command(
            [
                str(colmap),
                matcher_command,
                "--database_path",
                str(database_path),
                "--FeatureMatching.use_gpu",
                use_gpu,
                "--FeatureMatching.guided_matching",
                "1" if args.guided_matching else "0",
            ],
            log_path,
            args.dry_run,
        )
    )

    matched_pose_count = 0
    if not args.dry_run:
        database_images = _read_database_images(database_path)
        matched_pose_count = _write_seed_sparse_model(seed_sparse_dir, database_images, rows, calibration)
    else:
        seed_sparse_dir.mkdir(parents=True, exist_ok=True)

    triangulated_sparse_dir.mkdir(parents=True, exist_ok=True)
    commands.append(
        _run_command(
            [
                str(colmap),
                "point_triangulator",
                "--database_path",
                str(database_path),
                "--image_path",
                str(raw_images_dir),
                "--input_path",
                str(seed_sparse_dir),
                "--output_path",
                str(triangulated_sparse_dir),
                "--clear_points",
                "1",
                "--refine_intrinsics",
                "0",
            ],
            log_path,
            args.dry_run,
        )
    )

    sparse_for_undistortion = triangulated_sparse_dir
    if args.run_bundle_adjustment:
        refined_sparse_dir.mkdir(parents=True, exist_ok=True)
        commands.append(
            _run_command(
                [
                    str(colmap),
                    "bundle_adjuster",
                    "--input_path",
                    str(triangulated_sparse_dir),
                    "--output_path",
                    str(refined_sparse_dir),
                    "--BundleAdjustment.refine_focal_length",
                    "0",
                    "--BundleAdjustment.refine_principal_point",
                    "0",
                    "--BundleAdjustment.refine_extra_params",
                    "0",
                ],
                log_path,
                args.dry_run,
            )
        )
        sparse_for_undistortion = refined_sparse_dir

    if args.run_point_filtering:
        filtered_sparse_dir.mkdir(parents=True, exist_ok=True)
        commands.append(
            _run_command(
                [
                    str(colmap),
                    "point_filtering",
                    "--input_path",
                    str(sparse_for_undistortion),
                    "--output_path",
                    str(filtered_sparse_dir),
                    "--max_reproj_error",
                    str(args.point_filter_max_reproj_error),
                    "--min_track_len",
                    str(args.point_filter_min_track_len),
                    "--min_tri_angle",
                    str(args.point_filter_min_tri_angle),
                ],
                log_path,
                args.dry_run,
            )
        )
        sparse_for_undistortion = filtered_sparse_dir

    commands.append(
        _run_command(
            [
                str(colmap),
                "image_undistorter",
                "--image_path",
                str(raw_images_dir),
                "--input_path",
                str(sparse_for_undistortion),
                "--output_path",
                str(dataset_dir),
                "--output_type",
                "COLMAP",
            ],
            log_path,
            args.dry_run,
        )
    )

    final_sparse_dir = dataset_dir / "sparse" / "0"
    if not args.dry_run:
        _normalize_colmap_dataset_sparse_layout(dataset_dir)
    train_command = _write_train_command(output_dir, dataset_dir)
    summary = {
        "source_capture_dir": str(capture_dir),
        "source_mav0_dir": str(mav0_dir),
        "camera": args.camera,
        "raw_image_count": len(rows),
        "matched_pose_count": matched_pose_count,
        "colmap_executable": str(colmap),
        "matcher": args.matcher,
        "guided_matching": bool(args.guided_matching),
        "gpu_enabled": not args.no_gpu,
        "bundle_adjustment": bool(args.run_bundle_adjustment),
        "seed_sparse_dir": str(seed_sparse_dir),
        "triangulated_sparse_dir": str(triangulated_sparse_dir),
        "refined_sparse_dir": str(refined_sparse_dir) if args.run_bundle_adjustment else None,
        "filtered_sparse_dir": str(filtered_sparse_dir) if args.run_point_filtering else None,
        "dataset_dir": str(dataset_dir),
        "final_sparse_dir": str(final_sparse_dir),
        "triangulated_point_count": None if args.dry_run else _count_points3d(triangulated_sparse_dir),
        "filtered_point_count": None if args.dry_run or not args.run_point_filtering else _count_points3d(filtered_sparse_dir),
        "final_point_count": None if args.dry_run else _count_points3d(final_sparse_dir),
        "train_command": train_command,
        "commands": commands,
    }
    (output_dir / "sfm_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
