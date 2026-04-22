import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class CameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray
    body_from_sensor: np.ndarray

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def _quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = np.linalg.norm([qw, qx, qy, qz])
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)

    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        return _normalize_quaternion((1.0 - alpha) * q0 + alpha * q1)

    theta_0 = math.acos(max(min(dot, 1.0), -1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)

    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def _load_camera_calibration(mav0_dir: Path, camera_name: str) -> CameraCalibration:
    sensor_path = mav0_dir / camera_name / "sensor.yaml"
    if not sensor_path.exists():
        raise FileNotFoundError(f"Missing camera calibration: {sensor_path}")

    contents = yaml.safe_load(sensor_path.read_text(encoding="utf-8"))
    intrinsics = contents["intrinsics"]
    resolution = contents["resolution"]
    body_from_sensor = np.asarray(contents["T_BS"]["data"], dtype=np.float64).reshape((4, 4))

    return CameraCalibration(
        width=int(resolution[0]),
        height=int(resolution[1]),
        fx=float(intrinsics[0]),
        fy=float(intrinsics[1]),
        cx=float(intrinsics[2]),
        cy=float(intrinsics[3]),
        distortion=np.asarray(contents["distortion_coefficients"], dtype=np.float64),
        body_from_sensor=body_from_sensor,
    )


def _load_image_rows(mav0_dir: Path, camera_name: str) -> list[dict[str, str]]:
    csv_path = mav0_dir / camera_name / "data.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    rows.sort(key=lambda row: int(row["#timestamp [ns]"]))
    return rows


def _load_ground_truth(mav0_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_path = mav0_dir / "state_groundtruth_estimate0" / "data.csv"
    timestamps = []
    positions = []
    quaternions = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            timestamps.append(int(row[0]))
            positions.append([float(row[1]), float(row[2]), float(row[3])])
            quaternions.append([float(row[4]), float(row[5]), float(row[6]), float(row[7])])

    return (
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(quaternions, dtype=np.float64),
    )


def _interpolate_body_pose(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    query_ts: int,
) -> np.ndarray | None:
    if query_ts < timestamps[0] or query_ts > timestamps[-1]:
        return None

    idx = int(np.searchsorted(timestamps, query_ts))
    if idx < len(timestamps) and timestamps[idx] == query_ts:
        position = positions[idx]
        quaternion = quaternions[idx]
    else:
        if idx == 0 or idx >= len(timestamps):
            return None
        t0 = timestamps[idx - 1]
        t1 = timestamps[idx]
        alpha = float(query_ts - t0) / float(t1 - t0)
        position = (1.0 - alpha) * positions[idx - 1] + alpha * positions[idx]
        quaternion = _slerp(quaternions[idx - 1], quaternions[idx], alpha)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_to_matrix(
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
    )
    transform[:3, 3] = position
    return transform


def _blender_transform_from_camera_pose(camera_pose_world: np.ndarray) -> list[list[float]]:
    transform = camera_pose_world.copy()
    transform[:3, 1:3] *= -1.0
    return transform.tolist()


def _prepare_undistort_maps(
    calibration: CameraCalibration,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    if downsample < 1:
        raise ValueError(f"downsample must be >= 1, got {downsample}")

    undistorted_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        calibration.camera_matrix,
        calibration.distortion,
        (calibration.width, calibration.height),
        0.0,
        (calibration.width, calibration.height),
        centerPrincipalPoint=True,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        calibration.camera_matrix,
        calibration.distortion,
        None,
        undistorted_camera_matrix,
        (calibration.width, calibration.height),
        cv2.CV_32FC1,
    )

    output_width = calibration.width // downsample
    output_height = calibration.height // downsample
    scaled_camera_matrix = undistorted_camera_matrix.copy()
    scaled_camera_matrix[0, 0] /= downsample
    scaled_camera_matrix[1, 1] /= downsample
    scaled_camera_matrix[0, 2] /= downsample
    scaled_camera_matrix[1, 2] /= downsample

    return map_x, map_y, scaled_camera_matrix, (output_width, output_height)


def _undistort_and_downsample_image(
    input_path: Path,
    output_path: Path,
    map_x: np.ndarray,
    map_y: np.ndarray,
    output_size: tuple[int, int],
) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {input_path}")

    undistorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    if undistorted.shape[1] != output_size[0] or undistorted.shape[0] != output_size[1]:
        undistorted = cv2.resize(undistorted, output_size, interpolation=cv2.INTER_AREA)

    undistorted_rgb = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2RGB)
    if not cv2.imwrite(str(output_path), undistorted_rgb):
        raise RuntimeError(f"Failed to write image: {output_path}")


def _write_transforms_json(
    output_path: Path,
    width: int,
    height: int,
    camera_matrix: np.ndarray,
    frames: list[dict[str, object]],
) -> None:
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    contents = {
        "camera_angle_x": 2.0 * math.atan(width / (2.0 * fx)),
        "camera_angle_y": 2.0 * math.atan(height / (2.0 * fy)),
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "frames": frames,
    }
    output_path.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
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

    normals = np.zeros_like(points)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        handle.write("\n")
        for point, normal, color in zip(points, normals, colors):
            handle.write(
                f"{point[0]} {point[1]} {point[2]} "
                f"{normal[0]} {normal[1]} {normal[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _stream_sample_ascii_ply(
    input_path: Path,
    max_points: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if max_points <= 0:
        raise ValueError(f"max_points must be > 0, got {max_points}")

    rng = np.random.default_rng(rng_seed)
    points = np.empty((max_points, 3), dtype=np.float32)
    colors = np.empty((max_points, 3), dtype=np.uint8)
    kept = 0
    seen = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "end_header":
                break

        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 7:
                continue

            point = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
            intensity = float(parts[3])
            rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])], dtype=np.uint8)
            if int(rgb[0]) == 0 and int(rgb[1]) == 0 and int(rgb[2]) == 0:
                gray = int(np.clip(round(intensity * 255.0), 0, 255))
                color = np.array([gray, gray, gray], dtype=np.uint8)
            else:
                color = rgb

            if kept < max_points:
                points[kept] = point
                colors[kept] = color
                kept += 1
            else:
                replace_index = int(rng.integers(0, seen + 1))
                if replace_index < max_points:
                    points[replace_index] = point
                    colors[replace_index] = color
            seen += 1

    return points[:kept], colors[:kept], seen


def _generate_camera_box_seed_cloud(
    camera_centers: np.ndarray,
    num_points: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if num_points <= 0:
        raise ValueError(f"num_points must be > 0, got {num_points}")
    if camera_centers.size == 0:
        raise ValueError("camera_centers must not be empty")

    rng = np.random.default_rng(rng_seed)
    center = camera_centers.mean(axis=0)
    distances = np.linalg.norm(camera_centers - center[None, :], axis=1)
    radius = float(max(np.max(distances) * 1.1, 0.5))

    min_corner = center - radius
    max_corner = center + radius
    points = rng.uniform(min_corner, max_corner, size=(num_points, 3)).astype(np.float32)
    colors = np.full((num_points, 3), 127, dtype=np.uint8)

    return points, colors, {
        "type": "camera_box",
        "center": center.tolist(),
        "radius": radius,
        "min_corner": min_corner.tolist(),
        "max_corner": max_corner.tolist(),
        "output_point_count": int(num_points),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a trainable dataset from EuRoC raw data and ground-truth poses")
    parser.add_argument("--mav0-dir", required=True, help="Path to the EuRoC mav0 directory")
    parser.add_argument("--output-dir", required=True, help="Converted dataset output directory")
    parser.add_argument("--camera", default="cam0", choices=["cam0", "cam1"], help="Camera stream to use")
    parser.add_argument("--downsample", type=int, default=2, help="Image downsampling factor applied after undistortion")
    parser.add_argument("--frame-stride", type=int, default=8, help="Keep every Nth overlapping frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on converted frames after stride; 0 means no cap")
    parser.add_argument("--llffhold", type=int, default=8, help="Every Nth frame goes to transforms_test.json")
    parser.add_argument("--max-seed-points", type=int, default=250000, help="Reservoir-sampled point count from pointcloud0/data.ply")
    parser.add_argument("--seed-rng", type=int, default=0, help="Random seed for point-cloud sampling")
    parser.add_argument(
        "--seed-mode",
        default="pointcloud",
        choices=["pointcloud", "camera_box"],
        help="How to generate the initial points3d.ply seeds for triangle initialization",
    )
    args = parser.parse_args()

    mav0_dir = Path(args.mav0_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    calibration = _load_camera_calibration(mav0_dir, args.camera)
    image_rows = _load_image_rows(mav0_dir, args.camera)
    gt_timestamps, gt_positions, gt_quaternions = _load_ground_truth(mav0_dir)

    overlapping_rows = []
    for row in image_rows:
        timestamp_ns = int(row["#timestamp [ns]"])
        if gt_timestamps[0] <= timestamp_ns <= gt_timestamps[-1]:
            overlapping_rows.append(row)

    selected_rows = overlapping_rows[:: max(args.frame_stride, 1)]
    if args.max_frames > 0:
        selected_rows = selected_rows[: args.max_frames]

    if not selected_rows:
        raise RuntimeError("No image rows remain after GT overlap filtering and frame stride.")

    map_x, map_y, scaled_camera_matrix, output_size = _prepare_undistort_maps(
        calibration,
        downsample=args.downsample,
    )

    transformed_frames = []
    camera_centers = []
    dropped_rows = 0
    for row in selected_rows:
        timestamp_ns = int(row["#timestamp [ns]"])
        image_path = mav0_dir / args.camera / "data" / row["filename"]
        body_pose_world = _interpolate_body_pose(
            gt_timestamps,
            gt_positions,
            gt_quaternions,
            timestamp_ns,
        )
        if body_pose_world is None:
            dropped_rows += 1
            continue

        camera_pose_world = body_pose_world @ calibration.body_from_sensor
        camera_centers.append(camera_pose_world[:3, 3].copy())
        output_stem = image_path.stem
        output_image_path = images_dir / f"{output_stem}.png"
        _undistort_and_downsample_image(
            image_path,
            output_image_path,
            map_x,
            map_y,
            output_size,
        )
        transformed_frames.append(
            {
                "file_path": f"images/{output_stem}",
                "time_ns": timestamp_ns,
                "transform_matrix": _blender_transform_from_camera_pose(camera_pose_world),
            }
        )

    if not transformed_frames:
        raise RuntimeError("No frames with interpolated poses were produced.")

    train_frames = []
    test_frames = []
    for index, frame in enumerate(transformed_frames):
        if args.llffhold > 0 and index % args.llffhold == 0:
            test_frames.append(frame)
        else:
            train_frames.append(frame)

    _write_transforms_json(
        output_dir / "transforms_train.json",
        width=output_size[0],
        height=output_size[1],
        camera_matrix=scaled_camera_matrix,
        frames=train_frames,
    )
    _write_transforms_json(
        output_dir / "transforms_test.json",
        width=output_size[0],
        height=output_size[1],
        camera_matrix=scaled_camera_matrix,
        frames=test_frames,
    )

    if args.seed_mode == "pointcloud":
        pointcloud_input = mav0_dir / "pointcloud0" / "data.ply"
        sampled_points, sampled_colors, total_input_points = _stream_sample_ascii_ply(
            pointcloud_input,
            max_points=args.max_seed_points,
            rng_seed=args.seed_rng,
        )
        seed_summary: dict[str, object] = {
            "type": "pointcloud",
            "input_path": str(pointcloud_input),
            "input_point_count": int(total_input_points),
            "output_point_count": int(len(sampled_points)),
        }
    else:
        sampled_points, sampled_colors, seed_summary = _generate_camera_box_seed_cloud(
            np.asarray(camera_centers, dtype=np.float32),
            num_points=args.max_seed_points,
            rng_seed=args.seed_rng,
        )

    _write_ply(output_dir / "points3d.ply", sampled_points, sampled_colors)

    summary = {
        "source_mav0_dir": str(mav0_dir),
        "camera": args.camera,
        "downsample": int(args.downsample),
        "frame_stride": int(args.frame_stride),
        "max_frames": int(args.max_frames),
        "overlapping_image_count": int(len(overlapping_rows)),
        "converted_frame_count": int(len(transformed_frames)),
        "dropped_rows_after_selection": int(dropped_rows),
        "train_frame_count": int(len(train_frames)),
        "test_frame_count": int(len(test_frames)),
        "image_resolution": {"width": int(output_size[0]), "height": int(output_size[1])},
        "intrinsics": {
            "fx": float(scaled_camera_matrix[0, 0]),
            "fy": float(scaled_camera_matrix[1, 1]),
            "cx": float(scaled_camera_matrix[0, 2]),
            "cy": float(scaled_camera_matrix[1, 2]),
        },
        "seed_point_cloud": seed_summary,
    }
    (output_dir / "euroc_conversion.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
