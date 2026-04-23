import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    return Rotation.from_quat(quat / norm).as_matrix()


def _matrix_to_tum_row(timestamp_ns: int, transform: np.ndarray) -> str:
    translation = transform[:3, 3]
    quat_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    timestamp_s = timestamp_ns * 1e-9
    return (
        f"{timestamp_s:.9f} "
        f"{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
        f"{quat_xyzw[0]:.9f} {quat_xyzw[1]:.9f} {quat_xyzw[2]:.9f} {quat_xyzw[3]:.9f}"
    )


def _raw_capture_pose_from_row(row: dict[str, str]) -> tuple[int, np.ndarray] | None:
    required_fields = [
        "image_timestamp_ns",
        "position_x",
        "position_y",
        "position_z",
        "orientation_w",
        "orientation_x",
        "orientation_y",
        "orientation_z",
    ]
    if any(not row.get(field, "").strip() for field in required_fields):
        return None

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_to_matrix(
        float(row["orientation_w"]),
        float(row["orientation_x"]),
        float(row["orientation_y"]),
        float(row["orientation_z"]),
    )
    transform[:3, 3] = np.array(
        [
            float(row["position_x"]),
            float(row["position_y"]),
            float(row["position_z"]),
        ],
        dtype=np.float64,
    )
    return int(row["image_timestamp_ns"]), transform


def _load_raw_capture_trajectory(capture_dir: Path, camera_name: str) -> list[tuple[int, np.ndarray]]:
    csv_path = capture_dir / f"{camera_name}_frames.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        trajectory = []
        for row in reader:
            pose_entry = _raw_capture_pose_from_row(row)
            if pose_entry is not None:
                trajectory.append(pose_entry)
    trajectory.sort(key=lambda item: item[0])
    return trajectory


def _load_odometry_trajectory(capture_dir: Path) -> list[tuple[int, np.ndarray]]:
    csv_path = capture_dir / "odometry.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        trajectory = []
        seen_timestamps = set()
        for row in reader:
            required_fields = [
                "pose_timestamp_ns",
                "position_x",
                "position_y",
                "position_z",
                "orientation_w",
                "orientation_x",
                "orientation_y",
                "orientation_z",
            ]
            if any(not row.get(field, "").strip() for field in required_fields):
                continue

            timestamp_ns = int(row["pose_timestamp_ns"])
            if timestamp_ns in seen_timestamps:
                continue
            seen_timestamps.add(timestamp_ns)

            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = _quaternion_to_matrix(
                float(row["orientation_w"]),
                float(row["orientation_x"]),
                float(row["orientation_y"]),
                float(row["orientation_z"]),
            )
            transform[:3, 3] = np.array(
                [
                    float(row["position_x"]),
                    float(row["position_y"]),
                    float(row["position_z"]),
                ],
                dtype=np.float64,
            )
            trajectory.append((timestamp_ns, transform))

    trajectory.sort(key=lambda item: item[0])
    return trajectory


def _load_dataset_frames(dataset_dir: Path) -> list[dict]:
    frames = []
    for filename in ("transforms_train.json", "transforms_test.json"):
        path = dataset_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            contents = json.load(handle)
        frames.extend(contents.get("frames", []))
    frames.sort(key=lambda frame: int(frame.get("time_ns", 0)))
    return frames


def _stored_transform_from_frame(frame: dict) -> tuple[int, np.ndarray]:
    return int(frame["time_ns"]), np.asarray(frame["transform_matrix"], dtype=np.float64)


def _loaded_training_pose_from_frame(frame: dict) -> tuple[int, np.ndarray]:
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float64).copy()
    c2w[:3, 1:3] *= -1.0
    return int(frame["time_ns"]), c2w


def _write_tum_file(path: Path, trajectory: list[tuple[int, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for timestamp_ns, transform in trajectory:
            handle.write(_matrix_to_tum_row(timestamp_ns, transform))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Kimera and converted dataset poses to TUM format")
    parser.add_argument("--capture-dir", required=True, help="Path to the recorded Kimera capture")
    parser.add_argument("--dataset-dir", required=True, help="Path to the converted triangle-splatting dataset")
    parser.add_argument("--camera", default="cam0", choices=["cam0", "cam1"], help="Camera CSV to read from the capture")
    parser.add_argument("--output-dir", required=True, help="Directory to write exported TUM trajectories")
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    raw_trajectory = _load_raw_capture_trajectory(capture_dir, args.camera)
    odometry_trajectory = _load_odometry_trajectory(capture_dir)
    dataset_frames = _load_dataset_frames(dataset_dir)
    stored_trajectory = [_stored_transform_from_frame(frame) for frame in dataset_frames]
    loaded_trajectory = [_loaded_training_pose_from_frame(frame) for frame in dataset_frames]

    _write_tum_file(output_dir / "raw_capture_left_cam.tum", raw_trajectory)
    _write_tum_file(output_dir / "raw_capture_odometry_base_link.tum", odometry_trajectory)
    _write_tum_file(output_dir / "dataset_stored_transform_matrix.tum", stored_trajectory)
    _write_tum_file(output_dir / "dataset_loaded_training_pose.tum", loaded_trajectory)

    summary = {
        "capture_dir": str(capture_dir),
        "dataset_dir": str(dataset_dir),
        "camera": args.camera,
        "raw_pose_count": len(raw_trajectory),
        "odometry_pose_count": len(odometry_trajectory),
        "dataset_frame_count": len(dataset_frames),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
