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


def _load_euroc_calibration(mav0_dir: Path, camera_name: str) -> CameraCalibration:
    sensor_path = mav0_dir / camera_name / "sensor.yaml"
    if not sensor_path.exists():
        raise FileNotFoundError(f"Missing calibration file: {sensor_path}")

    contents = yaml.safe_load(sensor_path.read_text(encoding="utf-8"))
    intrinsics = contents["intrinsics"]
    resolution = contents["resolution"]
    distortion = np.asarray(contents["distortion_coefficients"], dtype=np.float64)

    return CameraCalibration(
        width=int(resolution[0]),
        height=int(resolution[1]),
        fx=float(intrinsics[0]),
        fy=float(intrinsics[1]),
        cx=float(intrinsics[2]),
        cy=float(intrinsics[3]),
        distortion=distortion,
    )


def _camera_pose_from_row(row: dict[str, str]) -> np.ndarray | None:
    required_fields = [
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
    return transform


def _iter_capture_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _load_frame_rows(capture_dir: Path, camera_name: str) -> list[dict[str, str]]:
    csv_path = capture_dir / f"{camera_name}_frames.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing frame CSV: {csv_path}")

    valid_rows = []
    for row in _iter_capture_rows(csv_path):
        pose = _camera_pose_from_row(row)
        if pose is None:
            continue

        image_path = capture_dir / "images" / camera_name / row["filename"]
        if not image_path.exists():
            continue

        row = dict(row)
        row["_image_path"] = str(image_path)
        row["_camera_pose_world"] = pose
        valid_rows.append(row)

    valid_rows.sort(key=lambda row: int(row["image_timestamp_ns"]))
    return valid_rows


def _blender_transform_from_capture_pose(camera_pose_world: np.ndarray) -> list[list[float]]:
    transform = camera_pose_world.copy()
    transform[:3, 1:3] *= -1.0
    return transform.tolist()


def _undistort_image(
    image_path: Path,
    output_path: Path,
    calibration: CameraCalibration,
    new_camera_matrix: np.ndarray,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    if image.shape[1] != calibration.width or image.shape[0] != calibration.height:
        raise RuntimeError(
            f"Image {image_path} has shape {image.shape[1]}x{image.shape[0]}, "
            f"expected {calibration.width}x{calibration.height}"
        )

    undistorted = cv2.undistort(
        image,
        calibration.camera_matrix,
        calibration.distortion,
        None,
        new_camera_matrix,
    )
    undistorted_rgb = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2RGB)
    if not cv2.imwrite(str(output_path), undistorted_rgb):
        raise RuntimeError(f"Failed to write undistorted image: {output_path}")


def _load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    vertices = []
    with mesh_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not vertices:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(vertices, dtype=np.float32)


def _write_point_cloud_ply(output_path: Path, points: np.ndarray) -> None:
    colors = np.full((len(points), 3), 180, dtype=np.uint8)
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

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        handle.write("\n")
        for point, normal, color in zip(points, normals, colors):
            handle.write(
                f"{point[0]} {point[1]} {point[2]} "
                f"{normal[0]} {normal[1]} {normal[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _build_mesh_seed_point_cloud(
    capture_dir: Path,
    output_path: Path,
    voxel_size: float,
) -> dict[str, object]:
    meshes_csv = capture_dir / "meshes.csv"
    if not meshes_csv.exists():
        return {"mesh_count": 0, "input_vertex_count": 0, "output_vertex_count": 0}

    all_vertices = []
    mesh_count = 0
    for row in _iter_capture_rows(meshes_csv):
        obj_filename = row.get("obj_filename", "").strip()
        if not obj_filename:
            continue
        mesh_path = capture_dir / "meshes" / obj_filename
        if not mesh_path.exists():
            continue
        vertices = _load_mesh_vertices(mesh_path)
        if len(vertices) == 0:
            continue
        mesh_count += 1
        all_vertices.append(vertices)

    if not all_vertices:
        return {"mesh_count": mesh_count, "input_vertex_count": 0, "output_vertex_count": 0}

    points = np.concatenate(all_vertices, axis=0)
    if voxel_size > 0.0:
        voxel_keys = np.floor(points / voxel_size).astype(np.int64)
        _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
        points = points[np.sort(unique_indices)]

    _write_point_cloud_ply(output_path, points)
    return {
        "mesh_count": mesh_count,
        "input_vertex_count": int(sum(len(vertices) for vertices in all_vertices)),
        "output_vertex_count": int(len(points)),
    }


def _write_transforms_json(
    output_path: Path,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    frames: list[dict[str, object]],
) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Kimera capture into a trainable triangle-splatting dataset")
    parser.add_argument("--capture-dir", required=True, help="Path to the recorded Kimera capture")
    parser.add_argument("--mav0-dir", required=True, help="Path to the EuRoC mav0 calibration directory")
    parser.add_argument("--output-dir", required=True, help="Where to write the converted dataset")
    parser.add_argument("--camera", default="cam0", choices=["cam0", "cam1"], help="Which camera stream to convert")
    parser.add_argument("--llffhold", type=int, default=8, help="Every Nth frame goes to transforms_test.json")
    parser.add_argument("--mesh-voxel-size", type=float, default=0.02, help="Downsample size for the mesh-seeded point cloud; <=0 disables downsampling")
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).expanduser().resolve()
    mav0_dir = Path(args.mav0_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    calibration = _load_euroc_calibration(mav0_dir, args.camera)
    frame_rows = _load_frame_rows(capture_dir, args.camera)
    if not frame_rows:
        raise RuntimeError(
            f"No frames with camera poses were found in {capture_dir / f'{args.camera}_frames.csv'}. "
            "This capture cannot be converted into a trainable dataset until pose rows are present."
        )

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        calibration.camera_matrix,
        calibration.distortion,
        (calibration.width, calibration.height),
        0.0,
        (calibration.width, calibration.height),
        centerPrincipalPoint=True,
    )
    new_fx = float(new_camera_matrix[0, 0])
    new_fy = float(new_camera_matrix[1, 1])
    new_cx = float(new_camera_matrix[0, 2])
    new_cy = float(new_camera_matrix[1, 2])

    transformed_frames = []
    for row in frame_rows:
        image_path = Path(row["_image_path"])
        output_stem = image_path.stem
        output_image_path = images_dir / f"{output_stem}.png"
        _undistort_image(image_path, output_image_path, calibration, new_camera_matrix)
        transformed_frames.append(
            {
                "file_path": f"images/{output_stem}",
                "time_ns": int(row["image_timestamp_ns"]),
                "transform_matrix": _blender_transform_from_capture_pose(row["_camera_pose_world"]),
            }
        )

    train_frames = []
    test_frames = []
    for index, frame in enumerate(transformed_frames):
        if args.llffhold > 0 and index % args.llffhold == 0:
            test_frames.append(frame)
        else:
            train_frames.append(frame)

    _write_transforms_json(
        output_dir / "transforms_train.json",
        calibration.width,
        calibration.height,
        new_fx,
        new_fy,
        new_cx,
        new_cy,
        train_frames,
    )
    _write_transforms_json(
        output_dir / "transforms_test.json",
        calibration.width,
        calibration.height,
        new_fx,
        new_fy,
        new_cx,
        new_cy,
        test_frames,
    )

    point_cloud_stats = _build_mesh_seed_point_cloud(
        capture_dir,
        output_dir / "points3d.ply",
        voxel_size=args.mesh_voxel_size,
    )

    conversion_summary = {
        "source_capture_dir": str(capture_dir),
        "source_mav0_dir": str(mav0_dir),
        "camera": args.camera,
        "frame_count": len(transformed_frames),
        "train_frame_count": len(train_frames),
        "test_frame_count": len(test_frames),
        "undistorted_intrinsics": {
            "fx": new_fx,
            "fy": new_fy,
            "cx": new_cx,
            "cy": new_cy,
            "width": calibration.width,
            "height": calibration.height,
        },
        "point_cloud": point_cloud_stats,
    }
    (output_dir / "kimera_conversion.json").write_text(
        json.dumps(conversion_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(conversion_summary, indent=2))


if __name__ == "__main__":
    main()
