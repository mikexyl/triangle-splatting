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


@dataclass
class MeshGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    texcoords: np.ndarray
    face_texcoords: np.ndarray
    vertex_colors: np.ndarray
    texture_path: Path | None


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


def _parse_obj_index(index_token: str, value_count: int, value_name: str) -> int:
    if not index_token:
        raise ValueError(f"Missing {value_name} index")

    index = int(index_token)
    if index < 0:
        index = value_count + index
    else:
        index -= 1

    if index < 0 or index >= value_count:
        raise ValueError(f"{value_name} index {index + 1} is out of range for mesh with {value_count} {value_name}s")
    return index


def _parse_obj_face_token(token: str, vertex_count: int, texcoord_count: int) -> tuple[int, int]:
    parts = token.split("/")
    vertex_index = _parse_obj_index(parts[0], vertex_count, "vertex")
    texcoord_index = -1
    if len(parts) > 1 and parts[1]:
        texcoord_index = _parse_obj_index(parts[1], texcoord_count, "texture coordinate")
    return vertex_index, texcoord_index


def _load_texture_from_mtl(mtl_path: Path) -> Path | None:
    if not mtl_path.exists():
        return None

    with mtl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0] != "map_Kd" or len(parts) < 2:
                continue
            return (mtl_path.parent / parts[-1]).resolve()
    return None


def _resolve_obj_texture_path(mesh_path: Path, mtl_filenames: list[str], texture_filename: str | None) -> Path | None:
    if texture_filename:
        texture_path = mesh_path.parent / texture_filename
        if texture_path.exists():
            return texture_path.resolve()

    for mtl_filename in mtl_filenames:
        texture_path = _load_texture_from_mtl(mesh_path.parent / mtl_filename)
        if texture_path is not None and texture_path.exists():
            return texture_path
    return None


def _load_mesh_geometry(mesh_path: Path, texture_filename: str | None = None) -> MeshGeometry:
    vertices = []
    vertex_colors = []
    has_vertex_colors = False
    texcoords = []
    faces = []
    face_texcoords = []
    mtl_filenames = []
    with mesh_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("mtllib "):
                mtl_filenames.extend(stripped.split()[1:])
                continue

            if stripped.startswith("v "):
                parts = stripped.split()
                if len(parts) < 4:
                    continue
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    vertex_colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
                    has_vertex_colors = True
                else:
                    vertex_colors.append([np.nan, np.nan, np.nan])
                continue

            if stripped.startswith("vt "):
                parts = stripped.split()
                if len(parts) >= 3:
                    texcoords.append([float(parts[1]), float(parts[2])])
                continue

            if not stripped.startswith("f "):
                continue

            parts = stripped.split()[1:]
            if len(parts) < 3:
                continue

            try:
                polygon = [_parse_obj_face_token(part, len(vertices), len(texcoords)) for part in parts]
            except ValueError:
                continue

            # OBJ faces may be quads or ngons; triangulate with a simple fan.
            for idx in range(1, len(polygon) - 1):
                face = [polygon[0], polygon[idx], polygon[idx + 1]]
                faces.append([face_vertex[0] for face_vertex in face])
                face_texcoords.append([face_vertex[1] for face_vertex in face])

    vertices_array = np.asarray(vertices, dtype=np.float32) if vertices else np.empty((0, 3), dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.int32) if faces else np.empty((0, 3), dtype=np.int32)
    texcoords_array = np.asarray(texcoords, dtype=np.float32) if texcoords else np.empty((0, 2), dtype=np.float32)
    face_texcoords_array = (
        np.asarray(face_texcoords, dtype=np.int32)
        if face_texcoords
        else np.empty((0, 3), dtype=np.int32)
    )

    vertex_colors_array = np.empty((0, 3), dtype=np.float32)
    if has_vertex_colors and len(vertex_colors) == len(vertices):
        vertex_colors_array = np.asarray(vertex_colors, dtype=np.float32)
        if not np.isnan(vertex_colors_array).any():
            if vertex_colors_array.size and float(vertex_colors_array.max()) > 1.0:
                vertex_colors_array = vertex_colors_array / 255.0
            vertex_colors_array = np.clip(vertex_colors_array, 0.0, 1.0).astype(np.float32, copy=False)
        else:
            vertex_colors_array = np.empty((0, 3), dtype=np.float32)

    return MeshGeometry(
        vertices=vertices_array,
        faces=faces_array,
        texcoords=texcoords_array,
        face_texcoords=face_texcoords_array,
        vertex_colors=vertex_colors_array,
        texture_path=_resolve_obj_texture_path(mesh_path, mtl_filenames, texture_filename),
    )


def _sample_mesh_surface_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    sample_spacing: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(vertices) == 0 or len(faces) == 0:
        return np.empty((0, 3), dtype=np.float32)

    triangles = vertices[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)

    sample_area = max(sample_spacing * sample_spacing * 0.5, 1e-8)
    samples_per_face = np.maximum(1, np.ceil(areas / sample_area).astype(np.int32))

    total_samples = int(samples_per_face.sum())
    sampled_points = np.empty((total_samples, 3), dtype=np.float32)
    offset = 0
    for triangle, count in zip(triangles, samples_per_face):
        random_a = rng.random(count, dtype=np.float32)
        random_b = rng.random(count, dtype=np.float32)
        sqrt_random_a = np.sqrt(random_a)

        barycentric_0 = 1.0 - sqrt_random_a
        barycentric_1 = sqrt_random_a * (1.0 - random_b)
        barycentric_2 = sqrt_random_a * random_b

        sampled_points[offset:offset + count] = (
            barycentric_0[:, None] * triangle[0]
            + barycentric_1[:, None] * triangle[1]
            + barycentric_2[:, None] * triangle[2]
        )
        offset += count

    return sampled_points


def _split_triangles_once(triangles: np.ndarray) -> np.ndarray:
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]

    ab = (a + b) * 0.5
    ac = (a + c) * 0.5
    bc = (b + c) * 0.5

    return np.concatenate(
        [
            np.stack([a, ab, ac], axis=1),
            np.stack([b, bc, ab], axis=1),
            np.stack([c, ac, bc], axis=1),
            np.stack([ab, bc, ac], axis=1),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _triangle_max_edges(triangles: np.ndarray) -> np.ndarray:
    edge_ab = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    edge_bc = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    edge_ca = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    return np.maximum(np.maximum(edge_ab, edge_bc), edge_ca)


def _subdivide_triangles_by_max_edge(
    triangles: np.ndarray,
    max_edge: float,
    max_count: int,
    vertex_attributes: list[np.ndarray | None] | None = None,
) -> tuple[np.ndarray, list[np.ndarray | None], bool]:
    vertex_attributes = vertex_attributes or []

    def truncate_attributes(count: int) -> list[np.ndarray | None]:
        return [
            None if attribute is None else attribute[:count].astype(np.float32, copy=False)
            for attribute in vertex_attributes
        ]

    if len(triangles) == 0 or max_edge <= 0.0:
        if max_count > 0 and len(triangles) > max_count:
            return triangles[:max_count].astype(np.float32, copy=False), truncate_attributes(max_count), True
        return triangles.astype(np.float32, copy=False), truncate_attributes(len(triangles)), False

    triangles = triangles.astype(np.float32, copy=False)
    vertex_attributes = [
        None if attribute is None else attribute.astype(np.float32, copy=False)
        for attribute in vertex_attributes
    ]
    cap_reached = False
    while True:
        max_edges = _triangle_max_edges(triangles)
        oversized_indices = np.nonzero(max_edges > max_edge)[0]
        if len(oversized_indices) == 0:
            break

        if max_count > 0:
            available_new_triangles = max_count - len(triangles)
            split_count = min(len(oversized_indices), available_new_triangles // 3)
            if split_count <= 0:
                cap_reached = True
                break

            largest_order = np.argsort(max_edges[oversized_indices])[::-1]
            selected_indices = oversized_indices[largest_order[:split_count]]
            if split_count < len(oversized_indices):
                cap_reached = True
        else:
            selected_indices = oversized_indices

        keep_mask = np.ones(len(triangles), dtype=bool)
        keep_mask[selected_indices] = False
        split_triangles = _split_triangles_once(triangles[selected_indices])
        triangles = np.concatenate(
            [triangles[keep_mask], split_triangles],
            axis=0,
        )
        vertex_attributes = [
            None
            if attribute is None
            else np.concatenate([attribute[keep_mask], _split_triangles_once(attribute[selected_indices])], axis=0)
            for attribute in vertex_attributes
        ]

        if cap_reached:
            break

    return triangles.astype(np.float32, copy=False), vertex_attributes, cap_reached


def _load_texture_image(texture_path: Path | None) -> np.ndarray | None:
    if texture_path is None or not texture_path.exists():
        return None

    image = cv2.imread(str(texture_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return None

    image = image.astype(np.float32)
    max_value = float(image.max()) if image.size else 1.0
    if max_value > 1.0:
        image /= 255.0 if max_value <= 255.0 else max_value
    return np.clip(image, 0.0, 1.0)


def _sample_texture_bilinear(texture: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    height, width = texture.shape[:2]
    u = np.clip(uvs[:, 0], 0.0, 1.0)
    v = np.clip(uvs[:, 1], 0.0, 1.0)

    x = u * (width - 1)
    # Kimera writes normalized image coordinates for these textures, so v=0 is
    # the top image row rather than the usual OBJ bottom-left convention.
    y = v * (height - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]

    top = texture[y0, x0] * (1.0 - wx) + texture[y0, x1] * wx
    bottom = texture[y1, x0] * (1.0 - wx) + texture[y1, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32, copy=False)


def _triangle_seed_colors(
    triangle_count: int,
    uv_triangles: np.ndarray | None,
    texture: np.ndarray | None,
    vertex_color_triangles: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    colors = np.full((triangle_count, 3), 180.0 / 255.0, dtype=np.float32)
    texture_mask = np.zeros((triangle_count,), dtype=bool)
    source = "neutral_gray"

    if texture is not None and uv_triangles is not None and len(uv_triangles) == triangle_count:
        finite_uv_mask = np.isfinite(uv_triangles).all(axis=(1, 2))
        if np.any(finite_uv_mask):
            centroid_uvs = uv_triangles[finite_uv_mask].mean(axis=1)
            colors[finite_uv_mask] = _sample_texture_bilinear(texture, centroid_uvs)
            texture_mask[finite_uv_mask] = True
            source = "texture"

    if not np.all(texture_mask) and vertex_color_triangles is not None and len(vertex_color_triangles) == triangle_count:
        finite_color_mask = np.isfinite(vertex_color_triangles).all(axis=(1, 2))
        if np.any(texture_mask):
            finite_color_mask &= ~np.isfinite(uv_triangles).all(axis=(1, 2))
        if np.any(finite_color_mask):
            colors[finite_color_mask] = np.clip(vertex_color_triangles[finite_color_mask].mean(axis=1), 0.0, 1.0)
            source = "texture_and_vertex_color" if np.any(texture_mask) else "vertex_color"

    return colors, texture_mask, source


def _write_triangle_soup_npz(output_path: Path, triangles: np.ndarray, colors: np.ndarray | None = None) -> None:
    if colors is None:
        colors = np.full((len(triangles), 3), 180.0 / 255.0, dtype=np.float32)
    else:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.shape != (len(triangles), 3):
            raise ValueError(f"Triangle colors must have shape {(len(triangles), 3)}, got {colors.shape}")
        if colors.size and colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)

    np.savez_compressed(
        output_path,
        triangles=triangles.astype(np.float32, copy=False),
        colors=colors.astype(np.float32, copy=False),
    )


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


def _downsample_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0.0:
        return points

    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def _triangle_reduction_summary(
    mode: str,
    input_count: int,
    output_count: int,
    voxel_size: float,
    normal_bins: int,
) -> dict[str, object]:
    return {
        "mode": mode,
        "input_triangle_count": int(input_count),
        "output_triangle_count": int(output_count),
        "removed_triangle_count": int(max(input_count - output_count, 0)),
        "voxel_size": float(voxel_size) if mode == "voxel" else 0.0,
        "normal_bins": int(normal_bins) if mode == "voxel" else 0,
    }


def _triangle_normals(triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norm, 1e-12)


def _reduce_triangle_soup(
    triangles: np.ndarray,
    colors: np.ndarray,
    texture_mask: np.ndarray,
    mode: str,
    voxel_size: float,
    normal_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    input_count = int(len(triangles))
    if input_count == 0 or mode == "concat" or voxel_size <= 0.0:
        return (
            triangles.astype(np.float32, copy=False),
            colors.astype(np.float32, copy=False),
            texture_mask.astype(bool, copy=False),
            _triangle_reduction_summary("concat", input_count, input_count, 0.0, 0),
        )

    centroids = triangles.mean(axis=1)
    keys = [np.floor(centroids / voxel_size).astype(np.int64)]
    if normal_bins > 0:
        normals = np.clip(_triangle_normals(triangles), -1.0, 1.0)
        normal_keys = np.floor((normals + 1.0) * normal_bins).astype(np.int64)
        normal_keys = np.clip(normal_keys, 0, normal_bins * 2 - 1)
        keys.append(normal_keys)
    voxel_keys = np.concatenate(keys, axis=1)

    _, first_indices, inverse = np.unique(voxel_keys, axis=0, return_index=True, return_inverse=True)
    group_count = len(first_indices)
    counts = np.bincount(inverse, minlength=group_count).astype(np.float32)

    color_sums = np.zeros((group_count, 3), dtype=np.float64)
    np.add.at(color_sums, inverse, colors.astype(np.float64, copy=False))
    reduced_colors = color_sums / np.maximum(counts[:, None], 1.0)

    if texture_mask.size:
        texture_weights = texture_mask.astype(np.float32, copy=False)
        texture_counts = np.bincount(inverse, weights=texture_weights, minlength=group_count).astype(np.float32)
        texture_color_sums = np.zeros((group_count, 3), dtype=np.float64)
        np.add.at(texture_color_sums, inverse, colors.astype(np.float64, copy=False) * texture_weights[:, None])
        textured_groups = texture_counts > 0.0
        reduced_colors[textured_groups] = (
            texture_color_sums[textured_groups] / texture_counts[textured_groups, None]
        )
        reduced_texture_mask = textured_groups
    else:
        reduced_texture_mask = np.zeros((group_count,), dtype=bool)

    # np.unique sorts by key. Restore first-observation order so the reduced soup
    # remains stable with respect to the original mesh timeline.
    output_order = np.argsort(first_indices)
    reduced_triangles = triangles[first_indices][output_order].astype(np.float32, copy=False)
    reduced_colors = np.clip(reduced_colors[output_order], 0.0, 1.0).astype(np.float32, copy=False)
    reduced_texture_mask = reduced_texture_mask[output_order].astype(bool, copy=False)
    return (
        reduced_triangles,
        reduced_colors,
        reduced_texture_mask,
        _triangle_reduction_summary(mode, input_count, len(reduced_triangles), voxel_size, normal_bins),
    )


def _build_mesh_seed_entries(
    capture_dir: Path,
    output_dir: Path,
    voxel_size: float,
    sample_spacing: float,
    triangle_max_edge: float,
    triangle_max_count: int,
    triangle_color_source: str,
    triangle_merge_mode: str,
    triangle_merge_voxel_size: float,
    triangle_merge_normal_bins: int,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    empty_reduction = _triangle_reduction_summary(
        triangle_merge_mode,
        0,
        0,
        triangle_merge_voxel_size,
        triangle_merge_normal_bins,
    )
    meshes_csv = capture_dir / "meshes.csv"
    if not meshes_csv.exists():
        return [], {
            "mesh_count": 0,
            "mesh_seed_file_count": 0,
            "input_vertex_count": 0,
            "input_face_count": 0,
            "sampled_point_count": 0,
            "output_point_count": 0,
        }, {
            "mesh_seed_file_count": 0,
            "output_triangle_count": 0,
            "cap_reached_file_count": 0,
            "max_edge": float(triangle_max_edge),
            "max_count": int(triangle_max_count),
            "textured_triangle_count": 0,
            "texture_seed_file_count": 0,
            "color_source": triangle_color_source,
            "reduction": empty_reduction,
        }, np.empty((0, 3), dtype=np.float32), np.empty((0, 3, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    all_points = []
    all_triangles = []
    all_triangle_colors = []
    all_triangle_texture_masks = []
    mesh_seed_entries = []
    mesh_count = 0
    total_vertex_count = 0
    total_face_count = 0
    total_sampled_point_count = 0
    total_output_triangle_count = 0
    total_textured_triangle_count = 0
    triangle_seed_file_count = 0
    texture_seed_file_count = 0
    triangle_cap_reached_file_count = 0
    effective_sample_spacing = sample_spacing
    if effective_sample_spacing <= 0.0:
        effective_sample_spacing = voxel_size * 0.5 if voxel_size > 0.0 else 0.01

    mesh_seed_dir = output_dir / "mesh_seed_points"
    mesh_seed_dir.mkdir(parents=True, exist_ok=True)
    triangle_seed_dir = output_dir / "mesh_seed_triangles"
    triangle_seed_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for row in _iter_capture_rows(meshes_csv):
        obj_filename = row.get("obj_filename", "").strip()
        if not obj_filename:
            continue
        mesh_path = capture_dir / "meshes" / obj_filename
        if not mesh_path.exists():
            continue
        mesh = _load_mesh_geometry(mesh_path, row.get("texture_filename", "").strip() or None)
        vertices = mesh.vertices
        faces = mesh.faces
        if len(vertices) == 0:
            continue

        triangle_seed_rel_path = None
        triangle_count = 0
        textured_triangle_count = 0
        triangle_cap_reached = False
        color_source = None
        if len(faces) > 0:
            mesh_triangles = vertices[faces]
            uv_triangles = None
            if len(mesh.texcoords) > 0 and mesh.face_texcoords.shape == faces.shape:
                uv_triangles = np.full((len(faces), 3, 2), np.nan, dtype=np.float32)
                valid_uv_mask = mesh.face_texcoords >= 0
                if np.any(valid_uv_mask):
                    uv_triangles[valid_uv_mask] = mesh.texcoords[mesh.face_texcoords[valid_uv_mask]]

            vertex_color_triangles = None
            if triangle_color_source in ("texture", "vertex") and len(mesh.vertex_colors) == len(vertices):
                vertex_color_triangles = mesh.vertex_colors[faces]

            mesh_triangles, subdivided_attributes, triangle_cap_reached = _subdivide_triangles_by_max_edge(
                mesh_triangles,
                max_edge=triangle_max_edge,
                max_count=triangle_max_count,
                vertex_attributes=[uv_triangles, vertex_color_triangles],
            )
            uv_triangles, vertex_color_triangles = subdivided_attributes
            if len(mesh_triangles) > 0:
                texture = _load_texture_image(mesh.texture_path) if triangle_color_source == "texture" else None
                if triangle_color_source == "gray":
                    vertex_color_triangles = None
                triangle_colors, triangle_texture_mask, color_source = _triangle_seed_colors(
                    len(mesh_triangles),
                    uv_triangles,
                    texture,
                    vertex_color_triangles,
                )
                textured_triangle_count = int(triangle_texture_mask.sum())
                triangle_seed_file_count += 1
                if textured_triangle_count > 0:
                    texture_seed_file_count += 1
                triangle_count = int(len(mesh_triangles))
                total_output_triangle_count += triangle_count
                total_textured_triangle_count += textured_triangle_count
                if triangle_cap_reached:
                    triangle_cap_reached_file_count += 1
                    cap_reason = (
                        f"before all edges reached {triangle_max_edge}"
                        if triangle_max_edge > 0.0
                        else "because --mesh-triangle-max-count truncated the mesh"
                    )
                    print(
                        f"[WARN] Triangle seed cap reached for {obj_filename}; "
                        f"wrote {triangle_count} triangles {cap_reason}."
                    )
                all_triangles.append(mesh_triangles)
                all_triangle_colors.append(triangle_colors)
                all_triangle_texture_masks.append(triangle_texture_mask)
                triangle_seed_rel_path = f"mesh_seed_triangles/{Path(obj_filename).stem}.npz"
                _write_triangle_soup_npz(output_dir / triangle_seed_rel_path, mesh_triangles, triangle_colors)

        surface_points = _sample_mesh_surface_points(vertices, faces, effective_sample_spacing, rng)
        mesh_points = np.concatenate([vertices, surface_points], axis=0) if len(surface_points) else vertices
        mesh_points = _downsample_points(mesh_points, voxel_size)
        if len(mesh_points) == 0:
            continue

        mesh_count += 1
        total_vertex_count += int(len(vertices))
        total_face_count += int(len(faces))
        total_sampled_point_count += int(len(surface_points))
        all_points.append(mesh_points)
        mesh_timestamp = row.get("mesh_timestamp_ns", "").strip() or row.get("pose_timestamp_ns", "").strip()
        seed_rel_path = f"mesh_seed_points/{Path(obj_filename).stem}.ply"
        _write_point_cloud_ply(output_dir / seed_rel_path, mesh_points)
        mesh_seed_entries.append(
            {
                "time_ns": int(mesh_timestamp) if mesh_timestamp else mesh_count - 1,
                "seed_rel_path": seed_rel_path,
                "triangle_seed_rel_path": triangle_seed_rel_path,
                "point_count": int(len(mesh_points)),
                "triangle_count": triangle_count,
                "textured_triangle_count": textured_triangle_count,
                "triangle_cap_reached": triangle_cap_reached,
                "texture_filename": row.get("texture_filename", "").strip() or None,
                "texture_path": str(mesh.texture_path) if mesh.texture_path else None,
                "triangle_color_source": color_source if triangle_count > 0 else None,
                "obj_filename": obj_filename,
            }
        )

    if not all_points:
        return mesh_seed_entries, {
            "mesh_count": mesh_count,
            "mesh_seed_file_count": len(mesh_seed_entries),
            "input_vertex_count": 0,
            "input_face_count": 0,
            "sampled_point_count": 0,
            "output_point_count": 0,
        }, {
            "mesh_seed_file_count": triangle_seed_file_count,
            "output_triangle_count": total_output_triangle_count,
            "cap_reached_file_count": triangle_cap_reached_file_count,
            "max_edge": float(triangle_max_edge),
            "max_count": int(triangle_max_count),
            "textured_triangle_count": total_textured_triangle_count,
            "texture_seed_file_count": texture_seed_file_count,
            "color_source": triangle_color_source,
            "reduction": empty_reduction,
        }, np.empty((0, 3), dtype=np.float32), np.empty((0, 3, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    points = _downsample_points(np.concatenate(all_points, axis=0), voxel_size)
    triangles = (
        np.concatenate(all_triangles, axis=0)
        if all_triangles
        else np.empty((0, 3, 3), dtype=np.float32)
    )
    triangle_colors = (
        np.concatenate(all_triangle_colors, axis=0)
        if all_triangle_colors
        else np.empty((0, 3), dtype=np.float32)
    )
    triangle_texture_mask = (
        np.concatenate(all_triangle_texture_masks, axis=0)
        if all_triangle_texture_masks
        else np.empty((0,), dtype=bool)
    )
    triangles, triangle_colors, triangle_texture_mask, triangle_reduction_stats = _reduce_triangle_soup(
        triangles,
        triangle_colors,
        triangle_texture_mask,
        mode=triangle_merge_mode,
        voxel_size=triangle_merge_voxel_size,
        normal_bins=triangle_merge_normal_bins,
    )
    mesh_seed_entries.sort(key=lambda entry: entry["time_ns"])
    return mesh_seed_entries, {
        "mesh_count": mesh_count,
        "mesh_seed_file_count": len(mesh_seed_entries),
        "input_vertex_count": total_vertex_count,
        "input_face_count": total_face_count,
        "sampled_point_count": total_sampled_point_count,
        "output_point_count": int(len(points)),
        "sample_spacing": float(effective_sample_spacing),
    }, {
        "mesh_seed_file_count": triangle_seed_file_count,
        "output_triangle_count": int(len(triangles)),
        "per_frame_triangle_count": int(total_output_triangle_count),
        "cap_reached_file_count": triangle_cap_reached_file_count,
        "max_edge": float(triangle_max_edge),
        "max_count": int(triangle_max_count),
        "textured_triangle_count": int(triangle_texture_mask.sum()),
        "per_frame_textured_triangle_count": int(total_textured_triangle_count),
        "texture_seed_file_count": int(texture_seed_file_count),
        "color_source": triangle_color_source,
        "reduction": triangle_reduction_stats,
    }, points, triangles, triangle_colors


def _select_mesh_seed_entry(frame_time_ns: int, mesh_seed_entries: list[dict[str, object]]) -> dict[str, object] | None:
    if not mesh_seed_entries:
        return None

    mesh_times = np.asarray([entry["time_ns"] for entry in mesh_seed_entries], dtype=np.int64)
    insert_idx = int(np.searchsorted(mesh_times, frame_time_ns))

    candidate_indices = []
    if insert_idx < len(mesh_seed_entries):
        candidate_indices.append(insert_idx)
    if insert_idx > 0:
        candidate_indices.append(insert_idx - 1)

    best_idx = min(candidate_indices, key=lambda idx: abs(int(mesh_times[idx]) - frame_time_ns))
    return mesh_seed_entries[best_idx]


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
    parser.add_argument(
        "--mesh-sample-spacing",
        type=float,
        default=0.0,
        help="Approximate spacing used to interpolate seed points across mesh faces before voxel downsampling; <=0 uses half the voxel size.",
    )
    parser.add_argument(
        "--mesh-triangle-max-edge",
        type=float,
        default=0.0,
        help="Split mesh seed triangles until their longest edge is at most this size in meters; <=0 disables subdivision.",
    )
    parser.add_argument(
        "--mesh-triangle-max-count",
        type=int,
        default=500000,
        help="Maximum triangles to write per mesh seed after subdivision; <=0 disables the cap.",
    )
    parser.add_argument(
        "--mesh-triangle-color-source",
        choices=["texture", "vertex", "gray"],
        default="texture",
        help="Appearance initialization for mesh triangle seeds. texture samples OBJ UV textures, vertex uses OBJ vertex colors, and gray preserves the previous neutral initialization.",
    )
    parser.add_argument(
        "--mesh-triangle-merge-mode",
        choices=["concat", "voxel"],
        default="concat",
        help="How to build mesh_seed_triangles/all.npz from per-frame Kimera meshes. concat preserves all frame meshes; voxel merges overlapping frame meshes by triangle centroid.",
    )
    parser.add_argument(
        "--mesh-triangle-merge-voxel-size",
        type=float,
        default=0.05,
        help="Centroid voxel size in meters for --mesh-triangle-merge-mode voxel.",
    )
    parser.add_argument(
        "--mesh-triangle-merge-normal-bins",
        type=int,
        default=0,
        help="Optional normal-direction bins for voxel triangle merging. 0 merges by centroid only; larger values preserve more differently oriented triangles in the same voxel.",
    )
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
    (
        mesh_seed_entries,
        point_cloud_stats,
        triangle_soup_stats,
        merged_seed_points,
        merged_seed_triangles,
        merged_seed_triangle_colors,
    ) = _build_mesh_seed_entries(
        capture_dir,
        output_dir,
        voxel_size=args.mesh_voxel_size,
        sample_spacing=args.mesh_sample_spacing,
        triangle_max_edge=args.mesh_triangle_max_edge,
        triangle_max_count=args.mesh_triangle_max_count,
        triangle_color_source=args.mesh_triangle_color_source,
        triangle_merge_mode=args.mesh_triangle_merge_mode,
        triangle_merge_voxel_size=args.mesh_triangle_merge_voxel_size,
        triangle_merge_normal_bins=args.mesh_triangle_merge_normal_bins,
    )
    if len(merged_seed_points) > 0:
        _write_point_cloud_ply(output_dir / "points3d.ply", merged_seed_points)
    if len(merged_seed_triangles) > 0:
        _write_triangle_soup_npz(
            output_dir / "mesh_seed_triangles" / "all.npz",
            merged_seed_triangles,
            merged_seed_triangle_colors,
        )

    transformed_frames = []
    for row in frame_rows:
        image_path = Path(row["_image_path"])
        output_stem = image_path.stem
        output_image_path = images_dir / f"{output_stem}.png"
        _undistort_image(image_path, output_image_path, calibration, new_camera_matrix)
        mesh_seed_entry = _select_mesh_seed_entry(int(row["image_timestamp_ns"]), mesh_seed_entries)
        mesh_seed_path = str(mesh_seed_entry["seed_rel_path"]) if mesh_seed_entry else None
        mesh_seed_triangle_path = (
            str(mesh_seed_entry.get("triangle_seed_rel_path"))
            if mesh_seed_entry and mesh_seed_entry.get("triangle_seed_rel_path")
            else None
        )
        transformed_frames.append(
            {
                "file_path": f"images/{output_stem}",
                "time_ns": int(row["image_timestamp_ns"]),
                "transform_matrix": _blender_transform_from_capture_pose(row["_camera_pose_world"]),
                "mesh_seed_path": mesh_seed_path,
                "mesh_seed_triangle_path": mesh_seed_triangle_path,
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

    conversion_summary = {
        "source_capture_dir": str(capture_dir),
        "source_mav0_dir": str(mav0_dir),
        "camera": args.camera,
        "frame_count": len(transformed_frames),
        "train_frame_count": len(train_frames),
        "test_frame_count": len(test_frames),
        "mesh_seed_directory": "mesh_seed_points" if mesh_seed_entries else None,
        "mesh_seed_triangle_directory": "mesh_seed_triangles" if triangle_soup_stats["mesh_seed_file_count"] else None,
        "undistorted_intrinsics": {
            "fx": new_fx,
            "fy": new_fy,
            "cx": new_cx,
            "cy": new_cy,
            "width": calibration.width,
            "height": calibration.height,
        },
        "point_cloud": point_cloud_stats,
        "triangle_soup": triangle_soup_stats,
    }
    (output_dir / "kimera_conversion.json").write_text(
        json.dumps(conversion_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(conversion_summary, indent=2))


if __name__ == "__main__":
    main()
