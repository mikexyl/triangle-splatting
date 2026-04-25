import argparse
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData
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


def _load_undistorted_rgb_image(
    image_path: Path,
    calibration: CameraCalibration,
    new_camera_matrix: np.ndarray,
) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    undistorted = cv2.undistort(
        image,
        calibration.camera_matrix,
        calibration.distortion,
        None,
        new_camera_matrix,
    )
    return cv2.cvtColor(undistorted, cv2.COLOR_GRAY2RGB)


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


def _triangle_areas(triangles: np.ndarray) -> np.ndarray:
    if len(triangles) == 0:
        return np.empty((0,), dtype=np.float32)
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    return (0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)).astype(np.float32, copy=False)


def _percentiles(values: np.ndarray, percentiles: tuple[int, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)) -> dict[str, float]:
    values = np.asarray(values)
    if values.size == 0:
        return {}
    computed = np.percentile(values.astype(np.float64, copy=False), percentiles)
    return {f"p{percentile:02d}": float(value) for percentile, value in zip(percentiles, computed)}


def _triangle_texture_samples(
    uv_triangles: np.ndarray | None,
    texture: np.ndarray | None,
    vertex_color_triangles: np.ndarray | None,
    fallback_triangle_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    triangle_count = fallback_triangle_count
    if uv_triangles is not None:
        triangle_count = len(uv_triangles)
    elif vertex_color_triangles is not None:
        triangle_count = len(vertex_color_triangles)

    samples = np.full((triangle_count, 7, 3), 180.0 / 255.0, dtype=np.float32)
    texture_mask = np.zeros((triangle_count,), dtype=bool)
    barycentric_samples = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.50, 0.50, 0.00],
            [0.50, 0.00, 0.50],
            [0.00, 0.50, 0.50],
        ],
        dtype=np.float32,
    )

    if texture is not None and uv_triangles is not None and len(uv_triangles) == triangle_count:
        finite_uv_mask = np.isfinite(uv_triangles).all(axis=(1, 2))
        if np.any(finite_uv_mask):
            sample_uvs = np.einsum(
                "sc,ncd->nsd",
                barycentric_samples,
                uv_triangles[finite_uv_mask],
                optimize=True,
            )
            samples[finite_uv_mask] = _sample_texture_bilinear(texture, sample_uvs.reshape(-1, 2)).reshape(-1, 7, 3)
            texture_mask[finite_uv_mask] = True

    if vertex_color_triangles is not None and len(vertex_color_triangles) == triangle_count:
        missing_texture = ~texture_mask
        finite_color_mask = missing_texture & np.isfinite(vertex_color_triangles).all(axis=(1, 2))
        if np.any(finite_color_mask):
            samples[finite_color_mask] = np.einsum(
                "sc,ncd->nsd",
                barycentric_samples,
                vertex_color_triangles[finite_color_mask],
                optimize=True,
            )

    return np.clip(samples, 0.0, 1.0).astype(np.float32, copy=False), texture_mask


def _triangle_color_complexity(samples: np.ndarray) -> np.ndarray:
    if len(samples) == 0:
        return np.empty((0,), dtype=np.float32)
    sample_std = np.std(samples.astype(np.float32, copy=False), axis=1)
    sample_range = np.ptp(samples.astype(np.float32, copy=False), axis=1)
    return np.maximum(np.linalg.norm(sample_std, axis=1), np.linalg.norm(sample_range, axis=1) * 0.25).astype(
        np.float32,
        copy=False,
    )


def _face_adjacency(faces: np.ndarray) -> list[list[int]]:
    adjacency = [set() for _ in range(len(faces))]
    edges_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for start, end in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(face[start]), int(face[end]))))
            edges_to_faces[edge].append(face_idx)

    for face_indices in edges_to_faces.values():
        if len(face_indices) < 2:
            continue
        for face_idx in face_indices:
            adjacency[face_idx].update(other_idx for other_idx in face_indices if other_idx != face_idx)
    return [sorted(neighbors) for neighbors in adjacency]


def _spatial_face_adjacency(centers: np.ndarray, radius: float) -> list[list[int]]:
    if len(centers) == 0 or radius <= 0.0:
        return [[] for _ in range(len(centers))]
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(centers)
        return [sorted(idx for idx in tree.query_ball_point(center, radius) if idx != center_idx) for center_idx, center in enumerate(centers)]
    except ImportError:
        adjacency = []
        radius2 = radius * radius
        for center_idx, center in enumerate(centers):
            dist2 = np.sum((centers - center[None, :]) ** 2, axis=1)
            neighbors = np.nonzero((dist2 <= radius2) & (np.arange(len(centers)) != center_idx))[0]
            adjacency.append(neighbors.tolist())
        return adjacency


def _merge_adjacency(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    merged = []
    for left_neighbors, right_neighbors in zip(left, right):
        merged.append(sorted(set(left_neighbors).union(right_neighbors)))
    return merged


def _patch_to_plane_triangles(
    patch_triangles: np.ndarray,
    max_plane_error: float,
    max_edge: float,
) -> tuple[np.ndarray | None, float]:
    points = np.unique(patch_triangles.reshape(-1, 3), axis=0).astype(np.float64, copy=False)
    if len(points) < 3:
        return None, 0.0

    center = points.mean(axis=0)
    centered = points - center[None, :]
    try:
        _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, 0.0
    if len(singular_values) < 2 or singular_values[1] <= 1e-9:
        return None, 0.0

    axis_x = vt[0]
    axis_y = vt[1]
    normal = vt[2] if vt.shape[0] >= 3 else np.cross(axis_x, axis_y)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    plane_distances = np.abs(centered @ normal)
    plane_error = float(np.max(plane_distances)) if len(plane_distances) else 0.0
    if max_plane_error > 0.0 and plane_error > max_plane_error:
        return None, plane_error

    coords = np.stack([centered @ axis_x, centered @ axis_y], axis=1).astype(np.float32)
    hull = cv2.convexHull(coords, returnPoints=True)
    hull_points = hull.reshape(-1, 2)
    if len(hull_points) < 3:
        return None, plane_error

    if len(hull_points) > 16:
        perimeter = cv2.arcLength(hull_points.reshape(-1, 1, 2), True)
        approximated = cv2.approxPolyDP(hull_points.reshape(-1, 1, 2), 0.015 * perimeter, True).reshape(-1, 2)
        if len(approximated) >= 3:
            hull_points = approximated

    hull_points_3d = (
        center[None, :]
        + hull_points[:, 0:1].astype(np.float64) * axis_x[None, :]
        + hull_points[:, 1:2].astype(np.float64) * axis_y[None, :]
    ).astype(np.float32)
    triangles = []
    for idx in range(1, len(hull_points_3d) - 1):
        triangles.append([hull_points_3d[0], hull_points_3d[idx], hull_points_3d[idx + 1]])
    if not triangles:
        return None, plane_error
    output_triangles = np.asarray(triangles, dtype=np.float32)
    if max_edge > 0.0:
        output_triangles, _attributes, _cap_reached = _subdivide_triangles_by_max_edge(
            output_triangles,
            max_edge=max_edge,
            max_count=0,
            vertex_attributes=[],
        )
    return output_triangles, plane_error


def _adaptive_mesh_triangles(
    triangles: np.ndarray,
    faces: np.ndarray,
    uv_triangles: np.ndarray | None,
    vertex_color_triangles: np.ndarray | None,
    texture: np.ndarray | None,
    triangle_color_source: str,
    min_edge: float,
    max_edge: float,
    color_threshold: float,
    normal_threshold_deg: float,
    plane_threshold: float,
    max_patch_diameter: float,
    max_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], bool]:
    if len(triangles) == 0:
        return (
            triangles.astype(np.float32, copy=False),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=bool),
            {"mode": "texture_adaptive", "input_triangle_count": 0, "output_triangle_count": 0},
            False,
        )

    if triangle_color_source == "gray":
        texture = None
        vertex_color_triangles = None

    samples, texture_mask = _triangle_texture_samples(uv_triangles, texture, vertex_color_triangles, len(triangles))
    face_colors = samples.mean(axis=1).astype(np.float32, copy=False)
    complexity = _triangle_color_complexity(samples)
    normals = _triangle_normals(triangles)
    centers = triangles.mean(axis=1)
    low_complexity = complexity <= max(color_threshold, 0.0)
    adjacency = _face_adjacency(faces) if len(faces) == len(triangles) else [[] for _ in range(len(triangles))]
    edge_neighbor_count = sum(len(neighbors) for neighbors in adjacency)
    if edge_neighbor_count < max(len(triangles) // 2, 1):
        spatial_radius = max(max_edge, max_patch_diameter * 0.25, 1e-6)
        adjacency = _merge_adjacency(adjacency, _spatial_face_adjacency(centers, spatial_radius))
    normal_cos_threshold = math.cos(math.radians(max(normal_threshold_deg, 0.0)))

    visited = np.zeros((len(triangles),), dtype=bool)
    detailed_indices = []
    patch_triangles = []
    patch_colors = []
    patch_texture_masks = []
    patch_sizes = []
    rejected_patch_count = 0
    rejected_plane_count = 0
    rejected_no_savings_count = 0
    plane_errors = []

    for seed_idx in range(len(triangles)):
        if visited[seed_idx]:
            continue

        if not low_complexity[seed_idx]:
            visited[seed_idx] = True
            detailed_indices.append(seed_idx)
            continue

        visited[seed_idx] = True
        patch = []
        queue = deque([seed_idx])
        seed_center = centers[seed_idx]
        seed_normal = normals[seed_idx]
        seed_color = face_colors[seed_idx]
        while queue:
            face_idx = queue.popleft()
            patch.append(face_idx)
            for neighbor_idx in adjacency[face_idx]:
                if visited[neighbor_idx] or not low_complexity[neighbor_idx]:
                    continue
                if float(np.dot(normals[neighbor_idx], seed_normal)) < normal_cos_threshold:
                    continue
                if np.linalg.norm(face_colors[neighbor_idx] - seed_color) > max(color_threshold * 2.0, 1e-6):
                    continue
                if max_patch_diameter > 0.0 and np.linalg.norm(centers[neighbor_idx] - seed_center) > max_patch_diameter:
                    continue
                visited[neighbor_idx] = True
                queue.append(neighbor_idx)

        if len(patch) < 2:
            detailed_indices.extend(patch)
            rejected_patch_count += 1
            continue

        patch = np.asarray(patch, dtype=np.int64)
        coarsened_triangles, plane_error = _patch_to_plane_triangles(triangles[patch], plane_threshold, max_edge)
        plane_errors.append(plane_error)
        if coarsened_triangles is None:
            detailed_indices.extend(patch.tolist())
            rejected_plane_count += 1
            continue
        if len(coarsened_triangles) >= len(patch):
            detailed_indices.extend(patch.tolist())
            rejected_no_savings_count += 1
            continue

        patch_triangles.append(coarsened_triangles)
        patch_colors.append(np.repeat(face_colors[patch].mean(axis=0, keepdims=True), len(coarsened_triangles), axis=0))
        patch_texture_masks.append(np.full((len(coarsened_triangles),), bool(np.any(texture_mask[patch])), dtype=bool))
        patch_sizes.append(int(len(patch)))

    cap_reached = False
    detailed_triangles = np.empty((0, 3, 3), dtype=np.float32)
    detailed_colors = np.empty((0, 3), dtype=np.float32)
    detailed_texture_mask = np.empty((0,), dtype=bool)
    if detailed_indices:
        detailed_indices_array = np.asarray(detailed_indices, dtype=np.int64)
        detailed_triangles = triangles[detailed_indices_array]
        detailed_uvs = uv_triangles[detailed_indices_array] if uv_triangles is not None else None
        detailed_vertex_colors = (
            vertex_color_triangles[detailed_indices_array] if vertex_color_triangles is not None else None
        )
        patch_count = int(sum(len(values) for values in patch_triangles))
        detailed_max_count = max_count - patch_count if max_count > 0 else 0
        detailed_triangles, detailed_attributes, cap_reached = _subdivide_triangles_by_max_edge(
            detailed_triangles,
            max_edge=max_edge,
            max_count=detailed_max_count,
            vertex_attributes=[detailed_uvs, detailed_vertex_colors],
        )
        detailed_uvs, detailed_vertex_colors = detailed_attributes
        detailed_colors, detailed_texture_mask, _source = _triangle_seed_colors(
            len(detailed_triangles),
            detailed_uvs,
            texture,
            detailed_vertex_colors,
        )

    if patch_triangles:
        output_triangles = np.concatenate([np.concatenate(patch_triangles, axis=0), detailed_triangles], axis=0)
        output_colors = np.concatenate([np.concatenate(patch_colors, axis=0), detailed_colors], axis=0)
        output_texture_mask = np.concatenate([np.concatenate(patch_texture_masks, axis=0), detailed_texture_mask], axis=0)
    else:
        output_triangles = detailed_triangles
        output_colors = detailed_colors
        output_texture_mask = detailed_texture_mask

    if max_count > 0 and len(output_triangles) > max_count:
        output_triangles = output_triangles[:max_count]
        output_colors = output_colors[:max_count]
        output_texture_mask = output_texture_mask[:max_count]
        cap_reached = True

    stats = {
        "mode": "texture_adaptive",
        "input_triangle_count": int(len(triangles)),
        "output_triangle_count": int(len(output_triangles)),
        "low_complexity_face_count": int(low_complexity.sum()),
        "high_complexity_face_count": int((~low_complexity).sum()),
        "coarsened_patch_count": int(len(patch_sizes)),
        "coarsened_source_face_count": int(sum(patch_sizes)),
        "detailed_source_face_count": int(len(detailed_indices)),
        "edge_neighbor_count": int(edge_neighbor_count),
        "rejected_small_patch_count": int(rejected_patch_count),
        "rejected_plane_patch_count": int(rejected_plane_count),
        "rejected_no_savings_patch_count": int(rejected_no_savings_count),
        "color_complexity": _percentiles(complexity),
        "patch_size": _percentiles(np.asarray(patch_sizes, dtype=np.float32)) if patch_sizes else {},
        "patch_plane_error": _percentiles(np.asarray(plane_errors, dtype=np.float32)) if plane_errors else {},
        "min_edge": float(min_edge),
        "max_edge": float(max_edge),
        "color_threshold": float(color_threshold),
        "normal_threshold_deg": float(normal_threshold_deg),
        "plane_threshold": float(plane_threshold),
        "max_patch_diameter": float(max_patch_diameter),
    }
    return (
        output_triangles.astype(np.float32, copy=False),
        np.clip(output_colors, 0.0, 1.0).astype(np.float32, copy=False),
        output_texture_mask.astype(bool, copy=False),
        stats,
        cap_reached,
    )


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


def _unlink_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


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


def _read_colored_point_cloud_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    if {"red", "green", "blue"}.issubset(vertices.data.dtype.names or ()):
        colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float32)
        if colors.size and colors.max() > 1.0:
            colors /= 255.0
    else:
        colors = np.full((len(points), 3), 180.0 / 255.0, dtype=np.float32)
    return points, np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False)


def _fibonacci_directions(nb_points: int) -> np.ndarray:
    if nb_points < 2:
        raise ValueError("Fallback triangle generation requires at least two vertices per shape")

    directions = []
    for point_idx in range(nb_points):
        z_coord = 1.0 - (2.0 * point_idx / (nb_points - 1))
        radius_xy = math.sqrt(max(1.0 - z_coord * z_coord, 0.0))
        theta = math.pi * (3.0 - math.sqrt(5.0)) * point_idx
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


def _nearest_neighbor_dist2(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.full((len(points),), 1e-7, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        try:
            distances, _indices = tree.query(points, k=2, workers=-1)
        except TypeError:
            distances, _indices = tree.query(points, k=2)
        return np.maximum(distances[:, 1] ** 2, 1e-7).astype(np.float32)
    except ImportError:
        import open3d as o3d

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        tree = o3d.geometry.KDTreeFlann(point_cloud)
        dist2 = np.empty((len(points),), dtype=np.float32)
        for point_idx, point in enumerate(points):
            count, _indices, distances = tree.search_knn_vector_3d(point.astype(np.float64), 2)
            dist2[point_idx] = distances[1] if count > 1 else 1e-7
        return np.maximum(dist2, 1e-7).astype(np.float32, copy=False)


def _points_to_triangle_soup(
    points: np.ndarray,
    triangle_size: float,
    nb_points: int,
    seed: int,
    chunk_size: int = 65536,
    max_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    nearest_dist2 = _nearest_neighbor_dist2(points)
    base_dirs = _fibonacci_directions(nb_points)
    radii = triangle_size * np.sqrt(np.maximum(nearest_dist2, 1e-7)).astype(np.float32)
    if max_radius > 0.0:
        radii = np.minimum(radii, max_radius).astype(np.float32, copy=False)
    triangles = np.empty((len(points), nb_points, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for start_idx in range(0, len(points), chunk_size):
        end_idx = min(start_idx + chunk_size, len(points))
        rotations = _random_rotation_matrices(end_idx - start_idx, rng)
        rotated = np.einsum("nij,pj->npi", rotations, base_dirs, optimize=True)
        triangles[start_idx:end_idx] = points[start_idx:end_idx, None, :] + rotated * radii[start_idx:end_idx, None, None]
    return triangles, radii


def _project_world_points(
    points: np.ndarray,
    camera_pose_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    downscale: int,
    near: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    pixels, valid, _depths = _project_world_points_with_depth(
        points,
        camera_pose_world,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        downscale,
        near=near,
    )
    return pixels, valid


def _project_world_points_with_depth(
    points: np.ndarray,
    camera_pose_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    downscale: int,
    near: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=bool), np.empty((0,), dtype=np.float32)

    downscale = max(int(downscale), 1)
    scaled_width = max(int(math.ceil(width / downscale)), 1)
    scaled_height = max(int(math.ceil(height / downscale)), 1)
    rotation = camera_pose_world[:3, :3].astype(np.float64, copy=False)
    translation = camera_pose_world[:3, 3].astype(np.float64, copy=False)
    camera_points = (points.astype(np.float64, copy=False) - translation[None, :]) @ rotation
    z = camera_points[:, 2]
    valid = z > near
    safe_z = np.where(valid, z, 1.0)
    u = (fx * camera_points[:, 0] / safe_z + cx) / downscale
    v = (fy * camera_points[:, 1] / safe_z + cy) / downscale
    pixels = np.stack([np.rint(u).astype(np.int32), np.rint(v).astype(np.int32)], axis=1)
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < scaled_width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < scaled_height)
    return pixels, valid, z.astype(np.float32, copy=False)


def _rasterize_triangle_coverage(
    triangles: np.ndarray,
    camera_pose_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    downscale: int,
    dilation_px: int,
) -> np.ndarray:
    downscale = max(int(downscale), 1)
    scaled_width = max(int(math.ceil(width / downscale)), 1)
    scaled_height = max(int(math.ceil(height / downscale)), 1)
    coverage = np.zeros((scaled_height, scaled_width), dtype=np.uint8)
    if len(triangles) == 0:
        return coverage.astype(bool)

    pixels, valid_vertices = _project_world_points(
        triangles.reshape(-1, 3),
        camera_pose_world,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        downscale,
    )
    triangle_pixels = pixels.reshape(-1, 3, 2)
    valid_triangles = valid_vertices.reshape(-1, 3).all(axis=1)
    for projected_triangle in triangle_pixels[valid_triangles]:
        min_xy = projected_triangle.min(axis=0)
        max_xy = projected_triangle.max(axis=0)
        if max_xy[0] < 0 or max_xy[1] < 0 or min_xy[0] >= scaled_width or min_xy[1] >= scaled_height:
            continue
        clipped_triangle = projected_triangle.copy()
        clipped_triangle[:, 0] = np.clip(clipped_triangle[:, 0], 0, scaled_width - 1)
        clipped_triangle[:, 1] = np.clip(clipped_triangle[:, 1], 0, scaled_height - 1)
        cv2.fillConvexPoly(coverage, clipped_triangle.astype(np.int32), 1)

    dilation_radius = int(math.ceil(max(dilation_px, 0) / downscale))
    if dilation_radius > 0 and np.any(coverage):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_radius + 1, 2 * dilation_radius + 1),
        )
        coverage = cv2.dilate(coverage, kernel)
    return coverage.astype(bool)


def _rasterize_triangle_coverage_and_depth(
    triangles: np.ndarray,
    camera_pose_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    downscale: int,
    dilation_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    downscale = max(int(downscale), 1)
    scaled_width = max(int(math.ceil(width / downscale)), 1)
    scaled_height = max(int(math.ceil(height / downscale)), 1)
    coverage = np.zeros((scaled_height, scaled_width), dtype=np.uint8)
    depth = np.full((scaled_height, scaled_width), np.inf, dtype=np.float32)
    if len(triangles) == 0:
        return coverage.astype(bool), depth

    pixels, valid_vertices, depths = _project_world_points_with_depth(
        triangles.reshape(-1, 3),
        camera_pose_world,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        downscale,
    )
    triangle_pixels = pixels.reshape(-1, 3, 2)
    triangle_depths = depths.reshape(-1, 3)
    valid_triangles = valid_vertices.reshape(-1, 3).all(axis=1)
    for projected_triangle, projected_depths in zip(triangle_pixels[valid_triangles], triangle_depths[valid_triangles]):
        min_xy = projected_triangle.min(axis=0)
        max_xy = projected_triangle.max(axis=0)
        if max_xy[0] < 0 or max_xy[1] < 0 or min_xy[0] >= scaled_width or min_xy[1] >= scaled_height:
            continue
        clipped_triangle = projected_triangle.copy()
        clipped_triangle[:, 0] = np.clip(clipped_triangle[:, 0], 0, scaled_width - 1)
        clipped_triangle[:, 1] = np.clip(clipped_triangle[:, 1], 0, scaled_height - 1)
        mask = np.zeros_like(coverage)
        cv2.fillConvexPoly(mask, clipped_triangle.astype(np.int32), 1)
        triangle_depth = float(np.mean(projected_depths))
        update_mask = (mask > 0) & (triangle_depth < depth)
        depth[update_mask] = triangle_depth
        coverage[update_mask] = 1

    dilation_radius = int(math.ceil(max(dilation_px, 0) / downscale))
    if dilation_radius > 0 and np.any(coverage):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_radius + 1, 2 * dilation_radius + 1),
        )
        coverage = cv2.dilate(coverage, kernel)
    return coverage.astype(bool), depth


def _backproject_image_plane_points(
    pixels_xy: np.ndarray,
    depths: np.ndarray,
    camera_pose_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    if len(pixels_xy) == 0:
        return np.empty((0, 3), dtype=np.float32)
    camera_points = np.stack(
        [
            (pixels_xy[:, 0] - cx) / fx * depths,
            (pixels_xy[:, 1] - cy) / fy * depths,
            depths,
        ],
        axis=1,
    )
    rotation = camera_pose_world[:3, :3].astype(np.float64, copy=False)
    translation = camera_pose_world[:3, 3].astype(np.float64, copy=False)
    return (camera_points @ rotation.T + translation[None, :]).astype(np.float32)


def _camera_facing_triangles(
    centers: np.ndarray,
    camera_pose_world: np.ndarray,
    radius: float,
) -> np.ndarray:
    if len(centers) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    rotation = camera_pose_world[:3, :3].astype(np.float32, copy=False)
    axis_x = rotation.T[0]
    axis_y = rotation.T[1]
    radius = max(float(radius), 1e-4)
    offsets = np.asarray(
        [
            [-1.0, -1.0 / math.sqrt(3.0)],
            [1.0, -1.0 / math.sqrt(3.0)],
            [0.0, 2.0 / math.sqrt(3.0)],
        ],
        dtype=np.float32,
    )
    world_offsets = offsets[:, 0:1] * axis_x[None, :] + offsets[:, 1:2] * axis_y[None, :]
    return centers[:, None, :] + radius * world_offsets[None, :, :]


def _image_plane_component_depth(component_mask: np.ndarray, finite_depth: np.ndarray, frame_median_depth: float) -> float:
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ring_mask = cv2.dilate(component_mask.astype(np.uint8), dilation_kernel).astype(bool) & ~component_mask
    ring_depths = finite_depth[ring_mask & np.isfinite(finite_depth)]
    if len(ring_depths) > 0:
        return float(np.median(ring_depths))
    if np.isfinite(frame_median_depth):
        return float(frame_median_depth)
    return 1.5


def _sample_image_plane_fallback_for_frame(
    frame_row: dict[str, str],
    triangles: np.ndarray,
    calibration: CameraCalibration,
    new_camera_matrix: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    coverage_downscale: int,
    coverage_dilation_px: int,
    min_component_area: int,
    sample_stride: int,
    triangle_radius: float,
    depth_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    coverage, depth = _rasterize_triangle_coverage_and_depth(
        triangles,
        frame_row["_camera_pose_world"],
        fx,
        fy,
        cx,
        cy,
        calibration.width,
        calibration.height,
        coverage_downscale,
        coverage_dilation_px,
    )
    finite_depth = np.where(np.isfinite(depth), depth, np.nan).astype(np.float32)
    frame_finite_depths = finite_depth[np.isfinite(finite_depth)]
    frame_median_depth = float(np.median(frame_finite_depths)) if len(frame_finite_depths) else float("nan")
    uncovered = ~coverage
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        uncovered.astype(np.uint8),
        connectivity=8,
    )
    image_rgb = _load_undistorted_rgb_image(Path(frame_row["_image_path"]), calibration, new_camera_matrix)

    fallback_triangles = []
    fallback_colors = []
    accepted_components = 0
    sampled_pixel_count = 0
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        component_mask = labels == component_id
        ys, xs = np.nonzero(component_mask)
        if len(xs) == 0:
            continue
        stride = max(int(sample_stride), 1)
        order = np.arange(len(xs))
        rng.shuffle(order)
        grid_mask = ((xs[order] % stride) == 0) & ((ys[order] % stride) == 0)
        selected = order[grid_mask]
        if len(selected) == 0:
            selected = order[:1]
        selected_x = xs[selected].astype(np.float32)
        selected_y = ys[selected].astype(np.float32)
        depths = np.full(
            (len(selected),),
            _image_plane_component_depth(component_mask, finite_depth, frame_median_depth) * depth_scale,
            dtype=np.float32,
        )
        full_res_pixels = np.stack(
            [
                (selected_x + 0.5) * max(int(coverage_downscale), 1),
                (selected_y + 0.5) * max(int(coverage_downscale), 1),
            ],
            axis=1,
        )
        centers = _backproject_image_plane_points(
            full_res_pixels,
            depths,
            frame_row["_camera_pose_world"],
            fx,
            fy,
            cx,
            cy,
        )
        fallback_triangles.append(_camera_facing_triangles(centers, frame_row["_camera_pose_world"], triangle_radius))
        if image_rgb is not None:
            sample_u = np.clip(np.rint(full_res_pixels[:, 0]).astype(np.int32), 0, image_rgb.shape[1] - 1)
            sample_v = np.clip(np.rint(full_res_pixels[:, 1]).astype(np.int32), 0, image_rgb.shape[0] - 1)
            colors = image_rgb[sample_v, sample_u].astype(np.float32) / 255.0
        else:
            colors = np.full((len(selected), 3), 180.0 / 255.0, dtype=np.float32)
        fallback_colors.append(colors)
        accepted_components += 1
        sampled_pixel_count += int(len(selected))

    if fallback_triangles:
        triangles_out = np.concatenate(fallback_triangles, axis=0).astype(np.float32, copy=False)
        colors_out = np.concatenate(fallback_colors, axis=0).astype(np.float32, copy=False)
    else:
        triangles_out = np.empty((0, 3, 3), dtype=np.float32)
        colors_out = np.empty((0, 3), dtype=np.float32)
    stats_out = {
        "coverage_ratio": float(coverage.mean()) if coverage.size else 0.0,
        "uncovered_ratio": float(uncovered.mean()) if uncovered.size else 0.0,
        "component_count": int(max(component_count - 1, 0)),
        "accepted_component_count": int(accepted_components),
        "sampled_pixel_count": int(sampled_pixel_count),
        "frame_median_depth": frame_median_depth if np.isfinite(frame_median_depth) else None,
    }
    return triangles_out, colors_out, stats_out


def _build_uncovered_fallback(
    frame_rows: list[dict[str, str]],
    mesh_seed_entries: list[dict[str, object]],
    output_dir: Path,
    sfm_points_path: Path | None,
    calibration: CameraCalibration,
    new_camera_matrix: np.ndarray,
    scene_bounds_min: np.ndarray | None,
    scene_bounds_max: np.ndarray | None,
    bounds_margin: float,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    mode: str,
    max_count: int,
    triangle_size: float,
    max_radius: float,
    coverage_downscale: int,
    coverage_dilation_px: int,
    frame_stride: int,
    seed: int,
    image_plane_min_component_area: int,
    image_plane_sample_stride: int,
    image_plane_triangle_radius: float,
    image_plane_depth_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    empty_triangles = np.empty((0, 3, 3), dtype=np.float32)
    empty_colors = np.empty((0, 3), dtype=np.float32)
    stats: dict[str, object] = {
        "mode": mode,
        "enabled": False,
        "input_point_count": 0,
        "selected_point_count": 0,
        "output_triangle_count": 0,
    }
    frame_indices = list(range(0, len(frame_rows), max(int(frame_stride), 1)))
    triangle_cache: dict[str, np.ndarray] = {}

    def load_frame_triangles(frame_row: dict[str, str]) -> np.ndarray | None:
        mesh_seed_entry = _select_mesh_seed_entry(int(frame_row["image_timestamp_ns"]), mesh_seed_entries)
        if not mesh_seed_entry or not mesh_seed_entry.get("triangle_seed_rel_path"):
            return None
        triangle_rel_path = str(mesh_seed_entry["triangle_seed_rel_path"])
        if triangle_rel_path not in triangle_cache:
            with np.load(output_dir / triangle_rel_path) as data:
                triangle_cache[triangle_rel_path] = data["triangles"].astype(np.float32)
        return triangle_cache[triangle_rel_path]

    if mode == "none":
        coverage_ratios = []
        for frame_idx in frame_indices:
            frame_row = frame_rows[frame_idx]
            frame_triangles = load_frame_triangles(frame_row)
            if frame_triangles is None:
                continue
            coverage = _rasterize_triangle_coverage(
                frame_triangles,
                frame_row["_camera_pose_world"],
                fx,
                fy,
                cx,
                cy,
                width,
                height,
                coverage_downscale,
                coverage_dilation_px,
            )
            coverage_ratios.append(float(coverage.mean()) if coverage.size else 0.0)
        stats.update(
            {
                "enabled": False,
                "processed_frame_count": int(len(frame_indices)),
                "frame_stride": int(max(frame_stride, 1)),
                "coverage_downscale": int(max(coverage_downscale, 1)),
                "coverage_dilation_px": int(max(coverage_dilation_px, 0)),
                "coverage_ratio": _percentiles(np.asarray(coverage_ratios, dtype=np.float32)),
                "uncovered_ratio": _percentiles(1.0 - np.asarray(coverage_ratios, dtype=np.float32))
                if coverage_ratios
                else {},
            }
        )
        return empty_triangles, empty_colors, stats

    if mode == "image_plane":
        rng = np.random.default_rng(seed)
        all_fallback_triangles = []
        all_fallback_colors = []
        frame_stats = []
        for frame_idx in frame_indices:
            frame_row = frame_rows[frame_idx]
            frame_triangles = load_frame_triangles(frame_row)
            if frame_triangles is None:
                continue
            fallback_frame_triangles, fallback_frame_colors, stats_frame = _sample_image_plane_fallback_for_frame(
                frame_row,
                frame_triangles,
                calibration,
                new_camera_matrix,
                fx,
                fy,
                cx,
                cy,
                coverage_downscale,
                coverage_dilation_px,
                image_plane_min_component_area,
                image_plane_sample_stride,
                image_plane_triangle_radius,
                image_plane_depth_scale,
                rng,
            )
            frame_stats.append(stats_frame)
            if len(fallback_frame_triangles) == 0:
                continue
            all_fallback_triangles.append(fallback_frame_triangles)
            all_fallback_colors.append(fallback_frame_colors)

        if all_fallback_triangles:
            fallback_triangles = np.concatenate(all_fallback_triangles, axis=0)
            fallback_colors = np.concatenate(all_fallback_colors, axis=0)
            if max_count > 0 and len(fallback_triangles) > max_count:
                selected_indices = np.sort(rng.choice(len(fallback_triangles), size=max_count, replace=False))
                fallback_triangles = fallback_triangles[selected_indices]
                fallback_colors = fallback_colors[selected_indices]
        else:
            fallback_triangles = empty_triangles
            fallback_colors = empty_colors

        coverage_ratios = np.asarray([item["coverage_ratio"] for item in frame_stats], dtype=np.float32)
        uncovered_ratios = np.asarray([item["uncovered_ratio"] for item in frame_stats], dtype=np.float32)
        sampled_pixel_counts = np.asarray([item["sampled_pixel_count"] for item in frame_stats], dtype=np.float32)
        accepted_component_counts = np.asarray(
            [item["accepted_component_count"] for item in frame_stats],
            dtype=np.float32,
        )
        frame_depths = np.asarray(
            [item["frame_median_depth"] for item in frame_stats if item["frame_median_depth"] is not None],
            dtype=np.float32,
        )
        stats.update(
            {
                "enabled": True,
                "input_point_count": 0,
                "selected_point_count": int(len(fallback_triangles)),
                "output_triangle_count": int(len(fallback_triangles)),
                "max_count": int(max_count),
                "processed_frame_count": int(len(frame_indices)),
                "frame_stride": int(max(frame_stride, 1)),
                "coverage_downscale": int(max(coverage_downscale, 1)),
                "coverage_dilation_px": int(max(coverage_dilation_px, 0)),
                "min_component_area": int(max(image_plane_min_component_area, 1)),
                "sample_stride": int(max(image_plane_sample_stride, 1)),
                "triangle_radius": float(image_plane_triangle_radius),
                "depth_scale": float(image_plane_depth_scale),
                "coverage_ratio": _percentiles(coverage_ratios),
                "uncovered_ratio": _percentiles(uncovered_ratios),
                "sampled_pixel_count_per_frame": _percentiles(sampled_pixel_counts),
                "accepted_component_count_per_frame": _percentiles(accepted_component_counts),
                "frame_median_depth": _percentiles(frame_depths),
            }
        )
        return fallback_triangles.astype(np.float32, copy=False), fallback_colors.astype(np.float32, copy=False), stats

    if mode != "sfm":
        raise ValueError(f"Unsupported uncovered seed mode: {mode}")

    if sfm_points_path is None or not sfm_points_path.exists():
        stats["reason"] = "missing_sfm_points"
        return empty_triangles, empty_colors, stats
    if not mesh_seed_entries:
        stats["reason"] = "missing_mesh_seed_entries"
        return empty_triangles, empty_colors, stats

    sfm_points, sfm_colors = _read_colored_point_cloud_ply(sfm_points_path)
    raw_sfm_point_count = int(len(sfm_points))
    if scene_bounds_min is not None and scene_bounds_max is not None and len(sfm_points):
        bounds_min = np.asarray(scene_bounds_min, dtype=np.float32) - max(bounds_margin, 0.0)
        bounds_max = np.asarray(scene_bounds_max, dtype=np.float32) + max(bounds_margin, 0.0)
        in_bounds = np.all((sfm_points >= bounds_min[None, :]) & (sfm_points <= bounds_max[None, :]), axis=1)
        sfm_points = sfm_points[in_bounds]
        sfm_colors = sfm_colors[in_bounds]
    hit_counts = np.zeros((len(sfm_points),), dtype=np.int32)
    visible_counts = np.zeros((len(sfm_points),), dtype=np.int32)
    coverage_ratios = []
    visible_sfm_ratios = []

    for frame_idx in frame_indices:
        frame_row = frame_rows[frame_idx]
        frame_triangles = load_frame_triangles(frame_row)
        if frame_triangles is None:
            continue
        coverage = _rasterize_triangle_coverage(
            frame_triangles,
            frame_row["_camera_pose_world"],
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            coverage_downscale,
            coverage_dilation_px,
        )
        coverage_ratios.append(float(coverage.mean()) if coverage.size else 0.0)
        pixels, valid = _project_world_points(
            sfm_points,
            frame_row["_camera_pose_world"],
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            coverage_downscale,
        )
        visible_counts[valid] += 1
        visible_sfm_ratios.append(float(valid.mean()) if len(valid) else 0.0)
        if not np.any(valid):
            continue
        valid_indices = np.nonzero(valid)[0]
        valid_pixels = pixels[valid]
        uncovered = ~coverage[valid_pixels[:, 1], valid_pixels[:, 0]]
        hit_counts[valid_indices[uncovered]] += 1

    selected_indices = np.nonzero(hit_counts > 0)[0]
    if len(selected_indices) > 0:
        order = np.lexsort((selected_indices, -hit_counts[selected_indices]))
        selected_indices = selected_indices[order]
    if max_count > 0 and len(selected_indices) > max_count:
        selected_indices = selected_indices[:max_count]

    if len(selected_indices) == 0:
        stats.update(
            {
                "enabled": True,
                "raw_input_point_count": int(raw_sfm_point_count),
                "input_point_count": int(len(sfm_points)),
                "processed_frame_count": int(len(frame_indices)),
                "coverage_ratio": _percentiles(np.asarray(coverage_ratios, dtype=np.float32)),
                "visible_sfm_ratio": _percentiles(np.asarray(visible_sfm_ratios, dtype=np.float32)),
                "reason": "no_sfm_points_projected_to_uncovered_regions",
            }
        )
        return empty_triangles, empty_colors, stats

    selected_points = sfm_points[selected_indices]
    selected_colors = sfm_colors[selected_indices]
    fallback_triangles, fallback_radii = _points_to_triangle_soup(
        selected_points,
        triangle_size=triangle_size,
        nb_points=3,
        seed=seed,
        max_radius=max_radius,
    )
    stats.update(
        {
            "enabled": True,
            "sfm_points_path": str(sfm_points_path),
            "raw_input_point_count": int(raw_sfm_point_count),
            "input_point_count": int(len(sfm_points)),
            "bounds_margin": float(bounds_margin),
            "selected_point_count": int(len(selected_indices)),
            "output_triangle_count": int(len(fallback_triangles)),
            "max_count": int(max_count),
            "triangle_size": float(triangle_size),
            "max_radius": float(max_radius),
            "processed_frame_count": int(len(frame_indices)),
            "frame_stride": int(max(frame_stride, 1)),
            "coverage_downscale": int(max(coverage_downscale, 1)),
            "coverage_dilation_px": int(max(coverage_dilation_px, 0)),
            "coverage_ratio": _percentiles(np.asarray(coverage_ratios, dtype=np.float32)),
            "uncovered_ratio": _percentiles(1.0 - np.asarray(coverage_ratios, dtype=np.float32))
            if coverage_ratios
            else {},
            "visible_sfm_ratio": _percentiles(np.asarray(visible_sfm_ratios, dtype=np.float32)),
            "uncovered_hit_count": _percentiles(hit_counts[selected_indices].astype(np.float32)),
            "fallback_radius": _percentiles(fallback_radii),
        }
    )
    return fallback_triangles.astype(np.float32, copy=False), selected_colors.astype(np.float32, copy=False), stats


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
    triangle_sizing_mode: str,
    adaptive_min_edge: float,
    adaptive_max_edge: float,
    adaptive_color_threshold: float,
    adaptive_normal_threshold_deg: float,
    adaptive_plane_threshold: float,
    adaptive_max_patch_diameter: float,
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
            "sizing_mode": triangle_sizing_mode,
            "reduction": empty_reduction,
        }, np.empty((0, 3), dtype=np.float32), np.empty((0, 3, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    all_points = []
    all_triangles = []
    all_triangle_colors = []
    all_triangle_texture_masks = []
    adaptive_file_stats = []
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
        adaptive_stats = None
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

            texture = _load_texture_image(mesh.texture_path) if triangle_color_source == "texture" else None
            if triangle_sizing_mode == "uniform":
                mesh_triangles, subdivided_attributes, triangle_cap_reached = _subdivide_triangles_by_max_edge(
                    mesh_triangles,
                    max_edge=triangle_max_edge,
                    max_count=triangle_max_count,
                    vertex_attributes=[uv_triangles, vertex_color_triangles],
                )
                uv_triangles, vertex_color_triangles = subdivided_attributes
            else:
                mesh_triangles, triangle_colors, triangle_texture_mask, adaptive_stats, triangle_cap_reached = (
                    _adaptive_mesh_triangles(
                        mesh_triangles,
                        faces,
                        uv_triangles,
                        vertex_color_triangles,
                        texture,
                        triangle_color_source,
                        min_edge=adaptive_min_edge,
                        max_edge=adaptive_max_edge,
                        color_threshold=adaptive_color_threshold,
                        normal_threshold_deg=adaptive_normal_threshold_deg,
                        plane_threshold=adaptive_plane_threshold,
                        max_patch_diameter=adaptive_max_patch_diameter,
                        max_count=triangle_max_count,
                    )
                )
            if len(mesh_triangles) > 0:
                if triangle_sizing_mode == "uniform":
                    if triangle_color_source == "gray":
                        vertex_color_triangles = None
                    triangle_colors, triangle_texture_mask, color_source = _triangle_seed_colors(
                        len(mesh_triangles),
                        uv_triangles,
                        texture,
                        vertex_color_triangles,
                    )
                else:
                    color_source = "adaptive_texture" if texture is not None else "adaptive_vertex_or_gray"
                    adaptive_file_stats.append(
                        {
                            "obj_filename": obj_filename,
                            **(adaptive_stats or {}),
                        }
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
            "sizing_mode": triangle_sizing_mode,
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
    adaptive_summary = {
        "enabled": triangle_sizing_mode != "uniform",
        "file_count": int(len(adaptive_file_stats)),
    }
    if adaptive_file_stats:
        adaptive_summary.update(
            {
                "input_triangle_count": int(sum(item.get("input_triangle_count", 0) for item in adaptive_file_stats)),
                "output_triangle_count": int(sum(item.get("output_triangle_count", 0) for item in adaptive_file_stats)),
                "coarsened_patch_count": int(sum(item.get("coarsened_patch_count", 0) for item in adaptive_file_stats)),
                "coarsened_source_face_count": int(
                    sum(item.get("coarsened_source_face_count", 0) for item in adaptive_file_stats)
                ),
                "detailed_source_face_count": int(
                    sum(item.get("detailed_source_face_count", 0) for item in adaptive_file_stats)
                ),
                "rejected_plane_patch_count": int(
                    sum(item.get("rejected_plane_patch_count", 0) for item in adaptive_file_stats)
                ),
                "rejected_no_savings_patch_count": int(
                    sum(item.get("rejected_no_savings_patch_count", 0) for item in adaptive_file_stats)
                ),
                "per_file_preview": adaptive_file_stats[:5],
            }
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
        "sizing_mode": triangle_sizing_mode,
        "adaptive": adaptive_summary,
        "area": _percentiles(_triangle_areas(triangles)),
        "max_edge_distribution": _percentiles(_triangle_max_edges(triangles)) if len(triangles) else {},
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
    parser.add_argument(
        "--mesh-triangle-sizing-mode",
        choices=["uniform", "texture_adaptive", "coverage_adaptive"],
        default="uniform",
        help="Triangle sizing strategy. uniform preserves the current global max-edge path; adaptive modes coarsen low-texture planar patches.",
    )
    parser.add_argument(
        "--mesh-triangle-adaptive-min-edge",
        type=float,
        default=0.08,
        help="Target max edge in meters for detailed/high-complexity triangles in adaptive sizing.",
    )
    parser.add_argument(
        "--mesh-triangle-adaptive-max-edge",
        type=float,
        default=0.60,
        help="Reserved adaptive coarse edge scale in meters; kept explicit for comparable configs and stats.",
    )
    parser.add_argument(
        "--mesh-triangle-adaptive-color-threshold",
        type=float,
        default=0.03,
        help="Maximum sampled RGB complexity for faces eligible for adaptive patch coarsening.",
    )
    parser.add_argument(
        "--mesh-triangle-adaptive-normal-threshold-deg",
        type=float,
        default=10.0,
        help="Maximum normal angle between neighboring faces in an adaptive patch.",
    )
    parser.add_argument(
        "--mesh-triangle-adaptive-plane-threshold",
        type=float,
        default=0.02,
        help="Maximum plane residual in meters for replacing a patch with larger planar triangles.",
    )
    parser.add_argument(
        "--mesh-triangle-adaptive-max-patch-diameter",
        type=float,
        default=0.75,
        help="Maximum center distance from a patch seed face when growing adaptive patches.",
    )
    parser.add_argument(
        "--mesh-coverage-dilation-px",
        type=int,
        default=8,
        help="Image-space dilation radius for projected mesh coverage masks.",
    )
    parser.add_argument(
        "--mesh-coverage-downscale",
        type=int,
        default=4,
        help="Downscale factor for mesh coverage estimation.",
    )
    parser.add_argument(
        "--mesh-coverage-frame-stride",
        type=int,
        default=4,
        help="Use every Nth frame when estimating uncovered SfM fallback regions.",
    )
    parser.add_argument(
        "--uncovered-seed-mode",
        choices=["none", "image_plane", "sfm"],
        default="none",
        help="Fallback seed source for image regions not covered by projected Kimera meshes. image_plane uses only Kimera poses/images/mesh coverage; sfm is ablation-only.",
    )
    parser.add_argument(
        "--uncovered-sfm-points",
        default=None,
        help="Kimera-frame SfM points3d.ply used when --uncovered-seed-mode sfm.",
    )
    parser.add_argument(
        "--uncovered-seed-max-count",
        type=int,
        default=20000,
        help="Maximum uncovered fallback triangles to append.",
    )
    parser.add_argument(
        "--uncovered-seed-bounds-margin",
        type=float,
        default=0.5,
        help="Meters added around the Kimera mesh bounds before accepting SfM fallback points.",
    )
    parser.add_argument(
        "--uncovered-seed-triangle-size",
        type=float,
        default=2.23,
        help="Nearest-neighbor triangle scale for uncovered SfM fallback triangles.",
    )
    parser.add_argument(
        "--uncovered-seed-max-radius",
        type=float,
        default=0.25,
        help="Maximum radius in meters for uncovered SfM fallback triangles; <=0 disables clamping.",
    )
    parser.add_argument(
        "--uncovered-seed-random-seed",
        type=int,
        default=0,
        help="Deterministic random seed for uncovered fallback triangle orientations.",
    )
    parser.add_argument(
        "--uncovered-image-plane-min-component-area",
        type=int,
        default=256,
        help="Minimum uncovered connected-component area in downscaled pixels for image-plane fallback.",
    )
    parser.add_argument(
        "--uncovered-image-plane-sample-stride",
        type=int,
        default=12,
        help="Downscaled-pixel grid stride for image-plane fallback samples.",
    )
    parser.add_argument(
        "--uncovered-image-plane-triangle-radius",
        type=float,
        default=0.08,
        help="Camera-facing triangle radius in meters for image-plane fallback.",
    )
    parser.add_argument(
        "--uncovered-image-plane-depth-scale",
        type=float,
        default=1.0,
        help="Scale applied to nearby Kimera mesh depth when placing image-plane fallback triangles.",
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
        triangle_sizing_mode=args.mesh_triangle_sizing_mode,
        adaptive_min_edge=args.mesh_triangle_adaptive_min_edge,
        adaptive_max_edge=args.mesh_triangle_adaptive_max_edge,
        adaptive_color_threshold=args.mesh_triangle_adaptive_color_threshold,
        adaptive_normal_threshold_deg=args.mesh_triangle_adaptive_normal_threshold_deg,
        adaptive_plane_threshold=args.mesh_triangle_adaptive_plane_threshold,
        adaptive_max_patch_diameter=args.mesh_triangle_adaptive_max_patch_diameter,
    )
    if len(merged_seed_points) > 0:
        _write_point_cloud_ply(output_dir / "points3d.ply", merged_seed_points)

    mesh_bounds_min = merged_seed_triangles.reshape(-1, 3).min(axis=0) if len(merged_seed_triangles) else None
    mesh_bounds_max = merged_seed_triangles.reshape(-1, 3).max(axis=0) if len(merged_seed_triangles) else None
    fallback_triangles, fallback_colors, fallback_stats = _build_uncovered_fallback(
        frame_rows,
        mesh_seed_entries,
        output_dir,
        Path(args.uncovered_sfm_points).expanduser().resolve() if args.uncovered_sfm_points else None,
        calibration,
        new_camera_matrix,
        mesh_bounds_min,
        mesh_bounds_max,
        args.uncovered_seed_bounds_margin,
        calibration.width,
        calibration.height,
        new_fx,
        new_fy,
        new_cx,
        new_cy,
        mode=args.uncovered_seed_mode if args.mesh_triangle_sizing_mode == "coverage_adaptive" else "none",
        max_count=args.uncovered_seed_max_count,
        triangle_size=args.uncovered_seed_triangle_size,
        max_radius=args.uncovered_seed_max_radius,
        coverage_downscale=args.mesh_coverage_downscale,
        coverage_dilation_px=args.mesh_coverage_dilation_px,
        frame_stride=args.mesh_coverage_frame_stride,
        seed=args.uncovered_seed_random_seed,
        image_plane_min_component_area=args.uncovered_image_plane_min_component_area,
        image_plane_sample_stride=args.uncovered_image_plane_sample_stride,
        image_plane_triangle_radius=args.uncovered_image_plane_triangle_radius,
        image_plane_depth_scale=args.uncovered_image_plane_depth_scale,
    )
    combined_seed_triangles = merged_seed_triangles
    combined_seed_colors = merged_seed_triangle_colors
    if len(fallback_triangles) > 0:
        combined_seed_triangles = (
            np.concatenate([merged_seed_triangles, fallback_triangles], axis=0)
            if len(merged_seed_triangles)
            else fallback_triangles
        )
        combined_seed_colors = (
            np.concatenate([merged_seed_triangle_colors, fallback_colors], axis=0)
            if len(merged_seed_triangle_colors)
            else fallback_colors
        )

    if len(merged_seed_triangles) > 0:
        _write_triangle_soup_npz(
            output_dir / "mesh_seed_triangles" / "adaptive_mesh.npz",
            merged_seed_triangles,
            merged_seed_triangle_colors,
        )
    fallback_rel_path = None
    if len(fallback_triangles) > 0:
        fallback_filename = (
            "uncovered_image_plane_fallback.npz"
            if fallback_stats.get("mode") == "image_plane"
            else "uncovered_sfm_fallback.npz"
        )
        fallback_rel_path = f"mesh_seed_triangles/{fallback_filename}"
        _write_triangle_soup_npz(
            output_dir / fallback_rel_path,
            fallback_triangles,
            fallback_colors,
        )
        for stale_name in ("uncovered_image_plane_fallback.npz", "uncovered_sfm_fallback.npz"):
            if stale_name != fallback_filename:
                _unlink_if_exists(output_dir / "mesh_seed_triangles" / stale_name)
    else:
        _unlink_if_exists(output_dir / "mesh_seed_triangles" / "uncovered_sfm_fallback.npz")
        _unlink_if_exists(output_dir / "mesh_seed_triangles" / "uncovered_image_plane_fallback.npz")
    if len(combined_seed_triangles) > 0:
        _write_triangle_soup_npz(
            output_dir / "mesh_seed_triangles" / "all.npz",
            combined_seed_triangles,
            combined_seed_colors,
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
        "uncovered_fallback": fallback_stats,
        "combined_seed": {
            "mesh_triangle_count": int(len(merged_seed_triangles)),
            "uncovered_fallback_triangle_count": int(len(fallback_triangles)),
            "output_triangle_count": int(len(combined_seed_triangles)),
            "all_npz": "mesh_seed_triangles/all.npz" if len(combined_seed_triangles) > 0 else None,
            "adaptive_mesh_npz": "mesh_seed_triangles/adaptive_mesh.npz" if len(merged_seed_triangles) > 0 else None,
            "uncovered_fallback_npz": fallback_rel_path,
            "uncovered_sfm_fallback_npz": fallback_rel_path if fallback_stats.get("mode") == "sfm" else None,
            "uncovered_image_plane_fallback_npz": (
                fallback_rel_path if fallback_stats.get("mode") == "image_plane" else None
            ),
        },
    }
    if len(combined_seed_triangles) > 0:
        seed_stats = {
            "sizing_mode": args.mesh_triangle_sizing_mode,
            "mesh_triangle_count": int(len(merged_seed_triangles)),
            "uncovered_fallback_triangle_count": int(len(fallback_triangles)),
            "output_triangle_count": int(len(combined_seed_triangles)),
            "mesh_triangle_area": _percentiles(_triangle_areas(merged_seed_triangles)),
            "mesh_triangle_max_edge": _percentiles(_triangle_max_edges(merged_seed_triangles))
            if len(merged_seed_triangles)
            else {},
            "fallback_triangle_area": _percentiles(_triangle_areas(fallback_triangles)),
            "fallback_triangle_max_edge": _percentiles(_triangle_max_edges(fallback_triangles))
            if len(fallback_triangles)
            else {},
            "combined_triangle_area": _percentiles(_triangle_areas(combined_seed_triangles)),
            "combined_triangle_max_edge": _percentiles(_triangle_max_edges(combined_seed_triangles)),
            "adaptive": triangle_soup_stats.get("adaptive", {}),
            "coverage": fallback_stats,
        }
        (output_dir / "mesh_seed_triangles" / "seed_stats.json").write_text(
            json.dumps(seed_stats, indent=2) + "\n",
            encoding="utf-8",
        )
    (output_dir / "kimera_conversion.json").write_text(
        json.dumps(conversion_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(conversion_summary, indent=2))


if __name__ == "__main__":
    main()
