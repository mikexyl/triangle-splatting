import argparse
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import rerun as rr
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.dataset_readers import readColmapSceneInfo
from utils.sh_utils import SH2RGB


def _sanitize_entity_name(name: str) -> str:
    sanitized = []
    for char in name:
        if char.isalnum() or char in ("-", "_"):
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_") or "camera"


def _camera_world_from_view(cam_info) -> tuple[np.ndarray, np.ndarray]:
    rt = np.eye(4, dtype=np.float32)
    rt[:3, :3] = cam_info.R.transpose().astype(np.float32)
    rt[:3, 3] = np.asarray(cam_info.T, dtype=np.float32)
    world_from_camera = np.linalg.inv(rt)
    return world_from_camera[:3, :3], world_from_camera[:3, 3]


def _sample_rows(array: np.ndarray, max_rows: int | None) -> np.ndarray:
    if max_rows is None or len(array) <= max_rows:
        return array
    keep = np.linspace(0, len(array) - 1, max_rows, dtype=np.int64)
    return array[keep]


def _read_triangle_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(f"Failed to load triangle mesh from {path}")
    return vertices, faces


def _mesh3d(vertex_positions: np.ndarray, triangle_indices: np.ndarray, vertex_colors: np.ndarray):
    try:
        return rr.Mesh3D(
            vertex_positions=vertex_positions,
            triangle_indices=triangle_indices,
            vertex_colors=vertex_colors,
        )
    except TypeError:
        return rr.Mesh3D(
            vertex_positions=vertex_positions,
            indices=triangle_indices,
            vertex_colors=vertex_colors,
        )


def _center_vertices(vertices: np.ndarray) -> np.ndarray:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    return vertices - center


def _log_camera_set(split_name: str, cam_infos: list) -> None:
    rr.log(f"results/{split_name}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for idx, cam in enumerate(cam_infos):
        image_name = getattr(cam, "image_name", f"camera_{idx:03d}")
        entity_name = _sanitize_entity_name(image_name)
        camera_path = f"results/{split_name}/cameras/{idx:03d}_{entity_name}"
        image_path = f"{camera_path}/image"
        rotation, translation = _camera_world_from_view(cam)

        width = float(cam.width)
        height = float(cam.height)
        focal_x = width / (2.0 * np.tan(float(cam.FovX) / 2.0))
        focal_y = height / (2.0 * np.tan(float(cam.FovY) / 2.0))
        principal_point = [width / 2.0, height / 2.0]
        image_from_camera = [
            [focal_x, 0.0, principal_point[0]],
            [0.0, focal_y, principal_point[1]],
            [0.0, 0.0, 1.0],
        ]

        rr.log(
            camera_path,
            rr.Transform3D(
                translation=translation,
                mat3x3=rotation,
                relation=rr.TransformRelation.ParentFromChild,
            ),
            rr.Pinhole(
                image_from_camera=image_from_camera,
                resolution=[width, height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=1.0,
                color=[255, 170, 40, 255],
                line_width=0.01,
            ),
            static=True,
        )
        rr.log(image_path, rr.Image(np.asarray(cam.image.convert("RGB"))), static=True)


def _log_trained_triangles(entity_path: str, checkpoint_path: str, max_triangles: int | None) -> tuple[np.ndarray, float]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    triangles = state["triangles_points"].detach().cpu().numpy()
    features_dc = state["features_dc"].detach().squeeze(1).cpu().numpy()
    opacity = torch.sigmoid(state["opacity"]).reshape(-1, 1).mul(255.0).round().to(torch.uint8).cpu().numpy()

    triangles = _sample_rows(triangles, max_triangles)
    features_dc = _sample_rows(features_dc, max_triangles)
    opacity = _sample_rows(opacity, max_triangles)

    triangle_colors = SH2RGB(features_dc)
    triangle_colors = np.clip(255.0 * triangle_colors, 0.0, 255.0).astype(np.uint8)
    triangle_rgba = np.concatenate([triangle_colors, opacity], axis=1)

    vertex_positions = triangles.reshape(-1, 3).astype(np.float32)
    vertex_positions = _center_vertices(vertex_positions)
    vertex_colors = np.repeat(triangle_rgba, 3, axis=0)
    triangle_indices = np.arange(len(vertex_positions), dtype=np.uint32).reshape(-1, 3)

    rr.log(entity_path, _mesh3d(vertex_positions, triangle_indices, vertex_colors), static=True)
    extent = float((vertex_positions.max(axis=0) - vertex_positions.min(axis=0)).max())
    return vertex_positions, extent


def _log_reference_mesh(entity_path: str, mesh_path: str, x_offset: float, max_faces: int | None) -> None:
    vertices, faces = _read_triangle_mesh(mesh_path)
    faces = _sample_rows(faces, max_faces)
    vertices = _center_vertices(vertices)
    vertices[:, 0] += x_offset
    vertex_colors = np.repeat(np.array([[37, 99, 235, 255]], dtype=np.uint8), len(vertices), axis=0)
    rr.log(entity_path, _mesh3d(vertices, faces.astype(np.uint32), vertex_colors), static=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log DTU training results to Rerun")
    parser.add_argument("--dataset", required=True, help="Path to the DTU scan directory")
    parser.add_argument("--checkpoint", required=True, help="Path to point_cloud_state_dict.pt")
    parser.add_argument("--reference-mesh", required=True, help="Path to provided/reference PLY mesh")
    parser.add_argument("--output-rrd", required=True, help="Output .rrd path")
    parser.add_argument("--max-trained-triangles", type=int, default=20000)
    parser.add_argument("--max-reference-faces", type=int, default=100000)
    parser.add_argument("--spawn", action="store_true", help="Spawn the Rerun viewer while logging")
    args = parser.parse_args()

    rr.init("triangle_splatting.dtu_result", spawn=args.spawn)
    rr.log("results", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    scene_info = readColmapSceneInfo(args.dataset, images=None, eval=True)
    _log_camera_set("train", scene_info.train_cameras)
    _log_camera_set("test", scene_info.test_cameras)

    rr.log("results/meshes/trained", rr.Transform3D(translation=[-1.0, 0.0, 0.0]), static=True)
    rr.log("results/meshes/reference", rr.Transform3D(translation=[1.0, 0.0, 0.0]), static=True)

    _, trained_extent = _log_trained_triangles(
        "results/meshes/trained/geometry",
        args.checkpoint,
        args.max_trained_triangles,
    )

    spacing = max(1.0, 0.75 * trained_extent)
    rr.log("results/meshes/trained", rr.Transform3D(translation=[-spacing, 0.0, 0.0]), static=True)
    rr.log("results/meshes/reference", rr.Transform3D(translation=[spacing, 0.0, 0.0]), static=True)
    _log_reference_mesh(
        "results/meshes/reference/geometry",
        args.reference_mesh,
        x_offset=0.0,
        max_faces=args.max_reference_faces,
    )

    output_path = Path(args.output_rrd)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(output_path))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
