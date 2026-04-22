import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import rerun as rr


def _read_triangle_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(f"Failed to load triangle mesh from {path}")
    return vertices, faces


def _sample_rows(array: np.ndarray, max_rows: int | None) -> np.ndarray:
    if max_rows is None or len(array) <= max_rows:
        return array
    keep = np.linspace(0, len(array) - 1, max_rows, dtype=np.int64)
    return array[keep]


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


def _extent_x(vertices: np.ndarray) -> float:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return float(maxs[0] - mins[0])


def _log_mesh(entity_path: str, vertices: np.ndarray, faces: np.ndarray, rgba: np.ndarray) -> None:
    vertex_colors = np.repeat(rgba[None, :], len(vertices), axis=0)
    rr.log(
        entity_path,
        _mesh3d(vertices, faces.astype(np.uint32), vertex_colors),
        static=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Log DTU mesh comparison results to Rerun")
    parser.add_argument("--fuse-mesh", required=True, help="Path to fuse.ply")
    parser.add_argument("--fuse-post-mesh", required=True, help="Path to fuse_post.ply")
    parser.add_argument("--reference-mesh", required=True, help="Path to the provided DTU reference mesh")
    parser.add_argument("--output-rrd", required=True, help="Output Rerun recording path")
    parser.add_argument("--max-fuse-faces", type=int, default=150000)
    parser.add_argument("--max-fuse-post-faces", type=int, default=120000)
    parser.add_argument("--max-reference-faces", type=int, default=150000)
    parser.add_argument("--spawn", action="store_true", help="Spawn the Rerun viewer while logging")
    args = parser.parse_args()

    rr.init("triangle_splatting.dtu_mesh_comparison", spawn=args.spawn)
    rr.log("comparison", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    mesh_specs = [
        (
            "fuse",
            args.fuse_mesh,
            args.max_fuse_faces,
            np.array([239, 68, 68, 220], dtype=np.uint8),
            "Fuse",
        ),
        (
            "fuse_post",
            args.fuse_post_mesh,
            args.max_fuse_post_faces,
            np.array([245, 158, 11, 220], dtype=np.uint8),
            "Fuse Post",
        ),
        (
            "reference",
            args.reference_mesh,
            args.max_reference_faces,
            np.array([37, 99, 235, 220], dtype=np.uint8),
            "Reference",
        ),
    ]

    loaded_meshes = []
    max_centered_extent_x = 1.0
    for mesh_name, mesh_path, max_faces, color, label in mesh_specs:
        vertices, faces = _read_triangle_mesh(mesh_path)
        faces = _sample_rows(faces, max_faces)
        centered_vertices = _center_vertices(vertices)
        max_centered_extent_x = max(max_centered_extent_x, _extent_x(centered_vertices))
        loaded_meshes.append(
            {
                "name": mesh_name,
                "path": mesh_path,
                "label": label,
                "vertices": vertices,
                "faces": faces,
                "centered_vertices": centered_vertices,
                "color": color,
            }
        )

    rr.log("comparison/overlay", rr.Boxes3D(centers=[[0.0, 0.0, 0.0]], half_sizes=[[0.0, 0.0, 0.0]]), static=True)
    for mesh in loaded_meshes:
        rr.log(
            f"comparison/overlay/{mesh['name']}",
            rr.Transform3D(translation=[0.0, 0.0, 0.0]),
            static=True,
        )
        _log_mesh(
            f"comparison/overlay/{mesh['name']}/geometry",
            mesh["vertices"],
            mesh["faces"],
            mesh["color"],
        )

    spacing = 0.8 * max_centered_extent_x + 0.25
    x_positions = [-spacing, 0.0, spacing]
    for mesh, x_pos in zip(loaded_meshes, x_positions):
        rr.log(
            f"comparison/side_by_side/{mesh['name']}",
            rr.Transform3D(translation=[x_pos, 0.0, 0.0]),
            static=True,
        )
        rr.log(
            f"comparison/side_by_side/{mesh['name']}/label",
            rr.TextDocument(f"{mesh['label']}\n{mesh['path']}", media_type="text/markdown"),
            static=True,
        )
        _log_mesh(
            f"comparison/side_by_side/{mesh['name']}/geometry",
            mesh["centered_vertices"],
            mesh["faces"],
            mesh["color"],
        )

    output_path = Path(args.output_rrd)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(output_path))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
