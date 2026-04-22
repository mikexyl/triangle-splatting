import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import rerun as rr


def _read_off_counts(handle) -> tuple[int, int]:
    while True:
        line = handle.readline()
        if not line:
            raise RuntimeError("Unexpected end of OFF file while reading counts")
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            parts = stripped.split()
            return int(parts[0]), int(parts[1])


def load_coff_mesh(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if header not in {"OFF", "COFF"}:
            raise RuntimeError(f"Unsupported OFF header: {header}")

        num_vertices, num_faces = _read_off_counts(handle)

        vertices = np.empty((num_vertices, 3), dtype=np.float32)
        for idx in range(num_vertices):
            parts = handle.readline().split()
            vertices[idx] = [float(parts[0]), float(parts[1]), float(parts[2])]

        faces = np.empty((num_faces, 3), dtype=np.int32)
        face_colors = np.empty((num_faces, 4), dtype=np.uint8)
        for idx in range(num_faces):
            parts = handle.readline().split()
            if int(parts[0]) != 3:
                raise RuntimeError("Only triangular OFF faces are supported")
            faces[idx] = [int(parts[1]), int(parts[2]), int(parts[3])]
            if len(parts) >= 8:
                face_colors[idx] = [
                    int(parts[4]),
                    int(parts[5]),
                    int(parts[6]),
                    int(parts[7]),
                ]
            else:
                face_colors[idx] = [217, 119, 6, 255]

    return vertices, faces, face_colors


def load_triangle_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    if len(vertices) == 0 or len(triangles) == 0:
        raise RuntimeError(f"Failed to load triangle mesh from {path}")
    return vertices, triangles


def center_mesh(vertices: np.ndarray) -> np.ndarray:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    return vertices - center


def sample_faces(
    faces: np.ndarray, face_colors: np.ndarray | None, max_faces: int | None
) -> tuple[np.ndarray, np.ndarray | None]:
    if max_faces is None or len(faces) <= max_faces:
        return faces, face_colors

    keep = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
    sampled_faces = faces[keep]
    sampled_colors = face_colors[keep] if face_colors is not None else None
    return sampled_faces, sampled_colors


def log_face_colored_mesh(
    entity_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
) -> None:
    vertex_positions = vertices[faces].reshape(-1, 3)
    vertex_colors = np.repeat(face_colors, 3, axis=0)
    triangle_indices = np.arange(len(vertex_positions), dtype=np.uint32).reshape(-1, 3)
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=vertex_positions,
            triangle_indices=triangle_indices,
            vertex_colors=vertex_colors,
        ),
        static=True,
    )


def log_vertex_colored_mesh(
    entity_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    rgba: np.ndarray,
) -> None:
    vertex_colors = np.repeat(rgba[None, :], len(vertices), axis=0)
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=faces.astype(np.uint32),
            vertex_colors=vertex_colors,
        ),
        static=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Log trained and reference meshes to Rerun")
    parser.add_argument("--trained-off", required=True, help="Path to the trained COFF/OFF mesh")
    parser.add_argument("--provided-mesh", required=True, help="Path to the reference PLY mesh")
    parser.add_argument("--output-rrd", required=True, help="Output Rerun recording path")
    parser.add_argument("--max-provided-faces", type=int, default=100000)
    parser.add_argument("--max-trained-faces", type=int, default=50000)
    parser.add_argument("--spawn", action="store_true", help="Spawn the Rerun viewer while logging")
    args = parser.parse_args()

    rr.init("triangle_splatting.dtu_meshes", spawn=args.spawn)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    trained_vertices, trained_faces, trained_face_colors = load_coff_mesh(args.trained_off)
    trained_vertices = center_mesh(trained_vertices)
    trained_faces, trained_face_colors = sample_faces(
        trained_faces, trained_face_colors, args.max_trained_faces
    )

    provided_vertices, provided_faces = load_triangle_mesh(args.provided_mesh)
    provided_vertices = center_mesh(provided_vertices)
    provided_faces, _ = sample_faces(provided_faces, None, args.max_provided_faces)

    trained_extent = trained_vertices.max(axis=0) - trained_vertices.min(axis=0)
    provided_extent = provided_vertices.max(axis=0) - provided_vertices.min(axis=0)
    spacing = 0.6 * max(float(trained_extent[0]), float(provided_extent[0]), 1.0)

    rr.log(
        "meshes/trained",
        rr.Transform3D(translation=[-spacing, 0.0, 0.0]),
        static=True,
    )
    rr.log(
        "meshes/provided",
        rr.Transform3D(translation=[spacing, 0.0, 0.0]),
        static=True,
    )

    log_face_colored_mesh("meshes/trained/geometry", trained_vertices, trained_faces, trained_face_colors)
    log_vertex_colored_mesh(
        "meshes/provided/geometry",
        provided_vertices,
        provided_faces,
        np.array([37, 99, 235, 255], dtype=np.uint8),
    )

    output_path = Path(args.output_rrd)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(output_path))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
