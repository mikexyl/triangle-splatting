import argparse
from pathlib import Path
import sys

import numpy as np
import rerun as rr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.dataset_readers import readNerfSyntheticInfo


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


def _camera_positions(cam_infos: list) -> np.ndarray:
    if not cam_infos:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack([_camera_world_from_view(cam)[1] for cam in cam_infos], axis=0).astype(np.float32)


def _log_camera_path(entity_path: str, cam_infos: list, color: list[int]) -> None:
    if len(cam_infos) < 2:
        return
    positions = _camera_positions(cam_infos)
    rr.log(
        entity_path,
        rr.LineStrips3D([positions], colors=np.asarray([color], dtype=np.uint8), radii=np.asarray([0.01], dtype=np.float32)),
        static=True,
    )


def _log_camera_points(entity_path: str, cam_infos: list, color: list[int]) -> None:
    if not cam_infos:
        return
    positions = _camera_positions(cam_infos)
    rr.log(
        entity_path,
        rr.Points3D(
            positions=positions,
            colors=np.tile(np.asarray(color, dtype=np.uint8), (positions.shape[0], 1)),
            radii=np.full((positions.shape[0],), 0.03, dtype=np.float32),
        ),
        static=True,
    )


def _log_camera_frustums(entity_path: str, cam_infos: list, color: list[int], stride: int) -> None:
    if stride <= 0:
        stride = 1

    for idx, cam in enumerate(cam_infos[::stride]):
        image_name = getattr(cam, "image_name", f"camera_{idx:03d}")
        camera_path = f"{entity_path}/{idx:03d}_{_sanitize_entity_name(image_name)}"
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
                color=[*color, 255],
                line_width=0.01,
            ),
            static=True,
        )


def _log_split(split_name: str, cam_infos: list, color: list[int], frustum_stride: int) -> None:
    split_root = f"dataset/{split_name}"
    rr.log(split_root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    _log_camera_points(f"{split_root}/points", cam_infos, color)
    _log_camera_path(f"{split_root}/path", cam_infos, color)
    _log_camera_frustums(f"{split_root}/frustums", cam_infos, color, frustum_stride)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log dataset camera poses to Rerun")
    parser.add_argument("--dataset", required=True, help="Path to a dataset with transforms_train.json")
    parser.add_argument("--output-rrd", required=True, help="Output .rrd path")
    parser.add_argument("--spawn", action="store_true", help="Spawn the Rerun viewer while logging")
    parser.add_argument("--eval", action="store_true", help="Keep the train/test split instead of merging all frames into train")
    parser.add_argument("--white-background", action="store_true", help="Load images with white alpha compositing")
    parser.add_argument("--extension", default=".png", help="Image extension referenced by transforms files")
    parser.add_argument(
        "--frustum-stride",
        type=int,
        default=10,
        help="Log every Nth camera frustum to keep the view readable; <=0 logs all frustums",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output_rrd).expanduser().resolve()

    scene_info = readNerfSyntheticInfo(
        str(dataset_path),
        white_background=args.white_background,
        eval=args.eval,
        extension=args.extension,
    )

    rr.init("triangle_splatting.dataset_cameras", spawn=args.spawn)
    rr.log("dataset", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    _log_split("train", scene_info.train_cameras, [90, 150, 255], args.frustum_stride)
    if scene_info.test_cameras:
        _log_split("test", scene_info.test_cameras, [160, 160, 160], args.frustum_stride)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(output_path))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
