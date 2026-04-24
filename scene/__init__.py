#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import fetchPly, fetchTriangleSoup, sceneLoadTypeCallbacks
from scene.triangle_model import TriangleModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


def _merge_point_clouds(point_clouds):
    valid_clouds = [pcd for pcd in point_clouds if pcd is not None and len(pcd.points) > 0]
    if not valid_clouds:
        return None

    return valid_clouds[0].__class__(
        points=np.concatenate([np.asarray(pcd.points) for pcd in valid_clouds], axis=0),
        colors=np.concatenate([np.asarray(pcd.colors) for pcd in valid_clouds], axis=0),
        normals=np.concatenate([np.asarray(pcd.normals) for pcd in valid_clouds], axis=0),
    )


def _merge_triangle_soups(triangle_soups):
    valid_soups = [soup for soup in triangle_soups if soup is not None and len(soup[0]) > 0]
    if not valid_soups:
        return None

    return (
        np.concatenate([soup[0] for soup in valid_soups], axis=0),
        np.concatenate([soup[1] for soup in valid_soups], axis=0),
    )


def _build_initial_online_seed_point_cloud(source_path, cam_infos, initial_count):
    if initial_count <= 0:
        return None

    seed_paths = []
    seen_paths = set()
    for cam_info in cam_infos[:min(initial_count, len(cam_infos))]:
        seed_rel_path = getattr(cam_info, "mesh_seed_path", None)
        if not seed_rel_path or seed_rel_path in seen_paths:
            continue
        seen_paths.add(seed_rel_path)
        seed_paths.append(os.path.join(source_path, seed_rel_path))

    if not seed_paths:
        return None

    return _merge_point_clouds([fetchPly(seed_path) for seed_path in seed_paths if os.path.exists(seed_path)])


def _build_initial_online_seed_triangle_soup(source_path, cam_infos, initial_count):
    if initial_count <= 0:
        return None

    seed_paths = []
    seen_paths = set()
    for cam_info in cam_infos[:min(initial_count, len(cam_infos))]:
        seed_rel_path = getattr(cam_info, "mesh_seed_triangle_path", None)
        if not seed_rel_path or seed_rel_path in seen_paths:
            continue
        seen_paths.add(seed_rel_path)
        seed_paths.append(os.path.join(source_path, seed_rel_path))

    if not seed_paths:
        return None

    return _merge_triangle_soups([fetchTriangleSoup(seed_path) for seed_path in seed_paths if os.path.exists(seed_path)])


def _load_offline_seed_triangle_soup(source_path):
    merged_path = os.path.join(source_path, "mesh_seed_triangles", "all.npz")
    if os.path.exists(merged_path):
        return fetchTriangleSoup(merged_path)
    return None


class Scene:

    triangles : TriangleModel

    def __init__(self, args : ModelParams, triangles : TriangleModel, init_opacity, init_size, nb_points, set_sigma, no_dome=False, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.triangles = triangles
        self.online_train_enabled = False
        self.online_train_initial_count = 0
        self.online_train_growth_interval = 0
        self.online_train_growth_count = 1
        self.online_train_window_size = 0
        self.online_train_counts = {}

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        seed_init_mode = getattr(args, "seed_init_mode", "point")
        if seed_init_mode not in ("point", "mesh_triangle"):
            raise ValueError(f"Unsupported seed_init_mode: {seed_init_mode}")

        initial_point_cloud = scene_info.point_cloud
        initial_triangle_soup = None
        if (
            not self.loaded_iter
            and getattr(args, "online_train_incremental_seed", False)
            and getattr(args, "online_train_initial_cameras", 0) > 0
        ):
            if seed_init_mode == "mesh_triangle":
                initial_triangle_soup = _build_initial_online_seed_triangle_soup(
                    args.source_path,
                    scene_info.train_cameras,
                    args.online_train_initial_cameras,
                )
                if initial_triangle_soup is not None:
                    print(
                        "Using incremental online triangle-soup seed initialization from the first "
                        f"{min(args.online_train_initial_cameras, len(scene_info.train_cameras))} revealed train frames"
                    )
            else:
                online_seed_point_cloud = _build_initial_online_seed_point_cloud(
                    args.source_path,
                    scene_info.train_cameras,
                    args.online_train_initial_cameras,
                )
                if online_seed_point_cloud is not None:
                    initial_point_cloud = online_seed_point_cloud
                    print(
                        "Using incremental online seed initialization from the first "
                        f"{min(args.online_train_initial_cameras, len(scene_info.train_cameras))} revealed train frames"
                    )
        elif not self.loaded_iter and seed_init_mode == "mesh_triangle":
            initial_triangle_soup = _load_offline_seed_triangle_soup(args.source_path)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.triangles.load(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)
                                                )
                                    )
        else:
            if seed_init_mode == "mesh_triangle":
                if initial_triangle_soup is None:
                    raise RuntimeError(
                        "seed_init_mode=mesh_triangle requires Kimera triangle seed files. "
                        "Run scripts/prepare_kimera_capture_dataset.py to generate mesh_seed_triangles/*.npz."
                    )
                self.triangles.create_from_triangle_soup(
                    initial_triangle_soup[0],
                    initial_triangle_soup[1],
                    self.cameras_extent,
                    init_opacity,
                    init_size,
                    nb_points,
                    set_sigma,
                    no_dome,
                )
            else:
                self.triangles.create_from_pcd(initial_point_cloud, self.cameras_extent, init_opacity, init_size, nb_points, set_sigma, no_dome)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.triangles.save(point_cloud_path)

    def enable_online_train_schedule(self, initial_count, growth_interval, growth_count, window_size=0):
        if initial_count <= 0:
            raise ValueError(f"initial_count must be > 0, got {initial_count}")
        if growth_interval <= 0:
            raise ValueError(f"growth_interval must be > 0, got {growth_interval}")
        if growth_count <= 0:
            raise ValueError(f"growth_count must be > 0, got {growth_count}")
        if window_size < 0:
            raise ValueError(f"window_size must be >= 0, got {window_size}")

        self.online_train_enabled = True
        self.online_train_initial_count = initial_count
        self.online_train_growth_interval = growth_interval
        self.online_train_growth_count = growth_count
        self.online_train_window_size = window_size
        self.online_train_counts = {
            scale: min(initial_count, len(cameras))
            for scale, cameras in self.train_cameras.items()
        }

    def update_online_train_set(self, iteration):
        if not self.online_train_enabled:
            return False

        updated = False
        steps = max(iteration - 1, 0) // self.online_train_growth_interval
        for scale, cameras in self.train_cameras.items():
            target_count = min(
                len(cameras),
                self.online_train_initial_count + steps * self.online_train_growth_count,
            )
            if target_count != self.online_train_counts[scale]:
                self.online_train_counts[scale] = target_count
                updated = True
        return updated

    def getActiveTrainCameraCount(self, scale=1.0):
        if not self.online_train_enabled:
            return len(self.train_cameras[scale])
        return self.online_train_counts[scale]

    def getActiveTrainWindowCount(self, scale=1.0):
        if not self.online_train_enabled:
            return len(self.train_cameras[scale])
        return len(self.getActiveTrainWindow(scale))

    def getTotalTrainCameraCount(self, scale=1.0):
        return len(self.train_cameras[scale])

    def getActiveTrainWindowStart(self, scale=1.0):
        if not self.online_train_enabled or self.online_train_window_size <= 0:
            return 0
        return max(0, self.online_train_counts[scale] - self.online_train_window_size)

    def getAllTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getRevealedTrainCameras(self, scale=1.0):
        if not self.online_train_enabled:
            return self.train_cameras[scale]
        return self.train_cameras[scale][:self.online_train_counts[scale]]

    def getNewlyRevealedTrainCameras(self, previous_count, scale=1.0):
        if not self.online_train_enabled:
            return []
        current_count = self.online_train_counts[scale]
        if previous_count >= current_count:
            return []
        return self.train_cameras[scale][previous_count:current_count]

    def getActiveTrainWindow(self, scale=1.0):
        if not self.online_train_enabled:
            return self.train_cameras[scale]
        window_end = self.online_train_counts[scale]
        window_start = self.getActiveTrainWindowStart(scale)
        return self.train_cameras[scale][window_start:window_end]

    def getTrainCameras(self, scale=1.0):
        return self.getActiveTrainWindow(scale)

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
