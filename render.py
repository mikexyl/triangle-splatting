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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from triangle_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from triangle_renderer import TriangleModel
from utils.rerun_utils import RerunLogger, create_rerun_config

def render_set(model_path, name, iteration, views, triangles, pipeline, background, rerun_logger=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if rerun_logger is not None:
        rerun_logger.log_render_cameras(name, views)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, triangles, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if rerun_logger is not None:
            rerun_logger.log_render_frame(
                split_name=name,
                frame_idx=idx,
                render_image=rendering,
                gt_image=gt,
                triangles_points=triangles.get_triangles_points,
                features_dc=triangles._features_dc,
                opacity=triangles.get_opacity,
            )

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    rerun_logger = RerunLogger("triangle_splatting.render", create_rerun_config(dataset))
    try:
        with torch.no_grad():
            triangles = TriangleModel(dataset.sh_degree)
            scene = Scene(args=dataset,
                      triangles=triangles,
                      init_opacity=None,
                      init_size=None,
                      nb_points=None,
                      set_sigma=None,
                      no_dome=False,
                      load_iteration=args.iteration,
                      shuffle=False)

            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            if not skip_train:
                 render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), triangles, pipeline, background, rerun_logger)

            if not skip_test:
                 render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), triangles, pipeline, background, rerun_logger)
    finally:
        rerun_logger.close()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rerun_spawn", action="store_true")
    parser.add_argument("--rerun_save", type=str, default=None)
    parser.add_argument(
        "--rerun_max_triangles",
        type=int,
        default=5000,
        help="Maximum number of triangles to log to Rerun. Use 0 to log all triangles.",
    )
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset_args = model.extract(args)
    for name in ("rerun", "rerun_spawn", "rerun_save", "rerun_max_triangles"):
        setattr(dataset_args, name, getattr(args, name))

    render_sets(dataset_args, args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
