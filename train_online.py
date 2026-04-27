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

import sys
from argparse import ArgumentParser

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state
from utils.train_runner import (
    ONLINE_TRAIN_ARG_NAMES,
    RERUN_ARG_NAMES,
    SEED_INIT_ARG_NAMES,
    TrainingRunConfig,
    add_common_training_args,
    add_online_training_args,
    add_rerun_args,
    copy_named_args,
    run_training,
)


if __name__ == "__main__":
    parser = ArgumentParser(description="Primitive online replay training parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    add_common_training_args(parser)
    add_rerun_args(parser)
    add_online_training_args(parser)

    raw_args = sys.argv[1:]
    iterations_explicit = any(arg == "--iterations" or arg.startswith("--iterations=") for arg in raw_args)
    args = parser.parse_args(raw_args)
    if args.online_train_unbounded or not iterations_explicit:
        args.online_train_unbounded = True
        args.iterations = 0
        args.online_train_stop_when_frames_exhausted = True
    if args.iterations > 0:
        args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    dataset_args = lp.extract(args)
    copy_named_args(dataset_args, args, RERUN_ARG_NAMES)
    copy_named_args(dataset_args, args, ONLINE_TRAIN_ARG_NAMES)
    copy_named_args(dataset_args, args, SEED_INIT_ARG_NAMES)
    dataset_args.online_train_incremental_seed = True

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    run_training(
        dataset_args,
        op.extract(args),
        pp.extract(args),
        args.no_dome,
        args.outdoor,
        args.test_iterations,
        args.save_iterations,
        args.start_checkpoint,
        args.debug_from,
        TrainingRunConfig(
            rerun_app_id="triangle_splatting.online",
            online_train=True,
            online_train_initial_cameras=args.online_train_initial_cameras,
            online_train_camera_growth_interval=args.online_train_camera_growth_interval,
            online_train_camera_growth_count=args.online_train_camera_growth_count,
            online_train_window_size=args.online_train_window_size,
            online_train_min_prune_cameras=args.online_train_min_prune_cameras,
            online_train_unbounded=args.online_train_unbounded,
            online_train_stop_when_frames_exhausted=args.online_train_stop_when_frames_exhausted,
        ),
    )

    print("\nTraining complete.")
