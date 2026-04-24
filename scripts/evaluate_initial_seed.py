#!/usr/bin/env python3
"""Render and score a scene immediately after triangle initialization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene, TriangleModel
from triangle_renderer import render
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.metric_utils import lpips_vgg
from utils.train_runner import SEED_INIT_ARG_NAMES, add_common_training_args, copy_named_args


def _evaluate_views(views, triangles, pipeline, background):
    ssims = []
    psnrs = []
    lpipss = []
    per_view = {}

    for view in tqdm(views, desc="Initial metric evaluation"):
        rendering = torch.clamp(render(view, triangles, pipeline, background)["render"], 0.0, 1.0).unsqueeze(0)
        gt = view.original_image[0:3, :, :].unsqueeze(0).cuda()
        view_ssim = ssim(rendering, gt)
        view_psnr = psnr(rendering, gt)
        view_lpips = lpips_vgg(rendering, gt)
        ssims.append(view_ssim)
        psnrs.append(view_psnr)
        lpipss.append(view_lpips)
        per_view[f"{view.image_name}.png"] = {
            "SSIM": float(view_ssim.item()),
            "PSNR": float(view_psnr.item()),
            "LPIPS": float(view_lpips.item()),
        }

    return {
        "SSIM": float(torch.tensor(ssims).mean().item()) if ssims else None,
        "PSNR": float(torch.tensor(psnrs).mean().item()) if psnrs else None,
        "LPIPS": float(torch.tensor(lpipss).mean().item()) if lpipss else None,
        "view_count": int(len(views)),
        "per_view": per_view,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a seed immediately after initialization")
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    add_common_training_args(parser)
    parser.add_argument("--output", required=True, help="Path to write initial metrics JSON")
    parser.add_argument("--split", choices=("test", "train"), default="test", help="Camera split to evaluate")
    args = parser.parse_args()

    safe_state(args.quiet)

    dataset_args = model.extract(args)
    copy_named_args(dataset_args, args, SEED_INIT_ARG_NAMES)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(dataset_args.model_path).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        triangles = TriangleModel(dataset_args.sh_degree)
        scene = Scene(
            dataset_args,
            triangles,
            optimization.set_opacity,
            optimization.triangle_size,
            optimization.nb_points,
            optimization.set_sigma,
            args.no_dome,
            shuffle=False,
        )
        views = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
        if not views:
            raise RuntimeError(f"No {args.split} cameras available for initial seed evaluation")

        bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        metrics = _evaluate_views(views, triangles, pipeline.extract(args), background)

    metrics.update(
        {
            "source_path": dataset_args.source_path,
            "model_path": dataset_args.model_path,
            "seed_init_mode": args.seed_init_mode,
            "split": args.split,
            "triangle_count": int(triangles.get_triangles_points.shape[0]),
        }
    )
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
