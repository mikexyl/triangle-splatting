#!/usr/bin/env python3
"""Evaluate untrained mesh-triangle seed renders across triangle scale factors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene, TriangleModel
from scene.dataset_readers import fetchTriangleSoup
from triangle_renderer import render
from utils.general_utils import safe_state
from utils.image_utils import psnr


def _load_recorded_scale(source_path: Path) -> float:
    for metadata_path in (
        source_path / "kimera_conversion.json",
        source_path / "mesh_seed_triangles" / "seed_stats.json",
    ):
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        for key_path in (
            ("mesh_triangle_scale",),
            ("triangle_soup", "scale"),
        ):
            value = metadata
            for key in key_path:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            if value is not None:
                return float(value)
    return 1.0


def _scale_triangles_about_centroids(triangles: np.ndarray, scale_ratio: float) -> np.ndarray:
    centers = triangles.mean(axis=1, keepdims=True)
    return centers + (triangles - centers) * np.float32(scale_ratio)


def _evaluate_psnr(views, triangles, pipeline, background, split_name: str) -> dict[str, float | int | None]:
    psnrs = []
    for view in tqdm(views, desc=f"Initial PSNR {split_name}", leave=False):
        rendering = torch.clamp(render(view, triangles, pipeline, background)["render"], 0.0, 1.0).unsqueeze(0)
        gt = view.original_image[0:3, :, :].unsqueeze(0).cuda()
        psnrs.append(psnr(rendering, gt).detach().cpu())

    return {
        "PSNR": float(torch.stack(psnrs).mean().item()) if psnrs else None,
        "view_count": int(len(views)),
    }


def _write_csv(path: Path, results: list[dict[str, object]]) -> None:
    fieldnames = [
        "scale",
        "relative_to_recorded_scale",
        "triangle_count",
        "train_PSNR",
        "train_view_count",
        "test_PSNR",
        "test_view_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "scale": result["scale"],
                    "relative_to_recorded_scale": result["relative_to_recorded_scale"],
                    "triangle_count": result["triangle_count"],
                    "train_PSNR": result.get("train", {}).get("PSNR"),
                    "train_view_count": result.get("train", {}).get("view_count"),
                    "test_PSNR": result.get("test", {}).get("PSNR"),
                    "test_view_count": result.get("test", {}).get("view_count"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep initial mesh-triangle seed scale and report PSNR")
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
    parser.add_argument("--output-dir", default="output/initial_triangle_scale_sweep")
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    parser.add_argument("--triangle-soup", default=None, help="Override mesh_seed_triangles/all.npz")
    parser.add_argument(
        "--recorded-scale",
        type=float,
        default=None,
        help="Scale already baked into --triangle-soup. Defaults to dataset metadata, or 1.0.",
    )
    parser.add_argument("--no_dome", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if any(scale <= 0.0 for scale in args.scales):
        raise ValueError("--scales entries must all be > 0")

    safe_state(args.quiet)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.model_path:
        args.model_path = str(output_dir / "scene_model")

    dataset_args = model.extract(args)
    dataset_args.seed_init_mode = "mesh_triangle"
    Path(dataset_args.model_path).mkdir(parents=True, exist_ok=True)
    source_path = Path(dataset_args.source_path)
    recorded_scale = float(args.recorded_scale) if args.recorded_scale is not None else _load_recorded_scale(source_path)
    if recorded_scale <= 0.0:
        raise ValueError(f"recorded scale must be > 0, got {recorded_scale}")

    triangle_soup_path = (
        Path(args.triangle_soup).expanduser().resolve()
        if args.triangle_soup
        else source_path / "mesh_seed_triangles" / "all.npz"
    )
    base_triangles, colors = fetchTriangleSoup(triangle_soup_path)

    scene_triangles = TriangleModel(dataset_args.sh_degree)
    scene = Scene(
        dataset_args,
        scene_triangles,
        optimization.set_opacity,
        optimization.triangle_size,
        optimization.nb_points,
        optimization.set_sigma,
        args.no_dome,
        shuffle=False,
    )

    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline_args = pipeline.extract(args)
    split_views = {
        "train": scene.getTrainCameras(),
        "test": scene.getTestCameras(),
    }

    results = []
    with torch.no_grad():
        for scale in args.scales:
            scale_ratio = float(scale) / recorded_scale
            scaled_triangles = _scale_triangles_about_centroids(base_triangles, scale_ratio)
            triangles = TriangleModel(dataset_args.sh_degree)
            triangles.create_from_triangle_soup(
                scaled_triangles,
                colors,
                scene.cameras_extent,
                optimization.set_opacity,
                optimization.triangle_size,
                optimization.nb_points,
                optimization.set_sigma,
                args.no_dome,
            )
            row = {
                "scale": float(scale),
                "recorded_source_scale": recorded_scale,
                "relative_to_recorded_scale": scale_ratio,
                "triangle_count": int(triangles.get_triangles_points.shape[0]),
            }
            print(f"Evaluating initial triangle scale {scale:g} ({scale_ratio:g}x recorded soup)")
            for split_name in args.splits:
                row[split_name] = _evaluate_psnr(
                    split_views[split_name],
                    triangles,
                    pipeline_args,
                    background,
                    split_name,
                )
            results.append(row)
            del triangles
            torch.cuda.empty_cache()

    summary = {
        "source_path": str(source_path),
        "triangle_soup": str(triangle_soup_path),
        "recorded_source_scale": recorded_scale,
        "splits": args.splits,
        "results": results,
    }
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, results)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
