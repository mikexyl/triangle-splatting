#!/usr/bin/env python3
"""Run the Vicon room seed/camera initialization ablation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AblationRun:
    name: str
    source_path: Path
    seed_init_mode: str
    cameras: str
    seed_geometry: str


def _run_command(command: list[str], log_path: Path) -> None:
    printable = " ".join(command)
    print(f"$ {printable}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ {printable}\n")
        log_file.flush()
        subprocess.run(command, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=True)


def _load_final_metrics(model_dir: Path, iteration: int) -> dict[str, float | None]:
    results_path = model_dir / "results.json"
    if not results_path.exists():
        return {"SSIM": None, "PSNR": None, "LPIPS": None}
    results = json.loads(results_path.read_text(encoding="utf-8"))
    method = f"ours_{iteration}"
    if method not in results:
        return {"SSIM": None, "PSNR": None, "LPIPS": None}
    return {
        "SSIM": results[method].get("SSIM"),
        "PSNR": results[method].get("PSNR"),
        "LPIPS": results[method].get("LPIPS"),
    }


def _load_initial_metrics(model_dir: Path) -> dict[str, float | None]:
    metrics_path = model_dir / "initial_metrics.json"
    if not metrics_path.exists():
        return {"SSIM": None, "PSNR": None, "LPIPS": None}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "SSIM": metrics.get("SSIM"),
        "PSNR": metrics.get("PSNR"),
        "LPIPS": metrics.get("LPIPS"),
    }


def _format_metric(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _write_summary(output_dir: Path, rows: list[dict[str, object]], alignment_summary: dict[str, object]) -> None:
    summary = {
        "alignment": alignment_summary,
        "runs": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Vicon Room Seed Initialization Ablation",
        "",
        f"- Matched cameras for Sim(3): {alignment_summary.get('matched_camera_count')}",
        f"- Camera-center RMSE after alignment: {_format_metric(alignment_summary.get('camera_center_rmse'))}",
        f"- Kimera-to-COLMAP scale: {_format_metric(alignment_summary.get('kimera_to_colmap', {}).get('scale'))}",
        "",
        "| Run | Cameras/images | Seed geometry | Initial PSNR | Final PSNR | Initial SSIM | Final SSIM | Initial LPIPS | Final LPIPS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        initial = row["initial_metrics"]
        final = row["final_metrics"]
        lines.append(
            "| {name} | {cameras} | {seed_geometry} | {initial_psnr} | {final_psnr} | "
            "{initial_ssim} | {final_ssim} | {initial_lpips} | {final_lpips} |".format(
                name=row["name"],
                cameras=row["cameras"],
                seed_geometry=row["seed_geometry"],
                initial_psnr=_format_metric(initial.get("PSNR")),
                final_psnr=_format_metric(final.get("PSNR")),
                initial_ssim=_format_metric(initial.get("SSIM")),
                final_ssim=_format_metric(final.get("SSIM")),
                initial_lpips=_format_metric(initial.get("LPIPS")),
                final_lpips=_format_metric(final.get("LPIPS")),
            )
        )
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vicon room seed initialization ablations")
    parser.add_argument(
        "--kimera-dataset",
        default="output/vicon_room_1_texture_triangle_dataset_reduced",
        help="Reduced Kimera mesh-triangle dataset",
    )
    parser.add_argument(
        "--sfm-dataset",
        default="output/vicon_room_1_kimera_pose_colmap_sfm/dataset",
        help="COLMAP SfM dataset",
    )
    parser.add_argument("--output-dir", default="output/vicon_room_1_seed_ablation", help="Where to write runs")
    parser.add_argument(
        "--hybrid-dataset-dir",
        default="output/vicon_room_1_seed_ablation_datasets",
        help="Where to write hybrid datasets",
    )
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--no-dome", action="store_true", help="Pass --no_dome to initial evaluation and training")
    parser.add_argument(
        "--disable-densification",
        action="store_true",
        help="Set --densification_interval 999999 for fixed-soup comparison",
    )
    parser.add_argument("--skip-training", action="store_true", help="Only prepare datasets and initial metrics")
    parser.add_argument("--skip-initial", action="store_true", help="Skip initial render metrics")
    parser.add_argument(
        "--only-runs",
        nargs="+",
        default=None,
        help="Optional run names to execute. Metrics for other runs are still included if already present.",
    )
    args = parser.parse_args()

    kimera_dataset = (REPO_ROOT / args.kimera_dataset).resolve()
    sfm_dataset = (REPO_ROOT / args.sfm_dataset).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    hybrid_dataset_dir = (REPO_ROOT / args.hybrid_dataset_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_log = output_dir / "prepare.log"
    _run_command(
        [
            sys.executable,
            "scripts/prepare_seed_ablation_datasets.py",
            "--kimera-dataset",
            str(kimera_dataset),
            "--sfm-dataset",
            str(sfm_dataset),
            "--output-dir",
            str(hybrid_dataset_dir),
        ],
        prepare_log,
    )
    alignment_summary = json.loads((hybrid_dataset_dir / "alignment_summary.json").read_text(encoding="utf-8"))

    runs = [
        AblationRun("colmap_sfm_point", sfm_dataset, "point", "COLMAP undistorted + BA/refined", "SfM point cloud"),
        AblationRun(
            "colmap_mesh_point",
            hybrid_dataset_dir / "colmap_mesh_point",
            "point",
            "COLMAP undistorted + BA/refined",
            "Kimera mesh upsampled points aligned to COLMAP",
        ),
        AblationRun(
            "colmap_mesh_triangle",
            hybrid_dataset_dir / "colmap_mesh_triangle",
            "mesh_triangle",
            "COLMAP undistorted + BA/refined",
            "Kimera mesh triangles aligned to COLMAP",
        ),
        AblationRun(
            "kimera_sfm_point",
            hybrid_dataset_dir / "kimera_sfm_point",
            "point",
            "Kimera poses + Kimera undistorted",
            "SfM point cloud aligned to Kimera",
        ),
        AblationRun(
            "kimera_mesh_point",
            kimera_dataset,
            "point",
            "Kimera poses + Kimera undistorted",
            "Kimera mesh upsampled points",
        ),
        AblationRun("kimera_mesh_triangle", kimera_dataset, "mesh_triangle", "Kimera poses + Kimera undistorted", "Kimera mesh triangles"),
    ]
    selected_runs = set(args.only_runs) if args.only_runs else None
    valid_run_names = {run.name for run in runs}
    if selected_runs and not selected_runs <= valid_run_names:
        unknown = ", ".join(sorted(selected_runs - valid_run_names))
        raise ValueError(f"Unknown --only-runs value(s): {unknown}")

    rows = []
    for run in runs:
        model_dir = output_dir / run.name
        model_dir.mkdir(parents=True, exist_ok=True)
        log_path = model_dir / "ablation.log"
        should_execute = selected_runs is None or run.name in selected_runs

        if should_execute and not args.skip_initial:
            initial_command = [
                sys.executable,
                "scripts/evaluate_initial_seed.py",
                "-s",
                str(run.source_path),
                "-m",
                str(model_dir),
                "--eval",
                "--seed_init_mode",
                run.seed_init_mode,
                "--output",
                str(model_dir / "initial_metrics.json"),
            ]
            if args.no_dome:
                initial_command.append("--no_dome")
            _run_command(initial_command, log_path)

        if should_execute and not args.skip_training:
            train_command = [
                sys.executable,
                "train.py",
                "-s",
                str(run.source_path),
                "-m",
                str(model_dir),
                "--eval",
                "--seed_init_mode",
                run.seed_init_mode,
                "--iterations",
                str(args.iterations),
                "--test_iterations",
                "1",
                str(args.iterations),
                "--save_iterations",
                str(args.iterations),
            ]
            if args.no_dome:
                train_command.append("--no_dome")
            if args.disable_densification:
                train_command.extend(["--densification_interval", "999999"])
            _run_command(train_command, log_path)

            _run_command([sys.executable, "render.py", "-m", str(model_dir), "--iteration", str(args.iterations)], log_path)
            _run_command([sys.executable, "metrics.py", "-m", str(model_dir)], log_path)

        rows.append(
            {
                "name": run.name,
                "source_path": str(run.source_path),
                "model_path": str(model_dir),
                "seed_init_mode": run.seed_init_mode,
                "cameras": run.cameras,
                "seed_geometry": run.seed_geometry,
                "initial_metrics": _load_initial_metrics(model_dir),
                "final_metrics": _load_final_metrics(model_dir, args.iterations),
            }
        )
        _write_summary(output_dir, rows, alignment_summary)

    _write_summary(output_dir, rows, alignment_summary)
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
