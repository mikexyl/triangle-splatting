# Repository Guidelines

## Project Structure & Module Organization
Top-level scripts drive the main workflow: `train.py`, `render.py`, `metrics.py`, `full_eval.py`, `train_game_engine.py`, `create_video.py`, and `create_off.py`. Core logic lives in `scene/`, `triangle_renderer/`, and `utils/`; CLI argument groups are defined in `arguments/`. Native dependencies are vendored in `submodules/` (`diff-triangle-rasterization`, `simple-knn`); COLMAP is provided by Pixi for SfM ablations. Benchmark helpers live under `scripts/eval_dtu/` and `scripts/eval_tnt/`, while `bash_scripts/` contains Slurm batch wrappers for full-scene evaluation. Keep large checkpoints, datasets, and rendered outputs out of Git.

## Build, Test, and Development Commands
Create the environment with `micromamba create -f requirements.yaml` (Python 3.11, CUDA 12.6). Build the rasterizer with `bash compile.sh`, then install KNN with `cd submodules/simple-knn && pip install .`. Main workflows:

- Put training, rendering, metric, and evaluation outputs under `output/` by default, for example `-m output/<run_name>`. Use `/tmp` only for explicitly temporary scratch runs.
- `pixi run colmap-help`, `pixi run kimera-sfm-vicon-room-1`, `pixi run visualize-kimera-sfm-vicon-room-1`, and `pixi run train-kimera-sfm-vicon-room-1` cover the Pixi COLMAP check, Kimera-pose SfM dataset generation, Rerun visualization, and baseline training workflow.
- `pixi run prepare-kimera-mesh-vicon-room-1-reduced` builds the reduced texture-initialized Kimera mesh triangle dataset using centroid-voxel merging to avoid repeated triangles from overlapping per-frame meshes; `pixi run train-kimera-mesh-vicon-room-1-reduced-2500` runs the matching fixed-soup smoke ablation.
- `python train.py -s <scene_dir> -m output/<run_name> --eval` trains a model.
- `python train.py -s <scene_dir> -m output/<run_name> --outdoor --eval` uses outdoor settings.
- `python render.py -m <model_dir>` renders train/test views.
- `python metrics.py -m <model_dir>` computes SSIM, PSNR, and LPIPS.
- `python full_eval.py --output_path output/<eval_name> -m360 <mipnerf360_root> -tat <tat_root>` reproduces the paper pipeline.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions, variables, and files, and `CapWords` for classes. Keep imports grouped as standard library, third-party, and local modules. Match the current CLI style in `arguments/__init__.py`: descriptive long flags, short aliases only when already established. No formatter or linter is configured, so keep changes small, consistent, and readable.

## Testing Guidelines
There is no dedicated unit-test suite yet. Validate changes with the smallest relevant smoke test: train a short run, render outputs, and rerun `python metrics.py -m <model_dir>`. For benchmark or evaluation changes, use `full_eval.py` or the appropriate script in `bash_scripts/` / `scripts/eval_*`. Document the dataset, command, and resulting metric deltas in your PR.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `Update README.md` and `Added game engine link`. Keep commit titles concise, imperative, and scoped to the changed area, for example `Fix metric export for missing renders`. PRs should describe the motivation, list reproduction commands, mention dataset paths used for validation, and include before/after metrics or rendered screenshots for visual changes. Call out submodule updates and dependency changes explicitly.
