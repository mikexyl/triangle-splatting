<h1 align="center">Triangle Splatting for Real-Time Radiance Field Rendering [3DV 2026]</h1>

<div align="center">
  <a href="https://trianglesplatting.github.io/">Project page</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2505.19175">Arxiv</a> &nbsp;|&nbsp;
</div>
<br>

<p align="center">
  Jan Held*, Renaud Vandeghen*, Adrien Deliege, Abdullah Hamdi, Daniel Rebain, Silvio Giancola, Anthony Cioppa, Andrea Vedaldi, Bernard Ghanem, Andrea Tagliasacchi, Marc Van Droogenbroeck
</p>

<br>

<div align="center">
  <img src="assets/teaser.png" width="800" height="304" alt="Abstract Image">
</div>


This repo contains the official implementation for the paper "Triangle Splatting for Real-Time Radiance Field Rendering". 

Our work represents a significant advancement in radiance field rendering by introducing 3D triangles as rendering primitive. By leveraging the same primitive used in classical mesh representations, our method bridges the gap between neural rendering and traditional graphics pipelines. Triangle Splatting offers a compelling alternative to volumetric and implicit methods, achieving high visual fidelity with faster rendering performance. These results establish Triangle Splatting as a promising step toward mesh-aware neural rendering, unifying decades of GPU-accelerated graphics with modern differentiable frameworks.

## Trained Models and Rendered Test Views on MipNeRF360
All trained models and rendered test views for MipNeRF360 are available at the [following link.](https://drive.google.com/drive/folders/1YrH9IVU8QWgfnIg_i0iRHq_M9rNCNPyV?usp=sharing)

## Cloning the Repository + Installation

The code has been used and tested with Python 3.11 and CUDA 12.6.

You should clone the repository with the different submodules by running the following command:

```bash
git clone https://github.com/trianglesplatting/triangle-splatting --recursive
cd triangle-splatting
```

Then, we suggest to use a virtual environment to install the dependencies.

```bash
micromamba create -f requirements.yaml
```

Finally, you can compile the custom CUDA kernels by running the following command:

```bash
bash compile.sh
cd submodules/simple-knn
pip install .
```

COLMAP is available through the Pixi environment for SfM initialization ablations:

```bash
pixi install
pixi run colmap-help
```

## Training
To train our model, you can use the following command:
```bash
python train.py -s <path_to_scenes> -m <output_model_path> --eval
```

If you want to train the model on outdoor scenes, you should add the following command:  
```bash
python train.py -s <path_to_scenes> -m <output_model_path> --outdoor --eval
```

## SfM Initialization Ablation

The Kimera/COLMAP SfM ablation uses raw Kimera images and Kimera camera poses to seed COLMAP, triangulates a sparse point cloud, undistorts the reconstruction into the standard COLMAP dataset layout, and trains triangle splatting from the existing point-cloud initializer.

```bash
python scripts/run_kimera_pose_colmap_sfm.py \
  --capture-dir /home/mikexyl/workspaces/kimera_ros2_ws/artifacts/vicon_room_1 \
  --mav0-dir /data/euroc/V1_01_easy_raw/mav0 \
  --camera cam0 \
  --matcher sequential \
  --output-dir output/vicon_room_1_kimera_pose_colmap_sfm \
  --copy-images \
  --no-gpu \
  --overwrite
```

Then run the generated training command, or invoke it directly:

```bash
python train.py \
  -s output/vicon_room_1_kimera_pose_colmap_sfm/dataset \
  -m output/vicon_room_1_kimera_pose_colmap_sfm/train \
  --eval \
  --seed_init_mode point
```

The same vicon-room baseline is available through Pixi tasks:

```bash
pixi run kimera-sfm-vicon-room-1-dry-run
pixi run kimera-sfm-vicon-room-1
pixi run visualize-kimera-sfm-vicon-room-1
pixi run visualize-seed-triangles-vicon-room-1
pixi run train-kimera-sfm-vicon-room-1
```

All generated COLMAP databases, sparse models, undistorted datasets, summaries, and training outputs should stay under `output/`.

## Kimera Mesh Triangle Initialization

Kimera publishes overlapping per-frame meshes, so blindly concatenating every mesh snapshot can create many repeated initial triangles. The dataset converter can reduce the offline triangle soup with centroid-voxel merging while averaging overlapping texture colors:

```bash
python scripts/prepare_kimera_capture_dataset.py \
  --capture-dir /home/mikexyl/workspaces/kimera_ros2_ws/artifacts/vicon_room_1 \
  --mav0-dir /data/euroc/V1_01_easy_raw/mav0 \
  --camera cam0 \
  --output-dir output/vicon_room_1_texture_triangle_dataset_reduced \
  --mesh-voxel-size 0.05 \
  --mesh-sample-spacing 0.05 \
  --mesh-triangle-max-edge 0.25 \
  --mesh-triangle-max-count 50000 \
  --mesh-triangle-color-source texture \
  --mesh-triangle-merge-mode voxel \
  --mesh-triangle-merge-voxel-size 0.05
```

Use `--mesh-triangle-merge-mode concat` to preserve the previous unreduced behavior. The vicon-room reduced mesh workflow is also available through Pixi:

```bash
pixi run prepare-kimera-mesh-vicon-room-1-reduced
pixi run visualize-seed-triangles-vicon-room-1
pixi run train-kimera-mesh-vicon-room-1-reduced-2500
```

The seed-triangle visualization writes `output/vicon_room_1_seed_triangle_soups/seed_triangle_soups.rrd` and a matching JSON summary. It logs the reduced Kimera mesh triangles directly and synthesizes the SfM point-seed triangles with the same nearest-neighbor sizing rule used by point initialization; SfM triangle rotations are deterministic in the export for repeatable visual comparisons.

For a seed-only check of coverage-aware adaptive mesh triangles, run:

```bash
pixi run prepare-kimera-mesh-vicon-room-1-coverage-adaptive
pixi run visualize-kimera-mesh-vicon-room-1-coverage-adaptive
```

This writes `output/vicon_room_1_coverage_adaptive_mesh_seed/mesh_seed_triangles/all.npz` for trainer-compatible mesh-triangle initialization, plus `adaptive_mesh.npz` and `seed_stats.json` sidecars for inspection. The adaptive path coarsens low-texture planar Kimera mesh patches into larger triangles, keeps high-detail regions smaller, and reports projected mesh coverage while leaving uncovered image regions unseeded.

For a Kimera-only fallback that does not require SfM, use the image-plane variant. It places bounded camera-facing proxy triangles in large uncovered image components using only Kimera poses, images, mesh coverage, and nearby mesh depth:

```bash
pixi run prepare-kimera-mesh-vicon-room-1-coverage-adaptive-image-fallback
pixi run visualize-kimera-mesh-vicon-room-1-coverage-adaptive-image-fallback
```

An explicit SfM-fallback variant is kept for ablation only. It writes a separate output directory and appends bounded SfM fallback triangles where projected Kimera mesh coverage is missing:

```bash
pixi run prepare-kimera-mesh-vicon-room-1-coverage-adaptive-sfm-fallback
pixi run visualize-kimera-mesh-vicon-room-1-coverage-adaptive-sfm-fallback
```

To separate camera/image alignment from seed geometry, run the four-cell Vicon room ablation:

```bash
pixi run ablate-vicon-room-1-seeds-2500
```

This builds hybrid datasets under `output/vicon_room_1_seed_ablation_datasets/`, evaluates initial renders, trains the fixed-soup 2500-iteration cells, and writes `output/vicon_room_1_seed_ablation/summary.md`. The ablation includes SfM points, Kimera mesh upsampled points, and Kimera mesh triangle seeds under both Kimera and COLMAP camera/image pipelines where applicable.

## Rendering
To render a scene, you can use the following command:
```bash
python render.py -m <path_to_model>
```

## Live Viewer
If you want to inspect training or rendering live, you can stream images, metrics, and a sampled triangle mesh to a local Rerun viewer.

Training example:
```bash
pixi run python train.py -s <path_to_scene> -m <output_model_path> --eval --rerun --rerun_spawn
```

Rendering example:
```bash
pixi run python render.py -m <path_to_model> --rerun --rerun_spawn
```

If you save a recording, you can reopen it later with:
```bash
pixi run rerun <path_to_recording.rrd>
```

Useful options:
- `--rerun_save <path.rrd>` saves the recording for later playback.
- `--rerun_image_every <N>` controls how often training images are logged.
- `--rerun_mesh_every <N>` controls how often the live triangle mesh is refreshed.
- `--rerun_max_triangles <N>` limits mesh logging to a sampled subset for responsiveness. Use `0` to log all triangles.

## Evaluation
To evaluate the model, you can use the following command:
```bash
python metrics.py -m <path_to_model>
```

## Video
To render a video, you can use the following command:
```bash
python create_video.py -m <path_to_model>
```

## Replication of the results
To replicate the results of our paper, you can use the following command:
```bash
python full_eval.py --output_path <output_path> -m360 <path_to_MipNeRF360> -tat <path_to_T&T>
```

## Game engine
To create your own .off file:

1. Train your scene using ```train_game_engine.py```. This version includes some modifications, such as pruning low-opacity triangles and applying an additional loss in the final training iterations to encourage higher opacity. This makes the result more compatible with how game engines render geometry. These modifications are experimental, so feel free to adjust them or try your own variants. (For example, increasing the normal loss often improves quality by making triangles better aligned and reducing black holes.)

2. Run ```create_off.py``` to convert the optimized triangles into a .off file that can be imported into a game engine. You only need to provide the path to the trained model (e.g., point_cloud_state_dict.pt) and specify the desired output file name (e.g., mesh_colored.off).

Note: The script generates fully opaque triangles. If you want to include per-triangle opacity, you can extract and activate the raw opacity values using:
```
opacity_raw = sd["opacity"]
opacity = torch.sigmoid(opacity_raw.view(-1))
opacity_uint8 = (opacity * 255).to(torch.uint8)
```
Each triangle has a single opacity value, so if needed, assign the same value to all three of its vertices when exporting with:
```
for i, face in enumerate(faces):
            r, g, b = colors[i].tolist()
            a = opacity_uint8[i].item()
            f.write(f"3 {face[0].item()} {face[1].item()} {face[2].item()} {r} {g} {b} {a}\n")
```

If you want to run some pretrained scene on a game engine for yourself, you can download the *Garden* and *Room* scenes from the [following link](https://drive.google.com/drive/folders/1_TMXEFTdEACpHHvsmc5UeZMM-cMgJ3xW?usp=sharing). 

## BibTeX
If you find our work interesting or use any part of it, please cite our paper:
```bibtex
@article{Held2025Triangle,
title = {Triangle Splatting for Real-Time Radiance Field Rendering},
author = {Held, Jan and Vandeghen, Renaud and Deliege, Adrien and Hamdi, Abdullah and Cioppa, Anthony and Giancola, Silvio and Vedaldi, Andrea and Ghanem, Bernard and Tagliasacchi, Andrea and Van Droogenbroeck, Marc},
journal = {arXiv},
year = {2025},
}
```

As Triangle Splatting builds heavily on top of 3D Convex Splatting, please also cite it.
```bibtex
@InProceedings{held20243d,
title={3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes},
  author={Held, Jan and Vandeghen, Renaud and Hamdi, Abdullah and Deliege, Adrien and Cioppa, Anthony and Giancola, Silvio and Vedaldi, Andrea and Ghanem, Bernard and Van Droogenbroeck, Marc},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025},
}
```

## Acknowledgements
This project is built upon 3D Convex Splatting and 3D Gaussian Splatting. We want to thank the authors for their contributions.

J. Held and A. Cioppa are funded by the F.R.S.-FNRS. The research reported in this publication was supported by funding from KAUST Center of Excellence on GenAI, under award number 5940. This work was also supported by KAUST Ibn Rushd Postdoc Fellowship program. The present research benefited from computational resources made available on Lucia, the Tier-1 supercomputer of the Walloon Region, infrastructure funded by the Walloon Region under the grant agreement n°1910247.
