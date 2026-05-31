# VLN-CE Navigation Evaluation

This directory contains the integrated VLN-CE evaluation infrastructure for OneVLA on R2R and RxR vision-and-language navigation benchmarks.

Based on [NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE) and [VLN-CE](https://github.com/jacobkrantz/VLN-CE).

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n onevla_vlnce python=3.9 -y
conda activate onevla_vlnce
```

### 2. Install Habitat-Sim (v0.1.7)

```bash
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7=py3.9_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7
```

If the above fails due to network issues, download the package directly from [conda website](https://anaconda.org/aihabitat/habitat-sim/0.1.7/download/linux-64/habitat-sim-0.1.7-py3.9_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2) and install manually:
```bash
conda install habitat-sim-0.1.7-*.tar.bz2
```

### 3. Install Habitat-Lab (v0.1.7)

```bash
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -r requirements.txt
pip install -r habitat_baselines/rl/requirements.txt
pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
cd ..
```

### 4. Install OneVLA

```bash
cd OneVLA
pip install -e .
pip install -r requirements.txt
```

### 5. Install Additional Dependencies

```bash
pip install jsonlines lmdb msgpack-python networkx
```

## Data Preparation

### Scene Datasets

Download the Matterport3D scene datasets following [VLN-CE instructions](https://github.com/jacobkrantz/VLN-CE?tab=readme-ov-file#data).

### Episode Datasets

Download R2R and/or RxR episodes:
- **R2R**: `R2R_VLNCE_v1-3_preprocessed` from [VLN-CE](https://github.com/jacobkrantz/VLN-CE)
- **RxR**: If the official link is unavailable, download from [here](https://1drv.ms/u/c/aa19f644cf9d8afb/ETQ8Co-hGLFMjwd5HckKsvABjWvZ3cbPsWwdzbhmQDoL1g?e=WtO8Lm)

### Configure Data Paths

Update `VLN_CE/habitat_extensions/config/vlnce_task_navid_r2r.yaml` with your data locations:

```yaml
DATASET:
  TYPE: VLN-CE-v1
  SPLIT: val_unseen
  DATA_PATH: /your/path/to/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz
  SCENES_DIR: /your/path/to/scene_datasets
```

### Recommended Directory Structure

```
data/
├── datasets/
│   ├── R2R_VLNCE_v1-3_preprocessed/
│   │   ├── train/
│   │   ├── val_seen/
│   │   └── val_unseen/
│   │       └── val_unseen.json.gz
│   └── RxR_VLNCE_v0/          # (optional, for RxR eval)
│       └── ...
└── scene_datasets/
    └── mp3d/
        ├── 1LXtFkjw3qL/
        ├── ...
        └── zsNo4HB9uLZ/
```

## Running Evaluation

Run from the **repository root**:

### R2R Evaluation

```bash
conda activate onevla_vlnce
bash examples/R2R/eval_r2r.sh /path/to/r2r_checkpoint.pt [stop_threshold] [chunk_stop_ratio]
```

### RxR Evaluation

Same script, use RxR-trained checkpoint:
```bash
bash examples/R2R/eval_r2r.sh /path/to/rxr_checkpoint.pt [stop_threshold] [chunk_stop_ratio]
```

### Parameters

- `stop_threshold` (optional, default: 0): Stop action confidence threshold. Set to 0 to use argmax.
- `chunk_stop_ratio` (optional, default: 0): Chunk voting ratio for stop decision.

### Multi-GPU Evaluation

The evaluation script automatically splits episodes across 8 GPUs. Adjust `CHUNKS=8` in `examples/R2R/eval_r2r.sh` to match your available GPUs.

### Monitor Results

Results are saved to `examples/R2R/r2r_eval_result/`. Each episode produces a JSON file with metrics:
- `distance_to_goal`: Distance to goal at episode end
- `success`: Whether agent reached within 3m of goal
- `spl`: Success weighted by Path Length
- `path_length`: Total path traversed
- `oracle_success`: Whether agent was ever within 3m of goal

## File Structure

```
VLN_CE/
├── vlnce_baselines/
│   ├── config/
│   │   ├── default.py                    # get_config() - unified config builder
│   │   └── r2r_baselines/
│   │       └── navid_r2r.yaml            # R2R evaluation config
│   └── common/
│       └── environments.py               # VLNCEDaggerEnv registration
├── habitat_extensions/
│   ├── config/
│   │   ├── default.py                    # Extended task config definitions
│   │   └── vlnce_task_navid_r2r.yaml     # Task config (data paths here)
│   ├── task.py                           # VLN-CE-v1 dataset registration
│   ├── measures.py                       # SPL, NDTW, PathLength, OracleSuccess, etc.
│   ├── sensors.py                        # ShortestPathSensor, InstructionSensor, etc.
│   ├── actions.py                        # GoTowardPoint action
│   ├── maps.py                           # Top-down map visualization
│   ├── obs_transformers.py               # CenterCropperPerSensor
│   ├── shortest_path_follower.py         # Oracle path follower
│   ├── discrete_planner.py               # Discrete path planning
│   └── utils.py                          # Helper functions
└── data/
    └── connectivity_graphs.pkl           # MP3D connectivity for map drawing
```

## Acknowledgments

- [NaVid](https://arxiv.org/pdf/2402.15852) - Video-based VLM for VLN
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE) - Vision-and-Language Navigation in Continuous Environments
- [Habitat](https://github.com/facebookresearch/habitat-lab) - Embodied AI simulation platform
