# SimplerEnv Evaluation

This directory contains the integrated SimplerEnv evaluation harness for OneVLA on WidowX Bridge robot manipulation tasks.

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n onevla_simpler python=3.10 -y
conda activate onevla_simpler
```

### 2. Install PyTorch

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install ManiSkill2_real2sim (Simulation Backend)

```bash
git clone https://github.com/simpler-env/ManiSkill2_real2sim.git
cd ManiSkill2_real2sim
pip install -e .
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
pip install gymnasium sapien==2.2.2 transforms3d mediapy tyro matplotlib websockets msgpack
pip install numpy==1.24.4
```

### Common Issues

When testing on NVIDIA A100, you may encounter:
```
libvulkan.so.1: cannot open shared object file: No such file or directory
```
Fix by following: [ManiSkill Installation Guide - Vulkan Section](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)

## Running Evaluation

Run from the **repository root**:

```bash
conda activate onevla_simpler
bash examples/SimplerEnv/eval_simplerenv.sh /path/to/checkpoint.pt
```

This evaluates the following WidowX Bridge tasks:
- **V1 Tasks**: StackGreenCubeOnYellowCube, PutCarrotOnPlate, PutSpoonOnTableCloth
- **V2 Tasks**: PutEggplantInBasket

Results (videos and success rates) are saved to `results/simplerenv/`.

## Evaluation Configuration

Key parameters in `examples/SimplerEnv/eval_simplerenv.sh`:
- `num_trials_per_task`: Number of episodes per task (default: 1)
- `--max-episode-steps`: Maximum steps per episode (default: 120)
- `--control-freq`: Control frequency in Hz (default: 5)
- `--obj-episode-range`: Object variation range (default: 0-24)

## File Structure

```
simpler_env/
├── __init__.py                  # Registers all ManiSkill2 environments
├── evaluation/
│   ├── maniskill2_evaluator.py  # Main evaluation loop
│   └── argparse.py              # Argument parsing
└── utils/
    ├── env/
    │   ├── env_builder.py       # build_maniskill2_env()
    │   └── observation_utils.py # Image extraction from obs
    ├── action/
    │   └── action_ensemble.py   # Action ensemble utilities
    ├── visualization.py         # Video saving
    └── metrics.py               # Success metrics
```
