#!/bin/bash
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

# NOTE: Activate your SimplerEnv environment before running, e.g.:
# conda activate simplerenv

# Set ONEVLA_HOME to the root of this repository
ONEVLA_HOME="$(cd "$(dirname "$0")/../.." && pwd)"

export PYTHONPATH=${ONEVLA_HOME}:${PYTHONPATH:-}

your_ckpt=$1
if [ -z "$your_ckpt" ]; then
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
export ONEVLA_QWEN_CHECKPOINT=${your_ckpt}

# Evaluation configuration
num_trials_per_task=1  # Number of episodes per task
video_out_path="results/simplerenv/${folder_name}"

# Robot and scene configuration for V1 tasks
scene_name_v1=bridge_table_1_v1
robot_v1=widowx
rgb_overlay_path_v1=${ONEVLA_HOME}/data/real_inpainting/bridge_real_eval_1.png
robot_init_x_v1=0.147
robot_init_y_v1=0.028

# Robot and scene configuration for V2 tasks
scene_name_v2=bridge_table_1_v2
robot_v2=widowx_sink_camera_setup
rgb_overlay_path_v2=${ONEVLA_HOME}/data/real_inpainting/bridge_sink.png
robot_init_x_v2=0.127
robot_init_y_v2=0.06

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "Evaluating SimplerEnv with checkpoint:"
echo "${your_ckpt}"
echo "=========================================="

# V1 Tasks
declare -a ENV_NAMES_V1=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)

echo "Running V1 tasks..."
for env in "${ENV_NAMES_V1[@]}"; do
  for ((run_idx=1; run_idx<=num_trials_per_task; run_idx++)); do
    echo "Launching task [${env}] run#${run_idx}"

    python examples/SimplerEnv/start_simpler_env.py \
      --ckpt-path ${your_ckpt} \
      --robot ${robot_v1} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name_v1} \
      --rgb-overlay-path ${rgb_overlay_path_v1} \
      --robot-init-x ${robot_init_x_v1} ${robot_init_x_v1} 1 \
      --robot-init-y ${robot_init_y_v1} ${robot_init_y_v1} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --logging-dir ${video_out_path} \
      --use-local
  done
done

# V2 Tasks
declare -a ENV_NAMES_V2=(
  PutEggplantInBasketScene-v0
)

echo "Running V2 tasks..."
for env in "${ENV_NAMES_V2[@]}"; do
  for ((run_idx=1; run_idx<=num_trials_per_task; run_idx++)); do
    echo "Launching V2 task [${env}] run#${run_idx}"

    python examples/SimplerEnv/start_simpler_env.py \
      --ckpt-path ${your_ckpt} \
      --robot ${robot_v2} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name_v2} \
      --rgb-overlay-path ${rgb_overlay_path_v2} \
      --robot-init-x ${robot_init_x_v2} ${robot_init_x_v2} 1 \
      --robot-init-y ${robot_init_y_v2} ${robot_init_y_v2} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --logging-dir ${video_out_path} \
      --use-local
  done
done

echo "Evaluation finished!"
echo "Results saved to: ${video_out_path}"
