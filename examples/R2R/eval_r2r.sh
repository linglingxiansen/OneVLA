#!/bin/bash

# ==========================================================================================
# R2R/RxR VLN-CE Evaluation for QwenGR00T_with_Language 14-dim models (Qwen3-VL backbone)
# With configurable stop threshold + chunk voting
# Usage: bash examples/R2R/eval_r2r.sh <ckpt_path> [stop_threshold] [chunk_stop_ratio]
# ==========================================================================================

export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

# NOTE: Activate your VLN-CE environment before running, e.g.:
# conda activate onevla_vlnce

###########################################################################################
# Set ONEVLA_HOME to the root of this repository
ONEVLA_HOME="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_DIR=${ONEVLA_HOME}/examples/R2R/eval_files

export PYTHONPATH=${ONEVLA_HOME}:${EVAL_DIR}:${PYTHONPATH:-}
###########################################################################################

MODEL_PATH=${1:?"Usage: bash examples/R2R/eval_r2r.sh <ckpt_path> [stop_threshold] [chunk_stop_ratio]"}

# Stop threshold config (set to 0 to disable, fallback to argmax)
export ONEVLA_STOP_THRESHOLD=${2:-0}
export ONEVLA_CHUNK_STOP_RATIO=${3:-0}

export ONEVLA_QWEN_CHECKPOINT="$MODEL_PATH"
export ONEVLA_ATTN_IMPL="sdpa"

CHUNKS=8
EXP_SAVE="video-data"
CONFIG_PATH="${ONEVLA_HOME}/VLN_CE/vlnce_baselines/config/r2r_baselines/onevla_r2r.yaml"

# Parse checkpoint info for save path
CKPT_FILENAME=$(basename "$MODEL_PATH")
CKPT_STEP=${CKPT_FILENAME#steps_}
CKPT_STEP=${CKPT_STEP%%_*}
MODEL_DIR=$(dirname "$(dirname "$MODEL_PATH")")
MODEL_NAME_DIR="${MODEL_DIR##*/}"
SAVE_PATH="${ONEVLA_HOME}/examples/R2R/r2r_eval_result/${MODEL_NAME_DIR}/${CKPT_STEP}_stop${ONEVLA_STOP_THRESHOLD}_ratio${ONEVLA_CHUNK_STOP_RATIO}"

mkdir -p ${SAVE_PATH}

echo "================================================"
echo "R2R Eval for 14-dim QwenGR00T_with_Language model"
echo "Checkpoint:       ${MODEL_PATH}"
echo "Stop threshold:   ${ONEVLA_STOP_THRESHOLD}"
echo "Chunk stop ratio: ${ONEVLA_CHUNK_STOP_RATIO}"
echo "Results:          ${SAVE_PATH}"
echo "================================================"

cd ${ONEVLA_HOME}

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Starting chunk ${IDX} on GPU $(( IDX % 8 ))"
    CUDA_VISIBLE_DEVICES=$(( IDX % 8 )) python ${EVAL_DIR}/run.py \
        --exp-config $CONFIG_PATH \
        --split-num $CHUNKS \
        --split-id $IDX \
        --model-path $MODEL_PATH \
        --result-path $SAVE_PATH \
        --exp-save $EXP_SAVE &
done

wait

echo "All ${CHUNKS} chunks completed!"
echo "Results at: ${SAVE_PATH}"
