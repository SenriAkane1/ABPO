#!/bin/bash


HF_MODEL_PATH="./DeepSeek-R1-Distill-Qwen-1.5B/"
LOCAL_DIR="checkpoints/actor"
TARGET_DIR="your_target_dir"




python3 scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path "$HF_MODEL_PATH" \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR"


