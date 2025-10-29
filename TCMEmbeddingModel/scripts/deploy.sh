#!/usr/bin/env bash
set -euo pipefail

# export CUDA_VISIBLE_DEVICES=3

# MODEL_PATH="../weights/argument/Qwen3-embedding-0.6B-finetuned_v4/checkpoint-1420/"
#MODEL_PATH="../weights/fine_tuned/Qwen3-embedding-0.6B-finetuned_v3/checkpoint-3375"
MODEL_PATH="NYCU-CGI-LLM/Qwen3-Embedding-0.6B-TCM-SD"
MODEL_NAME="Qwen3-Embedding-0.6B-finetuned-v3"
SERVE_PORT=8000

vllm serve ${MODEL_PATH} \
  --served_model_name ${MODEL_NAME} \
  --port ${SERVE_PORT} \
  --gpu-memory-util 0.7

# vllm serve ../weights/fine_tuned/Qwen3-embedding-0.6B-finetuned_v3/checkpoint-3375 \
#     --served_model_name Qwen3-Embedding-0.6B-finetuned-v3 \
#     --port 8000 \
#     --gpu-memory-util 0.7 \
#     # --host 0.0.0.0

