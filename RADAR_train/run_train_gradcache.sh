#!/usr/bin/env bash

# 减少变长张量分配产生的显存碎片；不增加容量，新启动的训练进程生效。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail

cd "$(dirname "$0")"

# 默认使用 GPU 3；多卡运行时可设置 CUDA_VISIBLE_DEVICES=2,3。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
IFS=, read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"

if (( ${#gpu_ids[@]} == 1 )); then
    exec python -u train_gradcache.py --cfg-path train_gradcache.yaml "$@"
fi

exec torchrun --standalone --nproc_per_node="${#gpu_ids[@]}" train_gradcache.py --cfg-path train_gradcache.yaml "$@"
