#!/usr/bin/env bash
set -e
cd /ssddata/CMMF/demo-radar

# NCCL 超时保存通信记录；torchrun 为每次启动分别保存两卡输出。
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1
export NCCL_DEBUG=WARN

CUDA_VISIBLE_DEVICES=2,3 torchrun \
  --standalone \
  --log-dir results/distributed_logs \
  --redirects 3 \
  --tee 3 \
  --nproc_per_node=2 \
  RADAR_train/train_gradcache.py \
  --cfg-path RADAR_train/train_merlin_finetune.yaml
