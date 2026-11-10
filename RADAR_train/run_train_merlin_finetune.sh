#!/usr/bin/env bash
set -e
cd /ssddata/CMMF/demo-radar

CUDA_VISIBLE_DEVICES=2,3 torchrun \
  --standalone \
  --nproc_per_node=2 \
  RADAR_train/train_gradcache.py \
  --cfg-path RADAR_train/train_merlin_finetune.yaml
