#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python RADAR_inference/inference_merlin_testset_all.py --cfg-path RADAR_inference/inference_merlin_testset_all.yaml "$@"
