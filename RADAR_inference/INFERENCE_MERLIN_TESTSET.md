# MERLIN test split inference

Both entry points select only `split=test` IDs from
`/data/dataset/json/merlin_report.json`, with images under
`/data/dataset/other_images`. The directory also contains validation scans;
these are excluded. Missing test images cause an explicit error.

The epoch-10 entry point uses zero-based `checkpoint_9.pth`; the official entry
point uses `ckpt/checkpoint_radar_plus.pth`. Each uses its own matched text
features with the same 20 MERLIN prompt groups, English tokenizer, HU clipping,
resampling and scoring from `inference_yg_images_epoch10.py`. Keep that shared module,
the exporter, original demo and network dependencies in place.

Validate without allocating a GPU:

```bash
docker exec -it -w /ssddata/CMMF/demo-radar radar-train-nvidia-pytorch-2610 \
  python RADAR_inference/inference_merlin_testset_epoch10.py --check-only --limit 1
docker exec -it -w /ssddata/CMMF/demo-radar radar-train-nvidia-pytorch-2610 \
  python RADAR_inference/inference_merlin_testset_radar_plus.py --check-only --limit 1
```

These commands verify all test image paths and inspect the first header.
Omit `--limit 1` to inspect all headers. Run sequentially on an available GPU,
for example physical GPU 0:

```bash
docker exec -it -e CUDA_VISIBLE_DEVICES=0 -w /ssddata/CMMF/demo-radar \
  radar-train-nvidia-pytorch-2610 \
  python RADAR_inference/inference_merlin_testset_epoch10.py --device cuda:0
docker exec -it -e CUDA_VISIBLE_DEVICES=0 -w /ssddata/CMMF/demo-radar \
  radar-train-nvidia-pytorch-2610 \
  python RADAR_inference/inference_merlin_testset_radar_plus.py --device cuda:0
```

Outputs under `results/merlin_testset`:

- `RADAR_infer_results_epoch10_merlin_testset.csv` and `.json`.
- `RADAR_infer_results_radar_plus_merlin_testset.csv` and `.json`.

Multi-GPU inference is supported by both entry points:

```bash
python RADAR_inference/inference_merlin_testset_epoch10.py --devices 0,1,2,3
python RADAR_inference/inference_merlin_testset_radar_plus.py --devices 0,1,2,3
```

Run these inside the container sequentially, with sufficient memory available on
each selected GPU. Indices refer to CUDA devices visible inside the container;
`CUDA_VISIBLE_DEVICES` can remap them. Use either `--device cuda:0` for a single
device or `--devices 0,1,2,3`, not both. Each GPU runs an independent process
with a complete copy of the model; sorted test cases are distributed round-robin.
`--limit` limits the total cases before distribution, not the cases per GPU.
CPU threads are per worker, so four workers with `--cpu-threads 4` can use 16 threads.

Each run creates a unique worker directory under the save directory containing
per-worker CSV, JSON and log files. After all workers succeed, the launcher
checks case coverage, order, duplicate filenames and matching settings, then
merges results into the normal output filenames in sorted test-case order.
Missing scores remain empty. A failed worker stops the other workers and leaves
the worker files for inspection; no merged output is published for that run.
Any merged output from an older run remains until a new run succeeds.

To check the multi-worker launch path on CPU without allocating GPU memory:

```bash
python RADAR_inference/inference_merlin_testset_epoch10.py --devices 0,1 --check-only --limit 2
```

Use `--limit 1 --save-tag trial_epoch10` (or `trial_radar_plus`) for a trial.
Paths can be overridden with `--checkpoint`, `--text-embeddings`, `--img-dir`,
`--report-json`, `--save-dir` and `--save-tag`. Changing checkpoints requires a
matching feature file with its metadata. The report JSON is used for split
selection, not for disease labels or accuracy/AUC computation. Missing scores
stay empty and reasons are recorded in JSON. These are prompt softmax scores,
not calibrated probabilities. Outputs are saved after each case; restarting
starts a new run rather than resuming existing rows.
