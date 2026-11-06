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

Segmentation probabilities are accumulated in FP32 on the inference GPU by
default, followed by GPU argmax. Only the final single-channel uint8 labels are
copied to the CPU for organ cropping. Window order and scoring are unchanged.
`--segmentation-device auto` uses at most a quarter of available GPU memory
(driver-free memory plus this worker's reusable PyTorch cache) for the full-volume
buffer, while reserving at least 8 GiB for inference; larger
scans fall back to CPU accumulation. GPU allocation failure also falls back to
CPU. Logs identify the actual accumulation device and buffer size per case.
Use `--segmentation-device cpu` to restore CPU accumulation or `cuda` to require
GPU accumulation. Forced CUDA mode can run out of memory. This option is passed
to every worker and also applies to the YG and official RADAR+ entry points.

GPU sliding-window batching is available through `--sw-batch-size` (default 1):

```bash
python RADAR_inference/inference_merlin_testset_epoch10.py --device cuda:0 --sw-batch-size 2
python RADAR_inference/inference_merlin_testset_epoch10.py --devices 0,1,2,3 --sw-batch-size 2
```

This batches consecutive windows from one scan in one GPU forward pass, then
scores them and stitches segmentation in original window order. Every GPU worker
uses the requested batch size. It trades additional GPU memory for throughput;
test a small number of cases first and lower it if memory is insufficient.
GPU batching may cause floating-point differences, so it does not promise
bitwise identical results. FP32, HU range, window dimensions and overlap are
unchanged. Threaded prefetch is automatically disabled with window batches above
one: the installed runtime produced invalid coverage when combining these paths.
Synchronous batched loading was checked against batch=1 on real test cases.

Independent case processes can also share one GPU:

```bash
python RADAR_inference/inference_merlin_testset_epoch10.py --devices 0 --workers-per-gpu 2 --sw-batch-size 1
```

The same option works with multiple devices and the official RADAR+ entry point.
Each process loads a separate model and maintains its own prefetch thread and
output shard. This requires substantially more GPU and CPU memory. It does not
guarantee speedup when a single process already saturates the GPU; benchmark
representative cases before a full run. Start with two processes and keep window
batch size at one rather than multiplying both kinds of concurrency at once.

One background thread preloads the next scan while the GPU computes the current
scan. Use `--no-prefetch` if host memory is constrained. CSV rows and a
`.cases.jsonl` journal are appended and flushed per case; the summary `.json`
is written after successful completion. Logs report case time, estimated time
remaining, total processing time and peak allocated GPU memory. Existing
completed rows are not automatically resumed after interruption.

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
