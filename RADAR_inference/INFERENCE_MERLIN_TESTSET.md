# MERLIN test split inference

All entry points select only `split=test` IDs from
`/data/dataset/json/merlin_report.json`, with images under
`/data/dataset/other_images`. The directory also contains validation scans;
these are excluded. Missing test images cause an explicit error.

All shared inference entry points export `file_name` followed by the first 20
official MERLIN columns, with exactly the Chinese names, English suffixes and
column order used by `inference_merlin_testset.py`. Scores are reordered only
when writing CSV; prompt order, model computation, score values and missing
values are preserved. The separate segmentation-based surgically absent
gallbladder item is excluded. Each output JSON records `csv_columns`.
The output schema is checked before model loading, and unexpected or duplicate
prompt items cause an error. Match cases by `file_name`, since row ordering can
differ from the original official output. Compare averages over the same 20
diseases and use the same missing-score policy for both models.

The epoch-10 entry point uses zero-based `checkpoint_9.pth`; the official entry
point uses `ckpt/checkpoint_radar_plus.pth`. Each uses its own matched text
features with the same 20 MERLIN prompt groups, English tokenizer, HU clipping,
resampling and scoring from `inference_yg_images_epoch10.py`. Keep that shared module,
the exporter, original demo and network dependencies in place.

## Epoch 29 (training round 30)

`inference_merlin_testset_epoch29.py` defaults to
`training_results/gradcache48/20261102_201102_310236/checkpoint_29.pth` and
requires checkpoint metadata `epoch=29` (zero-based, training round 30).
Once training has finished writing the checkpoint, the first run automatically
exports matching text features on CPU to
`ckpt/infer_text_embedding_epoch29_merlin_anatomy.pt` with a `.pt.json` sidecar.
Subsequent runs validate and reuse these files. Epoch-10 features are not reused.
If the checkpoint is not present yet, inference fails without waiting for it.

Run inside the training container, from `/ssddata/CMMF/demo-radar`:

```bash
python RADAR_inference/inference_merlin_testset_epoch29.py --check-only --limit 1
python RADAR_inference/inference_merlin_testset_epoch29.py --devices 0 --workers-per-gpu 2 --cpu-threads 2
```

For multiple GPUs, use `--devices 0,1,2,3`. The shared GPU accumulation and memory
fallback also apply to this entry point. Results are written to
`results/merlin_testset/RADAR_infer_results_epoch29_merlin_testset.csv` and `.json`.
Use `--checkpoint PATH` if the training directory changes; the expected epoch
remains 29. To export text features separately before inference:

```bash
python RADAR_inference/generate_text_embeddings.py \
  --prompt-source merlin --aggregation mean --max-length 100 \
  --checkpoint training_results/gradcache48/20261102_201102_310236/checkpoint_29.pth \
  --output ckpt/infer_text_embedding_epoch29_merlin_anatomy.pt --device cpu
```

## Other checkpoints

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
unchanged. Background loading is conservatively disabled with window batches above
one: the installed runtime produced invalid coverage with threaded batched loading.
Synchronous batched loading was checked against batch=1 on real test cases.

Independent case processes can also share one GPU:

```bash
python RADAR_inference/inference_merlin_testset_epoch10.py --devices 0 --workers-per-gpu 2 --sw-batch-size 1
```

The same option works with multiple devices and the official RADAR+ entry point.
Each process loads a separate model and maintains its own preprocessing workers and
output shard. This requires substantially more GPU and CPU memory. It does not
guarantee speedup when a single process already saturates the GPU; benchmark
representative cases before a full run. Start with two processes and keep window
batch size at one rather than multiplying both kinds of concurrency at once.

By default, `--loader-workers 2 --prefetch-factor 2` uses two independent CPU
preprocessing processes per inference process, queuing up to four cases. Workers
use the spawn start method and one PyTorch CPU thread each; they do not load the
inference model or initialize CUDA. Four inference processes therefore create
eight preprocessing processes. This uses additional host RAM and `/dev/shm`;
reduce `--prefetch-factor` or `--loader-workers` if these resources are limited.
Cases are consumed in original order. `--loader-workers 0` restores the original
single-thread prefetch. `--no-prefetch` disables background loading entirely.
Per-case logs separate data wait from inference, with totals printed at the end.
The first wait includes preprocessing worker startup. CSV rows and a
`.cases.jsonl` journal are appended and flushed per case; the summary `.json`
is written after successful completion. Logs report case time, estimated time
remaining, total processing time and peak allocated GPU memory. Existing
completed rows are not automatically resumed after interruption.

Example for two GPUs with two inference processes per GPU:

```bash
python -u RADAR_inference/inference_merlin_testset_epoch29.py \
  --devices 2,3 --workers-per-gpu 2 --cpu-threads 4 --sw-batch-size 1 \
  --loader-workers 2 --prefetch-factor 2
```

A sequential comparison on the same first 48 epoch-29 test cases with this
two-GPU configuration measured 167 seconds with threaded loading versus 152
seconds with process loading, including startup. The longest worker's processing
stage fell from 112.56 to 96.49 seconds. Across all four workers, data waiting
after each worker's first case fell from 251.21 to 33.10 seconds (summed process
time, not wall time). All 952 scores and eight missing entries matched exactly,
including comparison with the saved full-test results. This is a small-sample
measurement; throughput for the full dataset and different system loads can vary.
Benchmark files are under `results/performance_tests`, separate from final runs.

The shared inference engine also keeps organ features and disease scores on the
GPU until the final score transfer, and transfers organ boundary eligibility once
per window instead of repeatedly reading individual GPU scalars. Fallback organ
crops use axis projections instead of allocating full-volume coordinate grids.
Crop boundaries, preprocessing, organ selection and scoring remain the same.
Projection crops matched the original implementation on 150 synthetic masks;
empty masks are skipped.

A subsequent comparison on the same 48 epoch-29 cases, using GPUs 2 and 3 with
two inference processes per GPU, measured summed inference time of 158.89 seconds
before these changes and 112.35 seconds afterwards (29.3% lower). The longest
worker's processing stage decreased from 134.93 to 78.99 seconds. The latter
includes data waiting and can benefit from filesystem cache warming, so it should
not be interpreted as an isolated code speedup or a full-dataset estimate.
All 952 scores and eight missing entries matched both the baseline and the saved
full-test results exactly. Benchmark tags are `sync_baseline48` and
`sync_optimized48`. No preprocessed image cache is created. Remaining loading,
resampling, GPU memory transfers and internal model synchronization can still
leave gaps in GPU utilization.

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
