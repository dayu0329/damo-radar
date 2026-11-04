# RADAR contrastive gradient cache

Run from `RADAR_train` in `radar-train-nvidia-pytorch-2610`:

```bash
python -m pip install -r requirements-gradcache.txt
CUDA_VISIBLE_DEVICES=3 python validate_gradcache.py
CUDA_VISIBLE_DEVICES=3 python -u train_gradcache.py --cfg-path radar_gradcache.yaml --max-steps 2 --verify-replay
CUDA_VISIBLE_DEVICES=3 python -u train_gradcache.py --cfg-path radar_gradcache.yaml
```

Use a persistent terminal such as tmux for production runs. The trainer writes
`train.log`, `metrics.jsonl`, the resolved configuration and epoch checkpoints to
a unique directory under `run.output_dir`, including an `environment.json`
package-version snapshot. Timestamps use Asia/Shanghai. It disables core dumps and pinned
memory. No output capture with tee is required.

`batch_size_train: 2` is the GPU microbatch. `contrastive_batch_size: 48` means
24 microbatches contribute features to one contrastive objective and one
optimizer update. All 48 whole-image pairs share a 48-by-48 similarity matrix.
For each organ, only complete predicted organs are eligible; the entire pool
must contain an abnormal example and at least two eligible pairs. Organs whose
target matrix is uniform are skipped, matching the existing objective.
The effective number of organ pairs is therefore often below 48.

The implementation first computes features without retaining encoder graphs,
then computes the full contrastive loss and feature gradients, and finally
replays each microbatch with those gradients. Text branches are replayed
separately from the visual branch to limit activation memory. CPU/CUDA RNG
states are replayed so dropout is identical. BatchNorm buffers are restored
before replay and retained exactly once per microbatch. Augmented input tensors
are retained on CPU and reused. `--verify-replay` checks cached and replayed
features and is recommended for the smoke test.

Whole-image and organ objectives alternate per optimizer update, including
across epoch boundaries. The text momentum encoder updates once per organ
update; the original code updates it inside each organ loop. Segmentation loss
is averaged across the 48 cases. BatchNorm still uses microbatch statistics.
Consequently this implementation is not numerically identical to the original
24-GPU distributed run. It supports single-node torchrun with 1, 2, 3 or 4 GPUs.
Ordinary gradient accumulation must stay at 1.

## GPU count and configuration

Keep the same configuration for all four GPU counts:

```yaml
run:
  batch_size_train: 2
  contrastive_batch_size: 48
  accum_grad_iters: 1
  warmup_steps: 20
  init_lr: 5e-5
  precision: bf16
```

`contrastive_batch_size` is the global count, while `batch_size_train` is per
GPU. The global count must be divisible by GPU count times microbatch size.
With 1/2/3/4 GPUs, each rank handles 48/24/16/12 cases, requiring 24/12/8/6
microbatches. The loss uses pooled features across all ranks, and parameter
gradients are summed once after replay, including branches absent on a rank.
Segmentation gradients are weighted by the global case count. Temperature
gradients are counted once. Model initialization is broadcast from rank zero.
BatchNorm remains local; its buffers are not averaged across ranks.

```bash
# Select only GPUs available for your job.
CUDA_VISIBLE_DEVICES=3 python -u train_gradcache.py --cfg-path radar_gradcache.yaml
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train_gradcache.py --cfg-path radar_gradcache.yaml
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nproc_per_node=3 train_gradcache.py --cfg-path radar_gradcache.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_gradcache.py --cfg-path radar_gradcache.yaml
```

Only rank zero writes shared metrics, configuration and checkpoints. Other
ranks write `rank_N.log`. `num_workers` is per rank. Checkpoints store each
rank's random state; resuming with a different GPU count preserves optimizer
and schedule but reseeds random streams, so it is not an exact replay.
The saved BatchNorm buffers are from rank zero.

For a numerical distributed test, run `validate_distributed_gradcache.py` with
the same torchrun options. It checks the 48-case loss, gradients and optimizer
update against a direct reference, including an organ absent from rank zero.
For a real-data smoke test, add `--max-steps 2 --verify-replay` to the training
command. Check all rank logs if torchrun reports a worker failure.

For 15,309 cases, each epoch has 318 complete updates and drops the remaining
45 cases after shuffling (the omitted subset changes each epoch). Warmup is
measured in optimizer updates: 20 updates cover 960 cases, similar to 500 old
microbatch steps covering 1,000 cases. The learning-rate schedule starts cosine
decay by epoch after warmup. A value of 5e-5 is an experimental starting point,
not a claim of reproducing the author's hyperparameters or results.

`metrics.jsonl` records every update, its branch, patient IDs, valid organ pair
counts, exact loss, the last 100 losses for the same branch, diagonal/off-diagonal
similarities, bidirectional retrieval top-1, feature spread, temperature,
gradient norm, AMP scale/overflow skips, load/step times and current/peak memory.
Organ diagnostics use soft targets; retrieval top-1 is diagnostic, not a clinical
metric. Absent organs are omitted rather than displaying old values. Global
random-pair loss at 48 cases is approximately ln(48), not ln(2).

Epoch checkpoints contain student/teacher weights, optimizer/scaler and RNG
states. Set `resume_ckpt_path` to resume a completed gradient-cache epoch.
Set `init_checkpoint` to reuse older model weights with a new optimizer and
schedule; do not use the old optimizer's batch-2 state for gradient-cache resume.
The gradient-cache trainer enables strict dataset loading: a bad case raises
an error with its path instead of retrying indefinitely. Duplicate IDs in a
contrastive pool also cause an explicit failure. Image and label values are
checked on CPU before GPU computation. Validation-set evaluation
remains a separate step and is required to select the best checkpoint.
