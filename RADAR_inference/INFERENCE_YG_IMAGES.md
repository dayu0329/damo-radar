# YG images: MERLIN prompts with the epoch-10 checkpoint

Run in the existing training container, which contains the required Python dependencies:

```bash
docker exec -it -w /ssddata/CMMF/demo-radar radar-train-nvidia-pytorch-2610 \
  python RADAR_inference/inference_yg_images_epoch10.py --check-only
```

The defaults select `training_results/gradcache48/20261102_201102_310236/checkpoint_9.pth`
(zero-based epoch 9, training round 10), `ckpt/bert-base-uncased`, and the six
NIfTI images in `/data/dataset/yg_images`. The 20 prompt groups are taken from
`RADAR_train/infer_merlin_anatomy.py`; they are that script's selected evaluation
set, not every disease appearing in MERLIN training reports.

The script reuses `ckpt/infer_text_embedding_epoch10_merlin_anatomy.pt` and its
`.pt.json`. If absent, these are generated on CPU using the checkpoint's student
text encoder and projection. The earlier 146-item English templates are not used.
Source prompt sentences, spelling and wording are preserved, with backslashes
removed and `normal` added to the negative prompts as in the source script.
Encoding uses at most 100 tokens. Individual projected sentence features are
L2-normalized and averaged separately for negative/positive classes. The class
means are not renormalized: their dot products reproduce the source script's
mean-logit prompt ensemble before binary softmax.

`--check-only` strictly loads every inference weight, verifies feature provenance,
tokenizer vocabulary, prompt sentences, aggregation, disease keys and tensor
shapes, and reads all input headers. It does not allocate GPU memory or run CT
inference. Organ indices retain training order and map to English prompt regions
using the training model's `organ_dict`.

Once a GPU is available, run:

```bash
docker exec -it -w /ssddata/CMMF/demo-radar radar-train-nvidia-pytorch-2610 \
  python RADAR_inference/inference_yg_images_epoch10.py --device cuda:0
```

The GPU index is relative to the container's visible CUDA devices. Select a GPU
with sufficient free memory. `--limit 1` restricts a trial run to the first image.
Use `--device cpu` for slower CPU inference. The output is
`results/yg_images/RADAR_infer_results_epoch10_merlin.csv`, with a JSON sidecar
recording the checkpoint, protocol and reasons for unscored organs. Empty scores
are not negative predictions. Scores are binary prompt softmax scores, not
calibrated disease probabilities. There are no ground-truth reports or labels
for these six scans, so accuracy, sensitivity, specificity and AUC are not computed.

Use `--checkpoint`, `--text-embeddings`, `--bert-dir`, `--merlin-source`, `--img-dir`,
`--save-dir`, and `--save-tag` to override paths. Paths do not depend on the
current working directory. Changing checkpoints requires a matching text
embedding file and its exporter-generated `.pt.json` metadata; the script
rejects embeddings associated with another checkpoint or different prompt and
aggregation settings. The 146-item feature file is incompatible with this protocol.

CT preprocessing follows the MERLIN inference script: resize using its reference
spacing `(1, 1, 5)` and axis convention, transpose, clip to `[-1000, 1000]` HU,
min-max normalize, crop nonzero foreground and pad. This HU range also matches
the current training dataset. Invalid or constant images fail explicitly.
Sliding windows use `(96, 256, 384)` and overlap 0.25. Organ scoring follows the
demo; segmentation stitching runs on CPU to reduce GPU memory requirements.
The model loads only the trained visual encoder, organ attention,
query tokens, organ projections and temperature. Student text weights are used
by the text exporter. The teacher, optimizer and whole-image training head are
not needed for this organ-level prediction task. Every required inference key
must be present with the correct shape; incompatible checkpoints fail loading.

The source script's separate surgically-absent-gallbladder output is excluded:
it records segmented gallbladder voxel counts rather than a validated disease
probability. The 20 text-scored groups remain unchanged. Matching this prompt
protocol does not establish equivalence of complete pipelines or clinical quality.
