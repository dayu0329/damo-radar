# Checkpoint-matched inference text features

## MERLIN evaluation (recommended for the current training)

```bash
cd /ssddata/CMMF/demo-radar/RADAR_inference
python generate_text_embeddings.py \
  --prompt-source merlin \
  --checkpoint ../training_results/gradcache48/20261102_201102_310236/checkpoint_9.pth \
  --output ../ckpt/infer_text_embedding_epoch10_merlin_anatomy.pt
```

This mode extracts the actual `DiseasePrompts` definitions from
`RADAR_train/infer_merlin_anatomy.py` without importing or executing its inference
job. No handwritten JSON is necessary. It applies the same backslash removal,
adds `normal` to the negative prompts and defaults to length 100. Sentence text,
duplicates and ordering are retained, including any imperfections in the source.
Optional `--create-prompts PATH --prompt-source merlin` exports these sentences
for inspection. Use `--prompts PATH --prompt-source merlin` for an edited copy.

The output uses tuple keys `(region, disease-with-hyphens)` matching the training
MERLIN inference script. Each tensor is `[negative_mean, positive_mean]`. The
individual sentence embeddings are normalized, but their class means are **not
renormalized**. Dot products with these means exactly reproduce the original
script's mean-logit ensemble (up to floating-point rounding), followed by binary
softmax. Renormalizing would change scores. Do not normalize the exported tensors
again in the consumer. This is algebraic equivalence of the text scoring step,
not a claim of equality of complete inference pipelines.

This output is not a drop-in replacement for the Chinese-keyed 146-item demo
file or the Chinese-keyed `infer_text_embedding_merlin.pt`. They use different
item sets/key conventions. Use a consumer configured for these MERLIN tuple keys
and negative/positive rows. In `infer_merlin_anatomy.py`, you may replace its
entire text-feature construction section with:

```python
text_feat_dict = torch.load(PATH, map_location="cpu", weights_only=True)
text_feat_dict = {key: value.to(model.device) for key, value in text_feat_dict.items()}
```

That model's two-row scoring path already accepts these class means. Keep the
rest of its preprocessing and image checkpoint unchanged. The unmodified script
also constructs checkpoint-matched text features itself, so running it directly
does not require this export utility. Surgically absent gallbladder is separately
computed from segmentation by that script, not a text-prompt item.

## 146-item demo templates

Run inside the training container. The exporter reads only the student text
encoder and projection, not the visual encoder or momentum teacher. It does not
start training. CPU is the default so an existing GPU training job can continue.

```bash
cd /ssddata/CMMF/demo-radar/RADAR_inference
python generate_text_embeddings.py --create-prompts ../ckpt/radar_prompts_en.json
python generate_text_embeddings.py \
  --checkpoint ../training_results/gradcache48/20261102_201102_310236/checkpoint_9.pth \
  --prompts ../ckpt/radar_prompts_en.json \
  --reference ../ckpt/infer_text_embedding_radar.pt \
  --output ../ckpt/infer_text_embedding_epoch10.pt
```

`checkpoint_9.pth` is the end of training round 10 (epochs are zero-based).
Use `--device cuda:0` only with a GPU available for this job, or leave the CPU
default. `--bert-dir` defaults to the project's `ckpt/bert-base-uncased`, matching
the gradient-cache training code. Always provide the actual training tokenizer
and config if training used something different. Vocabulary size is checked,
but equal-sized vocabularies can still have different token ordering.

The first command creates editable English templates for the 146 disease keys
from `inference_demo.py`. These are **new templates, not the official original
prompts**. Original sentences cannot be recovered from a precomputed tensor.
If you have the original prompt sentences, put them in this JSON instead.
The prompt format is:

```json
{
  "肾_囊肿": {
    "negative_prompts": ["kidney: no cyst.", "normal."],
    "positive_prompts": ["kidney: cyst."]
  }
}
```

Each prompt is encoded with the checkpoint's student text encoder and text
projection in eval mode. Projected CLS features are normalized, averaged within
each class, then normalized again. Each output item is a CPU float32 `[2, 256]`
tensor: row 0 negative, row 1 positive. This matches the inference code's use of
softmax class index 1. The output is a plain dictionary, with metadata and all
prompt sentences saved separately to `OUTPUT.pt.json`. Existing output files
are not overwritten. `--reference` checks keys and shapes, not prompt semantics.
The generated features are not expected to equal the official ensemble when
different prompt sentences or averaging rules are used.

## Using the output

In the actual inference script, replace the embedding load path and explicitly
move features to the image feature device before computing similarity:

```python
text_feat_dict = torch.load("../ckpt/infer_text_embedding_epoch10.pt", map_location="cpu", weights_only=True)
```

At the point where `image_feat` is available:

```python
text_feat = text_feat_dict[item].to(device=image_feat.device, dtype=image_feat.dtype)
```

Keep the corresponding epoch-10 image weights and temperature. An older official
inference script that constructs a Chinese text encoder may reject the English
checkpoint's vocabulary dimensions even though inference uses cached features.
Fix that loading configuration or use an inference model that only loads visual
weights and temperature; silently dropping shape mismatches is not a solution.

Use identical prompts, preprocessing, disease order and aggregation when comparing
checkpoints. A new English-template result versus the official cached prompt
ensemble is not a controlled comparison. Calibrate thresholds on validation data;
softmax scores are not calibrated clinical probabilities.
