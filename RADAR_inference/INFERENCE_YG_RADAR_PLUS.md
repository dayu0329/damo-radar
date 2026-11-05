# Official RADAR+ comparison on the six YG scans

`docs/TRAINING.md` identifies `ckpt/checkpoint_radar_plus.pth` as the released
RADAR+ checkpoint trained from scratch on Merlin-CT-Train. It is distinct from
the RAD-CT checkpoint and the version fine-tuned on MERLIN.

Run inside the existing training container:

```bash
docker exec -it -w /ssddata/CMMF/demo-radar radar-train-nvidia-pytorch-2610 \
  python RADAR_inference/inference_yg_images_radar_plus.py --check-only

docker exec -it -w /ssddata/CMMF/demo-radar radar-train-nvidia-pytorch-2610 \
  python RADAR_inference/inference_yg_images_radar_plus.py --device cuda:0
```

Select an available GPU. The new entry point reuses `inference_yg_images_epoch10.py` so
that both checkpoints use the same six inputs, 20 repository MERLIN prompt
groups, intensity range, resampling, sliding windows and organ scoring. Command
line options remain available. Your epoch-10 default entry point is unchanged.

The official checkpoint uses the English `bert-base-uncased` vocabulary, confirmed
by its 30522-row embedding table. Its text features are generated independently
from its own student text encoder and projection and saved to
`ckpt/infer_text_embedding_radar_plus_merlin_anatomy.pt` with provenance metadata.
Individual sentence features are normalized; negative/positive class means are
not renormalized, preserving the repository's MERLIN mean-logit ensemble.
The epoch-10 feature file is not reused for this checkpoint.

Outputs under `results/yg_images`:

- `RADAR_infer_results_radar_plus_merlin.csv`: official-model scores.
- `RADAR_infer_results_radar_plus_merlin.json`: checkpoint, protocol and unscored organs.
- `RADAR_infer_results_radar_plus_merlin_vs_epoch10.csv`: one row per scan and
  disease, containing both scores, epoch10 minus official score and scoring status.
- `RADAR_infer_results_radar_plus_merlin_vs_epoch10.json`: comparison provenance
  and counts of jointly scored or missing entries.

Before comparing, the script checks disease columns, inference settings and
prompt sentences. Empty scores remain empty rather than being replaced with zero.
Each checkpoint predicts its own organ masks, so scoring coverage can differ.
The difference column describes score changes, not accuracy improvements.
These six scans have no ground-truth reports or labels, so no accuracy or AUC
is computed and the comparison cannot establish which checkpoint is better.
