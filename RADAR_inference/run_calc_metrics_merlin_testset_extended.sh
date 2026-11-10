#!/usr/bin/env bash
set -e
cd /ssddata/CMMF/demo-radar

python -u RADAR_inference/calc_metrics_merlin_testset_extended.py \
  --predictions results/merlin_testset_epoch29//RADAR_infer_results_epoch29_merlin_testset.csv \
  --labels /data/dataset/json/merlin_labels.json \
  --missing-policy zero \
  --bootstrap 2000 \
  --seed 42 \
  --target-specificity 0.95 \
  --target-sensitivity 0.90 \
  --gallbladder-cutoff 1000 \
  --output-dir results/merlin_metrics_epoch29 \
  --plot-roc \
  --search-thresholds \
  --thresholds RADAR_inference/merlin_thresholds.validation.json \
  --dataset-role test_exploratory

# --predictions：模型预测 CSV，必须包含 file_name 列。
# --labels：Merlin 标签 JSON。
# --thresholds：验证集确定的疾病阈值 JSON。
# --missing-policy：缺失预测处理方式，可选 zero 或 exclude。
# --bootstrap：患者级 bootstrap 次数；快速检查可改为 0。
# --seed：bootstrap 随机种子。
# --target-specificity：ROC 工作点目标特异度。
# --target-sensitivity：ROC 工作点目标敏感度。
# --gallbladder-cutoff：胆囊分割值小于该值时判为缺失。
# --output-dir：指标结果输出目录。
# --plot-roc：生成 ROC 图；不需要时删除该行。
# --search-thresholds：生成阈值比较表；不需要时删除该行。
# --dataset-role：数据用途，可选 test_exploratory 或 validation。
# --roc-font：如需指定字体，必须填写 .ttf/.otf 文件路径。
