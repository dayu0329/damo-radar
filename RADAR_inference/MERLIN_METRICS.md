# Merlin 测试集扩展评估

`calc_metrics_merlin_testset_extended.py` 是独立新增脚本，原脚本保持不变。
仅依赖项目已有的 numpy、pandas、scikit-learn，不加载模型或运行推理。

## 输入与运行

在远程项目根目录 `/ssddata/CMMF/demo-radar` 运行：

```bash
# 远程 base 环境缺少 numpy；已验证 AnatoSeg_env 具有评估所需依赖。
conda activate AnatoSeg_env

# 标签文件由用户自行放到 ckpt/merlin_labels.json。
# 默认对齐原脚本的缺失预测填零策略，计算 2000 次患者级 bootstrap。
python RADAR_inference/calc_metrics_merlin_testset_extended.py

# 快速检查输入和点估计，暂不计算置信区间。
python RADAR_inference/calc_metrics_merlin_testset_extended.py --bootstrap 0

# 复制带中文说明的阈值样本，然后填写独立验证集上确定的阈值。
cp RADAR_inference/merlin_thresholds.example.json RADAR_inference/merlin_thresholds.validation.json

# 使用已填写的阈值；只有配置了数值阈值的疾病才计算固定阈值指标。
python RADAR_inference/calc_metrics_merlin_testset_extended.py \
  --predictions results/RADAR_infer_results_MerlinTestset.csv \
  --labels ckpt/merlin_labels.json \
  --thresholds RADAR_inference/merlin_thresholds.validation.json \
  --missing-policy zero \
  --bootstrap 2000 \
  --seed 42 \
  --target-specificity 0.95 \
  --target-sensitivity 0.90 \
  --output-dir results/merlin_metrics

# 可额外排除缺失预测做敏感性分析，输出到另一个目录以便比较。
python RADAR_inference/calc_metrics_merlin_testset_extended.py \
  --missing-policy exclude --output-dir results/merlin_metrics_exclude
```

默认文件路径以脚本所在项目为基准，因此可从任意目录启动。
显式传入的相对路径相对于启动时的当前目录。输出目录中同名结果文件会被覆盖。

预测 CSV 需要 `file_name` 列及疾病分数列，兼容原来的中文列名及英文括号后缀。
常规疾病分数要求在 `[0, 1]` 内；胆囊缺失列保存非负原始分割数值。
标签结构沿用原脚本：顶层为 Merlin 英文疾病名，每个疾病对应患者 ID 到 0/1/-1 的字典。
患者 ID 从文件名去除 `.nii.gz` 或 `.nii` 后得到；`-1`、无效或缺失标签均不参与评估。
本次不创建或修改标签文件。

阈值样本覆盖全部 21 项，全部初始化为 `null`，防止示例值被误当成正式阈值。
JSON 内的中文注释通过 `_comment` 等顶层字段表达，是合法 JSON，解析时会保存为元数据。
如果没有验证集阈值，可完全不传 `--thresholds`。

## 输出字段

### 阈值搜索比较表

增加 `--search-thresholds` 导出一份 `threshold_candidates.csv`，20 种疾病各 10 行，
排除胆囊缺失。疾病内顺序固定为：最大 Youden 指数；特异度至少 95%、85%、75%、65%、55%；
召回率至少 90%、80%、70%、60%。各疾病先后顺序沿用原映射。

默认 `--dataset-role test_exploratory`，明确标记为测试集探索性选阈值。
只有输入确实为独立验证集时才指定 `--dataset-role validation`。
所有阈值及同行指标都来自同一份选阈值数据，不代表独立测试集上的效果。
疾病数据、AUROC/AP 位于左侧及中部，方案、目标、阈值、指标和混淆矩阵位于右侧；全部双语表头。

特异度约束优先最大化召回率，再最大化特异度；召回率约束反之。
Youden 最大化召回率加特异度减一，平局依次优先召回率、特异度；最终平局均选更高阈值。
在完整经验 ROC 候选点上搜索，不插值、不拆分同分样本。不同方案可选到同一阈值。
全部判阴性/全部判阳性会明确标记；无穷阈值留空并以 `always_negative` 表示。
缺少预测或没有双类别标签时仍保留该疾病的 10 行，以状态和空指标解释。

```bash
# 示例：正式指标文件保留原样，可先输出到独立目录查看候选阈值。
python RADAR_inference/calc_metrics_merlin_testset_extended.py \
  --labels /data/dataset/json/merlin_labels.json \
  --search-thresholds --dataset-role test_exploratory \
  --bootstrap 0 --output-dir results_calc/merlin_threshold_search
```

增加 `--plot-roc` 可同时保存 `roc_curves_20_diseases.jpg`：4 行 5 列，排除胆囊缺失，
使用中文疾病标题，每个子图横纵坐标均为 0～1、间隔 0.1，标注 AUROC。
绘图直接复用指标计算时清洗后的样本，遵循相同的标签排除和预测缺失策略。
远程需安装 matplotlib 及中文字体；可用 `--roc-font /path/to/font.ttf` 指定字体。
函数 `plot_disease_roc_curves(roc_data, output_path, font_path=None)` 也可独立调用，
其中 `roc_data` 为 `{英文疾病名: (有效标签数组, 预测分数数组)}`。

CSV 使用单行中英文表头，例如 `sensitivity（敏感度即召回率TPR）`。
列从左到右依次为：中文/英文疾病名、状态、原始列名和样本统计；预测覆盖情况；
分数类型、AUROC、置信区间与 AP；ROC 约束工作点、固定阈值及其混淆矩阵和分类指标。
ROC 工作点表头会显示本次实际设置的敏感度/特异度目标。
读取 CSV 的程序需使用新的完整表头；汇总 JSON 继续使用原有英文键名。

- `metrics_per_disease.csv`：有效样本数、患者数、阳性/阴性数、阳性率、缺失预测数及覆盖率；AUROC、AP、AUROC 95% CI、有效 bootstrap 次数；两个 ROC 工作点；有阈值时的 TP、FP、TN、FN、敏感度、特异度、PPV、NPV、F1。
- `metrics_summary.json`：实际运行参数、阈值文件说明、疾病状态计数、各指标宏平均及各自纳入疾病数、评估解释。

`status` 含义：`ok` 为正常；`unknown_disease` 为未知疾病列；`missing_disease_labels` 为标签文件缺少该疾病；`no_valid_samples` 为无可评估样本；`single_class` 为有效标签只有一个类别。
CSV 空白表示未计算或无定义，JSON 中对应空值为 `null`。单类别仍可输出固定阈值下有定义的指标，但不输出 AUROC、AP 或 ROC 工作点。

## 解读时注意

- AP 由 `average_precision_score` 计算，不是梯形积分 PR-AUC；比较时同时看阳性率。
- 工作点分别表示经验 ROC 上“特异度至少达到目标时的最高敏感度”和“敏感度至少达到目标时的最高特异度”。不插值，不导出测试集阈值用于部署。
- 固定阈值必须事先在验证集上确定。概率分数大于等于阈值判为阳性。
- 缺失预测包括空值、非数值和无穷值。缺失率的分母是标签有效的样本数，覆盖率在填零或排除前统计。
- `zero` 与原脚本一致；胆囊缺失先填 0 再按 `< 1000` 二值化，缺失分割会被判为胆囊缺失。建议同时运行 `exclude` 比较影响。
- 胆囊缺失可通过 `--gallbladder-cutoff` 修改分割阈值。它的 AUROC 等于敏感度和特异度的平均值，AP/ROC 也基于二值结果解读。
- 指标点估计以扫描行为单位，置信区间按患者整组重采样，默认 2000 次、95% 百分位区间。只有一个类别的重采样不计入，样本很少时区间不稳定，应检查 `bootstrap_valid`。
- 宏平均按疾病等权，跳过无定义值，并报告实际纳入数；不计算宏平均置信区间，也不把逐病置信区间端点简单平均。
- 此版本不做概率校准评估或自动阈值搜索。
