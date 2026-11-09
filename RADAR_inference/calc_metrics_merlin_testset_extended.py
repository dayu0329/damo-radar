"""Merlin 测试集扩展评估：逐疾病 AUROC、AP、工作点和固定阈值指标。

本文件独立运行，不导入或修改原评估脚本。运行示例见 MERLIN_METRICS.md。
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABSENT_GALLBLADDER = "surgically_absent_gallbladder"
DISEASE_MAPPING = {
    "主动脉_主动脉瘤": "abdominal_aortic_aneurysm",
    "主动脉_粥样硬化": "atherosclerosis",
    "大肠_粘膜下水肿": "submucosal_edema",
    "大肠_阑尾炎": "appendicitis",
    "小肠_梗阻": "bowel_obstruction",
    "心脏_主动脉瓣钙化": "aortic_valve_calcification",
    "心脏_心影（脏）增大": "cardiomegaly",
    "肝_肝内胆管扩张": "biliary_ductal_dilation",
    "肝_肝大": "hepatomegaly",
    "肝_脂肪肝": "hepatic_steatosis",
    "肺_胸腔积液": "pleural_effusion",
    "肺_膨胀不全": "atelectasis",
    "肾_低密度影": "renal_hypodensities",
    "肾_囊肿": "renal_cyst",
    "肾_肾积水": "hydronephrosis",
    "胆囊_结石": "gallstones",
    "胰腺_萎缩": "pancreatic_atrophy",
    "脾_脾大": "splenomegaly",
    "腰椎_骨折": "fracture",
    "食管_裂孔疝": "hiatal_hernia",
    "胆囊_术后胆囊缺失": ABSENT_GALLBLADDER,
}
MACRO_METRICS = [
    "roc_auc", "average_precision", "sensitivity_at_specificity",
    "specificity_at_sensitivity", "sensitivity", "specificity", "ppv", "npv", "f1",
]


def format_metrics_for_csv(metrics, target_specificity, target_sensitivity):
    """只调整导出展示；内部计算及汇总 JSON 继续使用英文键名。"""
    metrics = metrics.copy()
    metrics["disease_cn"] = metrics["column"].map(
        lambda column: "_".join(column.split("_")[:2])
    )
    # 字典顺序就是导出顺序：数据与覆盖情况、整体排序指标、阈值及工作点指标。
    descriptions = {
        "disease_cn": "中文疾病名称",
        "disease": "英文疾病名称",
        "status": "评估状态",
        "column": "原始预测列名",
        "n_input": "输入扫描数",
        "n_unknown_label": "标签缺失或无效扫描数",
        "n_ignored_label": "标签为负一的排除扫描数",
        "n_label_valid": "标签有效扫描数",
        "n_evaluated": "实际评估扫描数",
        "n_patients": "实际评估患者数",
        "n_positive": "实际评估阳性扫描数",
        "n_negative": "实际评估阴性扫描数",
        "positive_rate": "实际评估阳性率",
        "n_prediction_missing": "标签有效但预测缺失扫描数",
        "prediction_missing_rate": "标签有效样本中的预测缺失率",
        "prediction_coverage": "标签有效样本中的预测覆盖率",
        "score_type": "模型分数类型",
        "roc_auc": "ROC曲线下面积",
        "roc_auc_ci95_lower": "AUROC的95%置信区间下限",
        "roc_auc_ci95_upper": "AUROC的95%置信区间上限",
        "bootstrap_valid": "有效患者级重采样次数",
        "average_precision": "平均精确率AP",
        "sensitivity_at_specificity": f"特异度至少{target_specificity:.1%}时的最高召回率",
        "specificity_at_sensitivity": f"召回率至少{target_sensitivity:.1%}时的最高特异度",
        "threshold": "固定判阳阈值",
        "threshold_rule": "固定阈值判阳规则",
        "tp": "真阳性扫描数",
        "fp": "假阳性扫描数",
        "tn": "真阴性扫描数",
        "fn": "假阴性扫描数",
        "sensitivity": "敏感度即召回率TPR",
        "specificity": "特异度TNR",
        "ppv": "阳性预测值即精确率",
        "npv": "阴性预测值",
        "f1": "精确率与召回率的调和平均",
    }
    # 缺列时补空值；显式列出所有字段，避免依赖行字典的初始化顺序。
    return metrics.reindex(columns=list(descriptions)).rename(
        columns={name: f"{name}（{description}）" for name, description in descriptions.items()}
    )


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def patient_id_from_filename(filename):
    # 按实际扩展名去除后缀；.nii.gz 与原脚本去掉末尾 7 个字符的规则一致。
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def safe_ratio(numerator, denominator):
    # 分母为零时指标没有定义，保留 NaN，不能用 0 冒充有效结果。
    return float(numerator / denominator) if denominator else float("nan")


def binary_metrics(labels, predictions):
    positive = labels == 1
    predicted_positive = predictions == 1
    true_positive = int(np.sum(positive & predicted_positive))
    false_positive = int(np.sum(~positive & predicted_positive))
    true_negative = int(np.sum(~positive & ~predicted_positive))
    false_negative = int(np.sum(positive & ~predicted_positive))
    return {
        "tp": true_positive, "fp": false_positive,
        "tn": true_negative, "fn": false_negative,
        "sensitivity": safe_ratio(true_positive, true_positive + false_negative),
        "specificity": safe_ratio(true_negative, true_negative + false_positive),
        "ppv": safe_ratio(true_positive, true_positive + false_positive),
        "npv": safe_ratio(true_negative, true_negative + false_negative),
        "f1": safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
    }


def bootstrap_auc(labels, scores, patient_ids, repetitions, rng):
    # 按患者整组重采样：同一患者如有多个扫描，不能当成互相独立的患者抽样。
    # AUC 点估计仍按扫描行计算；置信区间考虑患者内相关性。
    unique_patients, group_index = np.unique(patient_ids, return_inverse=True)
    groups = [np.flatnonzero(group_index == index) for index in range(len(unique_patients))]
    bootstrap_values = []
    for _ in range(repetitions):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        sampled_rows = np.concatenate([groups[index] for index in sampled_groups])
        sampled_labels = labels[sampled_rows]
        # 重采样后只剩一个类别时无法计算 AUC，记录有效重采样次数供检查。
        if np.unique(sampled_labels).size == 2:
            bootstrap_values.append(roc_auc_score(sampled_labels, scores[sampled_rows]))
    if len(bootstrap_values) < 2:
        return float("nan"), float("nan"), len(bootstrap_values)
    lower, upper = np.percentile(bootstrap_values, [2.5, 97.5])
    return float(lower), float(upper), len(bootstrap_values)


def load_thresholds(path):
    if path is None:
        return {}, {}
    config = read_json(path)
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("阈值文件必须包含 thresholds 对象，格式见阈值样本。")
    for disease, threshold in thresholds.items():
        if disease not in DISEASE_MAPPING.values():
            raise ValueError(f"未知阈值疾病名：{disease}")
        if disease == ABSENT_GALLBLADDER and threshold is not None:
            raise ValueError("胆囊缺失使用分割阈值，请通过 --gallbladder-cutoff 设置。")
        # null 表示尚未确定阈值，不能替换成默认 0.5。
        if threshold is not None and (
            isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold) or not 0 <= threshold <= 1
        ):
            raise ValueError(f"{disease} 的阈值必须是 [0, 1] 数值或 null。")
    metadata = {key: value for key, value in config.items() if key != "thresholds"}
    return thresholds, metadata


def evaluate_column(frame, column, all_labels, thresholds, args, rng, roc_data=None):
    chinese_name = "_".join(column.split("_")[:2])
    disease = DISEASE_MAPPING.get(chinese_name)
    # 即使所有疾病都无法计算某项指标，CSV 仍保留固定表头，以空值表示未定义。
    row = {metric: float("nan") for metric in MACRO_METRICS}
    row.update({
        "roc_auc_ci95_lower": float("nan"), "roc_auc_ci95_upper": float("nan"),
        "bootstrap_valid": 0, "threshold": float("nan"), "threshold_rule": "",
        "tp": float("nan"), "fp": float("nan"),
        "tn": float("nan"), "fn": float("nan"),
        "score_type": "", "n_patients": 0, "n_positive": 0, "n_negative": 0,
        "positive_rate": float("nan"), "prediction_missing_rate": float("nan"),
        "prediction_coverage": float("nan"),
    })
    row.update({
        "column": column, "disease": disease or "", "status": "ok",
        "n_input": len(frame), "n_unknown_label": 0, "n_ignored_label": 0,
        "n_label_valid": 0, "n_prediction_missing": 0, "n_evaluated": 0,
    })
    if disease is None:
        row["status"] = "unknown_disease"
        return row
    disease_labels = all_labels.get(disease)
    if not isinstance(disease_labels, dict):
        row["status"] = "missing_disease_labels"
        return row

    labels, scores, patient_ids = [], [], []
    for filename, prediction in zip(frame["file_name"], frame[column]):
        patient_id = patient_id_from_filename(filename)
        try:
            label = float(disease_labels[patient_id])
        except (KeyError, TypeError, ValueError, OverflowError):
            row["n_unknown_label"] += 1
            continue
        if label == -1:
            row["n_ignored_label"] += 1
            continue
        if label not in (0, 1):
            row["n_unknown_label"] += 1
            continue
        row["n_label_valid"] += 1
        try:
            score = float(prediction)
        except (TypeError, ValueError, OverflowError):
            score = float("nan")
        # 缺失率在标签有效的样本中统计，且在填零或排除之前计算。
        if not np.isfinite(score):
            row["n_prediction_missing"] += 1
            if args.missing_policy == "exclude":
                continue
            score = 0.0
        if disease != ABSENT_GALLBLADDER and not 0 <= score <= 1:
            raise ValueError(f"列 {column}、患者 {patient_id} 的分数超出 [0, 1]：{score}")
        if disease == ABSENT_GALLBLADDER and score < 0:
            raise ValueError(f"患者 {patient_id} 的胆囊分割数值不能为负。")
        labels.append(int(label))
        scores.append(score)
        patient_ids.append(patient_id)

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    row.update({
        "n_evaluated": len(labels), "n_patients": len(set(patient_ids)),
        "n_positive": int(np.sum(labels == 1)), "n_negative": int(np.sum(labels == 0)),
        "positive_rate": safe_ratio(np.sum(labels == 1), len(labels)),
        "prediction_missing_rate": safe_ratio(row["n_prediction_missing"], row["n_label_valid"]),
        "prediction_coverage": safe_ratio(
            row["n_label_valid"] - row["n_prediction_missing"], row["n_label_valid"]
        ),
    })
    if not len(labels):
        row["status"] = "no_valid_samples"
        return row

    # 直接复用本次评估清洗后的标签和分数，确保绘图与指标使用完全相同的样本。
    if roc_data is not None and disease != ABSENT_GALLBLADDER:
        roc_data[disease] = (labels.copy(), scores.copy())

    if disease == ABSENT_GALLBLADDER:
        # 延续原脚本：分割值 < 1000 判为胆囊缺失。这里不是概率阈值。
        # zero 策略下，缺失值填 0 会被判成“缺失”，因此必须同时查看缺失率。
        scores = (scores < args.gallbladder_cutoff).astype(float)
        row.update({"score_type": "binary_segmentation", "threshold": args.gallbladder_cutoff,
                    "threshold_rule": "raw_segmentation_value < threshold"})
        row.update(binary_metrics(labels, scores.astype(int)))
    else:
        row["score_type"] = "continuous_score"
        threshold = thresholds.get(disease)
        if threshold is not None:
            row.update({"threshold": threshold, "threshold_rule": "score >= threshold"})
            row.update(binary_metrics(labels, (scores >= threshold).astype(int)))

    if np.unique(labels).size < 2:
        # 单类别时不计算 AUROC/AP/ROC 工作点，但已配置阈值的计数仍可输出。
        row["status"] = "single_class"
        return row
    row["roc_auc"] = float(roc_auc_score(labels, scores))
    row["average_precision"] = float(average_precision_score(labels, scores))
    lower, upper, valid_repetitions = bootstrap_auc(
        labels, scores, np.asarray(patient_ids), args.bootstrap, rng
    )
    row.update({"roc_auc_ci95_lower": lower, "roc_auc_ci95_upper": upper,
                "bootstrap_valid": valid_repetitions})
    if args.bootstrap and valid_repetitions < 0.8 * args.bootstrap:
        warnings.warn(f"{disease} 有效 bootstrap 次数不足 80%，请谨慎解读置信区间。")

    # 工作点只取经验 ROC 上满足约束的点，不插值，不把测试集阈值用于部署。
    false_positive_rates, true_positive_rates, _ = roc_curve(labels, scores, drop_intermediate=False)
    specificities = 1 - false_positive_rates
    row["sensitivity_at_specificity"] = float(np.max(
        true_positive_rates[specificities >= args.target_specificity]
    ))
    row["specificity_at_sensitivity"] = float(np.max(
        specificities[true_positive_rates >= args.target_sensitivity]
    ))
    return row


def json_safe(value):
    # 标准 JSON 不接受 NaN/Infinity，统一转成 null；CSV 则留空。
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def plot_disease_roc_curves(roc_data, output_path, font_path=None):
    """绘制除胆囊缺失以外的 20 种疾病 ROC，保存为 4 行 5 列 JPG。"""
    import matplotlib

    # 使用无界面后端，远程服务器不需要桌面环境。
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

    # 检查字体是否包含所有标题汉字，避免生成中文乱码的图片。
    titles = [name for name, disease in DISEASE_MAPPING.items() if disease != ABSENT_GALLBLADDER]
    required_characters = {ord(character) for character in "".join(titles) if ord(character) > 127}
    candidates = [str(font_path)] if font_path else sorted(font_manager.findSystemFonts())
    selected_font = None
    for candidate in candidates:
        try:
            if required_characters.issubset(FT2Font(candidate).get_charmap()):
                selected_font = font_manager.FontProperties(fname=candidate, size=12)
                break
        except (OSError, RuntimeError):
            continue
    if selected_font is None:
        raise ValueError("未找到能显示全部中文疾病名的字体，请用 --roc-font 指定中文字体文件。")

    figure, axes = plt.subplots(4, 5, figsize=(25, 20), constrained_layout=True)
    ticks = np.linspace(0, 1, 11)
    for axis, chinese_name in zip(axes.flat, titles):
        disease = DISEASE_MAPPING[chinese_name]
        data = roc_data.get(disease)
        if data is not None and np.unique(data[0]).size == 2:
            labels, scores = data
            false_positive_rates, true_positive_rates, _ = roc_curve(labels, scores, drop_intermediate=False)
            auc = roc_auc_score(labels, scores)
            axis.plot(false_positive_rates, true_positive_rates, color="#1565C0", linewidth=2,
                      label=f"AUROC = {auc:.3f}")
            axis.legend(loc="lower right", fontsize=10)
        else:
            # 无有效双类别标签时明确留字说明，不能画一条虚构的 ROC 曲线。
            axis.text(.5, .5, "ROC unavailable", ha="center", va="center")
        axis.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
        axis.set_title(chinese_name, fontproperties=selected_font, pad=10)
        axis.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
        axis.set_ylabel("True Positive Rate (Recall)", fontsize=10)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.grid(True, color="#dddddd", linewidth=.6, alpha=.8)
        axis.set_aspect("equal", adjustable="box")
        axis.tick_params(labelsize=9)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, format="jpg", dpi=200, facecolor="white")
    finally:
        plt.close(figure)
    print(f"ROC 曲线图已保存至：{output_path.resolve()}")
    return output_path


def search_disease_thresholds(roc_data, metrics, output_path, dataset_role="test_exploratory"):
    """逐疾病比较十套阈值；只在输入数据上搜索，不把结果冒充独立测试表现。"""
    schemes = [("youden", "最大Youden指数", None)]
    schemes += [("specificity", f"特异度≥{target:.0%}", target)
                for target in (.95, .85, .75, .65, .55)]
    schemes += [("sensitivity", f"召回率≥{target:.0%}", target)
                for target in (.90, .80, .70, .60)]
    descriptions = {
        "disease_cn": "中文疾病名称", "disease": "英文疾病名称", "status": "评估状态",
        "dataset_role": "选阈值数据用途", "n_input": "输入扫描数",
        "n_evaluated": "有效评估扫描数", "n_patients": "有效评估患者数",
        "n_positive": "阳性扫描数", "n_negative": "阴性扫描数",
        "positive_rate": "阳性率", "prediction_coverage": "预测覆盖率",
        "roc_auc": "ROC曲线下面积", "average_precision": "平均精确率AP",
        "scheme_order": "疾病内方案顺序", "scheme": "阈值选择方案",
        "target_specificity": "目标特异度", "target_sensitivity": "目标召回率",
        "target_met": "约束目标是否达到", "threshold": "选定阈值",
        "threshold_rule": "判阳规则", "degenerate_rule": "是否全部判为同一类别",
        "youden": "Youden指数", "sensitivity": "敏感度即召回率TPR",
        "specificity": "特异度TNR", "false_positive_rate": "假阳性率FPR",
        "ppv": "阳性预测值即精确率", "npv": "阴性预测值",
        "f1": "精确率与召回率的调和平均", "tp": "真阳性扫描数",
        "fp": "假阳性扫描数", "tn": "真阴性扫描数", "fn": "假阴性扫描数",
    }
    base_rows = {row["disease"]: row for row in metrics.to_dict("records")}
    output_rows = []
    for chinese_name, disease in DISEASE_MAPPING.items():
        if disease == ABSENT_GALLBLADDER:
            continue
        base = base_rows.get(disease, {})
        data = roc_data.get(disease)
        usable = data is not None and np.unique(data[0]).size == 2
        candidates = []
        if usable:
            labels, scores = data
            # 同分数样本必须一起判阳，不能为凑目标拆开并列分数；保留所有经验 ROC 点。
            false_rates, recalls, cutoffs = roc_curve(labels, scores, drop_intermediate=False)
            positives = int(np.sum(labels == 1))
            negatives = int(np.sum(labels == 0))
            for false_rate, recall, cutoff in zip(false_rates, recalls, cutoffs):
                specificity = 1 - false_rate
                true_positive = int(round(recall * positives))
                false_positive = int(round(false_rate * negatives))
                # 从 ROC 计数计算所有指标，避免逐个阈值重新扫描样本造成平方复杂度。
                candidate = {
                    "threshold": float(cutoff), "sensitivity": float(recall),
                    "specificity": float(specificity), "false_positive_rate": float(false_rate),
                    "youden": float(recall + specificity - 1),
                    "tp": true_positive, "fp": false_positive,
                    "tn": negatives - false_positive, "fn": positives - true_positive,
                    "ppv": safe_ratio(true_positive, true_positive + false_positive),
                    "npv": safe_ratio(negatives - false_positive, negatives - false_positive + positives - true_positive),
                    "f1": safe_ratio(2 * true_positive, true_positive + false_positive + positives),
                }
                candidates.append(candidate)
        for order, (kind, title, target) in enumerate(schemes, start=1):
            row = {key: base.get(key, float("nan")) for key in (
                "n_input", "n_evaluated", "n_patients", "n_positive", "n_negative",
                "positive_rate", "prediction_coverage", "roc_auc", "average_precision"
            )}
            row.update({"disease_cn": chinese_name, "disease": disease,
                        "status": base.get("status", "missing_prediction_column"),
                        "dataset_role": dataset_role, "scheme_order": order, "scheme": title,
                        "target_specificity": target if kind == "specificity" else float("nan"),
                        "target_sensitivity": target if kind == "sensitivity" else float("nan")})
            if usable:
                eligible = [item for item in candidates
                            if kind == "youden" or item[kind] >= target]
                # 平局依次比较另一个指标、阈值，固定规则确保重复运行可复现。
                if kind == "youden":
                    key = lambda item: (item["youden"], item["sensitivity"], item["specificity"], item["threshold"])
                elif kind == "specificity":
                    key = lambda item: (item["sensitivity"], item["specificity"], item["threshold"])
                else:
                    key = lambda item: (item["specificity"], item["sensitivity"], item["threshold"])
                if eligible:
                    best = max(eligible, key=key)
                    row.update(best)
                    row["target_met"] = "不适用" if kind == "youden" else "是"
                    row["threshold_rule"] = "score >= threshold"
                    predicted_positive = best["tp"] + best["fp"]
                    if predicted_positive == 0:
                        row["degenerate_rule"] = "全部判阴性"
                        # ROC 的无穷大阈值用明确规则表达，避免把 inf 当成正常概率阈值。
                        if not np.isfinite(best["threshold"]):
                            row["threshold"] = float("nan")
                            row["threshold_rule"] = "always_negative"
                    elif predicted_positive == len(data[0]):
                        row["degenerate_rule"] = "全部判阳性"
                    else:
                        row["degenerate_rule"] = "否"
                else:
                    row.update({"status": "target_unreachable", "target_met": "否"})
            output_rows.append(row)
    exported = pd.DataFrame(output_rows).reindex(columns=list(descriptions)).rename(
        columns={key: f"{key}（{value}）" for key, value in descriptions.items()}
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output_path, index=False, encoding="utf-8-sig", na_rep="")
    print(f"阈值比较表已保存至：{output_path.resolve()}，共 {len(exported)} 行")
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # 默认路径以本脚本所在项目为基准，显式传入的相对路径以启动目录为基准。
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "results/RADAR_infer_results_MerlinTestset.csv", help="预测 CSV，必须有 file_name 列")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "ckpt/merlin_labels.json", help="真实标签 JSON，由用户提供")
    parser.add_argument("--thresholds", type=Path, help="验证集确定的逐疾病阈值 JSON；可不提供")
    parser.add_argument("--missing-policy", choices=["zero", "exclude"], default="zero", help="zero 延续原脚本填零；exclude 排除缺失预测")
    parser.add_argument("--bootstrap", type=int, default=2000, help="患者级重采样次数；0 表示不计算置信区间")
    parser.add_argument("--seed", type=int, default=42, help="重采样随机种子")
    parser.add_argument("--target-specificity", type=float, default=0.95, help="描述性 ROC 工作点的特异度目标")
    parser.add_argument("--target-sensitivity", type=float, default=0.90, help="描述性 ROC 工作点的敏感度目标")
    parser.add_argument("--gallbladder-cutoff", type=float, default=1000.0, help="胆囊分割数值阈值，小于此值判为缺失")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/merlin_metrics", help="结果目录；同名结果文件会覆盖")
    parser.add_argument("--plot-roc", action="store_true", help="额外生成 4×5 疾病 ROC 图，排除胆囊缺失")
    parser.add_argument("--search-thresholds", action="store_true", help="导出20种疾病各10套阈值方案的比较表")
    parser.add_argument("--dataset-role", choices=["test_exploratory", "validation"], default="test_exploratory", help="选阈值数据用途，默认标为测试集探索；仅独立验证集可指定 validation")
    parser.add_argument("--roc-font", type=Path, help="可选：中文字体文件路径，默认自动查找系统字体")
    args = parser.parse_args()
    if args.bootstrap < 0 or args.seed < 0:
        parser.error("bootstrap 和 seed 必须是非负整数。")
    if args.bootstrap == 1:
        parser.error("bootstrap 请设为 0 或至少 2；正式评估建议 2000。")
    for name in ("target_specificity", "target_sensitivity"):
        if not 0 < getattr(args, name) <= 1:
            parser.error(f"{name} 必须在 (0, 1] 内。")
    if not np.isfinite(args.gallbladder_cutoff) or args.gallbladder_cutoff <= 0:
        parser.error("gallbladder-cutoff 必须是有限正数。")

    try:
        all_labels = read_json(args.labels)
        if not isinstance(all_labels, dict):
            raise ValueError("标签 JSON 顶层必须是疾病名到患者标签字典的映射。")
        thresholds, threshold_metadata = load_thresholds(args.thresholds)
        frame = pd.read_csv(args.predictions, dtype={"file_name": str})
        if "file_name" not in frame or len(frame.columns) < 2:
            raise ValueError("预测 CSV 必须包含 file_name 和至少一个疾病列。")
        if frame["file_name"].isna().any():
            raise ValueError("预测 CSV 中 file_name 不能缺失。")
        # 每个疾病使用独立随机流，避免前一疾病的样本量影响后续重采样。
        columns = [column for column in frame.columns if column != "file_name"]
        random_streams = np.random.SeedSequence(args.seed).spawn(len(columns))
        roc_data = {}
        rows = [evaluate_column(frame, column, all_labels, thresholds, args, np.random.default_rng(stream), roc_data)
                for column, stream in zip(columns, random_streams)]
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"评估失败：{error}\n")

    metrics = pd.DataFrame(rows)
    # 每项宏平均都附带纳入疾病数；无定义的指标不参与平均。
    macro = {}
    for metric in MACRO_METRICS:
        values = pd.to_numeric(metrics.get(metric, pd.Series(dtype=float)), errors="coerce")
        values = values[np.isfinite(values)]
        macro[metric] = {"mean": float(values.mean()) if len(values) else None,
                         "n_diseases": len(values)}
    summary = {
        "config": {key: str(value.resolve()) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "threshold_metadata": threshold_metadata,
        "n_disease_columns": len(rows),
        "status_counts": metrics["status"].value_counts().to_dict(),
        "macro": macro,
        "notes": [
            "指标以扫描行为单位；AUROC 95% CI 为患者整组 bootstrap 百分位区间。",
            "AP 使用 average_precision_score，不是梯形积分 PR-AUC。",
            "ROC 工作点来自测试集经验曲线，属于描述性指标；部署阈值应由验证集确定。",
            "胆囊缺失先二值化；其 AUROC 等于敏感度与特异度的平均值。",
            "zero 策略会把缺失胆囊分割值判为胆囊缺失；请结合覆盖率解读。",
            "未提供阈值的常规疾病不计算固定阈值指标；单类别不计算 AUROC/AP/工作点。",
            "缺少标签、未知疾病列及无有效样本均通过 status 标记，请检查纳入疾病数量。",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_metrics = format_metrics_for_csv(metrics, args.target_specificity, args.target_sensitivity)
    export_metrics.to_csv(args.output_dir / "metrics_per_disease.csv", index=False, encoding="utf-8-sig", na_rep="")
    with (args.output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2, allow_nan=False)
    if args.plot_roc:
        plot_disease_roc_curves(roc_data, args.output_dir / "roc_curves_20_diseases.jpg", args.roc_font)
    if args.search_thresholds:
        search_disease_thresholds(roc_data, metrics, args.output_dir / "threshold_candidates.csv", args.dataset_role)
    for row in rows:
        print(f"{row['disease'] or row['column']}: status={row['status']}, "
              f"n={row['n_evaluated']}, AUROC={row.get('roc_auc', float('nan')):.4f}, "
              f"AP={row.get('average_precision', float('nan')):.4f}")
    print(f"宏平均 AUROC: {macro['roc_auc']['mean']}，纳入疾病数: {macro['roc_auc']['n_diseases']}")
    print(f"结果已保存至：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
