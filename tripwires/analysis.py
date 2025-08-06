import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator


def writejsonl(data, datapath):
    os.makedirs(os.path.dirname(datapath), exist_ok=True)
    print(f"saving file at {datapath}")
    with open(datapath, "w", encoding="utf-8") as f:
        for item in data:
            json_item = json.dumps(item, ensure_ascii=False)
            f.write(json_item + "\n")


def writejson(data, datapath):
    os.makedirs(os.path.dirname(datapath), exist_ok=True)
    print(f"saving file at {datapath}")
    json_str = json.dumps(data, indent=4, ensure_ascii=False)
    with open(datapath, "w", encoding="utf-8") as json_file:
        json_file.write(json_str)


def readjsonl(datapath):
    res = []
    print(f"reading file at {datapath}")
    with open(datapath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            res.append(json.loads(line))
    return res


def readjson(datapath):
    print(f"reading file at {datapath}")
    with open(datapath, "r", encoding="utf-8") as f:
        res = json.load(f)
    return res


# Set publication quality figure parameters
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# Define output directory for figures
from pathlib import Path

current_path = Path(os.path.dirname(os.path.abspath(__file__)))
output_dir = current_path / "figures"
os.makedirs(output_dir, exist_ok=True)

# Get list of models
models = os.listdir(current_path / "infer")
print(f"Found {len(models)} models: {models}")

# Dictionary to store results
results = {}

# 收集数据
for model in models:
    results[model] = {}
    model_dir = current_path / "infer" / model
    indices = os.listdir(model_dir)
    indices = sorted([idx for idx in indices if idx.isdigit()], key=int)
    print(f"Model {model} has {len(indices)} checkpoints: {indices}")

    for idx in indices:
        data_path = model_dir / idx / "probe_vllm.json"
        if data_path.exists():
            data = readjson(data_path)
            results[model][idx] = data["metrics"]
        else:
            print(f"Warning: File not found: {data_path}")

# 整理数据到DataFrame
data_list = []
for model in results:
    for idx in results[model]:
        metrics = results[model][idx]
        row = {
            "model": model,
            "checkpoint": int(idx),
            "macro_hack_rate": metrics.get("macro_hack_rate", None),
            "macro_hack_rate_hard": metrics.get("macro_hack_rate_hard", None),
            "micro_hack_rate": metrics.get("micro_hack_rate", None),
        }
        data_list.append(row)

df = pd.DataFrame(data_list)

# 保存数据
df.to_csv(output_dir / "hacking_prob_results.csv", index=False, float_format="%.15g")

# 创建模型名称的简洁版本用于可视化
model_name_mapping = {}
for i, full_name in enumerate(sorted(df["model"].unique())):
    # 提取关键部分
    parts = full_name.split("_")
    if "nokl" in full_name:
        kl = "w/o KL"
    elif "kl005" in full_name:
        kl = "KL=0.005"
    else:
        kl = "Unknown"

    if "qwen7Bjudge" in full_name:
        judge = "7B as IntentCheck"
    elif "no-IntentCheck" in full_name:
        judge = "w/o IntentCheck"
    elif "overoptimization" in full_name:
        judge = "w/o IntentCheck, w/o Criteria"
    elif "qwen32Bjudge" in full_name:
        judge = "32B as IntentCheck"
    elif "qwen72Bjudge" in full_name:
        judge = "72B as IntentCheck"
    else:
        judge = "Unknown"

    short_name = f"{judge}, {kl}"
    model_name_mapping[full_name] = short_name
    # model_name_mapping[full_name] = full_name

df["model_short"] = df["model"].map(model_name_mapping)

# 使用更加区分度高的颜色映射
# 定义具有高对比度的颜色列表
distinct_colors = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#000000",  # black
    "#006400",  # dark green
    "#8B008B",  # dark magenta
    "#FF1493",  # deep pink
    "#FFD700",  # gold
    "#00CED1",  # dark turquoise
]

# 定义不同形状的标记列表
markers = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "<",
    ">",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "1",
    "2",
    "3",
    "4",
]

unique_models = sorted(df["model_short"].unique())
if len(unique_models) > len(distinct_colors):
    # 如果模型数量超过颜色列表长度，增加更多颜色
    additional_colors = sns.color_palette(
        "tab20", len(unique_models) - len(distinct_colors)
    )
    distinct_colors.extend(additional_colors)

if len(unique_models) > len(markers):
    # 循环使用标记
    markers = markers * (len(unique_models) // len(markers) + 1)

model_color_map = dict(zip(unique_models, distinct_colors[: len(unique_models)]))
model_marker_map = dict(zip(unique_models, markers[: len(unique_models)]))

# 图1: 按checkpoint展示各模型的macro_hack_rate变化
plt.figure(figsize=(12, 7))
for model, group in df.groupby("model"):
    group = group.sort_values("checkpoint")
    short_name = model_name_mapping[model]
    plt.plot(
        group["checkpoint"],
        group["macro_hack_rate"],
        marker=model_marker_map[short_name],
        linewidth=2,
        markersize=8,
        label=short_name,
        color=model_color_map[short_name],
    )

# 添加基线 - qwen2.5-7b-instruct 的结果
baseline_data_path = current_path / "infer" / "qwen2.5-7b-instruct" / "probe_vllm.json"
if baseline_data_path.exists():
    baseline_data = readjson(baseline_data_path)
    baseline_macro_hack_rate = baseline_data["metrics"]["macro_hack_rate"]
    xmin, xmax = plt.xlim()
    plt.hlines(
        baseline_macro_hack_rate,
        xmin,
        xmax,
        colors="black",
        linestyles="dashed",
        linewidth=2,
        label="Qwen2.5-7B-Instruct Baseline",
    )
    plt.text(
        xmax * 0.02,
        baseline_macro_hack_rate * 1.02,
        f"                        Qwen2.5-7B-Instruct: {baseline_macro_hack_rate:.4f}",
        fontweight="bold",
        verticalalignment="bottom",
    )

plt.xlabel("Training Checkpoint")
plt.ylabel("Macro Hack Rate")
plt.title("Macro Hack Rate by Training Checkpoint")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(output_dir / "macro_hack_rate_by_checkpoint.pdf")
plt.savefig(output_dir / "macro_hack_rate_by_checkpoint.png")

# 图2: 按checkpoint展示各模型的macro_hack_rate_hard变化
plt.figure(figsize=(12, 7))
for model, group in df.groupby("model"):
    group = group.sort_values("checkpoint")
    short_name = model_name_mapping[model]
    plt.plot(
        group["checkpoint"],
        group["macro_hack_rate_hard"],
        marker=model_marker_map[short_name],
        linewidth=2,
        markersize=8,
        label=short_name,
        color=model_color_map[short_name],
    )

# 添加基线 - qwen2.5-7b-instruct 的结果
if baseline_data_path.exists():
    baseline_macro_hack_rate_hard = baseline_data["metrics"]["macro_hack_rate_hard"]
    xmin, xmax = plt.xlim()
    plt.hlines(
        baseline_macro_hack_rate_hard,
        xmin,
        xmax,
        colors="black",
        linestyles="dashed",
        linewidth=2,
        label="Qwen2.5-7B-Instruct Baseline",
    )
    plt.text(
        xmax * 0.02,
        baseline_macro_hack_rate_hard * 1.02,
        f"Baseline: {baseline_macro_hack_rate_hard:.4f}",
        fontweight="bold",
        verticalalignment="bottom",
    )

plt.xlabel("Training Checkpoint")
plt.ylabel("Macro Hack Rate (Probe Level)")
plt.title("Macro Hack Rate for Hard Problems by Training Checkpoint")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(output_dir / "macro_hack_rate_hard_by_checkpoint.pdf")
plt.savefig(output_dir / "macro_hack_rate_hard_by_checkpoint.png")

# 图3: 按checkpoint展示各模型的micro_hack_rate变化
plt.figure(figsize=(12, 7))
for model, group in df.groupby("model"):
    group = group.sort_values("checkpoint")
    short_name = model_name_mapping[model]
    plt.plot(
        group["checkpoint"],
        group["micro_hack_rate"],
        marker=model_marker_map[short_name],
        linewidth=2,
        markersize=8,
        label=short_name,
        color=model_color_map[short_name],
    )

# 添加基线 - qwen2.5-7b-instruct 的结果
if baseline_data_path.exists():
    baseline_micro_hack_rate = baseline_data["metrics"]["micro_hack_rate"]
    xmin, xmax = plt.xlim()
    plt.hlines(
        baseline_micro_hack_rate,
        xmin,
        xmax,
        colors="black",
        linestyles="dashed",
        linewidth=2,
        label="Qwen2.5-7B-Instruct Baseline",
    )
    plt.text(
        xmax * 0.02,
        baseline_micro_hack_rate * 1.02,
        f"Baseline: {baseline_micro_hack_rate:.4f}",
        fontweight="bold",
        verticalalignment="bottom",
    )

plt.xlabel("Training Checkpoint")
plt.ylabel("Micro Hack Rate")
plt.title("Micro Hack Rate by Training Checkpoint")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(output_dir / "micro_hack_rate_by_checkpoint.pdf")
plt.savefig(output_dir / "micro_hack_rate_by_checkpoint.png")

# 图4: 最终checkpoint的性能对比（柱状图）
last_checkpoints = df.groupby("model").apply(lambda x: x.loc[x["checkpoint"].idxmax()])
last_checkpoints = last_checkpoints.reset_index(drop=True)

plt.figure(figsize=(12, 6))
x = np.arange(len(last_checkpoints))
width = 0.25

plt.bar(
    x - width,
    last_checkpoints["macro_hack_rate"],
    width,
    label="Macro Hack Rate",
    color="#1f77b4",
)
plt.bar(
    x,
    last_checkpoints["macro_hack_rate_hard"],
    width,
    label="Macro Hack Rate (Hard)",
    color="#ff7f0e",
)
plt.bar(
    x + width,
    last_checkpoints["micro_hack_rate"],
    width,
    label="Micro Hack Rate",
    color="#2ca02c",
)

plt.xlabel("Model Configuration")
plt.ylabel("Hack Rate")
plt.title("Performance Comparison of Different Model Configurations (Final Checkpoint)")
plt.xticks(x, last_checkpoints["model_short"], rotation=45, ha="right")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(output_dir / "final_performance_comparison.pdf")
plt.savefig(output_dir / "final_performance_comparison.png")

# 图5: 热力图展示不同参数对性能的影响
pivot_data = df.pivot_table(
    index=["model_short"],
    columns=["checkpoint"],
    values=["macro_hack_rate", "macro_hack_rate_hard", "micro_hack_rate"],
    aggfunc="mean",
)


# 打印分析结果
print("\n=== Analysis Results ===\n")

# 找出表现最好的模型配置
best_macro = df.loc[df["macro_hack_rate"].idxmin()]
best_macro_hard = df.loc[df["macro_hack_rate_hard"].idxmin()]
best_micro = df.loc[df["micro_hack_rate"].idxmin()]

print(
    f"Best model for Macro Hack Rate: {best_macro['model_short']} at checkpoint {best_macro['checkpoint']} with value {best_macro['macro_hack_rate']:.4f}"
)
print(
    f"Best model for Macro Hack Rate (Hard): {best_macro_hard['model_short']} at checkpoint {best_macro_hard['checkpoint']} with value {best_macro_hard['macro_hack_rate_hard']:.4f}"
)
print(
    f"Best model for Micro Hack Rate: {best_micro['model_short']} at checkpoint {best_micro['checkpoint']} with value {best_micro['micro_hack_rate']:.4f}"
)

# 趋势分析
for model, group in df.groupby("model"):
    group = group.sort_values("checkpoint")
    first_cp = group.iloc[0]
    last_cp = group.iloc[-1]

    macro_diff = last_cp["macro_hack_rate"] - first_cp["macro_hack_rate"]
    macro_hard_diff = last_cp["macro_hack_rate_hard"] - first_cp["macro_hack_rate_hard"]
    micro_diff = last_cp["micro_hack_rate"] - first_cp["micro_hack_rate"]

    short_name = model_name_mapping[model]
    print(
        f"\nModel {short_name} trend from checkpoint {first_cp['checkpoint']} to {last_cp['checkpoint']}:"
    )
    print(
        f"  Macro Hack Rate: {first_cp['macro_hack_rate']:.4f} → {last_cp['macro_hack_rate']:.4f} ({macro_diff:.4f})"
    )
    print(
        f"  Macro Hack Rate (Hard): {first_cp['macro_hack_rate_hard']:.4f} → {last_cp['macro_hack_rate_hard']:.4f} ({macro_hard_diff:.4f})"
    )
    print(
        f"  Micro Hack Rate: {first_cp['micro_hack_rate']:.4f} → {last_cp['micro_hack_rate']:.4f} ({micro_diff:.4f})"
    )

print(f"\nAll figures saved to {output_dir}")
