import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RAW_DIR = Path("results/raw")
PLOT_DIR = Path("results/plots")
AGG_PATH = Path("results/aggregated_outputs.csv")
CALIBRATION_PLOT = PLOT_DIR / "calibration_curve.png"
CONF_HIST_PLOT = PLOT_DIR / "confidence_histogram.png"
BAR_PLOT = PLOT_DIR / "performance_bar.png"
ERROR_PATH = Path("results/error_cases.md")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def prepare_dataframe() -> pd.DataFrame:
    baseline = load_jsonl(RAW_DIR / "baseline.jsonl")
    sanction = load_jsonl(RAW_DIR / "sanction.jsonl")
    df = pd.DataFrame(baseline + sanction)
    df["pred_idx"] = df["decision"].apply(decision_to_index)
    df["correct_flag"] = np.where(
        (df["pred_idx"].notna()) & (~df["abstain"]),
        (df["pred_idx"] == df["correct_choice_idx"]).astype(float),
        np.nan,
    )
    df["penalty"] = df.apply(calc_penalty, axis=1)
    return df


def decision_to_index(value):
    if value is None or not isinstance(value, str):
        return np.nan
    value = value.strip().upper()
    if not value or not value[0].isalpha():
        return np.nan
    return ord(value[0]) - ord("A")


def performance_bar(df: pd.DataFrame) -> None:
    answered = df[~df["abstain"] & df["pred_idx"].notna() & (~df["correct_flag"].isna())]
    summary = answered.groupby("protocol").agg(
        accuracy=("correct_flag", "mean"),
        mean_conf=("confidence", "mean"),
    )
    brier = (
        df.dropna(subset=["confidence", "correct_flag"])
        .assign(sq=lambda frame: (frame["confidence"] - frame["correct_flag"]) ** 2)
        .groupby("protocol")["sq"]
        .mean()
    )
    summary["brier"] = brier
    summary["abstention"] = df.groupby("protocol")["abstain"].mean()
    melted = summary.reset_index().melt(
        id_vars="protocol",
        value_vars=["accuracy", "mean_conf", "brier", "abstention"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="protocol", palette="deep")
    plt.title("Performance summary by protocol")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(BAR_PLOT, dpi=200)
    plt.close()


def confidence_hist(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    for protocol in df["protocol"].unique():
        subset = df[(df["protocol"] == protocol) & (~df["confidence"].isna())]
        sns.kdeplot(subset["confidence"], label=protocol, fill=True, common_norm=False, alpha=0.25)
    plt.title("Confidence density by protocol")
    plt.xlabel("Self-reported confidence")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONF_HIST_PLOT, dpi=200)
    plt.close()


def calibration_curve(df: pd.DataFrame, bins: int = 10) -> None:
    plt.figure(figsize=(6, 6))
    diagonal = np.linspace(0, 1, 100)
    plt.plot(diagonal, diagonal, linestyle="--", color="gray", label="Perfect calibration")
    for protocol in df["protocol"].unique():
        subset = df[(df["protocol"] == protocol) & (~df["confidence"].isna()) & (~df["correct_flag"].isna())]
        if subset.empty:
            continue
        bin_ids = np.minimum((subset["confidence"].values * bins).astype(int), bins - 1)
        points = []
        for b in range(bins):
            idx = bin_ids == b
            if not np.any(idx):
                continue
            points.append(
                (
                    subset["confidence"].values[idx].mean(),
                    subset["correct_flag"].values[idx].mean(),
                )
            )
        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], marker="o", label=protocol)
    plt.title("Reliability curve")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CALIBRATION_PLOT, dpi=200)
    plt.close()


def write_error_cases(df: pd.DataFrame, k: int = 5) -> None:
    wrong = df[(~df["abstain"]) & (df["pred_idx"].notna()) & (df["pred_idx"] != df["correct_choice_idx"])]
    samples = wrong.sort_values("confidence", ascending=False).groupby("protocol").head(k)
    lines = ["# High-confidence errors\n"]
    for _, row in samples.iterrows():
        correct_choice = row["choices"][row["correct_choice_idx"]]
        pred_idx = int(row["pred_idx"]) if not pd.isna(row["pred_idx"]) else None
        pred_choice = row["choices"][pred_idx] if pred_idx is not None else "N/A"
        lines.append(
            f"## {row['protocol'].title()} – Q{row['question_idx']}\n"
            f"- **Question**: {row['question']}\n"
            f"- **Predicted**: {pred_choice} (confidence {row['confidence']})\n"
            f"- **Correct**: {correct_choice}\n"
            f"- **Rationale**: {row['rationale']}\n"
        )
    ERROR_PATH.write_text("\n".join(lines))


def calc_penalty(row):
    confidence = row.get("confidence")
    if pd.isna(confidence) or row.get("abstain"):
        return 0.0
    pred_idx = row.get("pred_idx")
    if pd.isna(pred_idx):
        return 0.0
    correct = int(pred_idx == row["correct_choice_idx"])
    threshold = 0.7
    audit_rate = 0.35
    if correct:
        underconfidence = max(0.0, threshold - confidence)
        return 0.1 * underconfidence
    if confidence >= threshold:
        return audit_rate * 2.0 * confidence
    return audit_rate * confidence


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_dataframe()
    df.to_csv(AGG_PATH, index=False)
    performance_bar(df)
    confidence_hist(df)
    calibration_curve(df)

    write_error_cases(df)


if __name__ == "__main__":
    main()
