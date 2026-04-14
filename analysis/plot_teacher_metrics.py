#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


MAIN_METRICS = [
    "mean_spearman",
    "mean_kendall_tau",
    "pairwise_accuracy",
    "success_auc",
    "top1_hit_rate",
]

MAIN_CONDITION_ORDER = [
    "base",
    "solution",
    "another_solution",
    "solution+another_solution",
    "failure_solution",
    "solution+failure_solution",
]

BUCKET_ORDER = ["easy", "medium", "hard"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot teacher signal metrics from aggregate JSON.")
    parser.add_argument("--input", required=True, help="Path to teacher_metrics.json.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--sample-set",
        choices=["all_samples", "effective_only"],
        default="all_samples",
        help="Which sample set from teacher_metrics.json to plot.",
    )
    parser.add_argument(
        "--paired-target-type",
        help="Optional target_type to plot from paired_by_target_type instead of the full condition set.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def select_metrics_view(metrics: dict, sample_set: str, paired_target_type: str | None) -> dict:
    sample_sets = metrics.get("sample_sets")
    if sample_sets is None:
        return metrics

    split = sample_sets[sample_set]
    if paired_target_type is None:
        return split
    return split["paired_by_target_type"][paired_target_type]


def ordered_conditions(metrics_view: dict) -> list[str]:
    conditions = list(metrics_view["conditions"].keys())
    ordered = [c for c in MAIN_CONDITION_ORDER if c in conditions]
    ordered.extend(sorted(c for c in conditions if c not in ordered))
    return ordered


def plot_main(metrics_view: dict, output_path: Path, title: str) -> None:
    conditions = ordered_conditions(metrics_view)
    fig, axes = plt.subplots(1, len(MAIN_METRICS), figsize=(22, 4.8), constrained_layout=True)

    for ax, metric in zip(axes, MAIN_METRICS):
        values = [metrics_view["conditions"][cond][metric] for cond in conditions]
        ax.bar(range(len(conditions)), values, color="#2f6db3")
        ax.set_title(metric)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha="right")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_buckets(metrics_view: dict, output_path: Path, title: str) -> None:
    conditions = ordered_conditions(metrics_view)
    fig, axes = plt.subplots(len(BUCKET_ORDER), len(MAIN_METRICS), figsize=(22, 12), constrained_layout=True)

    for row_idx, bucket in enumerate(BUCKET_ORDER):
        for col_idx, metric in enumerate(MAIN_METRICS):
            ax = axes[row_idx][col_idx]
            values = []
            for cond in conditions:
                bucket_metrics = metrics_view["conditions"][cond]["by_bucket"].get(bucket)
                values.append(bucket_metrics[metric] if bucket_metrics else 0.0)
            ax.bar(range(len(conditions)), values, color="#c65f46")
            if row_idx == 0:
                ax.set_title(metric)
            if col_idx == 0:
                ax.set_ylabel(bucket)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=45, ha="right")
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    metrics = load_metrics(Path(args.input))
    output_dir = Path(args.output_dir)
    metrics_view = select_metrics_view(metrics, args.sample_set, args.paired_target_type)
    title_suffix = args.sample_set
    if args.paired_target_type is not None:
        title_suffix = f"{title_suffix}, paired target_type={args.paired_target_type}"

    plot_main(
        metrics_view,
        output_dir / "teacher_metrics_main.png",
        f"Teacher Signal Metrics By Condition ({title_suffix})",
    )
    plot_buckets(
        metrics_view,
        output_dir / "teacher_metrics_by_bucket.png",
        f"Teacher Signal Metrics By Condition And Difficulty Bucket ({title_suffix})",
    )
    print(f"Wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
