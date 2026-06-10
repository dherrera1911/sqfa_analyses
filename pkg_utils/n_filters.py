"""Shared helpers for the n_filters review analyses."""

import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import artifact_path
from .evaluation import collect_metric_across_runs


REVIEW_MODEL_SPECS = [
    ("SQFA", "sqfa"),
    ("smSQFA", "smsqfa"),
    ("SQFA-H", "hellinger"),
    ("SQFA-B", "bhattacharyya"),
    ("SQFA-W", "wasserstein"),
    ("SQFA-J", "jeffreys"),
    ("LDA", "lda"),
    ("SPCA", "spca"),
    ("LFDA", "lfda"),
    ("WDA", "wda"),
    ("LMNN", "lmnn"),
    ("PCA", "pca"),
]


def _to_float(value):
    """Convert scalar tensor-like outputs to plain floats."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def max_supported_filter_count(filter_range, max_rank):
    """Return the largest requested filter count supported by a model rank."""
    valid_counts = [n_filters for n_filters in filter_range if n_filters <= max_rank]
    return max(valid_counts, default=0)


def summarize_training_time_table(
    filters_dir="filters_review",
    model_specs=REVIEW_MODEL_SPECS,
):
    """Build a table of median training time by filter count and model."""
    n_filters_vals = []
    for _model_name, model_key in model_specs:
        pattern = re.compile(rf"^{re.escape(model_key)}_time_n(\d+)\.npy$")
        for file_name in os.listdir(filters_dir):
            match = pattern.match(file_name)
            if match is not None:
                n_filters_vals.append(int(match.group(1)))

    n_filters_vals = sorted(set(n_filters_vals))
    time_table = pd.DataFrame(index=n_filters_vals)
    time_table.index.name = "n_filters"

    for model_name, model_key in model_specs:
        medians = []
        for n_filters in n_filters_vals:
            time_path = artifact_path(
                filters_dir,
                model_key,
                "time",
                n_filters=n_filters,
            )
            if os.path.exists(time_path):
                times = np.load(time_path)
                medians.append(float(np.median(times)))
            else:
                medians.append(np.nan)

        time_table[model_name] = medians

    return time_table


def summarize_metric_results(
    model_specs,
    filter_range,
    filters_dir,
    score_fn,
    model_max_filters=None,
):
    """Summarize per-model metric values across filter counts."""
    model_max_filters = {} if model_max_filters is None else dict(model_max_filters)
    results = []

    for model_name, model_key in model_specs:
        max_filters = model_max_filters.get(model_key)
        for n_filters in filter_range:
            if max_filters is not None and n_filters > max_filters:
                continue

            current_filter_path = artifact_path(
                filters_dir,
                model_key,
                "filters",
                n_filters=n_filters,
            )
            if not os.path.exists(current_filter_path):
                continue

            filters = np.load(current_filter_path)
            scores = collect_metric_across_runs(
                filters,
                lambda filt: _to_float(score_fn(filt)),
            )
            scores = np.asarray(scores, dtype=float) * 100.0
            median = float(np.median(scores))
            if scores.size > 2:
                sorted_scores = np.sort(scores)
                q10 = sorted_scores[1]
                q90 = sorted_scores[-2]
            else:
                q10 = median
                q90 = median

            results.append(
                {
                    "model_name": model_name,
                    "model_key": model_key,
                    "n_filters": int(n_filters),
                    "n_runs": int(scores.size),
                    "mean_percent": float(np.mean(scores)),
                    "std_percent": float(np.std(scores)),
                    "median_percent": median,
                    "q10_percent": float(q10),
                    "q90_percent": float(q90),
                }
            )

    return results


def export_metric_results_csv(metric_results, metric_name, output_path):
    """Write summarized metric results to CSV."""
    fieldnames = [
        "metric",
        "model_name",
        "model_key",
        "n_filters",
        "n_runs",
        "mean_percent",
        "std_percent",
        "median_percent",
        "q10_percent",
        "q90_percent",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in metric_results:
            writer.writerow({"metric": metric_name, **result})


def plot_metric_results(
    model_specs,
    filter_range,
    metric_results,
    ylabel,
    output_path,
    ylim,
):
    """Plot median performance with 10th-90th percentile bands."""
    colors = plt.get_cmap("tab10")(np.arange(len(model_specs)))
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    if len(model_specs) > 1:
        jitter_offsets = np.linspace(-0.18, 0.18, len(model_specs))
    else:
        jitter_offsets = np.array([0.0])

    for jitter_offset, color, (model_name, model_key) in zip(
        jitter_offsets,
        colors,
        model_specs,
    ):
        model_results = [
            result for result in metric_results if result["model_key"] == model_key
        ]
        if not model_results:
            continue

        x_vals = [result["n_filters"] for result in model_results]
        x_vals = np.asarray(x_vals, dtype=float) + jitter_offset
        medians = [result["median_percent"] for result in model_results]
        q10_vals = [result["q10_percent"] for result in model_results]
        q90_vals = [result["q90_percent"] for result in model_results]

        ax.plot(
            x_vals,
            medians,
            color=color,
            marker="o",
            linewidth=2,
            label=model_name,
            markersize=7,
            markerfacecolor=(*color[:3], 0.35),
            markeredgecolor=(*color[:3], 0.6),
            markeredgewidth=1.0,
        )
        ax.fill_between(x_vals, q10_vals, q90_vals, color=color, alpha=0.15)

    ax.set_xlabel("Number of Filters", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(filter_range)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
