"""Plotting helpers shared by experiments."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_filter_grid(
    filters,
    image_shape: Tuple[int, int],
    *,
    n_cols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None,
):
    """Create a grid plot of flattened filters.

    Returns the created ``(fig, ax)`` tuple so callers can save/close.
    """
    filters = np.asarray(filters)
    n_filters = filters.shape[0]
    if n_cols is None:
        n_cols = n_filters
    n_rows = int(np.ceil(n_filters / n_cols))

    if figsize is None:
        figsize = (n_cols * 1.0, n_rows * 1.0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= n_filters:
            continue
        ax.imshow(filters[idx].reshape(image_shape), cmap="gray")
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


def plot_metric_with_errorbars(
    model_names,
    score_sets,
    ylabel,
    filename,
    *,
    scale=1.0,
    ylim=None,
    unit="",
    value_fmt="{:.1f}",
    spread_fmt="{:.1f}",
    yscale='linear',
    offset_ratio=0.02,
    min_offset=1.0,
    show_errorbars=True,
    figsize=(7, 3),
):
    """Plot bar chart with optional error bars for repeated measurements."""
    centers = []
    spreads = []
    for scores in score_sets:
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if scores.size == 0:
            centers.append(float("nan"))
            spreads.append(None)
            continue
        scaled = scores * scale
        if scaled.size > 1:
            median = float(np.median(scaled))
            q25, q75 = np.percentile(scaled, [25, 75])
            centers.append(median)
            spreads.append((float(median - q25), float(q75 - median)))
        else:
            centers.append(float(scaled[0]))
            spreads.append(None)

    positions = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(positions, centers)

    if show_errorbars:
        for idx, spread in enumerate(spreads):
            if spread is None:
                continue
            lower, upper = spread
            ax.errorbar(
                positions[idx],
                centers[idx],
                yerr=np.array([[lower], [upper]]),
                fmt='none',
                ecolor='black',
                capsize=4,
                linewidth=1.2,
            )

    for idx, (center, spread) in enumerate(zip(centers, spreads)):
        label = f"{value_fmt.format(center)}{unit}"
        offset = max(abs(center) * offset_ratio, min_offset)
        text_y = center + offset if center >= 0 else center - offset
        ax.text(positions[idx], text_y, label, ha='center', fontsize=10)

    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Model", fontsize=14)
    ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
