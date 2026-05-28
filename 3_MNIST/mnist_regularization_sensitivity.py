import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import sqfa

sys.path.append("..")  # Add parent directory to path
from pkg_utils import (
    artifact_path,
    collect_metric_across_runs,
    has_saved_artifacts,
    load_cached_filters,
    qda_accuracy,
    scale_and_center,
    save_training_artifacts,
    train_sqfa_repeated,
)


N_FILTERS = 9
NOISE_VALS = torch.tensor([0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
N_REPS = 3
FILTERS_DIR = "filters_regularization"
FIGURES_DIR = "figures_review"
RESULTS_DIR = "results_review"
SQFA_FIT_KWARGS = {"max_epochs": 300, "show_progress": False}
WASSERSTEIN_DTYPES = (torch.float64,)
torch.manual_seed(3)

os.makedirs(FILTERS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def _to_float(value):
    """Convert scalar tensor-like outputs to plain floats."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def summarize_metric_results(model_specs, score_fn):
    """Summarize performance across regularization values and repetitions."""
    results = []

    for model_name, model_key, _model_factory, _dtypes, _fit_kwargs in model_specs:
        for noise in NOISE_VALS:
            noise_value = float(noise.item())
            filter_path = artifact_path(
                FILTERS_DIR,
                model_key,
                "filters",
                n_filters=noise_to_tag(noise_value),
            )
            if not os.path.exists(filter_path):
                continue

            filters = np.load(filter_path)
            scores = collect_metric_across_runs(
                filters,
                lambda filt: _to_float(score_fn(filt)),
            )
            scores = np.asarray(scores, dtype=float) * 100.0
            median = float(np.median(scores))
            if scores.size > 1:
                q25, q75 = np.percentile(scores, [25, 75])
            else:
                q25 = median
                q75 = median

            results.append(
                {
                    "model_name": model_name,
                    "model_key": model_key,
                    "regularization": noise_value,
                    "n_runs": int(scores.size),
                    "mean_percent": float(np.mean(scores)),
                    "std_percent": float(np.std(scores)),
                    "median_percent": median,
                    "q25_percent": float(q25),
                    "q75_percent": float(q75),
                }
            )

    return results


def export_metric_results_csv(metric_results, metric_name, output_path):
    """Write summarized metric results to CSV."""
    fieldnames = [
        "metric",
        "model_name",
        "model_key",
        "regularization",
        "n_runs",
        "mean_percent",
        "std_percent",
        "median_percent",
        "q25_percent",
        "q75_percent",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in metric_results:
            writer.writerow({"metric": metric_name, **result})


def plot_metric_results(model_specs, metric_results, ylabel, output_path, ylim):
    """Plot test accuracy versus regularization value."""
    colors = plt.get_cmap("tab10")(np.arange(len(model_specs)))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if len(model_specs) > 1:
        jitter_scales = np.geomspace(10 ** (-0.02), 10 ** 0.02, len(model_specs))
    else:
        jitter_scales = np.array([1.0])

    for jitter_scale, color, (
        model_name,
        model_key,
        _model_factory,
        _dtypes,
        _fit_kwargs,
    ) in zip(jitter_scales, colors, model_specs):
        model_results = [
            result for result in metric_results if result["model_key"] == model_key
        ]
        if not model_results:
            continue

        model_results = sorted(model_results, key=lambda result: result["regularization"])
        x_vals = [result["regularization"] for result in model_results]
        x_vals = np.asarray(x_vals, dtype=float) * jitter_scale
        medians = [result["median_percent"] for result in model_results]
        q25_vals = [result["q25_percent"] for result in model_results]
        q75_vals = [result["q75_percent"] for result in model_results]

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
        ax.fill_between(x_vals, q25_vals, q75_vals, color=color, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("Regularization Value", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def noise_to_tag(noise_value):
    """Convert a float regularization value to a stable artifact tag."""
    return str(noise_value).replace(".", "p")


#############################
#
# LOAD AND PROCESS DATA
#
#############################

trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True)

n_samples, n_row, n_col = trainset.data.shape
x_train = torch.as_tensor(trainset.data).float().reshape(-1, n_row * n_col)
y_train = torch.as_tensor(trainset.targets, dtype=torch.long)
x_test = torch.as_tensor(testset.data).float().reshape(-1, n_row * n_col)
y_test = torch.as_tensor(testset.targets, dtype=torch.long)

x_train, x_test = scale_and_center(x_train, x_test)


#############################
#
# TRAIN MODELS
#
#############################

model_specs = [
    (
        "SQFA",
        "sqfa",
        lambda noise: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=noise,
        ),
        (torch.float32, torch.float64),
        SQFA_FIT_KWARGS,
    ),
    (
        "SQFA-H",
        "hellinger",
        lambda noise: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=noise,
            distance_fun=sqfa.distances.hellinger,
        ),
        (torch.float32, torch.float64),
        SQFA_FIT_KWARGS,
    ),
    (
        "SQFA-B",
        "bhattacharyya",
        lambda noise: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=noise,
            distance_fun=sqfa.distances.bhattacharyya,
        ),
        (torch.float32, torch.float64),
        SQFA_FIT_KWARGS,
    ),
    (
        "SQFA-W",
        "wasserstein",
        lambda noise: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=noise,
            distance_fun=sqfa.distances.wasserstein,
            constraint="orthogonal",
        ),
        WASSERSTEIN_DTYPES,
        SQFA_FIT_KWARGS,
    ),
    (
        "SQFA-J",
        "jeffreys",
        lambda noise: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=noise,
            distance_fun=sqfa.distances.jeffreys,
        ),
        (torch.float32, torch.float64),
        SQFA_FIT_KWARGS,
    ),
]

for model_name, model_key, model_factory, dtypes, fit_kwargs in model_specs:
    print(f"Training {model_name} across regularization values")

    for noise in NOISE_VALS:
        noise_value = float(noise.item())
        noise_tag = noise_to_tag(noise_value)
        current_filter_path = artifact_path(
            FILTERS_DIR,
            model_key,
            "filters",
            n_filters=noise_tag,
        )
        current_time_path = artifact_path(
            FILTERS_DIR,
            model_key,
            "time",
            n_filters=noise_tag,
        )

        if not has_saved_artifacts(current_filter_path, current_time_path):
            filters, times = train_sqfa_repeated(
                model_factory=lambda noise_value=noise_value: model_factory(noise_value),
                x_train=x_train,
                y_train=y_train,
                n_reps=N_REPS,
                fit_kwargs=fit_kwargs,
                dtypes=dtypes,
                run_label=f"{model_key} sensitivity, noise={noise_value}",
            )
            save_training_artifacts(current_filter_path, current_time_path, filters, times)
        else:
            load_cached_filters(
                current_filter_path,
                description=f"{model_key} filters for regularization={noise_value}",
            )


#############################
#
# PLOT QDA ACCURACY VS REGULARIZATION
#
#############################

qda_results = summarize_metric_results(
    model_specs,
    lambda filt: qda_accuracy(
        x_train,
        y_train,
        x_test,
        y_test,
        filt,
        eval_qda_noise=0.0,
        eval_qda_reg=1.0e-5,
    ),
)
export_metric_results_csv(
    qda_results,
    "qda",
    f"{RESULTS_DIR}/mnist_regularization_sensitivity_qda.csv",
)
plot_metric_results(
    model_specs,
    qda_results,
    "QDA Accuracy (%)",
    f"{FIGURES_DIR}/mnist_regularization_sensitivity_qda.pdf",
    ylim=(60, 100),
)
