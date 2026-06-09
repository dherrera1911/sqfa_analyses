import csv
import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sqfa_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from metric_learn import LMNN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sqfa

sys.path.append("..")  # Add parent directory to path

from pkg_utils import (
    artifact_path,
    balanced_subset_indices,
    collect_metric_across_runs,
    has_saved_artifacts,
    knn_accuracy,
    load_cached_filters,
    load_or_validate_lda_shrinkage,
    load_or_validate_noise,
    plot_metric_with_errorbars,
    qda_accuracy,
    scale_and_center,
    save_training_artifacts,
    train_metric_learn_repeated,
    train_sqfa_repeated,
    train_val_split,
)


ACTIVE_MODEL_KEYS = ("sqfa", "lda")
FILTER_RANGE = (2, 4, 8, 16)
EMBEDDING_STEM = "cifar100_resnet18"
NOISE_VALS = torch.tensor([0.005, 0.05, 0.2, 1.0, 5.0, 20.0])
N_REPS = 3
SQFA_FIT_KWARGS = {"max_epochs": 800, "show_progress": False}
LDA_SHRINKAGE_VALS = np.array([0.05, 0.1, 0.2, 0.4, 0.8], dtype=float)
KNN_N_NEIGHBORS = 5
LMNN_PCA_DIM = 200
LMNN_SAMPLES_PER_CLASS = 50
FILTERS_DIR = "filters_n_filters_all_methods"
FIGURES_DIR = "figures_n_filters_all_methods"
RESULTS_DIR = "results_n_filters_all_methods"
EMBEDDINGS_DIR = "embeddings"
ALL_MODEL_SPECS = [
    ("SQFA", "sqfa"),
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
MODEL_SPECS = [
    (model_name, model_key)
    for model_name, model_key in ALL_MODEL_SPECS
    if model_key in ACTIVE_MODEL_KEYS
]
torch.manual_seed(5)

os.makedirs(FILTERS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def _to_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def max_supported_filter_count(filter_range, max_rank):
    valid_counts = [n_filters for n_filters in filter_range if n_filters <= max_rank]
    return max(valid_counts, default=0)


def load_embeddings(split_name):
    split_path = os.path.join(EMBEDDINGS_DIR, f"{EMBEDDING_STEM}_{split_name}.pt")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Missing {split_path}. Run extract_embeddings.py first."
        )

    saved = torch.load(split_path, map_location="cpu")
    x = torch.as_tensor(saved["X"], dtype=torch.float32)
    y = torch.as_tensor(saved["y"], dtype=torch.long)
    return x, y


def summarize_metric_results(model_specs, score_fn, model_max_filters=None):
    model_max_filters = {} if model_max_filters is None else dict(model_max_filters)
    results = []

    for model_name, model_key in model_specs:
        max_filters = model_max_filters.get(model_key)
        for n_filters in FILTER_RANGE:
            if max_filters is not None and n_filters > max_filters:
                continue

            current_filter_path = artifact_path(
                FILTERS_DIR,
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
            if scores.size > 1:
                q25, q75 = np.percentile(scores, [25, 75])
            else:
                q25 = median
                q75 = median

            results.append(
                {
                    "model_name": model_name,
                    "model_key": model_key,
                    "n_filters": int(n_filters),
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
    fieldnames = [
        "metric",
        "model_name",
        "model_key",
        "n_filters",
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

    ax.set_xlabel("Number of Filters", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(FILTER_RANGE)
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


x_train, y_train = load_embeddings("train")
x_test, y_test = load_embeddings("test")
x_train, x_test = scale_and_center(x_train, x_test)
N_CLASSES = int(torch.unique(y_train).numel())
MAX_LDA_FILTERS = max_supported_filter_count(FILTER_RANGE, N_CLASSES - 1)

x_train_reg, y_train_reg, x_val, y_val = train_val_split(
    x_train,
    y_train,
    val_size=0.15,
)


for n_filters in FILTER_RANGE:
    print(f"Training models with n_filters={n_filters}")

    sqfa_filter_path = artifact_path(FILTERS_DIR, "sqfa", "filters", n_filters=n_filters)
    sqfa_time_path = artifact_path(FILTERS_DIR, "sqfa", "time", n_filters=n_filters)
    sqfa_noise_path = artifact_path(FILTERS_DIR, "sqfa", "noise", n_filters=n_filters)
    pca_filter_path = artifact_path(FILTERS_DIR, "pca", "filters", n_filters=n_filters)
    pca_time_path = artifact_path(FILTERS_DIR, "pca", "time", n_filters=n_filters)
    hellinger_filter_path = artifact_path(
        FILTERS_DIR, "hellinger", "filters", n_filters=n_filters
    )
    hellinger_time_path = artifact_path(
        FILTERS_DIR, "hellinger", "time", n_filters=n_filters
    )
    hellinger_noise_path = artifact_path(
        FILTERS_DIR, "hellinger", "noise", n_filters=n_filters
    )
    lda_filter_path = artifact_path(FILTERS_DIR, "lda", "filters", n_filters=n_filters)
    lda_time_path = artifact_path(FILTERS_DIR, "lda", "time", n_filters=n_filters)
    lda_shrinkage_path = artifact_path(
        FILTERS_DIR, "lda", "shrinkage", n_filters=n_filters
    )
    lmnn_filter_path = artifact_path(
        FILTERS_DIR, "lmnn", "filters", n_filters=n_filters
    )
    lmnn_time_path = artifact_path(FILTERS_DIR, "lmnn", "time", n_filters=n_filters)

    if "sqfa" in ACTIVE_MODEL_KEYS:
        sqfa_noise = load_or_validate_noise(
            noise_path=sqfa_noise_path,
            model_factory=lambda noise: sqfa.model.SQFA(
                n_dim=x_train.shape[1],
                n_filters=n_filters,
                feature_noise=noise,
            ),
            x_train=x_train_reg,
            y_train=y_train_reg,
            x_val=x_val,
            y_val=y_val,
            noise_vals=NOISE_VALS,
            fit_kwargs=SQFA_FIT_KWARGS,
            run_label=f"sqfa validation for n_filters={n_filters}",
        )
        if not has_saved_artifacts(sqfa_filter_path, sqfa_time_path):
            sqfa_filters, sqfa_times = train_sqfa_repeated(
                model_factory=lambda: sqfa.model.SQFA(
                    n_dim=x_train.shape[1],
                    n_filters=n_filters,
                    feature_noise=sqfa_noise,
                ),
                x_train=x_train,
                y_train=y_train,
                n_reps=N_REPS,
                fit_kwargs=SQFA_FIT_KWARGS,
                run_label=f"sqfa training for n_filters={n_filters}",
            )
            save_training_artifacts(
                sqfa_filter_path,
                sqfa_time_path,
                sqfa_filters,
                sqfa_times,
            )
        else:
            load_cached_filters(
                sqfa_filter_path,
                description=f"sqfa filters for n_filters={n_filters}",
            )

    if "pca" in ACTIVE_MODEL_KEYS:
        if not has_saved_artifacts(pca_filter_path, pca_time_path):
            pca = PCA(n_components=n_filters)
            start = time.time()
            pca.fit(x_train)
            save_training_artifacts(
                pca_filter_path,
                pca_time_path,
                pca.components_,
                time.time() - start,
            )
        else:
            load_cached_filters(
                pca_filter_path,
                description=f"pca filters for n_filters={n_filters}",
            )

    if "hellinger" in ACTIVE_MODEL_KEYS:
        hellinger_noise = load_or_validate_noise(
            noise_path=hellinger_noise_path,
            model_factory=lambda noise: sqfa.model.SQFA(
                n_dim=x_train.shape[1],
                n_filters=n_filters,
                feature_noise=noise,
                distance_fun=sqfa.distances.hellinger,
            ),
            x_train=x_train_reg,
            y_train=y_train_reg,
            x_val=x_val,
            y_val=y_val,
            noise_vals=NOISE_VALS,
            fit_kwargs=SQFA_FIT_KWARGS,
            run_label=f"hellinger validation for n_filters={n_filters}",
        )
        if not has_saved_artifacts(hellinger_filter_path, hellinger_time_path):
            hellinger_filters, hellinger_times = train_sqfa_repeated(
                model_factory=lambda: sqfa.model.SQFA(
                    n_dim=x_train.shape[1],
                    n_filters=n_filters,
                    feature_noise=hellinger_noise,
                    distance_fun=sqfa.distances.hellinger,
                ),
                x_train=x_train,
                y_train=y_train,
                n_reps=N_REPS,
                fit_kwargs=SQFA_FIT_KWARGS,
                run_label=f"hellinger training for n_filters={n_filters}",
            )
            save_training_artifacts(
                hellinger_filter_path,
                hellinger_time_path,
                hellinger_filters,
                hellinger_times,
            )
        else:
            load_cached_filters(
                hellinger_filter_path,
                description=f"hellinger filters for n_filters={n_filters}",
            )

    if "lda" in ACTIVE_MODEL_KEYS:
        if n_filters > MAX_LDA_FILTERS:
            print(
                f"Skipping LDA for n_filters={n_filters}: supported maximum is {MAX_LDA_FILTERS}."
            )
        else:
            lda_shrinkage = load_or_validate_lda_shrinkage(
                shrinkage_path=lda_shrinkage_path,
                x_train=x_train_reg,
                y_train=y_train_reg,
                x_val=x_val,
                y_val=y_val,
                shrinkage_vals=LDA_SHRINKAGE_VALS,
                n_filters=n_filters,
                eval_qda_reg=1.0e-5,
            )
            if not has_saved_artifacts(lda_filter_path, lda_time_path):
                lda = LinearDiscriminantAnalysis(
                    solver="eigen",
                    shrinkage=lda_shrinkage,
                )
                start = time.time()
                lda.fit(x_train, y_train)
                lda_time = time.time() - start
                lda_filters = lda.scalings_.T[:MAX_LDA_FILTERS]
                if lda_filters.shape[0] >= n_filters:
                    save_training_artifacts(
                        lda_filter_path,
                        lda_time_path,
                        lda_filters[:n_filters],
                        lda_time,
                    )
            else:
                load_cached_filters(
                    lda_filter_path,
                    description=f"lda filters for n_filters={n_filters}",
                )

    if "lmnn" in ACTIVE_MODEL_KEYS:
        if not has_saved_artifacts(lmnn_filter_path, lmnn_time_path):
            pca_subsample = PCA(n_components=LMNN_PCA_DIM)
            x_train_np = x_train.detach().cpu().numpy()
            pca_subsample.fit(x_train_np)
            x_transformed = pca_subsample.transform(x_train_np)
            lmnn_subset_idx = balanced_subset_indices(
                y_train.detach().cpu().numpy(),
                samples_per_class=LMNN_SAMPLES_PER_CLASS,
                seed=5,
            )
            x_transformed_sub = x_transformed[lmnn_subset_idx]
            y_train_sub = y_train[lmnn_subset_idx]

            lmnn_filters, lmnn_times = train_metric_learn_repeated(
                estimator_factory=lambda: LMNN(
                    n_neighbors=3,
                    learn_rate=1.0e-6,
                    n_components=n_filters,
                    init="pca",
                    verbose=True,
                    max_iter=2000,
                    convergence_tol=0.1,
                ),
                x_train=x_train,
                y_train=y_train_sub,
                n_reps=1,
                fit_x=x_transformed_sub,
                extract_filters=lambda estimator: pca_subsample.inverse_transform(
                    estimator.components_
                ),
                run_label=f"lmnn training for n_filters={n_filters}",
            )
            save_training_artifacts(
                lmnn_filter_path,
                lmnn_time_path,
                lmnn_filters,
                lmnn_times,
            )
        else:
            load_cached_filters(
                lmnn_filter_path,
                description=f"lmnn filters for n_filters={n_filters}",
            )


qda_results = summarize_metric_results(
    MODEL_SPECS,
    lambda filt: qda_accuracy(
        x_train,
        y_train,
        x_test,
        y_test,
        filt,
        eval_qda_noise=0.0,
        eval_qda_reg=1.0e-5,
    ),
    model_max_filters={"lda": MAX_LDA_FILTERS},
)
export_metric_results_csv(
    qda_results,
    "qda",
    os.path.join(RESULTS_DIR, "cifar100_n_filters_qda_all_methods.csv"),
)
plot_metric_results(
    MODEL_SPECS,
    qda_results,
    "QDA Accuracy (%)",
    os.path.join(FIGURES_DIR, "cifar100_accuracies_n_filters_all_methods.pdf"),
    ylim=(0, 100),
)

knn_results = summarize_metric_results(
    MODEL_SPECS,
    lambda filt: knn_accuracy(
        x_train,
        y_train,
        x_test,
        y_test,
        filt,
        n_neighbors=KNN_N_NEIGHBORS,
    ),
    model_max_filters={"lda": MAX_LDA_FILTERS},
)
export_metric_results_csv(
    knn_results,
    "knn",
    os.path.join(RESULTS_DIR, "cifar100_n_filters_knn_all_methods.csv"),
)
plot_metric_results(
    MODEL_SPECS,
    knn_results,
    "KNN Accuracy (%)",
    os.path.join(FIGURES_DIR, "cifar100_accuracies_n_filters_knn_all_methods.pdf"),
    ylim=(0, 100),
)

time_sets = []
model_names = [model_name for model_name, _model_key in MODEL_SPECS]
for _model_name, model_key in MODEL_SPECS:
    per_filter_times = []
    for n_filters in FILTER_RANGE:
        current_time_path = artifact_path(FILTERS_DIR, model_key, "time", n_filters=n_filters)
        if os.path.exists(current_time_path):
            per_filter_times.append(
                np.asarray(np.load(current_time_path), dtype=float).reshape(-1)
            )
    if per_filter_times:
        time_sets.append(np.concatenate(per_filter_times))
    else:
        time_sets.append(np.asarray([], dtype=float))

non_empty_time_sets = [times for times in time_sets if times.size > 0]
if non_empty_time_sets:
    all_times = np.concatenate(non_empty_time_sets)
    plot_metric_with_errorbars(
        model_names,
        time_sets,
        "Training Time (s)",
        os.path.join(FIGURES_DIR, "cifar100_training_times_n_filters_all_methods.pdf"),
        unit=" s",
        value_fmt="{:.2f}",
        spread_fmt="{:.2f}",
        yscale="log",
        ylim=(max(float(all_times.min()) * 0.5, 1.0e-3), float(all_times.max()) * 5),
        offset_ratio=0.1,
        min_offset=0.05,
    )
