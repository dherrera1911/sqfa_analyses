import csv
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import sqfa
from zand_kohn_utils import (
    CONDITIONS,
    CONSTRAINT,
    EVAL_QDA_REG,
    SPLIT_SEED,
    SQFA_DTYPE,
    SQFA_FIT_KWARGS,
    load_processed_sessions,
    normalize_from_train,
    zand_regularization_artifact_path,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append("..")

from pkg_utils import (
    has_saved_artifacts,
    load_cached_filters,
    qda_accuracy,
    save_training_artifacts,
    train_sqfa_repeated,
)


N_FILTERS = 2
AREA_KEY = "V1"
CONDITION = CONDITIONS[0]["condition"]
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, "processed_data")
FILTERS_DIR = os.path.join(SCRIPT_DIR, "filters_regularization_single_split")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures_review")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_review")
N_REPS = 1
TEST_SIZE = 0.20
NOISE_VALS = torch.tensor([0.0002, 0.002, 0.02, 0.2, 0.5, 1.0, 2.0, 5.0])

torch.manual_seed(6)
np.random.seed(6)
os.makedirs(FILTERS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


model_specs = [
    (
        "SQFA",
        "sqfa",
        lambda n_dim, noise: sqfa.model.SQFA(
            n_dim=n_dim,
            n_filters=N_FILTERS,
            feature_noise=noise,
            constraint=CONSTRAINT,
        ),
    ),
    (
        "SQFA-H",
        "hellinger",
        lambda n_dim, noise: sqfa.model.SQFA(
            n_dim=n_dim,
            n_filters=N_FILTERS,
            feature_noise=noise,
            distance_fun=sqfa.distances.hellinger,
            constraint=CONSTRAINT,
        ),
    ),
    (
        "SQFA-B",
        "bhattacharyya",
        lambda n_dim, noise: sqfa.model.SQFA(
            n_dim=n_dim,
            n_filters=N_FILTERS,
            feature_noise=noise,
            distance_fun=sqfa.distances.bhattacharyya,
            constraint=CONSTRAINT,
        ),
    ),
#    (
#        "SQFA-W",
#        "wasserstein",
#        lambda n_dim, noise: sqfa.model.SQFA(
#            n_dim=n_dim,
#            n_filters=N_FILTERS,
#            feature_noise=noise,
#            distance_fun=sqfa.distances.wasserstein,
#            constraint=CONSTRAINT,
#        ),
#    ),
#    (
#        "SQFA-J",
#        "jeffreys",
#        lambda n_dim, noise: sqfa.model.SQFA(
#            n_dim=n_dim,
#            n_filters=N_FILTERS,
#            feature_noise=noise,
#            distance_fun=sqfa.distances.jeffreys,
#            constraint=CONSTRAINT,
#        ),
#    ),
]


def iter_filter_sets(filter_bank):
    filters = np.asarray(filter_bank)
    if filters.ndim == 2:
        yield 0, filters
    else:
        for rep_i, current_filters in enumerate(filters):
            yield rep_i, current_filters


def normalized_train_test_split(data):
    x_train, x_test, y_train, y_test = train_test_split(
        data["x"],
        data["y"],
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED + 1000 * data["session"],
        stratify=data["y"],
    )
    x_train, _x_val, x_test = normalize_from_train(x_train, y_train, x_train, x_test)
    return x_train, y_train, x_test, y_test


def train_model(model_key, model_factory, noise, metadata, x_train, y_train):
    filter_path = zand_regularization_artifact_path(
        FILTERS_DIR,
        model_key,
        "filters",
        metadata["session"],
        metadata["split"],
        noise,
    )
    time_path = zand_regularization_artifact_path(
        FILTERS_DIR,
        model_key,
        "time",
        metadata["session"],
        metadata["split"],
        noise,
    )
    if not has_saved_artifacts(filter_path, time_path):
        filters, times = train_sqfa_repeated(
            model_factory=lambda: model_factory(x_train.shape[1], noise),
            x_train=x_train,
            y_train=y_train,
            n_reps=N_REPS,
            fit_kwargs=SQFA_FIT_KWARGS,
            dtypes=(SQFA_DTYPE,),
            run_label=(
                f"{model_key} regularization, session={metadata['session']}, "
                f"split={metadata['split']}, noise={noise}"
            ),
        )
        save_training_artifacts(filter_path, time_path, filters, times)
    else:
        load_cached_filters(
            filter_path,
            description=f"{model_key} filters for regularization={noise}",
        )
    return filter_path


def score_filters(score_rows, model_name, model_key, filter_path, metadata, x_train, y_train, x_test, y_test):
    filters = np.load(filter_path)
    for rep_i, current_filters in iter_filter_sets(filters):
        qda_score = qda_accuracy(
            x_train,
            y_train,
            x_test,
            y_test,
            current_filters,
            eval_qda_reg=EVAL_QDA_REG,
        ).item()
        score_rows.append({
            **metadata,
            "rep": rep_i,
            "model_name": model_name,
            "model_key": model_key,
            "qda_accuracy": qda_score,
        })


def summarize_results(score_rows):
    results = []
    for model_name, model_key, _model_factory in model_specs:
        for noise in NOISE_VALS:
            noise_value = float(noise.item())
            scores = [
                row["qda_accuracy"] * 100.0
                for row in score_rows
                if row["model_key"] == model_key
                and np.isclose(row["regularization"], noise_value)
            ]
            if len(scores) == 0:
                continue
            scores = np.asarray(scores, dtype=float)
            median = float(np.median(scores))
            if scores.size > 1:
                q25, q75 = np.percentile(scores, [25, 75])
            else:
                q25 = median
                q75 = median
            results.append({
                "model_name": model_name,
                "model_key": model_key,
                "regularization": noise_value,
                "n_runs": int(scores.size),
                "mean_percent": float(np.mean(scores)),
                "std_percent": float(np.std(scores)),
                "median_percent": median,
                "q25_percent": float(q25),
                "q75_percent": float(q75),
            })
    return results


def write_scores_csv(score_rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(score_rows[0].keys()))
        writer.writeheader()
        writer.writerows(score_rows)


def export_summary_csv(results, output_path):
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
        for result in results:
            writer.writerow({"metric": "qda", **result})


def plot_results(results, output_path):
    colors = plt.get_cmap("tab10")(np.arange(len(model_specs)))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    jitter_scales = np.geomspace(10 ** (-0.02), 10 ** 0.02, len(model_specs))

    for jitter_scale, color, (model_name, model_key, _model_factory) in zip(
        jitter_scales,
        colors,
        model_specs,
    ):
        model_results = [
            result for result in results if result["model_key"] == model_key
        ]
        if not model_results:
            continue
        model_results = sorted(model_results, key=lambda result: result["regularization"])
        x_vals = np.asarray(
            [result["regularization"] for result in model_results],
            dtype=float,
        ) * jitter_scale
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
            markerfacecolor=(*color[:3], 1.00),
            markeredgecolor=(*color[:3], 1.0),
            markeredgewidth=1.0,
        )
        ax.fill_between(x_vals, q25_vals, q75_vals, color=color, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("Regularization Value", fontsize=12)
    ax.set_ylabel("QDA Accuracy (%)", fontsize=12)
    ax.set_ylim(90, 100)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


sessions = load_processed_sessions(PROCESSED_DATA_DIR, area_key=AREA_KEY, condition_name=CONDITION)
score_rows = []

data = sessions[4]
split_i = 1
x_train, y_train, x_test, y_test = normalized_train_test_split(data)
for noise in NOISE_VALS:
    noise_value = float(noise.item())
    metadata = {
        "area": data["area"],
        "condition": data["condition"],
        "session": int(data["session"]),
        "file_name": data["file_name"],
        "split": split_i,
        "regularization": noise_value,
        "n_filters": N_FILTERS,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_neurons": int(x_train.shape[1]),
        "n_classes": int(torch.unique(data["y"]).numel()),
    }
    print(
        f"Training regularization sweep, session={data['session']}, "
        f"split={split_i}, noise={noise_value}",
        flush=True,
    )
    for model_name, model_key, model_factory in model_specs:
        filter_path = train_model(
            model_key,
            model_factory,
            noise_value,
            metadata,
            x_train,
            y_train,
        )
        score_filters(
            score_rows,
            model_name,
            model_key,
            filter_path,
            metadata,
            x_train,
            y_train,
            x_test,
            y_test,
        )

write_scores_csv(
    score_rows,
    os.path.join(RESULTS_DIR, "zand_kohn_regularization_sensitivity_all_scores.csv"),
)
summary_results = summarize_results(score_rows)
export_summary_csv(
    summary_results,
    os.path.join(RESULTS_DIR, "zand_kohn_regularization_sensitivity_qda.csv"),
)
plot_results(
    summary_results,
    os.path.join(FIGURES_DIR, "zand_kohn_regularization_sensitivity_qda.pdf"),
)
