import csv
import os
import sys

import numpy as np
import torch
import torchvision

import sqfa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append("..")

from pkg_utils import artifact_path, scale_and_center
from pkg_utils.fisher_rao import (
    compute_calvo_oller_matrix,
    compute_mean_true_fisher_rao,
    mean_pairwise_distance,
    plot_calvo_oller_vs_fisher_rao,
    save_pairwise_distance_csv,
    select_cached_filter_set,
)


FILTER_RANGE = (2, 4, 8, 16)
CALVO_OLLER_N_FILTERS = 8
METHOD_SPECS = (
    ("sqfa", "sqfa"),
    ("hellinger", "hellinger"),
    ("bhattacharyya", "bhattacharyya"),
    ("wasserstein", "wasserstein"),
    ("jeffreys", "jeffreys"),
    ("pca", "pca"),
    ("lda", "lda"),
)
FILTERS_DIR = os.path.join(SCRIPT_DIR, "filters_review")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_true_FR_n_filters")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures_review")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_summary_csv(summary_rows, output_path):
    """Write mean Fisher-Rao distances as a filter-count-by-method table."""
    fieldnames = ["n_filters", *[column for column, _model_key in METHOD_SPECS]]
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            formatted_row = {"n_filters": row["n_filters"]}
            for fieldname in fieldnames[1:]:
                value = row[fieldname]
                formatted_row[fieldname] = (
                    ""
                    if np.isnan(value)
                    else f"{float(value):.3f}"
                )
            writer.writerow(formatted_row)


trainset = torchvision.datasets.MNIST(root="./data", train=True, download=False)
n_samples, n_row, n_col = trainset.data.shape
x_train = torch.as_tensor(trainset.data).float().reshape(-1, n_row * n_col)
y_train = torch.as_tensor(trainset.targets, dtype=torch.long)
x_train, _x_unused = scale_and_center(x_train, x_train.clone())
class_stats = sqfa.statistics.class_statistics(points=x_train, labels=y_train)

summary_rows = []

for n_filters in FILTER_RANGE:
    print(f"Computing Fisher-Rao matrices for n_filters={n_filters}")
    row = {"n_filters": n_filters}

    sqfa_noise_path = artifact_path(FILTERS_DIR, "sqfa", "noise", n_filters=n_filters)
    if not os.path.exists(sqfa_noise_path):
        print(
            f"Skipping n_filters={n_filters}: missing SQFA noise artifact at "
            f"{sqfa_noise_path}"
        )
        for column, _model_key in METHOD_SPECS:
            row[column] = np.nan
        summary_rows.append(row)
        continue

    sqfa_noise = float(np.load(sqfa_noise_path).item())

    for column, model_key in METHOD_SPECS:
        filter_path = artifact_path(FILTERS_DIR, model_key, "filters", n_filters=n_filters)
        if not os.path.exists(filter_path):
            print(f"Missing cached filters for {model_key}, n_filters={n_filters}")
            row[column] = np.nan
            continue

        matrix_path = artifact_path(
            RESULTS_DIR,
            model_key,
            "fisherrao_matrix",
            n_filters=n_filters,
        )
        if os.path.exists(matrix_path):
            distance_matrix = np.load(matrix_path)
            mean_distance = mean_pairwise_distance(distance_matrix)
            row[column] = mean_distance
            print(
                f"Reusing saved {model_key} Fisher-Rao matrix for n_filters={n_filters} "
                f"(mean={mean_distance:.6f})"
            )
            continue

        cached_filters = np.load(filter_path)
        filters = select_cached_filter_set(cached_filters, run_idx=0)
        distance_matrix, mean_distance = compute_mean_true_fisher_rao(
            class_stats=class_stats,
            filters=filters,
            covariance_noise=sqfa_noise,
        )

        np.save(matrix_path, distance_matrix)
        row[column] = mean_distance
        print(
            f"Saved {model_key} Fisher-Rao matrix for n_filters={n_filters} "
            f"(mean={mean_distance:.6f})"
        )

    summary_rows.append(row)

save_summary_csv(
    summary_rows,
    os.path.join(RESULTS_DIR, "mean_fisherrao_by_n_filters.csv"),
)


print(
    f"Computing SQFA Calvo-Oller distances for n_filters={CALVO_OLLER_N_FILTERS}"
)
sqfa_filter_path = artifact_path(
    FILTERS_DIR,
    "sqfa",
    "filters",
    n_filters=CALVO_OLLER_N_FILTERS,
)
sqfa_noise_path = artifact_path(
    FILTERS_DIR,
    "sqfa",
    "noise",
    n_filters=CALVO_OLLER_N_FILTERS,
)
sqfa_filters = select_cached_filter_set(np.load(sqfa_filter_path), run_idx=0)
sqfa_noise = float(np.load(sqfa_noise_path).item())

sqfa_fisher_rao_matrix, _mean_distance = compute_mean_true_fisher_rao(
    class_stats=class_stats,
    filters=sqfa_filters,
    covariance_noise=sqfa_noise,
)
np.save(
    artifact_path(
        RESULTS_DIR,
        "sqfa",
        "fisherrao_matrix",
        n_filters=CALVO_OLLER_N_FILTERS,
    ),
    sqfa_fisher_rao_matrix,
)
calvo_oller_matrix = compute_calvo_oller_matrix(
    class_stats=class_stats,
    filters=sqfa_filters,
    covariance_noise=sqfa_noise,
)

save_pairwise_distance_csv(
    fisher_rao_matrix=sqfa_fisher_rao_matrix,
    calvo_oller_matrix=calvo_oller_matrix,
    output_path=os.path.join(
        RESULTS_DIR,
        f"mnist_sqfa_calvo_oller_pairwise_n{CALVO_OLLER_N_FILTERS}.csv",
    ),
    extra_fields={"n_filters": CALVO_OLLER_N_FILTERS},
)
plot_calvo_oller_vs_fisher_rao(
    fisher_rao_matrix=sqfa_fisher_rao_matrix,
    calvo_oller_matrix=calvo_oller_matrix,
    output_path=os.path.join(
        FIGURES_DIR,
        f"mnist_sqfa_calvo_oller_vs_fisherrao_n{CALVO_OLLER_N_FILTERS}.pdf",
    ),
)
