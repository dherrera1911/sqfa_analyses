import csv
import os
import sys

import numpy as np
import torch

import sqfa
from zand_kohn_utils import (
    CONDITIONS,
    load_processed_sessions,
    normalized_split,
    zand_artifact_path,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append("..")

from pkg_utils.fisher_rao import (
    compute_calvo_oller_matrix,
    compute_mean_true_fisher_rao,
    mean_pairwise_distance,
    plot_calvo_oller_vs_fisher_rao,
    save_pairwise_distance_csv,
    select_cached_filter_set,
)


AREA_KEY = "V1"
CONDITION = CONDITIONS[0]["condition"]
FILTER_RANGE = (2, 4, 6, 8)
CALVO_OLLER_N_FILTERS = 6
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, "processed_data")
FILTERS_DIR = os.path.join(SCRIPT_DIR, "filters_review")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_true_FR_n_filters")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures_review")
METHOD_SPECS = (
    ("sqfa", "sqfa"),
    ("smsqfa", "smsqfa"),
    ("hellinger", "hellinger"),
    ("bhattacharyya", "bhattacharyya"),
    ("wasserstein", "wasserstein"),
    ("jeffreys", "jeffreys"),
    ("pca", "pca"),
    ("lda", "lda"),
    ("spca", "spca"),
    ("lfda", "lfda"),
    ("wda", "wda"),
    ("lmnn", "lmnn"),
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_summary_csv(summary_rows, output_path):
    fieldnames = ["n_filters", *[column for column, _model_key in METHOD_SPECS]]
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            formatted_row = {"n_filters": row["n_filters"]}
            for column, _model_key in METHOD_SPECS:
                value = row[column]
                formatted_row[column] = (
                    ""
                    if np.isnan(value)
                    else f"{float(value):.3f}"
                )
            writer.writerow(formatted_row)


sessions = load_processed_sessions(PROCESSED_DATA_DIR, area_key=AREA_KEY, condition_name=CONDITION)
summary_rows = []

data = sessions[0]
split_i = 0
x_train, y_train, _x_val, _y_val, _x_test, _y_test = normalized_split(data, split_i)
class_stats = sqfa.statistics.class_statistics(points=x_train, labels=y_train)

for n_filters in FILTER_RANGE:
    print(f"Computing Fisher-Rao matrices for n_filters={n_filters}", flush=True)
    row = {"n_filters": n_filters}

    sqfa_noise_path = zand_artifact_path(
        FILTERS_DIR,
        "sqfa",
        "noise",
        data["session"],
        split_i,
        n_filters,
    )
    if not os.path.exists(sqfa_noise_path):
        print(
            f"Skipping session={data['session']}, split={split_i}, "
            f"n_filters={n_filters}: missing SQFA noise artifact",
            flush=True,
        )
        for column, _model_key in METHOD_SPECS:
            row[column] = np.nan
        summary_rows.append(row)
        continue
    covariance_noise = float(np.load(sqfa_noise_path).item())

    for column, model_key in METHOD_SPECS:
        filter_path = zand_artifact_path(
            FILTERS_DIR,
            model_key,
            "filters",
            data["session"],
            split_i,
            n_filters,
        )
        if not os.path.exists(filter_path):
            row[column] = np.nan
            continue

        matrix_path = zand_artifact_path(
            RESULTS_DIR,
            model_key,
            "fisherrao_matrix",
            data["session"],
            split_i,
            n_filters,
        )
        if os.path.exists(matrix_path):
            distance_matrix = np.load(matrix_path)
            mean_distance = mean_pairwise_distance(distance_matrix)
            row[column] = mean_distance
            print(
                f"Reusing saved {model_key} Fisher-Rao matrix for "
                f"session={data['session']}, split={split_i}, n_filters={n_filters} "
                f"(mean={mean_distance:.6f})",
                flush=True,
            )
            continue
        else:
            cached_filters = np.load(filter_path)
            filters = select_cached_filter_set(cached_filters, run_idx=0)
            distance_matrix, mean_distance = compute_mean_true_fisher_rao(
                class_stats=class_stats,
                filters=filters,
                covariance_noise=covariance_noise,
            )
            np.save(matrix_path, distance_matrix)

        row[column] = mean_distance
        print(
            f"Saved {model_key} Fisher-Rao matrix for "
            f"session={data['session']}, split={split_i}, n_filters={n_filters} "
            f"(mean={mean_distance:.6f})",
            flush=True,
        )

    summary_rows.append(row)

save_summary_csv(
    summary_rows,
    os.path.join(RESULTS_DIR, "mean_fisherrao_by_n_filters.csv"),
)


print(
    f"Computing SQFA Calvo-Oller distances for n_filters={CALVO_OLLER_N_FILTERS}",
    flush=True,
)
sqfa_filter_path = zand_artifact_path(
    FILTERS_DIR,
    "sqfa",
    "filters",
    data["session"],
    split_i,
    CALVO_OLLER_N_FILTERS,
)
sqfa_noise_path = zand_artifact_path(
    FILTERS_DIR,
    "sqfa",
    "noise",
    data["session"],
    split_i,
    CALVO_OLLER_N_FILTERS,
)
sqfa_filters = select_cached_filter_set(np.load(sqfa_filter_path), run_idx=0)
sqfa_noise = float(np.load(sqfa_noise_path).item())

sqfa_fisher_rao_path = zand_artifact_path(
    RESULTS_DIR,
    "sqfa",
    "fisherrao_matrix",
    data["session"],
    split_i,
    CALVO_OLLER_N_FILTERS,
)
if os.path.exists(sqfa_fisher_rao_path):
    sqfa_fisher_rao_matrix = np.load(sqfa_fisher_rao_path)
else:
    sqfa_fisher_rao_matrix, _mean_distance = compute_mean_true_fisher_rao(
        class_stats=class_stats,
        filters=sqfa_filters,
        covariance_noise=sqfa_noise,
    )
    np.save(sqfa_fisher_rao_path, sqfa_fisher_rao_matrix)

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
        "zand_kohn_sqfa_calvo_oller_pairwise"
        f"_n{CALVO_OLLER_N_FILTERS}_session{data['session']}_split{split_i}.csv",
    ),
    extra_fields={
        "session": data["session"],
        "split": split_i,
        "n_filters": CALVO_OLLER_N_FILTERS,
    },
)
plot_calvo_oller_vs_fisher_rao(
    fisher_rao_matrix=sqfa_fisher_rao_matrix,
    calvo_oller_matrix=calvo_oller_matrix,
    output_path=os.path.join(
        FIGURES_DIR,
        "zand_kohn_sqfa_calvo_oller_vs_fisherrao"
        f"_n{CALVO_OLLER_N_FILTERS}_session{data['session']}_split{split_i}.pdf",
    ),
)
