import csv
import glob
import os
import re

import numpy as np
import torch
from sklearn.model_selection import train_test_split


AREA_SPECS = {
    "V1": [0],
}
CONDITIONS = [
    {"condition": "early_50_100ms", "start_ms": 50, "end_ms": 100},
]
MIN_RATE = 2.0
CLEAN_OUTLIER_COMPONENTS = 10
CLEAN_SUPPORT_FRACTION = 0.8
TRIAL_OUTLIER_STD_MULT = 4.0
#NEURON_OUTLIER_THRESHOLD = 4.0
NEURON_OUTLIER_THRESHOLD = 3.0
N_OUTLIER_MAX = 60

N_SPLITS = 5
TEST_SIZE = 0.10
VAL_SIZE = 0.10
SPLIT_SEED = 12
EVAL_QDA_REG = 1.0e-5
SQFA_DTYPE = torch.float64
CONSTRAINT = "sphere"
SQFA_FIT_KWARGS = {
    "max_epochs": 300,
    "show_progress": False,
    "estimator": "empirical",
    "atol": 1e-9,
    "line_search_fn": "strong_wolfe",
}
NOISE_VALS = torch.tensor([0.0002, 0.002, 0.02, 0.2, 0.5, 0.8])
LDA_SHRINKAGE_VALS = np.array([0.01, 0.1, 0.2, 0.4, 0.8], dtype=float)


def processed_data_path(processed_dir, session_i, area_key, condition_name):
    return os.path.join(
        processed_dir,
        f"zand_session{session_i}_{area_key}_{condition_name}.pt",
    )


def zand_artifact_path(artifacts_dir, model_key, artifact_kind, session_i, split_i, n_filters):
    return (
        f"{artifacts_dir}/{model_key}_{artifact_kind}"
        f"_n{n_filters}_session{session_i}_split{split_i}.npy"
    )


def zand_regularization_artifact_path(
    artifacts_dir,
    model_key,
    artifact_kind,
    session_i,
    split_i,
    noise_value,
):
    return (
        f"{artifacts_dir}/{model_key}_{artifact_kind}"
        f"_noise{noise_to_tag(noise_value)}_session{session_i}_split{split_i}.npy"
    )


def load_processed_sessions(processed_dir, area_key="V1", condition_name="early_50_100ms"):
    pattern = processed_data_path(processed_dir, "*", area_key, condition_name)
    paths = sorted(
        glob.glob(pattern),
        key=lambda path: int(re.search(r"zand_session(\d+)_", os.path.basename(path)).group(1)),
    )
    sessions = []
    for path in paths:
        data = torch.load(path, weights_only=False)
        data["path"] = path
        sessions.append(data)
    return sessions


def split_train_val_test(x, y, session_i, split_i):
    random_state = SPLIT_SEED + 1000 * session_i + split_i
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y,
    )
    val_fraction_of_train_val = VAL_SIZE / (1.0 - TEST_SIZE)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_fraction_of_train_val,
        random_state=random_state + 100,
        stratify=y_train_val,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def within_class_std(x_train, y_train):
    residuals = torch.zeros_like(x_train)
    for label in y_train.unique():
        class_mask = y_train == label
        class_mean = x_train[class_mask].mean(dim=0, keepdim=True)
        residuals[class_mask] = x_train[class_mask] - class_mean
    return residuals.std(dim=0, keepdim=True)


def normalize_from_train(x_train, y_train, x_val, x_test):
    train_mean = x_train.mean(dim=0, keepdim=True)
    train_std = within_class_std(x_train, y_train)
    x_train = (x_train - train_mean) / (train_std + 1e-6)
    x_val = (x_val - train_mean) / (train_std + 1e-6)
    x_test = (x_test - train_mean) / (train_std + 1e-6)
    return x_train, x_val, x_test


def normalized_split(data, split_i):
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
        data["x"],
        data["y"],
        data["session"],
        split_i,
    )
    x_train, x_val, x_test = normalize_from_train(x_train, y_train, x_val, x_test)
    return x_train, y_train, x_val, y_val, x_test, y_test


def write_csv(rows, output_path):
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def noise_to_tag(noise_value):
    return str(float(noise_value)).replace(".", "p")


def summarize_score_rows(rows, model_specs, filter_range, metric_key):
    results = []
    for model_name, model_key in model_specs:
        for n_filters in filter_range:
            scores = [
                float(row[metric_key]) * 100.0
                for row in rows
                if row["model_key"] == model_key and row["n_filters"] == n_filters
            ]
            if len(scores) == 0:
                continue

            scores = np.asarray(scores, dtype=float)
            median = float(np.median(scores))
            if scores.size > 2:
                sorted_scores = np.sort(scores)
                q10 = sorted_scores[1]
                q90 = sorted_scores[-2]
            else:
                q10 = median
                q90 = median

            results.append({
                "model_name": model_name,
                "model_key": model_key,
                "n_filters": int(n_filters),
                "n_runs": int(scores.size),
                "mean_percent": float(np.mean(scores)),
                "std_percent": float(np.std(scores)),
                "median_percent": median,
                "q10_percent": float(q10),
                "q90_percent": float(q90),
            })
    return results
