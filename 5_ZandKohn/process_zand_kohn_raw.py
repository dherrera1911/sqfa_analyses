import csv
import os

import numpy as np
import scipy.io
import torch
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA

from zand_kohn_utils import (
    AREA_SPECS,
    CLEAN_OUTLIER_COMPONENTS,
    CLEAN_SUPPORT_FRACTION,
    CONDITIONS,
    MIN_RATE,
    NEURON_OUTLIER_THRESHOLD,
    N_OUTLIER_MAX,
    TRIAL_OUTLIER_STD_MULT,
    processed_data_path,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

RAW_DATA_DIR = "raw_data"
PROCESSED_DATA_DIR = "processed_data"
PREFERRED_FILE_ORDER = [
    "106r001p26.mat",
    "106r002p70.mat",
    "105l001p16.mat",
    "107l002p67.mat",
    "107l003p143.mat",
]
ORIENTATIONS = np.array([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5])

torch.manual_seed(6)
np.random.seed(6)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def ordered_raw_files():
    raw_files = [
        file_name for file_name in os.listdir(RAW_DATA_DIR) if file_name.endswith(".mat")
    ]
    ordered_files = [file_name for file_name in PREFERRED_FILE_ORDER if file_name in raw_files]
    ordered_files.extend(sorted(set(raw_files) - set(ordered_files)))
    return ordered_files


def neuron_z_scores(x, y):
    z_scores = torch.zeros_like(x)
    for label in y.unique():
        class_mask = y == label
        xc = x[class_mask]
        mean = xc.mean(dim=0, keepdim=True)
        std = xc.std(dim=0, keepdim=True)
        z_scores[class_mask] = (xc - mean) / (std + 1e-6)
    return z_scores


def n_outlier_trials(x, y, threshold=2.5):
    z_scores = neuron_z_scores(x, y)
    return (z_scores.abs() > threshold).sum(dim=0)


def outlier_trials_mcd(x, y, n_components=10, support_fraction=0.9):
    z_scores = neuron_z_scores(x, y)
    trial_inds = torch.arange(len(y))
    outlier_mask = torch.zeros(len(y), dtype=torch.bool)
    mahalanobis_distances = torch.zeros(len(y))

    for label in y.unique():
        class_inds = trial_inds[y == label]
        zc = z_scores[y == label].numpy()
        pca = PCA(n_components=n_components)
        zc_pca = pca.fit_transform(zc)
        mcd = MinCovDet(support_fraction=support_fraction).fit(zc_pca)
        outlier_mask[class_inds] = torch.as_tensor(mcd.support_ == False)
        mahalanobis_distances[class_inds] = torch.as_tensor(
            mcd.dist_,
            dtype=torch.float32,
        )

    return outlier_mask, mahalanobis_distances


def load_zand_file(file_name, areas, start_ms, end_ms, min_rate):
    mat = scipy.io.loadmat(
        os.path.join(RAW_DATA_DIR, file_name),
        squeeze_me=True,
    )["neuralData"]
    spike_rasters = mat["spikeRasters"].item()
    stim = np.asarray(mat["stim"].item())
    valid_trials = stim != 0
    stim_valid = stim[valid_trials]

    x_by_area = []
    n_neurons_raw_by_area = []
    for area in areas:
        counts = []
        for trial_i in range(len(spike_rasters)):
            counts.append(spike_rasters[trial_i][area][:, start_ms:end_ms].sum(axis=1))

        counts = np.asarray(counts).squeeze()
        n_neurons_raw_by_area.append(int(counts.shape[1]))
        firing_rates = counts[valid_trials, :] / ((end_ms - start_ms) / 1000.0)
        x_by_area.append(firing_rates)

    x = np.concatenate(x_by_area, axis=1)
    keep_neurons = x.mean(axis=0) > min_rate
    x = x[:, keep_neurons]
    _stim_values, stim_id = np.unique(stim_valid, return_inverse=True)

    return (
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(stim_id, dtype=torch.long),
        ORIENTATIONS[stim_id],
        n_neurons_raw_by_area,
        int(valid_trials.sum()),
        int(len(stim)),
    )


def clean_zand_data(x, y):
    min_class_count = int(torch.bincount(y).min().item())
    n_components = min(CLEAN_OUTLIER_COMPONENTS, x.shape[1], min_class_count - 1)
    _outlier_mask, dist = outlier_trials_mcd(
        x,
        y,
        n_components=n_components,
        support_fraction=CLEAN_SUPPORT_FRACTION,
    )
    outlier_mask = dist > (dist.median() + TRIAL_OUTLIER_STD_MULT * dist.std())
    x = x[~outlier_mask]
    y = y[~outlier_mask]

    neuron_outliers = n_outlier_trials(x, y, threshold=NEURON_OUTLIER_THRESHOLD)
    x = x[:, neuron_outliers < N_OUTLIER_MAX]
    return x, y, outlier_mask, neuron_outliers


summary_rows = []
raw_files = ordered_raw_files()

for session_i, file_name in enumerate(raw_files):
    for area_key, areas in AREA_SPECS.items():
        for condition in CONDITIONS:
            print(
                f"Processing session {session_i} ({file_name}), {area_key}, "
                f"{condition['condition']}",
                flush=True,
            )
            x, y, orientations, n_neurons_raw_by_area, n_valid_trials, n_trials_raw = (
                load_zand_file(
                    file_name=file_name,
                    areas=areas,
                    start_ms=condition["start_ms"],
                    end_ms=condition["end_ms"],
                    min_rate=MIN_RATE,
                )
            )
            n_trials_before_cleaning = int(x.shape[0])
            n_neurons_before_cleaning = int(x.shape[1])
            x, y, trial_outlier_mask, neuron_outliers = clean_zand_data(x, y)
            orientations = orientations[~trial_outlier_mask.numpy()]
            output_path = processed_data_path(
                PROCESSED_DATA_DIR,
                session_i,
                area_key,
                condition["condition"],
            )
            data = {
                "x": x,
                "y": y,
                "session": int(session_i),
                "file_name": file_name,
                "area": area_key,
                "areas": areas,
                "condition": condition["condition"],
                "start_ms": int(condition["start_ms"]),
                "end_ms": int(condition["end_ms"]),
                "min_rate": float(MIN_RATE),
                "orientations": orientations,
                "orientation_by_class": ORIENTATIONS,
                "n_trials_raw": n_trials_raw,
                "n_valid_trials": n_valid_trials,
                "n_neurons_raw_by_area": n_neurons_raw_by_area,
                "n_trials_before_cleaning": n_trials_before_cleaning,
                "n_neurons_before_cleaning": n_neurons_before_cleaning,
                "n_trials": int(x.shape[0]),
                "n_neurons": int(x.shape[1]),
                "n_classes": int(torch.unique(y).numel()),
                "n_trial_outliers": int(trial_outlier_mask.sum().item()),
                "n_neuron_outliers": int((neuron_outliers >= N_OUTLIER_MAX).sum().item()),
            }
            torch.save(data, output_path)
            summary_rows.append({
                "session": session_i,
                "file_name": file_name,
                "area": area_key,
                "condition": condition["condition"],
                "n_trials_raw": n_trials_raw,
                "n_valid_trials": n_valid_trials,
                "n_neurons_before_cleaning": n_neurons_before_cleaning,
                "n_trials": int(x.shape[0]),
                "n_neurons": int(x.shape[1]),
                "n_classes": int(torch.unique(y).numel()),
                "n_trial_outliers": int(trial_outlier_mask.sum().item()),
                "n_neuron_outliers": int((neuron_outliers >= N_OUTLIER_MAX).sum().item()),
            })

summary_path = os.path.join(PROCESSED_DATA_DIR, "zand_kohn_processed_summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Saved {len(summary_rows)} processed datasets to {PROCESSED_DATA_DIR}")
