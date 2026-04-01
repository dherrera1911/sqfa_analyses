"""Simple SVHN WDA comparison script for repeated test runs."""

import csv
import os
import sys

import numpy as np
import sqfa
import torch
import torchvision
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append("..")

from pkg_utils import (
    fit_sqfa_adaptive_precision,
    load_or_validate_noise,
    qda_accuracy,
    scale_and_center,
)
from pkg_utils.wda import fit_wda


#############################
#
# PARAMETERS
#
#############################

SEED = 2
VAL_SIZE = 0.15

N_FILTERS = 9
EVAL_QDA_REG = 1.0e-5

WDA_REG = 1.0
WDA_SINKHORN_ITERS = 10
WDA_MAXITER = 100
WDA_SINKHORN_METHOD = "sinkhorn"
WDA_SOLVER = "steepest"
WDA_NORMALIZE = True

EXP1_PCA_DIM = 100
EXP1_SAMPLES_PER_CLASS = 500
EXP1_N_REPS = 3

EXP2_PCA_DIMS = [50, 100]
EXP2_SAMPLES_PER_CLASS = [100, 200, 500, 1000]
EXP2_N_REPS = 3

NOISE_VALS = torch.tensor([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
SQFA_FIT_KWARGS = {"max_epochs": 500, "show_progress": False}

WDA_FILTERS_DIR = "wda_filters"
WDA_FIGURES_DIR = "wda_figures"
DATA_DIR = "data"

EXP1_CSV = os.path.join(WDA_FIGURES_DIR, "svhn_wdatest_repeated_wda.csv")
EXP2_CSV = os.path.join(WDA_FIGURES_DIR, "svhn_wdatest_wda_vs_sqfa.csv")
SQFA_NOISE_PATH = os.path.join(WDA_FILTERS_DIR, "sqfa_noise_wdatest.npy")


#############################
#
# SMALL HELPERS
#
#############################

torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(WDA_FILTERS_DIR, exist_ok=True)
os.makedirs(WDA_FIGURES_DIR, exist_ok=True)


def train_val_split_fixed(x, y, val_size=VAL_SIZE, seed=SEED):
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(x.numpy(), y.numpy()))
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def save_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="ascii") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def artifact_path(stem, suffix):
    return os.path.join(WDA_FILTERS_DIR, f"{stem}_{suffix}.npy")


def wda_artifact_stem(experiment_name, pca_dim, samples_per_class, seed):
    return (
        f"{experiment_name}_wda_pca{pca_dim}_samples{samples_per_class}_seed{seed}"
    )


def sqfa_artifact_stem(experiment_name, pca_dim, samples_per_class, seed):
    return (
        f"{experiment_name}_sqfa_pca{pca_dim}_samples{samples_per_class}_seed{seed}"
    )


def load_saved_artifacts(stem):
    filters_path = artifact_path(stem, "filters")
    time_path = artifact_path(stem, "time")
    if os.path.exists(filters_path) and os.path.exists(time_path):
        filters = np.load(filters_path)
        fit_time = float(np.load(time_path))
        return filters, fit_time
    return None, None


def save_artifacts(stem, filters, fit_time):
    np.save(artifact_path(stem, "filters"), np.asarray(filters))
    np.save(artifact_path(stem, "time"), np.asarray(fit_time))


def run_wda_cached(pca_dim, samples_per_class, seed, experiment_name):
    stem = wda_artifact_stem(experiment_name, pca_dim, samples_per_class, seed)
    filters, fit_time = load_saved_artifacts(stem)

    if filters is None:
        filters, fit_time = fit_wda(
            x_train=x_train,
            y_train=y_train,
            n_filters=N_FILTERS,
            n_pca_components=pca_dim,
            reg=WDA_REG,
            samples_per_class=samples_per_class,
            seed=seed,
            sinkhorn_iters=WDA_SINKHORN_ITERS,
            maxiter=WDA_MAXITER,
            sinkhorn_method=WDA_SINKHORN_METHOD,
            solver=WDA_SOLVER,
            normalize=WDA_NORMALIZE,
        )
        save_artifacts(stem, filters, fit_time)

    qda_acc = qda_accuracy(
        x_train,
        y_train,
        x_test,
        y_test,
        filters,
        eval_qda_reg=EVAL_QDA_REG,
    ).item()
    return fit_time, qda_acc


def run_sqfa_cached(seed, pca_dim, samples_per_class, experiment_name):
    stem = sqfa_artifact_stem(experiment_name, pca_dim, samples_per_class, seed)
    filters, fit_time = load_saved_artifacts(stem)

    if filters is None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, fit_time, train_dtype = fit_sqfa_adaptive_precision(
            model_factory=lambda: sqfa.model.SQFA(
                n_dim=x_train.shape[1],
                n_filters=N_FILTERS,
                feature_noise=sqfa_noise,
            ),
            x_train=x_train,
            y_train=y_train,
            fit_kwargs=SQFA_FIT_KWARGS,
            run_label=f"sqfa seed={seed}",
        )
        filters = model.filters.detach().to(dtype=torch.float32).cpu().numpy()
        save_artifacts(stem, filters, fit_time)
    else:
        train_dtype = torch.float32

    qda_acc = qda_accuracy(
        x_train.to(dtype=train_dtype),
        y_train,
        x_test.to(dtype=train_dtype),
        y_test,
        filters,
        eval_qda_reg=EVAL_QDA_REG,
    ).item()
    return fit_time, qda_acc


def add_repeat_summaries(row, prefix, n_reps):
    times = [row[f"{prefix}_time_{rep + 1}"] for rep in range(n_reps)]
    qdas = [row[f"{prefix}_qda_{rep + 1}"] for rep in range(n_reps)]
    row[f"{prefix}_time_median"] = float(np.median(times))
    row[f"{prefix}_time_max"] = float(np.max(times))
    row[f"{prefix}_qda_median"] = float(np.median(qdas))
    row[f"{prefix}_qda_max"] = float(np.max(qdas))


#############################
#
# LOAD DATA
#
#############################

trainset = torchvision.datasets.SVHN(root=DATA_DIR, split="train", download=True)
testset = torchvision.datasets.SVHN(root=DATA_DIR, split="test", download=True)

n_samples, n_channels, n_row, n_col = trainset.data.shape

x_train = torch.as_tensor(trainset.data).float().mean(dim=1).reshape(-1, n_row * n_col)
y_train = torch.as_tensor(trainset.labels, dtype=torch.long)
x_test = torch.as_tensor(testset.data).float().mean(dim=1).reshape(-1, n_row * n_col)
y_test = torch.as_tensor(testset.labels, dtype=torch.long)

x_train, x_test = scale_and_center(x_train, x_test)
x_train_reg, y_train_reg, x_val, y_val = train_val_split_fixed(x_train, y_train)

print("Data loaded")
print("Train shape:", tuple(x_train.shape))
print("Validation shape:", tuple(x_val.shape))
print("Test shape:", tuple(x_test.shape))


#############################
#
# SQFA NOISE FROM SVHN REVIEWS SETUP
#
#############################

sqfa_noise = load_or_validate_noise(
    noise_path=SQFA_NOISE_PATH,
    model_factory=lambda noise: sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=noise,
    ),
    x_train=x_train_reg,
    y_train=y_train_reg,
    x_val=x_val,
    y_val=y_val,
    noise_vals=NOISE_VALS,
    fit_kwargs=SQFA_FIT_KWARGS,
    run_label="svhn_wdatest sqfa validation",
)

print("SQFA noise:", sqfa_noise)


#############################
#
# EXPERIMENT 1
#
#############################

exp1_rows = []

for rep in range(EXP1_N_REPS):
    current_seed = SEED + rep
    fit_time, qda_acc = run_wda_cached(
        pca_dim=EXP1_PCA_DIM,
        samples_per_class=EXP1_SAMPLES_PER_CLASS,
        seed=current_seed,
        experiment_name="exp1",
    )
    exp1_rows.append(
        {
            "rep": rep + 1,
            "seed": current_seed,
            "pca_dim": EXP1_PCA_DIM,
            "samples_per_class": EXP1_SAMPLES_PER_CLASS,
            "reg": WDA_REG,
            "sinkhorn_method": WDA_SINKHORN_METHOD,
            "fit_time": fit_time,
            "qda_accuracy": qda_acc,
        }
    )
    print(
        f"Experiment 1, rep={rep + 1}, seed={current_seed}, "
        f"time={fit_time:.2f}s, qda={100 * qda_acc:.2f}%"
    )

save_csv(exp1_rows, EXP1_CSV)
print("Saved:", EXP1_CSV)


#############################
#
# EXPERIMENT 2
#
#############################

exp2_rows = []

for combo_idx, pca_dim in enumerate(EXP2_PCA_DIMS):
    for samples_per_class in EXP2_SAMPLES_PER_CLASS:
        row = {
            "pca_dim": pca_dim,
            "samples_per_class": samples_per_class,
            "wda_reg": WDA_REG,
            "sqfa_noise": sqfa_noise,
        }

        for rep in range(EXP2_N_REPS):
            wda_seed = SEED + 100 * combo_idx + 10 * samples_per_class + rep
            wda_time, wda_qda = run_wda_cached(
                pca_dim=pca_dim,
                samples_per_class=samples_per_class,
                seed=wda_seed,
                experiment_name="exp2",
            )
            row[f"wda_seed_{rep + 1}"] = wda_seed
            row[f"wda_time_{rep + 1}"] = wda_time
            row[f"wda_qda_{rep + 1}"] = wda_qda

        for rep in range(EXP2_N_REPS):
            sqfa_seed = SEED + 1000 * combo_idx + 100 * samples_per_class + rep
            sqfa_time, sqfa_qda = run_sqfa_cached(
                seed=sqfa_seed,
                pca_dim=pca_dim,
                samples_per_class=samples_per_class,
                experiment_name="exp2",
            )
            row[f"sqfa_seed_{rep + 1}"] = sqfa_seed
            row[f"sqfa_time_{rep + 1}"] = sqfa_time
            row[f"sqfa_qda_{rep + 1}"] = sqfa_qda

        add_repeat_summaries(row, "wda", EXP2_N_REPS)
        add_repeat_summaries(row, "sqfa", EXP2_N_REPS)
        exp2_rows.append(row)
        print(
            f"Experiment 2, pca_dim={pca_dim}, samples_per_class={samples_per_class}, "
            f"wda_median={100 * row['wda_qda_median']:.2f}%, "
            f"wda_max={100 * row['wda_qda_max']:.2f}%, "
            f"sqfa_median={100 * row['sqfa_qda_median']:.2f}%, "
            f"sqfa_max={100 * row['sqfa_qda_max']:.2f}%"
        )

save_csv(exp2_rows, EXP2_CSV)
print("Saved:", EXP2_CSV)
