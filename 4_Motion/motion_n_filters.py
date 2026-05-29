import os
import sys
import time

import numpy as np
import torch
from metric_learn import LMNN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sqfa
from functions import load_data, normalize_stim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, "ama_dir"))
from ama_model import AMAGauss
from optim import fit as fit_ama

sys.path.append("..")  # Add parent directory to path
from pkg_utils import (
    artifact_path,
    balanced_subset_indices,
    export_metric_results_csv,
    fit_wda,
    has_saved_artifacts,
    knn_accuracy,
    load_cached_filters,
    load_or_validate_lda_shrinkage,
    load_or_validate_wda_reg,
    plot_metric_results,
    qda_accuracy,
    save_training_artifacts,
    summarize_metric_results,
    SupervisedPCA,
    train_lfda_repeated,
    train_metric_learn_repeated,
    train_sqfa_repeated,
    train_val_split,
    validate_lfda_k,
)


FILTER_RANGE = (2, 4, 6, 8)
RESPONSE_NOISE = 0.001
C50 = 0.8
N_REPS = 10
FILTERS_DIR = os.path.join(SCRIPT_DIR, "filters_review")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures_review")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_review")
SQFA_FIT_KWARGS = {
    "max_epochs": 300,
    "show_progress": False,
    "estimator": "empirical",
    "pairwise": False,
}
AMA_FIT_KWARGS = {
    "max_epochs": 100,
    "lr": 0.2,
    "show_progress": False,
    "pairwise": True,
}
LDA_SHRINKAGE_VALS = np.array([0.05, 0.1, 0.2, 0.4, 0.8], dtype=float)
LFDA_K_VALS = torch.tensor([3, 5, 9, 17])
LFDA_PCA_DIM = 100
LFDA_EMBEDDING_TYPE = "orthonormalized"
LMNN_PCA_DIM = 200
KNN_N_NEIGHBORS = 7
WASSERSTEIN_DTYPES = (torch.float64,)
WDA_REG_VALS = np.array([1.0, 5.0, 10.0, 20.0, 50.0], dtype=float)
WDA_PCA_DIM = 50
WDA_SAMPLES_PER_CLASS = 60
WDA_SINKHORN_ITERS = 10
WDA_MAXITER = 100
WDA_SINKHORN_METHOD = "sinkhorn_log"
WDA_DTYPE = torch.float32
torch.manual_seed(2)

os.makedirs(FILTERS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def ensure_fixed_scalar(path, value, description):
    """Persist a fixed scalar artifact, overwriting stale cached values."""
    fixed_value = float(value)
    if os.path.exists(path):
        cached_value = float(np.load(path).item())
        if np.isclose(cached_value, fixed_value):
            return fixed_value
        print(
            f"Overwriting cached {description}: "
            f"{cached_value} -> {fixed_value}"
        )
    np.save(path, np.asarray(fixed_value))
    return fixed_value


def train_ama_repeated(x_train, y_train, n_filters, n_reps, fit_kwargs, run_label):
    """Train AMA repeatedly and collect filters and times."""
    x_train_ch = x_train.unsqueeze(1)
    filters = []
    times = []

    for rep in range(n_reps):
        torch.manual_seed(302 + rep)
        ama = AMAGauss(
            stimuli=x_train_ch,
            labels=y_train,
            n_filters=n_filters,
            response_noise=RESPONSE_NOISE,
        )
        start = time.time()
        fit_ama(
            model=ama,
            stimuli=x_train_ch,
            labels=y_train,
            **fit_kwargs,
        )
        elapsed = time.time() - start
        filters.append(
            np.asarray(ama.filters.squeeze().detach().cpu().numpy(), dtype=np.float32)
        )
        times.append(elapsed)
        print(f"{run_label}, rep={rep}, elapsed={elapsed:.3f}s")

    return np.asarray(filters), np.asarray(times)


#############################
#
# LOAD AND PROCESS DATA
#
#############################

x_train_raw, y_train, _ = load_data("train")
x_test_raw, y_test, _ = load_data("test")

x_train = normalize_stim(x_train_raw, C50).to(dtype=torch.float32)
x_test = normalize_stim(x_test_raw, C50).to(dtype=torch.float32)
y_train = y_train.to(dtype=torch.long)
y_test = y_test.to(dtype=torch.long)

x_train_reg, y_train_reg, x_val, y_val = train_val_split(
    x_train,
    y_train,
    val_size=0.15,
)


#############################
#
# TRAIN MODELS
#
#############################

for n_filters in FILTER_RANGE:
    print(f"Training models with n_filters={n_filters}")

    sqfa_filter_path = artifact_path(FILTERS_DIR, "sqfa", "filters", n_filters=n_filters)
    sqfa_time_path = artifact_path(FILTERS_DIR, "sqfa", "time", n_filters=n_filters)
    sqfa_noise_path = artifact_path(FILTERS_DIR, "sqfa", "noise", n_filters=n_filters)
    smsqfa_filter_path = artifact_path(FILTERS_DIR, "smsqfa", "filters", n_filters=n_filters)
    smsqfa_time_path = artifact_path(FILTERS_DIR, "smsqfa", "time", n_filters=n_filters)
    smsqfa_noise_path = artifact_path(FILTERS_DIR, "smsqfa", "noise", n_filters=n_filters)
    pca_filter_path = artifact_path(FILTERS_DIR, "pca", "filters", n_filters=n_filters)
    pca_time_path = artifact_path(FILTERS_DIR, "pca", "time", n_filters=n_filters)
    spca_filter_path = artifact_path(FILTERS_DIR, "spca", "filters", n_filters=n_filters)
    spca_time_path = artifact_path(FILTERS_DIR, "spca", "time", n_filters=n_filters)
    bhattacharyya_filter_path = artifact_path(
        FILTERS_DIR, "bhattacharyya", "filters", n_filters=n_filters
    )
    bhattacharyya_time_path = artifact_path(
        FILTERS_DIR, "bhattacharyya", "time", n_filters=n_filters
    )
    bhattacharyya_noise_path = artifact_path(
        FILTERS_DIR, "bhattacharyya", "noise", n_filters=n_filters
    )
    hellinger_filter_path = artifact_path(
        FILTERS_DIR, "hellinger", "filters", n_filters=n_filters
    )
    hellinger_time_path = artifact_path(
        FILTERS_DIR, "hellinger", "time", n_filters=n_filters
    )
    hellinger_noise_path = artifact_path(
        FILTERS_DIR, "hellinger", "noise", n_filters=n_filters
    )
    wasserstein_filter_path = artifact_path(
        FILTERS_DIR, "wasserstein", "filters", n_filters=n_filters
    )
    wasserstein_time_path = artifact_path(
        FILTERS_DIR, "wasserstein", "time", n_filters=n_filters
    )
    wasserstein_noise_path = artifact_path(
        FILTERS_DIR, "wasserstein", "noise", n_filters=n_filters
    )
    jeffreys_filter_path = artifact_path(
        FILTERS_DIR, "jeffreys", "filters", n_filters=n_filters
    )
    jeffreys_time_path = artifact_path(
        FILTERS_DIR, "jeffreys", "time", n_filters=n_filters
    )
    jeffreys_noise_path = artifact_path(
        FILTERS_DIR, "jeffreys", "noise", n_filters=n_filters
    )
    ama_filter_path = artifact_path(FILTERS_DIR, "ama", "filters", n_filters=n_filters)
    ama_time_path = artifact_path(FILTERS_DIR, "ama", "time", n_filters=n_filters)
    wda_filter_path = artifact_path(FILTERS_DIR, "wda", "filters", n_filters=n_filters)
    wda_time_path = artifact_path(FILTERS_DIR, "wda", "time", n_filters=n_filters)
    wda_reg_path = artifact_path(FILTERS_DIR, "wda", "reg", n_filters=n_filters)
    lda_filter_path = artifact_path(FILTERS_DIR, "lda", "filters", n_filters=n_filters)
    lda_time_path = artifact_path(FILTERS_DIR, "lda", "time", n_filters=n_filters)
    lda_shrinkage_path = artifact_path(
        FILTERS_DIR,
        "lda",
        "shrinkage",
        n_filters=n_filters,
    )

    sqfa_noise = ensure_fixed_scalar(
        sqfa_noise_path,
        RESPONSE_NOISE,
        f"sqfa noise for n_filters={n_filters}",
    )
    smsqfa_noise = ensure_fixed_scalar(
        smsqfa_noise_path,
        RESPONSE_NOISE,
        f"smsqfa noise for n_filters={n_filters}",
    )
    bhattacharyya_noise = ensure_fixed_scalar(
        bhattacharyya_noise_path,
        RESPONSE_NOISE,
        f"bhattacharyya noise for n_filters={n_filters}",
    )
    hellinger_noise = ensure_fixed_scalar(
        hellinger_noise_path,
        RESPONSE_NOISE,
        f"hellinger noise for n_filters={n_filters}",
    )
    wasserstein_noise = ensure_fixed_scalar(
        wasserstein_noise_path,
        RESPONSE_NOISE,
        f"wasserstein noise for n_filters={n_filters}",
    )
    jeffreys_noise = ensure_fixed_scalar(
        jeffreys_noise_path,
        RESPONSE_NOISE,
        f"jeffreys noise for n_filters={n_filters}",
    )

    # ------------------------------
    # Train SQFA
    # ------------------------------
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

    # ------------------------------
    # Train smSQFA
    # ------------------------------
    if not has_saved_artifacts(smsqfa_filter_path, smsqfa_time_path):
        smsqfa_filters, smsqfa_times = train_sqfa_repeated(
            model_factory=lambda: sqfa.model.SecondMomentsSQFA(
                n_dim=x_train.shape[1],
                n_filters=n_filters,
                feature_noise=smsqfa_noise,
            ),
            x_train=x_train,
            y_train=y_train,
            n_reps=N_REPS,
            fit_kwargs=SQFA_FIT_KWARGS,
            run_label=f"smsqfa training for n_filters={n_filters}",
        )
        save_training_artifacts(
            smsqfa_filter_path,
            smsqfa_time_path,
            smsqfa_filters,
            smsqfa_times,
        )
    else:
        load_cached_filters(
            smsqfa_filter_path,
            description=f"smsqfa filters for n_filters={n_filters}",
        )

    # ------------------------------
    # Train PCA
    # ------------------------------
    if not has_saved_artifacts(pca_filter_path, pca_time_path):
        pca = PCA(n_components=n_filters)
        start = time.time()
        pca.fit(x_train)
        pca_time = time.time() - start
        save_training_artifacts(
            pca_filter_path,
            pca_time_path,
            pca.components_,
            pca_time,
        )
    else:
        load_cached_filters(
            pca_filter_path,
            description=f"pca filters for n_filters={n_filters}",
        )

    # ------------------------------
    # Train Supervised PCA
    # ------------------------------
    if not has_saved_artifacts(spca_filter_path, spca_time_path):
        x_subsampled = x_train[::5]
        y_subsampled = y_train[::5]
        spca = SupervisedPCA(n_components=n_filters, label_kernel="delta")
        start = time.time()
        spca.fit(x_subsampled, y_subsampled)
        spca_time = time.time() - start
        save_training_artifacts(
            spca_filter_path,
            spca_time_path,
            spca.components_,
            spca_time,
        )
    else:
        load_cached_filters(
            spca_filter_path,
            description=f"spca filters for n_filters={n_filters}",
        )

    # ------------------------------
    # Train SQFA-B
    # ------------------------------
    if not has_saved_artifacts(bhattacharyya_filter_path, bhattacharyya_time_path):
        bhattacharyya_filters, bhattacharyya_times = train_sqfa_repeated(
            model_factory=lambda: sqfa.model.SQFA(
                n_dim=x_train.shape[1],
                n_filters=n_filters,
                feature_noise=bhattacharyya_noise,
                distance_fun=sqfa.distances.bhattacharyya,
            ),
            x_train=x_train,
            y_train=y_train,
            n_reps=N_REPS,
            fit_kwargs=SQFA_FIT_KWARGS,
            run_label=f"bhattacharyya training for n_filters={n_filters}",
        )
        save_training_artifacts(
            bhattacharyya_filter_path,
            bhattacharyya_time_path,
            bhattacharyya_filters,
            bhattacharyya_times,
        )
    else:
        load_cached_filters(
            bhattacharyya_filter_path,
            description=f"bhattacharyya filters for n_filters={n_filters}",
        )

    # ------------------------------
    # Train SQFA-H
    # ------------------------------
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

    # ------------------------------
    # Train SQFA-W
    # ------------------------------
    if not has_saved_artifacts(wasserstein_filter_path, wasserstein_time_path):
        wasserstein_filters, wasserstein_times = train_sqfa_repeated(
            model_factory=lambda: sqfa.model.SQFA(
                n_dim=x_train.shape[1],
                n_filters=n_filters,
                feature_noise=wasserstein_noise,
                distance_fun=sqfa.distances.wasserstein,
                constraint="orthogonal",
            ),
            x_train=x_train,
            y_train=y_train,
            n_reps=N_REPS,
            fit_kwargs=SQFA_FIT_KWARGS,
            dtypes=WASSERSTEIN_DTYPES,
            run_label=f"wasserstein training for n_filters={n_filters}",
        )
        save_training_artifacts(
            wasserstein_filter_path,
            wasserstein_time_path,
            wasserstein_filters,
            wasserstein_times,
        )
    else:
        load_cached_filters(
            wasserstein_filter_path,
            description=f"wasserstein filters for n_filters={n_filters}",
        )

    # ------------------------------
    # Train SQFA-J
    # ------------------------------
    if not has_saved_artifacts(jeffreys_filter_path, jeffreys_time_path):
        jeffreys_filters, jeffreys_times = train_sqfa_repeated(
            model_factory=lambda: sqfa.model.SQFA(
                n_dim=x_train.shape[1],
                n_filters=n_filters,
                feature_noise=jeffreys_noise,
                distance_fun=sqfa.distances.jeffreys,
            ),
            x_train=x_train,
            y_train=y_train,
            n_reps=N_REPS,
            fit_kwargs=SQFA_FIT_KWARGS,
            run_label=f"jeffreys training for n_filters={n_filters}",
        )
        save_training_artifacts(
            jeffreys_filter_path,
            jeffreys_time_path,
            jeffreys_filters,
            jeffreys_times,
        )
    else:
        load_cached_filters(
            jeffreys_filter_path,
            description=f"jeffreys filters for n_filters={n_filters}",
        )

    # ------------------------------
    # Train AMA
    # ------------------------------
    if not has_saved_artifacts(ama_filter_path, ama_time_path):
        ama_filters, ama_times = train_ama_repeated(
            x_train=x_train,
            y_train=y_train,
            n_filters=n_filters,
            n_reps=N_REPS,
            fit_kwargs=AMA_FIT_KWARGS,
            run_label=f"ama training for n_filters={n_filters}",
        )
        save_training_artifacts(
            ama_filter_path,
            ama_time_path,
            ama_filters,
            ama_times,
        )
    else:
        load_cached_filters(
            ama_filter_path,
            description=f"ama filters for n_filters={n_filters}",
        )

    # ------------------------------
    # Train LDA
    # ------------------------------
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

        lda_filters = lda.scalings_.T
        if lda_filters.shape[0] < n_filters:
            print(
                f"Skipping LDA save for n_filters={n_filters}: "
                f"only {lda_filters.shape[0]} filters are available."
            )
        else:
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

    # ------------------------------
    # Train LFDA
    # ------------------------------
    lfda_filter_path = artifact_path(FILTERS_DIR, "lfda", "filters", n_filters=n_filters)
    lfda_time_path = artifact_path(FILTERS_DIR, "lfda", "time", n_filters=n_filters)
    if not has_saved_artifacts(lfda_filter_path, lfda_time_path):
        lfda_accs = validate_lfda_k(
            x_train=x_train,
            y_train=y_train,
            k_vals=LFDA_K_VALS,
            n_filters=n_filters,
            n_pca_components=LFDA_PCA_DIM,
            eval_qda_reg=1.0e-5,
            val_size=0.15,
            embedding_type=LFDA_EMBEDDING_TYPE,
        )
        best_k = int(LFDA_K_VALS[torch.argmax(lfda_accs)].item())
        lfda_filters, lfda_times = train_lfda_repeated(
            x_train=x_train,
            y_train=y_train,
            n_reps=N_REPS,
            n_filters=n_filters,
            k=best_k,
            n_pca_components=LFDA_PCA_DIM,
            embedding_type=LFDA_EMBEDDING_TYPE,
            run_label=f"lfda training for n_filters={n_filters}",
        )
        save_training_artifacts(
            lfda_filter_path,
            lfda_time_path,
            lfda_filters,
            lfda_times,
        )
    else:
        load_cached_filters(
            lfda_filter_path,
            description=f"lfda filters for n_filters={n_filters}",
        )

    # ------------------------------
    # Train WDA
    # ------------------------------
    wda_reg = load_or_validate_wda_reg(
        reg_path=wda_reg_path,
        x_train=x_train_reg,
        y_train=y_train_reg,
        x_val=x_val,
        y_val=y_val,
        reg_vals=WDA_REG_VALS,
        n_filters=n_filters,
        n_pca_components=WDA_PCA_DIM,
        samples_per_class=WDA_SAMPLES_PER_CLASS,
        eval_qda_reg=1.0e-5,
        seed=2,
        sinkhorn_iters=WDA_SINKHORN_ITERS,
        sinkhorn_method=WDA_SINKHORN_METHOD,
        maxiter=WDA_MAXITER,
        dtype=WDA_DTYPE,
    )
    if not has_saved_artifacts(wda_filter_path, wda_time_path):
        wda_filters = []
        wda_times = []
        for rep in range(N_REPS):
            current_seed = 2 + rep
            current_filters, current_time = fit_wda(
                x_train=x_train,
                y_train=y_train,
                n_filters=n_filters,
                n_pca_components=WDA_PCA_DIM,
                reg=wda_reg,
                samples_per_class=WDA_SAMPLES_PER_CLASS,
                seed=current_seed,
                sinkhorn_iters=WDA_SINKHORN_ITERS,
                sinkhorn_method=WDA_SINKHORN_METHOD,
                maxiter=WDA_MAXITER,
                dtype=WDA_DTYPE,
            )
            wda_filters.append(current_filters)
            wda_times.append(current_time)
            print(
                f"wda training for n_filters={n_filters}, rep={rep}, "
                f"elapsed={current_time:.3f}s"
            )
        save_training_artifacts(
            wda_filter_path,
            wda_time_path,
            np.asarray(wda_filters),
            np.asarray(wda_times),
        )
    else:
        load_cached_filters(
            wda_filter_path,
            description=f"wda filters for n_filters={n_filters}",
        )

    # ------------------------------
    # Train LMNN
    # ------------------------------
    lmnn_filter_path = artifact_path(FILTERS_DIR, "lmnn", "filters", n_filters=n_filters)
    lmnn_time_path = artifact_path(FILTERS_DIR, "lmnn", "time", n_filters=n_filters)
    if not has_saved_artifacts(lmnn_filter_path, lmnn_time_path):
        pca_subsample = PCA(n_components=LMNN_PCA_DIM)
        x_train_np = x_train.detach().cpu().numpy()
        pca_subsample.fit(x_train_np)
        x_transformed = pca_subsample.transform(x_train_np)
        lmnn_subset_idx = balanced_subset_indices(
            y_train.detach().cpu().numpy(),
            samples_per_class=WDA_SAMPLES_PER_CLASS,
            seed=2,
        )
        x_transformed_sub = x_transformed[lmnn_subset_idx]
        y_train_sub = y_train[lmnn_subset_idx]

        lmnn_filters, lmnn_times = train_metric_learn_repeated(
            estimator_factory=lambda: LMNN(
                n_neighbors=3,
                learn_rate=2e-6,
                n_components=n_filters,
                init="pca",
                verbose=True,
                max_iter=1000,
                convergence_tol=1.0,
            ),
            x_train=x_train,
            y_train=y_train_sub,
            n_reps=N_REPS,
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


#############################
#
# PLOT QDA ACCURACIES VS N_FILTERS
#
#############################

model_specs = [
    ("SQFA", "sqfa"),
    ("smSQFA", "smsqfa"),
    ("SQFA-H", "hellinger"),
    ("SQFA-B", "bhattacharyya"),
    ("SQFA-W", "wasserstein"),
    ("SQFA-J", "jeffreys"),
    ("AMA", "ama"),
    ("LDA", "lda"),
    ("SPCA", "spca"),
    ("LFDA", "lfda"),
    ("WDA", "wda"),
    ("LMNN", "lmnn"),
    ("PCA", "pca"),
]

qda_results = summarize_metric_results(
    model_specs,
    FILTER_RANGE,
    FILTERS_DIR,
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
    os.path.join(RESULTS_DIR, "motion_n_filters_qda_review.csv"),
)
plot_metric_results(
    model_specs,
    FILTER_RANGE,
    qda_results,
    "QDA Accuracy (%)",
    os.path.join(FIGURES_DIR, "motion_accuracies_n_filters_review.pdf"),
    ylim=(0, 100),
)


#############################
#
# PLOT KNN ACCURACIES VS N_FILTERS
#
#############################

knn_results = summarize_metric_results(
    model_specs,
    FILTER_RANGE,
    FILTERS_DIR,
    lambda filt: knn_accuracy(
        x_train,
        y_train,
        x_test,
        y_test,
        filt,
        n_neighbors=KNN_N_NEIGHBORS,
    ),
)
export_metric_results_csv(
    knn_results,
    "knn",
    os.path.join(RESULTS_DIR, "motion_n_filters_knn_review.csv"),
)
plot_metric_results(
    model_specs,
    FILTER_RANGE,
    knn_results,
    "KNN Accuracy (%)",
    os.path.join(FIGURES_DIR, "motion_accuracies_n_filters_knn_review.pdf"),
    ylim=(0, 100),
)
