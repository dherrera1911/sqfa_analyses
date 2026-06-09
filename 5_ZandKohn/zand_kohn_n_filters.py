import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from metric_learn import LMNN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sqfa
from zand_kohn_utils import (
    CONDITIONS,
    CONSTRAINT,
    EVAL_QDA_REG,
    LDA_SHRINKAGE_VALS,
    NOISE_VALS,
    N_SPLITS,
    SQFA_DTYPE,
    SQFA_FIT_KWARGS,
    load_processed_sessions,
    normalized_split,
    summarize_score_rows,
    write_csv,
    zand_artifact_path,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append("..")

from pkg_utils import (
    balanced_subset_indices,
    export_metric_results_csv,
    fit_wda,
    has_saved_artifacts,
    knn_accuracy,
    load_cached_filters,
    load_or_validate_lda_shrinkage,
    load_or_validate_noise,
    load_or_validate_wda_reg,
    max_supported_filter_count,
    plot_metric_results,
    qda_accuracy,
    save_training_artifacts,
    SupervisedPCA,
    train_lfda_repeated,
    train_metric_learn_repeated,
    train_sqfa_repeated,
    validate_lfda_k,
)


FILTER_RANGE = (2, 4, 6)
AREA_KEY = "V1"
CONDITION = CONDITIONS[0]["condition"]
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, "processed_data")
FILTERS_DIR = os.path.join(SCRIPT_DIR, "filters_review")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures_review")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_review")
N_REPS = 1
EXPENSIVE_METHOD_SPLITS = (0,)
EVAL_QDA_REG = 1.0e-3
LFDA_K_VALS = torch.tensor([3, 5, 9, 17])
LFDA_PCA_DIM = 50
LFDA_EMBEDDING_TYPE = "orthonormalized"
LMNN_PCA_DIM = 50
KNN_N_NEIGHBORS = 5
WDA_REG_VALS = np.array([0.01, 0.05, 0.1, 0.2, 0.5], dtype=float)
WDA_PCA_DIM = 50
WDA_SAMPLES_PER_CLASS = 250
WDA_SINKHORN_ITERS = 10
WDA_MAXITER = 100
WDA_SINKHORN_METHOD = "sinkhorn_log"
WDA_DTYPE = torch.float64

torch.manual_seed(6)
np.random.seed(6)
os.makedirs(FILTERS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


MODEL_SPECS = [
    ("SQFA", "sqfa"),
    ("smSQFA", "smsqfa"),
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


def iter_filter_sets(filter_bank):
    filters = np.asarray(filter_bank)
    if filters.ndim == 2:
        yield 0, filters
    else:
        for rep_i, current_filters in enumerate(filters):
            yield rep_i, current_filters


def score_saved_filters(rows, model_name, model_key, filter_path, metadata, x_train, y_train, x_test, y_test):
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
        knn_score = knn_accuracy(
            x_train,
            y_train,
            x_test,
            y_test,
            current_filters,
            n_neighbors=KNN_N_NEIGHBORS,
        ).item()
        rows.append({
            **metadata,
            "rep": rep_i,
            "model_name": model_name,
            "model_key": model_key,
            "qda_accuracy": qda_score,
            "knn_accuracy": knn_score,
        })


def train_sqfa_method(model_key, model_factory, x_train, y_train, x_val, y_val, metadata):
    filter_path = zand_artifact_path(
        FILTERS_DIR,
        model_key,
        "filters",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    time_path = zand_artifact_path(
        FILTERS_DIR,
        model_key,
        "time",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    noise_path = zand_artifact_path(
        FILTERS_DIR,
        model_key,
        "noise",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    n_filters = metadata["n_filters"]
    n_dim = x_train.shape[1]
    noise = load_or_validate_noise(
        noise_path=noise_path,
        model_factory=lambda current_noise: model_factory(n_dim, n_filters, current_noise),
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        noise_vals=NOISE_VALS,
        fit_kwargs=SQFA_FIT_KWARGS,
        dtypes=(SQFA_DTYPE,),
        run_label=(
            f"{model_key} validation, session={metadata['session']}, "
            f"split={metadata['split']}, n_filters={n_filters}"
        ),
        eval_qda_reg=EVAL_QDA_REG,
    )

    if not has_saved_artifacts(filter_path, time_path):
        filters, times = train_sqfa_repeated(
            model_factory=lambda: model_factory(n_dim, n_filters, noise),
            x_train=x_train,
            y_train=y_train,
            n_reps=N_REPS,
            fit_kwargs=SQFA_FIT_KWARGS,
            dtypes=(SQFA_DTYPE,),
            run_label=(
                f"{model_key} training, session={metadata['session']}, "
                f"split={metadata['split']}, n_filters={n_filters}"
            ),
        )
        save_training_artifacts(filter_path, time_path, filters, times)
    else:
        load_cached_filters(filter_path, description=f"{model_key} filters")

    return filter_path


def train_pca(x_train, metadata):
    filter_path = zand_artifact_path(
        FILTERS_DIR,
        "pca",
        "filters",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    time_path = zand_artifact_path(
        FILTERS_DIR,
        "pca",
        "time",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    if not has_saved_artifacts(filter_path, time_path):
        pca = PCA(n_components=metadata["n_filters"])
        start = time.time()
        pca.fit(x_train)
        save_training_artifacts(filter_path, time_path, pca.components_, time.time() - start)
    else:
        load_cached_filters(filter_path, description="pca filters")
    return filter_path


def train_spca(x_train, y_train, metadata):
    filter_path = zand_artifact_path(
        FILTERS_DIR,
        "spca",
        "filters",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    time_path = zand_artifact_path(
        FILTERS_DIR,
        "spca",
        "time",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    if not has_saved_artifacts(filter_path, time_path):
        spca = SupervisedPCA(n_components=metadata["n_filters"], label_kernel="delta")
        start = time.time()
        spca.fit(x_train, y_train)
        save_training_artifacts(filter_path, time_path, spca.components_, time.time() - start)
    else:
        load_cached_filters(filter_path, description="spca filters")
    return filter_path


def train_lda(x_train, y_train, x_val, y_val, metadata):
    filter_path = zand_artifact_path(
        FILTERS_DIR,
        "lda",
        "filters",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    time_path = zand_artifact_path(
        FILTERS_DIR,
        "lda",
        "time",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    shrinkage_path = zand_artifact_path(
        FILTERS_DIR,
        "lda",
        "shrinkage",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    shrinkage = load_or_validate_lda_shrinkage(
        shrinkage_path=shrinkage_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        shrinkage_vals=LDA_SHRINKAGE_VALS,
        n_filters=metadata["n_filters"],
        eval_qda_reg=EVAL_QDA_REG,
    )
    if not has_saved_artifacts(filter_path, time_path):
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrinkage)
        start = time.time()
        lda.fit(x_train, y_train)
        lda_filters = lda.scalings_.T[:metadata["n_filters"]]
        save_training_artifacts(filter_path, time_path, lda_filters, time.time() - start)
    else:
        load_cached_filters(filter_path, description="lda filters")
    return filter_path


def train_lfda(x_train, y_train, metadata):
    filter_path = zand_artifact_path(
        FILTERS_DIR,
        "lfda",
        "filters",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    time_path = zand_artifact_path(
        FILTERS_DIR,
        "lfda",
        "time",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    if not has_saved_artifacts(filter_path, time_path):
        n_pca_components = min(LFDA_PCA_DIM, x_train.shape[1])
        lfda_accs = validate_lfda_k(
            x_train=x_train,
            y_train=y_train,
            k_vals=LFDA_K_VALS,
            n_filters=metadata["n_filters"],
            n_pca_components=n_pca_components,
            eval_qda_reg=EVAL_QDA_REG,
            val_size=0.15,
            embedding_type=LFDA_EMBEDDING_TYPE,
        )
        best_k = int(LFDA_K_VALS[torch.argmax(lfda_accs)].item())
        filters, times = train_lfda_repeated(
            x_train=x_train,
            y_train=y_train,
            n_reps=N_REPS,
            n_filters=metadata["n_filters"],
            k=best_k,
            n_pca_components=n_pca_components,
            embedding_type=LFDA_EMBEDDING_TYPE,
            run_label=(
                f"lfda training, session={metadata['session']}, "
                f"split={metadata['split']}, n_filters={metadata['n_filters']}"
            ),
        )
        save_training_artifacts(filter_path, time_path, filters, times)
    else:
        load_cached_filters(filter_path, description="lfda filters")
    return filter_path


def train_wda(x_train, y_train, x_val, y_val, metadata):
    filter_path = zand_artifact_path(
        FILTERS_DIR,
        "wda",
        "filters",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    time_path = zand_artifact_path(
        FILTERS_DIR,
        "wda",
        "time",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    reg_path = zand_artifact_path(
        FILTERS_DIR,
        "wda",
        "reg",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    n_pca_components = min(WDA_PCA_DIM, x_train.shape[1])
    reg = load_or_validate_wda_reg(
        reg_path=reg_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        reg_vals=WDA_REG_VALS,
        n_filters=metadata["n_filters"],
        n_pca_components=n_pca_components,
        samples_per_class=WDA_SAMPLES_PER_CLASS,
        eval_qda_reg=EVAL_QDA_REG,
        seed=6 + metadata["session"] * 100 + metadata["split"],
        sinkhorn_iters=WDA_SINKHORN_ITERS,
        sinkhorn_method=WDA_SINKHORN_METHOD,
        maxiter=WDA_MAXITER,
        dtype=WDA_DTYPE,
    )
    if not has_saved_artifacts(filter_path, time_path):
        filters = []
        times = []
        for rep in range(N_REPS):
            current_filters, current_time = fit_wda(
                x_train=x_train,
                y_train=y_train,
                n_filters=metadata["n_filters"],
                n_pca_components=n_pca_components,
                reg=reg,
                samples_per_class=WDA_SAMPLES_PER_CLASS,
                seed=6 + metadata["session"] * 100 + metadata["split"] + rep,
                sinkhorn_iters=WDA_SINKHORN_ITERS,
                sinkhorn_method=WDA_SINKHORN_METHOD,
                maxiter=WDA_MAXITER,
                dtype=WDA_DTYPE,
            )
            filters.append(current_filters)
            times.append(current_time)
            print(
                f"wda training, session={metadata['session']}, "
                f"split={metadata['split']}, n_filters={metadata['n_filters']}, "
                f"rep={rep}, elapsed={current_time:.3f}s"
            )
        save_training_artifacts(filter_path, time_path, np.asarray(filters), np.asarray(times))
    else:
        load_cached_filters(filter_path, description="wda filters")
    return filter_path


def train_lmnn(x_train, y_train, metadata):
    filter_path = zand_artifact_path(
        FILTERS_DIR,
        "lmnn",
        "filters",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    time_path = zand_artifact_path(
        FILTERS_DIR,
        "lmnn",
        "time",
        metadata["session"],
        metadata["split"],
        metadata["n_filters"],
    )
    if not has_saved_artifacts(filter_path, time_path):
        n_pca_components = min(LMNN_PCA_DIM, x_train.shape[1])
        pca_subsample = PCA(n_components=n_pca_components)
        x_train_np = x_train.detach().cpu().numpy()
        pca_subsample.fit(x_train_np)
        x_transformed = pca_subsample.transform(x_train_np)
        subset_idx = balanced_subset_indices(
            y_train.detach().cpu().numpy(),
            samples_per_class=WDA_SAMPLES_PER_CLASS,
            seed=6 + metadata["session"] * 100 + metadata["split"],
        )
        x_transformed_sub = x_transformed[subset_idx]
        y_train_sub = y_train[subset_idx]
        filters, times = train_metric_learn_repeated(
            estimator_factory=lambda: LMNN(
                n_neighbors=3,
                learn_rate=1e-6,
                n_components=metadata["n_filters"],
                init="pca",
                verbose=True,
                max_iter=2000,
                convergence_tol=1.0,
            ),
            x_train=x_train,
            y_train=y_train_sub,
            n_reps=N_REPS,
            fit_x=x_transformed_sub,
            extract_filters=lambda estimator: pca_subsample.inverse_transform(
                estimator.components_
            ),
            run_label=(
                f"lmnn training, session={metadata['session']}, "
                f"split={metadata['split']}, n_filters={metadata['n_filters']}"
            ),
        )
        save_training_artifacts(filter_path, time_path, filters, times)
    else:
        load_cached_filters(filter_path, description="lmnn filters")
    return filter_path


sqfa_model_specs = [
    (
        "SQFA",
        "sqfa",
        lambda n_dim, n_filters, noise: sqfa.model.SQFA(
            n_dim=n_dim,
            n_filters=n_filters,
            feature_noise=noise,
            constraint=CONSTRAINT,
        ),
    ),
    (
        "smSQFA",
        "smsqfa",
        lambda n_dim, n_filters, noise: sqfa.model.SecondMomentsSQFA(
            n_dim=n_dim,
            n_filters=n_filters,
            feature_noise=noise,
            constraint=CONSTRAINT,
        ),
    ),
    (
        "SQFA-H",
        "hellinger",
        lambda n_dim, n_filters, noise: sqfa.model.SQFA(
            n_dim=n_dim,
            n_filters=n_filters,
            feature_noise=noise,
            distance_fun=sqfa.distances.hellinger,
            constraint=CONSTRAINT,
        ),
    ),
    (
        "SQFA-B",
        "bhattacharyya",
        lambda n_dim, n_filters, noise: sqfa.model.SQFA(
            n_dim=n_dim,
            n_filters=n_filters,
            feature_noise=noise,
            distance_fun=sqfa.distances.bhattacharyya,
            constraint=CONSTRAINT,
        ),
    ),
    (
        "SQFA-W",
        "wasserstein",
        lambda n_dim, n_filters, noise: sqfa.model.SQFA(
            n_dim=n_dim,
            n_filters=n_filters,
            feature_noise=noise,
            distance_fun=sqfa.distances.wasserstein,
            constraint=CONSTRAINT,
        ),
    ),
    (
        "SQFA-J",
        "jeffreys",
        lambda n_dim, n_filters, noise: sqfa.model.SQFA(
            n_dim=n_dim,
            n_filters=n_filters,
            feature_noise=noise,
            distance_fun=sqfa.distances.jeffreys,
            constraint=CONSTRAINT,
        ),
    ),
]


sessions = load_processed_sessions(PROCESSED_DATA_DIR, area_key=AREA_KEY, condition_name=CONDITION)
score_rows = []

for data in sessions:
    n_classes = int(torch.unique(data["y"]).numel())
    max_lda_filters = max_supported_filter_count(FILTER_RANGE, n_classes - 1)
    for split_i in range(N_SPLITS):
        print(f"Preparing session={data['session']}, split={split_i}", flush=True)
        x_train, y_train, x_val, y_val, x_test, y_test = normalized_split(data, split_i)

        for n_filters in FILTER_RANGE:
            if n_filters > x_train.shape[1]:
                continue
            print(
                f"Training models, session={data['session']}, split={split_i}, "
                f"n_filters={n_filters}",
                flush=True,
            )
            metadata = {
                "area": data["area"],
                "condition": data["condition"],
                "session": int(data["session"]),
                "file_name": data["file_name"],
                "split": split_i,
                "n_filters": n_filters,
                "n_train": int(x_train.shape[0]),
                "n_val": int(x_val.shape[0]),
                "n_test": int(x_test.shape[0]),
                "n_neurons": int(x_train.shape[1]),
                "n_classes": n_classes,
            }

            for model_name, model_key, model_factory in sqfa_model_specs:
                filter_path = train_sqfa_method(
                    model_key,
                    model_factory,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    metadata,
                )
                score_saved_filters(
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

            pca_path = train_pca(x_train, metadata)
            score_saved_filters(
                score_rows,
                "PCA",
                "pca",
                pca_path,
                metadata,
                x_train,
                y_train,
                x_test,
                y_test,
            )

            spca_path = train_spca(x_train, y_train, metadata)
            score_saved_filters(
                score_rows,
                "SPCA",
                "spca",
                spca_path,
                metadata,
                x_train,
                y_train,
                x_test,
                y_test,
            )

            if n_filters <= max_lda_filters:
                lda_path = train_lda(x_train, y_train, x_val, y_val, metadata)
                score_saved_filters(
                    score_rows,
                    "LDA",
                    "lda",
                    lda_path,
                    metadata,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                )

            lfda_path = train_lfda(x_train, y_train, metadata)
            score_saved_filters(
                score_rows,
                "LFDA",
                "lfda",
                lfda_path,
                metadata,
                x_train,
                y_train,
                x_test,
                y_test,
            )

            if split_i in EXPENSIVE_METHOD_SPLITS:
                wda_path = train_wda(x_train, y_train, x_val, y_val, metadata)
                score_saved_filters(
                    score_rows,
                    "WDA",
                    "wda",
                    wda_path,
                    metadata,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                )

                lmnn_path = train_lmnn(x_train, y_train, metadata)
                score_saved_filters(
                    score_rows,
                    "LMNN",
                    "lmnn",
                    lmnn_path,
                    metadata,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                )

write_csv(
    score_rows,
    os.path.join(RESULTS_DIR, "zand_kohn_n_filters_all_scores.csv"),
)

qda_results = summarize_score_rows(
    score_rows,
    MODEL_SPECS,
    FILTER_RANGE,
    metric_key="qda_accuracy",
)
export_metric_results_csv(
    qda_results,
    "qda",
    os.path.join(RESULTS_DIR, "zand_kohn_n_filters_qda_review.csv"),
)
plot_metric_results(
    MODEL_SPECS,
    FILTER_RANGE,
    qda_results,
    "QDA Accuracy (%)",
    os.path.join(FIGURES_DIR, "zand_kohn_accuracies_n_filters_review.pdf"),
    ylim=(0, 100),
)

knn_results = summarize_score_rows(
    score_rows,
    MODEL_SPECS,
    FILTER_RANGE,
    metric_key="knn_accuracy",
)
export_metric_results_csv(
    knn_results,
    "knn",
    os.path.join(RESULTS_DIR, "zand_kohn_n_filters_knn_review.csv"),
)
plot_metric_results(
    MODEL_SPECS,
    FILTER_RANGE,
    knn_results,
    "KNN Accuracy (%)",
    os.path.join(FIGURES_DIR, "zand_kohn_accuracies_n_filters_knn_review.pdf"),
    ylim=(0, 100),
)
