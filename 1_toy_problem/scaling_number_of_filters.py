import torch

from scaling_toy_utils import (
    embed_statistics_in_ambient_space,
    FILTERS_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    fit_sqfa_from_statistics,
    make_informative_statistics,
    make_random_subspace_basis,
    orthonormalize_filters,
    plot_accuracy_comparison,
    project_statistics,
    save_csv,
    save_filters,
    simulate_qda_accuracy_from_statistics,
)


torch.manual_seed(11)

N_CLASSES = 5
AMBIENT_DIM = 5000
INFORMATIVE_DIMS = [10, 50, 100, 500]
FEATURE_NOISE = 0.001
NULL_VARIANCE = 1.0
SQFA_CONSTRAINT = "sphere"
FIT_KWARGS = {
  "max_epochs": 200,
  "show_progress": True,
  "pairwise": False,
  "line_search_fn": "strong_wolfe",
  "history_size": 10,
  "max_iter": 10,
}
N_TRAIN_PER_CLASS = 20000
N_TEST_PER_CLASS = 20000


results = []
for informative_dim in INFORMATIVE_DIMS:
    print(f"Running filter-count analysis for informative_dim={informative_dim}")
    basis = make_random_subspace_basis(
        n_dim=AMBIENT_DIM,
        subspace_dim=informative_dim,
        seed=informative_dim,
    )
    cov_axes = torch.tensor([1 + 6.0 / torch.sqrt(torch.tensor(informative_dim)), 1.0])
    means_info, covariances_info = make_informative_statistics(
        n_classes=N_CLASSES,
        informative_dim=informative_dim,
        mean_scale=2.0,
        cov_axes=cov_axes
    )
    means_full, covariances_full = embed_statistics_in_ambient_space(
        means=means_info,
        covariances=covariances_info,
        basis=basis,
        null_variance=NULL_VARIANCE,
    )
    model, fit_time = fit_sqfa_from_statistics(
        means=means_full,
        covariances=covariances_full,
        n_filters=informative_dim,
        feature_noise=FEATURE_NOISE,
        constraint=SQFA_CONSTRAINT,
        fit_kwargs=FIT_KWARGS,
        seed=informative_dim,
    )
    filters_full = orthonormalize_filters(
        model.filters.detach().to(dtype=torch.float32)
    )
    save_filters(
        filters_full,
        FILTERS_DIR / f"scaling_number_of_filters_filters_k{informative_dim}.npy",
    )

    true_acc = simulate_qda_accuracy_from_statistics(
        means=means_info,
        covariances=covariances_info,
        n_train_per_class=N_TRAIN_PER_CLASS,
        n_test_per_class=N_TEST_PER_CLASS,
        seed=informative_dim,
    )
    sqfa_means, sqfa_covariances = project_statistics(
        means=means_full,
        covariances=covariances_full,
        filters=filters_full,
    )
    sqfa_acc = simulate_qda_accuracy_from_statistics(
        means=sqfa_means,
        covariances=sqfa_covariances,
        n_train_per_class=N_TRAIN_PER_CLASS,
        n_test_per_class=N_TEST_PER_CLASS,
        seed=informative_dim,
    )
    results.append(
        {
            "ambient_dim": AMBIENT_DIM,
            "informative_dim": informative_dim,
            "n_filters": informative_dim,
            "true_subspace_accuracy_percent": 100.0 * true_acc,
            "sqfa_accuracy_percent": 100.0 * sqfa_acc,
            "fit_time_seconds": fit_time,
        }
    )

save_csv(
    rows=results,
    output_path=RESULTS_DIR / "scaling_number_of_filters.csv",
    fieldnames=[
        "ambient_dim",
        "informative_dim",
        "n_filters",
        "true_subspace_accuracy_percent",
        "sqfa_accuracy_percent",
        "fit_time_seconds",
    ],
)
plot_accuracy_comparison(
    x_values=[row["n_filters"] for row in results],
    true_accs=[row["true_subspace_accuracy_percent"] / 100.0 for row in results],
    sqfa_accs=[row["sqfa_accuracy_percent"] / 100.0 for row in results],
    xlabel="Number of Filters",
    output_path=FIGURES_DIR / "scaling_number_of_filters.pdf",
    xscale="log",
)
