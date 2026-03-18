"""Helpers for artifact naming, loading, and persistence."""

import os

import numpy as np


def artifact_path(artifacts_dir, model_key, artifact_kind, n_filters=None):
    """Build an artifact path for a model, artifact kind, and optional filter count."""
    suffix = "" if n_filters is None else f"_n{n_filters}"
    return f"{artifacts_dir}/{model_key}_{artifact_kind}{suffix}.npy"


def load_cached_filters(filter_path, description=None):
    """Load cached filters if available and optionally print a short message."""
    if not os.path.exists(filter_path):
        return None

    if description is not None:
        print(f"Loading cached {description}")
    return np.load(filter_path)


def has_saved_artifacts(filter_path, time_path):
    """Check whether both filter and timing artifacts already exist."""
    return os.path.exists(filter_path) and os.path.exists(time_path)


def save_training_artifacts(filter_path, time_path, filters, times):
    """Save learned filters and their associated training times."""
    np.save(filter_path, np.asarray(filters))
    np.save(time_path, np.asarray(times))
