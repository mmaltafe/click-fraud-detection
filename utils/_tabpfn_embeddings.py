from __future__ import annotations

import numpy as np


def ensure_2d_embeddings(embeddings: np.ndarray, expected_rows: int) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim > 1 and array.shape[0] != expected_rows:
        matching_axes = [axis for axis, size in enumerate(array.shape) if size == expected_rows]
        if matching_axes:
            array = np.moveaxis(array, matching_axes[0], 0)
        else:
            raise ValueError(
                "Could not align TabPFN embeddings with input rows: "
                f"embedding_shape={array.shape}, expected_rows={expected_rows}"
            )
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    if array.shape[0] != expected_rows:
        raise ValueError(
            "TabPFN embeddings have incompatible row count after reshape: "
            f"embedding_shape={array.shape}, expected_rows={expected_rows}"
        )
    return array


def tabpfn_embeddings_or_proba(classifier, X: np.ndarray, expected_rows: int) -> tuple[np.ndarray, str]:
    try:
        embeddings = classifier.get_embeddings(X, data_source="test")
        return ensure_2d_embeddings(embeddings, expected_rows), "tabpfn_embeddings"
    except Exception:
        try:
            embeddings = classifier.get_embeddings(X)
            return ensure_2d_embeddings(embeddings, expected_rows), "tabpfn_embeddings_no_data_source"
        except Exception:
            proba = classifier.predict_proba(X)
            return ensure_2d_embeddings(proba, expected_rows), "predict_proba_fallback"
