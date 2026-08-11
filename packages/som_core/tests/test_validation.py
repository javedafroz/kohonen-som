"""Input and hyperparameter validation."""

import numpy as np
import pytest

from som_core import SOM


def test_rejects_nan():
    som = SOM(width=3, height=3, seed=0)
    X = np.array([[1.0, 2.0], [np.nan, 3.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        som.fit(X, n_iterations=5)


def test_rejects_inf():
    som = SOM(width=3, height=3, seed=0)
    X = np.array([[1.0, 2.0], [np.inf, 3.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        som.fit(X, n_iterations=5)


def test_rejects_bad_dims():
    with pytest.raises(ValueError, match="width and height"):
        SOM(width=1, height=5)
    with pytest.raises(ValueError, match="alpha0"):
        SOM(width=3, height=3, alpha0=0)
    with pytest.raises(ValueError, match="n_iterations"):
        SOM(width=3, height=3).fit(np.ones((5, 2)), n_iterations=0)


def test_rejects_wrong_feature_count_after_fit():
    som = SOM(width=3, height=3, seed=1)
    som.fit(np.ones((10, 2)), n_iterations=5, online=True)
    with pytest.raises(ValueError, match="Expected 2 features"):
        som.transform(np.ones((3, 3)))
