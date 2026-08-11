"""Quantization / topographic error and training history."""

import numpy as np

from som_core import SOM


def test_metrics_and_history():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    som = SOM(width=6, height=6, seed=3, scale=True, alpha0=0.2)
    som.fit(X, n_iterations=40, online=False, log_every=10)

    assert len(som.history_) >= 2
    assert all(
        set(row) >= {"iteration", "qe", "sigma", "alpha"} for row in som.history_
    )
    assert som.history_[0]["iteration"] == 1
    assert som.history_[-1]["iteration"] == 40

    qes = [row["qe"] for row in som.history_]
    # Rough decrease: final QE should not be much worse than early
    assert qes[-1] <= qes[0] * 1.5 + 0.5

    qe = som.quantization_error(X)
    te = som.topographic_error(X)
    assert 0.0 <= te <= 1.0
    assert qe >= 0.0
    assert som.quantization_error_ is not None
    assert som.topographic_error_ is not None
    assert abs(qe - som.quantization_error_) < 1e-9
    assert abs(te - som.topographic_error_) < 1e-9
