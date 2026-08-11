"""BMU / transform shape checks on tiny synthetic data."""

import numpy as np

from som_core import SOM


def test_fit_transform_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    som = SOM(width=5, height=4, seed=7, scale=True)
    som.fit(X, n_iterations=30, online=True, log_every=10)

    coords = som.transform(X)
    assert coords.shape == (40, 2)
    assert coords.dtype.kind in "iu"
    assert coords[:, 0].min() >= 0
    assert coords[:, 0].max() < 5
    assert coords[:, 1].min() >= 0
    assert coords[:, 1].max() < 4

    indices = som.predict(X)
    assert indices.shape == (40,)
    assert np.all(indices == coords[:, 0] * som.height + coords[:, 1])

    bx, by = som.bmu(X[0])
    assert (bx, by) == tuple(coords[0])
