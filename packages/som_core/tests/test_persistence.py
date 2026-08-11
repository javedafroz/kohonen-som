"""Save / load round-trip preserves transform coordinates."""

import numpy as np

from som_core import SOM


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(25, 4))
    som = SOM(width=4, height=5, seed=99, scale=True)
    som.fit(
        X,
        n_iterations=20,
        online=True,
        feature_names=["a", "b", "c", "d"],
        log_every=5,
    )
    coords_before = som.transform(X)
    indices_before = som.predict(X)

    path = tmp_path / "model.npz"
    som.save(path)
    loaded = SOM.load(path)

    assert loaded.width == som.width
    assert loaded.height == som.height
    assert loaded.n_features == som.n_features
    assert loaded.feature_names == ["a", "b", "c", "d"]
    assert np.allclose(loaded.weights, som.weights)
    assert np.allclose(loaded.mean_, som.mean_)
    assert np.allclose(loaded.std_, som.std_)
    assert loaded.history_ == som.history_

    coords_after = loaded.transform(X)
    indices_after = loaded.predict(X)
    assert np.array_equal(coords_before, coords_after)
    assert np.array_equal(indices_before, indices_after)
