# kohonen.py
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def train(input_data, n_max_iterations, width, height, log_every=None):
    if log_every is None:
        log_every = max(1, n_max_iterations // 10)

    σ0 = max(width, height) / 2
    α0 = 0.1
    weights = np.random.random((width, height, 3))
    λ = n_max_iterations / np.log(σ0)

    logger.info(
        "Training start: map=%sx%s samples=%s iters=%s σ0=%.3f λ=%.3f",
        width,
        height,
        len(input_data),
        n_max_iterations,
        σ0,
        λ,
    )
    t0 = time.perf_counter()

    for t in range(n_max_iterations):
        σt = σ0 * np.exp(-t / λ)
        αt = α0 * np.exp(-t / λ)
        for vt in input_data:
            bmu = np.argmin(np.sum((weights - vt) ** 2, axis=2))
            bmu_x, bmu_y = np.unravel_index(bmu, (width, height))
            for x in range(width):
                for y in range(height):
                    di = np.sqrt(((x - bmu_x) ** 2) + ((y - bmu_y) ** 2))
                    θt = np.exp(-(di ** 2) / (2 * (σt ** 2)))
                    weights[x, y] += αt * θt * (vt - weights[x, y])

        if t % log_every == 0 or t == n_max_iterations - 1:
            pct = 100.0 * (t + 1) / n_max_iterations
            elapsed = time.perf_counter() - t0
            logger.info(
                "Progress %5.1f%% (iter %d/%d) σ=%.4f α=%.4f elapsed=%.1fs",
                pct,
                t + 1,
                n_max_iterations,
                σt,
                αt,
                elapsed,
            )

    logger.info("Training done in %.1fs", time.perf_counter() - t0)
    return weights


def train_vectorized(input_data, n_max_iterations, width, height, log_every=None):
    """Same training semantics as train(), with vectorized neighbourhood updates."""
    if log_every is None:
        log_every = max(1, n_max_iterations // 10)

    σ0 = max(width, height) / 2
    α0 = 0.1
    weights = np.random.random((width, height, 3))
    λ = n_max_iterations / np.log(σ0)
    xs, ys = np.indices((width, height))

    logger.info(
        "Training start (vectorized): map=%sx%s samples=%s iters=%s σ0=%.3f λ=%.3f",
        width,
        height,
        len(input_data),
        n_max_iterations,
        σ0,
        λ,
    )
    t0 = time.perf_counter()

    for t in range(n_max_iterations):
        σt = σ0 * np.exp(-t / λ)
        αt = α0 * np.exp(-t / λ)
        σt_safe = max(σt, 1e-8)
        for vt in input_data:
            bmu = np.argmin(np.sum((weights - vt) ** 2, axis=2))
            bmu_x, bmu_y = np.unravel_index(bmu, (width, height))
            di_sq = (xs - bmu_x) ** 2 + (ys - bmu_y) ** 2
            θt = np.exp(-di_sq / (2 * (σt_safe ** 2)))
            weights += αt * θt[..., None] * (vt - weights)

        if t % log_every == 0 or t == n_max_iterations - 1:
            pct = 100.0 * (t + 1) / n_max_iterations
            elapsed = time.perf_counter() - t0
            logger.info(
                "Progress %5.1f%% (iter %d/%d) σ=%.4f α=%.4f elapsed=%.1fs",
                pct,
                t + 1,
                n_max_iterations,
                σt,
                αt,
                elapsed,
            )

    logger.info("Training done in %.1fs", time.perf_counter() - t0)
    return weights


def load_numeric_csv(path, feature_columns=None, label_column=None):
    """
    Load any CSV into a numeric feature matrix.

    - If feature_columns is None, use all columns except label_column.
    - Non-numeric feature values raise ValueError.
    - Returns (X, y, feature_names) where y may be None.
    """
    path = Path(path)
    with path.open(newline="") as f:
        header = f.readline().strip().split(",")
        rows = [line.strip().split(",") for line in f if line.strip()]

    if not header:
        raise ValueError(f"Empty CSV: {path}")

    if label_column is not None and label_column not in header:
        raise ValueError(f"label_column {label_column!r} not in {header}")

    if feature_columns is None:
        feature_columns = [c for c in header if c != label_column]
    missing = [c for c in feature_columns if c not in header]
    if missing:
        raise ValueError(f"Unknown feature columns: {missing}")

    feat_idx = [header.index(c) for c in feature_columns]
    try:
        X = np.array([[float(row[i]) for i in feat_idx] for row in rows], dtype=float)
    except ValueError as exc:
        raise ValueError(
            "Feature columns must be numeric after encoding; "
            "encode categoricals before calling load_numeric_csv."
        ) from exc

    y = None
    if label_column is not None:
        label_idx = header.index(label_column)
        y = np.array([row[label_idx] for row in rows])

    logger.info(
        "Loaded %s: samples=%s features=%s label=%s",
        path.name,
        X.shape[0],
        feature_columns,
        label_column,
    )
    return X, y, feature_columns


class SOM:
    """
    Generic Self-Organising Map for any numeric dataset shaped (n_samples, n_features).

    Weights have shape (width, height, n_features). Optional z-score scaling is
    fit on training data and reused in transform / quantization_error.
    """

    def __init__(
        self,
        width,
        height,
        n_features=None,
        alpha0=0.1,
        seed=None,
        scale=True,
    ):
        self.width = width
        self.height = height
        self.n_features = n_features
        self.alpha0 = alpha0
        self.scale = scale
        self.rng = np.random.default_rng(seed)
        self.weights = None
        self.mean_ = None
        self.std_ = None
        self._xs, self._ys = np.indices((width, height))

    def _validate_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features); got shape {X.shape}")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if self.n_features is not None and X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}"
            )
        return X

    def _fit_scaler(self, X):
        if not self.scale:
            self.mean_ = np.zeros(X.shape[1])
            self.std_ = np.ones(X.shape[1])
            return X
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-12, 1.0, self.std_)
        return (X - self.mean_) / self.std_

    def _transform_scale(self, X):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("SOM must be fit before transforming data")
        return (X - self.mean_) / self.std_

    def fit(self, X, n_iterations, log_every=None, online=False):
        """
        Train on any numeric matrix X with shape (n_samples, n_features).

        online=False: classic epoch-style pass over all samples each iteration
        online=True:  one random sample per iteration (better for large N)
        """
        X = self._validate_X(X)
        self.n_features = X.shape[1]
        if log_every is None:
            log_every = max(1, n_iterations // 10)

        X_scaled = self._fit_scaler(X)
        # Init weights near data scale (scaled space ≈ N(0,1) when scale=True)
        self.weights = self.rng.normal(
            loc=0.0, scale=1.0, size=(self.width, self.height, self.n_features)
        )

        sigma0 = max(self.width, self.height) / 2
        if sigma0 <= 1:
            sigma0 = 1.0
        lam = n_iterations / np.log(sigma0)

        mode = "online" if online else "epoch"
        logger.info(
            "SOM fit (%s): map=%sx%s samples=%s features=%s iters=%s σ0=%.3f",
            mode,
            self.width,
            self.height,
            X_scaled.shape[0],
            self.n_features,
            n_iterations,
            sigma0,
        )
        t0 = time.perf_counter()
        n_samples = X_scaled.shape[0]

        for t in range(n_iterations):
            sigma_t = max(sigma0 * np.exp(-t / lam), 1e-8)
            alpha_t = self.alpha0 * np.exp(-t / lam)

            if online:
                vt = X_scaled[self.rng.integers(0, n_samples)]
                self._update(vt, sigma_t, alpha_t)
            else:
                for vt in X_scaled:
                    self._update(vt, sigma_t, alpha_t)

            if t % log_every == 0 or t == n_iterations - 1:
                qe = self._quantization_error_scaled(X_scaled)
                pct = 100.0 * (t + 1) / n_iterations
                elapsed = time.perf_counter() - t0
                logger.info(
                    "Progress %5.1f%% (iter %d/%d) σ=%.4f α=%.4f QE=%.4f elapsed=%.1fs",
                    pct,
                    t + 1,
                    n_iterations,
                    sigma_t,
                    alpha_t,
                    qe,
                    elapsed,
                )

        logger.info("SOM fit done in %.1fs", time.perf_counter() - t0)
        return self

    def _update(self, vt, sigma_t, alpha_t):
        bmu = np.argmin(np.sum((self.weights - vt) ** 2, axis=2))
        bmu_x, bmu_y = np.unravel_index(bmu, (self.width, self.height))
        di_sq = (self._xs - bmu_x) ** 2 + (self._ys - bmu_y) ** 2
        theta = np.exp(-di_sq / (2 * (sigma_t ** 2)))
        self.weights += alpha_t * theta[..., None] * (vt - self.weights)

    def bmu(self, x):
        """Return (x_idx, y_idx) BMU for a single feature vector."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[0]}")
        xs = self._transform_scale(x.reshape(1, -1))[0]
        idx = np.argmin(np.sum((self.weights - xs) ** 2, axis=2))
        return np.unravel_index(idx, (self.width, self.height))

    def transform(self, X):
        """Return BMU coordinates for each sample, shape (n_samples, 2)."""
        X = self._validate_X(X)
        X_scaled = self._transform_scale(X)
        coords = np.empty((X_scaled.shape[0], 2), dtype=int)
        flat_weights = self.weights.reshape(-1, self.n_features)
        for i, vt in enumerate(X_scaled):
            idx = np.argmin(np.sum((flat_weights - vt) ** 2, axis=1))
            coords[i] = np.unravel_index(idx, (self.width, self.height))
        return coords

    def _quantization_error_scaled(self, X_scaled):
        flat_weights = self.weights.reshape(-1, self.n_features)
        errors = [
            np.min(np.sum((flat_weights - vt) ** 2, axis=1)) ** 0.5 for vt in X_scaled
        ]
        return float(np.mean(errors))

    def quantization_error(self, X):
        X = self._validate_X(X)
        return self._quantization_error_scaled(self._transform_scale(X))


def plot_component_planes(weights, feature_names, path):
    """Save one heatmap per feature dimension from SOM weights."""
    n_features = weights.shape[2]
    cols = min(n_features, 4)
    rows = int(np.ceil(n_features / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n_features):
        ax = axes[i]
        im = ax.imshow(weights[:, :, i].T, origin="lower", aspect="auto")
        title = feature_names[i] if feature_names and i < len(feature_names) else f"f{i}"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    for j in range(n_features, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved component planes to %s", path)


def plot_bmu_labels(coords, labels, width, height, path):
    """Scatter BMUs jittered on the grid, colored by label (if present)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.25, 0.25, size=coords.shape)

    if labels is None:
        ax.scatter(coords[:, 0] + jitter[:, 0], coords[:, 1] + jitter[:, 1], s=20, alpha=0.7)
    else:
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(
                coords[mask, 0] + jitter[mask, 0],
                coords[mask, 1] + jitter[mask, 1],
                s=28,
                alpha=0.75,
                label=str(label),
            )
        ax.legend(loc="best", fontsize=8)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_xlabel("SOM x")
    ax.set_ylabel("SOM y")
    ax.set_title("BMU projection")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved BMU label map to %s", path)


def train_som_from_csv(
    csv_path,
    feature_columns,
    label_column=None,
    *,
    width=10,
    height=10,
    n_iterations=1000,
    seed=42,
    scale=True,
    online=False,
    output_dir=".",
    alpha0=0.1,
):
    """
    End-to-end API: load a CSV, train a SOM, save plots, return metrics.

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file.
    feature_columns : list[str]
        Numeric feature column names.
    label_column : str | None
        Optional label column (used only for BMU coloring, not training).
    width, height : int
        SOM grid size.
    n_iterations : int
        Training iterations.
    seed : int | None
        RNG seed for reproducibility.
    scale : bool
        If True, z-score features before training.
    online : bool
        If True, one random sample per iteration (better for large N).
    output_dir : str | Path
        Directory for plot artifacts.
    alpha0 : float
        Initial learning rate.

    Returns
    -------
    dict
        Metrics, artifact paths, and the fitted SOM instance under key "som".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = load_numeric_csv(
        csv_path,
        feature_columns=feature_columns,
        label_column=label_column,
    )

    som = SOM(
        width=width,
        height=height,
        seed=seed,
        scale=scale,
        alpha0=alpha0,
    )
    som.fit(X, n_iterations=n_iterations, online=online)
    coords = som.transform(X)
    qe = som.quantization_error(X)

    components_path = output_dir / "som_components.png"
    bmu_path = output_dir / "som_bmu.png"
    plot_component_planes(som.weights, feature_names, components_path)
    plot_bmu_labels(coords, y, som.width, som.height, bmu_path)

    result = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_columns": list(feature_names),
        "label_column": label_column,
        "map_size": [width, height],
        "n_iterations": n_iterations,
        "online": online,
        "quantization_error": float(qe),
        "weights_shape": list(som.weights.shape),
        "bmu_coords": coords,
        "artifacts": {
            "components": str(components_path),
            "bmu": str(bmu_path),
        },
        "som": som,
    }
    logger.info(
        "train_som_from_csv done: samples=%s features=%s QE=%.4f",
        result["n_samples"],
        result["n_features"],
        result["quantization_error"],
    )
    return result