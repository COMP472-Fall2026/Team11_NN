import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CustomDecisionTree:

    def __init__(
        self,
        max_depth=8,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features=None,
        n_thresholds=8,
        random_state=42
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("Invalid training data shape")

        if not np.isfinite(X).all():
            raise ValueError("Features contain NaN or infinity")

        self.classes_, encoded = np.unique(
            y, return_inverse=True
        )

        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.rng_ = np.random.default_rng(self.random_state)

        self.tree_ = self._grow(X, encoded, 0)
        return self

    def _feature_count(self):
        m = self.max_features

        if m is None:
            return self.n_features_

        if m == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))

        if isinstance(m, float):
            return max(
                1, min(self.n_features_, int(m * self.n_features_))
            )

        return max(1, min(self.n_features_, int(m)))

    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=self.n_classes_)

        node = {
            "prediction": int(np.argmax(counts)),
            "n_samples": len(y)
        }

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
            or np.count_nonzero(counts) == 1
        ):
            return node

        parent_gini = 1.0 - np.sum(
            (counts / len(y)) ** 2
        )

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        features = self.rng_.choice(
            self.n_features_,
            self._feature_count(),
            replace=False
        )

        for feature in features:
            values = X[:, feature]

            thresholds = np.unique(
                np.quantile(
                    values,
                    np.linspace(
                        0, 1, self.n_thresholds + 2
                    )[1:-1]
                )
            )

            if len(thresholds) == 0:
                continue

            bins = np.searchsorted(
                thresholds, values, side="right"
            )

            hist = np.bincount(
                bins * self.n_classes_ + y,
                minlength=(
                    (len(thresholds) + 1) * self.n_classes_
                )
            ).reshape(-1, self.n_classes_)

            left_counts = np.cumsum(hist, axis=0)[:-1]
            right_counts = counts[None, :] - left_counts

            nl = left_counts.sum(axis=1)
            nr = len(y) - nl

            valid = (
                (nl >= self.min_samples_leaf)
                & (nr >= self.min_samples_leaf)
            )

            if not valid.any():
                continue

            left_gini = 1 - (
                (left_counts ** 2).sum(axis=1)
                / np.maximum(nl, 1) ** 2
            )

            right_gini = 1 - (
                (right_counts ** 2).sum(axis=1)
                / np.maximum(nr, 1) ** 2
            )

            gains = parent_gini - (
                nl * left_gini + nr * right_gini
            ) / len(y)

            gains[~valid] = -np.inf

            j = int(np.argmax(gains))

            if gains[j] > best_gain + 1e-12:
                best_gain = float(gains[j])
                best_feature = int(feature)
                best_threshold = float(thresholds[j])

        if best_feature is None:
            return node

        mask = X[:, best_feature] <= best_threshold

        node.update(
            feature=best_feature,
            threshold=best_threshold,
            left=self._grow(X[mask], y[mask], depth + 1),
            right=self._grow(X[~mask], y[~mask], depth + 1)
        )

        return node

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)

        predictions = np.empty(
            len(X), dtype=np.int64
        )

        for i, row in enumerate(X):
            node = self.tree_

            while "feature" in node:
                if row[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]

            predictions[i] = self.classes_[
                node["prediction"]
            ]

        return predictions