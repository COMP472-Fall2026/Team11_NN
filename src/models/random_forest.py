import numpy as np
from .decision_tree import CustomDecisionTree
class CustomRandomForest:

    def __init__(
        self,
        n_estimators=15,
        max_depth=10,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features="sqrt",
        n_thresholds=8,
        bootstrap_fraction=0.8,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.bootstrap_fraction = bootstrap_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y)
        self.trees_ = []

        rng = np.random.default_rng(self.random_state)

        n_boot = max(
            2, int(len(X) * self.bootstrap_fraction)
        )

        for i in range(self.n_estimators):

            sample_idx = rng.integers(
                0, len(X), size=n_boot
            )

            tree = CustomDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_thresholds=self.n_thresholds,
                random_state=int(
                    rng.integers(0, 2**31 - 1)
                )
            )

            tree.fit(X[sample_idx], y[sample_idx])
            self.trees_.append(tree)

        return self

    def predict(self, X):

        if not self.trees_:
            raise ValueError("Train the forest first")

        votes = np.stack(
            [tree.predict(X) for tree in self.trees_],
            axis=0
        )

        counts = np.stack(
            [
                (votes == label).sum(axis=0)
                for label in self.classes_
            ],
            axis=0
        )

        return self.classes_[counts.argmax(axis=0)]