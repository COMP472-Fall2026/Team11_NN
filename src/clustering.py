import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

def assign_clusters(X, centers):
    #each column contains the distances from all rows to one center.
    squared_distances = np.column_stack([
        ((X - center) ** 2).sum(axis=1)
        for center in centers
    ])

    #for each row, return the index of its closest center.
    return squared_distances.argmin(axis=1)


def fit_kmeans(X, k=3, max_iterations=100, seed=42):
    rng = np.random.default_rng(seed)

    #pick k different training rows as the initial centers.
    initial_indices = rng.choice(len(X), size=k, replace=False)
    centers = X[initial_indices].copy()

    previous_assignments = None

    for _ in range(max_iterations):
        # assign each training row to its closest center.
        assignments = assign_clusters(X, centers)

        # stop once the groups stay the same.
        if np.array_equal(assignments, previous_assignments):
            break

        # move each center to the average of its assigned rows.
        for cluster_id in range(k):
            members = X[assignments == cluster_id]

            if len(members) > 0:
                centers[cluster_id] = members.mean(axis=0)
            #if a cluster is empty, keep its previous center.

        previous_assignments = assignments

    return centers

def calculate_inertia(X, centers):
    # find each review's closest center.
    assignments = assign_clusters(X, centers)

    # centers[assignments] gives us one matching center per review.
    differences = X - centers[assignments]
    return float(np.sum(differences ** 2, dtype=np.float64))

def analyze_clusters(X_val, y_val, centers):
    cluster_ids = assign_clusters(X_val, centers)

    counts = pd.crosstab(
        pd.Series(cluster_ids, name="Cluster"),
        pd.Series(y_val, name="True class"),
    ).reindex(index=range(len(centers)), fill_value=0)

    sizes = counts.sum(axis=1)
    percentages = counts.div(sizes.where(sizes > 0), axis=0) * 100
    ari = adjusted_rand_score(y_val, cluster_ids)

    return counts, percentages, ari