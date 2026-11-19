import numpy as np

def predict_knn(X_train, y_train, X_query, k=3):
    predictions = []

    for query in X_query:
        #Compute one squared distance per training row.
        differences = X_train - query
        squared_distances = (differences ** 2).sum(axis=1)

        #Find the k closest training rows and look up their labels.
        nearest_indices = np.argsort(squared_distances)[:k]
        nearest_labels = y_train[nearest_indices]

        #Count the votes and pick the label with the most votes.
        votes = np.bincount(nearest_labels)
        predicted_label = votes.argmax()
        predictions.append(predicted_label)

    return np.array(predictions)

def normalize_rows(X):
    lengths = np.linalg.norm(X, axis=1, keepdims=True)

    if np.any(lengths == 0):
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    return X / lengths

def predict_knn_weighted(X_train, y_train, X_query, k=3):
    predictions = []

    for query in X_query:
        differences = X_train - query
        squared_distances = (differences ** 2).sum(axis=1)

        nearest_indices = np.argsort(squared_distances)[:k]
        nearest_labels = y_train[nearest_indices]
        nearest_distances = np.sqrt(squared_distances[nearest_indices])

        exact_matches = nearest_distances == 0
        if exact_matches.any():
            # If there are exact matches, let only those neighbours vote.
            votes = np.bincount(nearest_labels[exact_matches])
        else:
            weights = 1 / nearest_distances
            votes = np.bincount(nearest_labels, weights=weights)

        predictions.append(votes.argmax())

    return np.array(predictions)

