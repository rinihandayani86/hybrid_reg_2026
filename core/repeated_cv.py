from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
import numpy as np

def evaluate_repeated_cv(model_name, best_params, X, y):

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = []

    for train_idx, test_idx in rkf.split(X):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = get_model(model_name, best_params)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        scores.append(r2_score(y_test, pred))

    return np.mean(scores), np.std(scores)