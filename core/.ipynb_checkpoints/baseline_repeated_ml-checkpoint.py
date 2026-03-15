import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# =====================================================
# MODEL FACTORY (DEFAULT PARAMETERS)
# =====================================================

def get_baseline_model(model_name):

    if model_name == "DT":
        return DecisionTreeRegressor(random_state=42)

    elif model_name == "BaggingDT":
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(),
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "RF":
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "AdaBoost":
        return AdaBoostRegressor(
            random_state=42
        )

    elif model_name == "GradBoost":
        return GradientBoostingRegressor(
            random_state=42
        )

    elif model_name == "XGB":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

    elif model_name == "LGBM":
        return LGBMRegressor(
            random_state=42
        )

    else:
        raise ValueError("Unknown model name")


# =====================================================
# REPEATED CV BASELINE
# =====================================================

def run_baseline_repeated_regression(X, y):

    models = [
        "DT",
        "BaggingDT",
        "RF",
        "AdaBoost",
        "GradBoost",
        "XGB",
        "LGBM"
    ]

    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

    results = {}

    for model_name in models:

        scores = []

        for train_idx, test_idx in rkf.split(X):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = get_baseline_model(model_name)
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            scores.append(r2_score(y_test, pred))

        results[model_name] = (
            np.mean(scores),
            np.std(scores)
        )

    return results