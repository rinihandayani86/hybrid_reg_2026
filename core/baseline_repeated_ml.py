import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

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
    y_true_all = []
    y_pred_all = []
    scores = []
    
    for model_name in models:

        scores = []

        
    for train_idx, test_idx in rkf.split(X):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(pred)

        scores.append(r2_score(y_test, pred))

    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

        # ===== Plot only for RF and XGB =====
    if model_name in ["RF", "XGB"]:

        plt.figure(figsize=(6,6))
        plt.scatter(y_true_all, y_pred_all, alpha=0.3)

        min_val = min(y_true_all.min(), y_pred_all.min())
        max_val = max(y_true_all.max(), y_pred_all.max())

        plt.plot([min_val, max_val],
                 [min_val, max_val],
                 'r--', linewidth=2)

        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        #plt.title(f"{model_name} - Repeated 5x3\nR² = {mean_r2:.4f}")
        plt.grid(True)
        plt.savefig(f"{model_name}-.pdf", dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    return mean_r2, std_r2