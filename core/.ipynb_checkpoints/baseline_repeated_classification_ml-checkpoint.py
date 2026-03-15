import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =====================================================
# MODEL FACTORY
# =====================================================

def get_baseline_model_clf(model_name):

    if model_name == "DT":
        return DecisionTreeClassifier(random_state=42)

    elif model_name == "BaggingDT":
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "RF":
        return RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "AdaBoost":
        return AdaBoostClassifier(random_state=42)

    elif model_name == "GradBoost":
        return GradientBoostingClassifier(random_state=42)

    elif model_name == "XGB":
        return XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False
        )

    elif model_name == "LGBM":
        return LGBMClassifier(random_state=42)

    else:
        raise ValueError("Unknown model name")


# =====================================================
# REPEATED CV CLASSIFICATION
# =====================================================

def run_baseline_repeated_classification(X, y):

    models = [
        "DT",
        "BaggingDT",
        "RF",
        "AdaBoost",
        "GradBoost",
        "XGB",
        "LGBM"
    ]

    rskf = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

    results = {}

    for model_name in models:

        scores = []

        for train_idx, test_idx in rskf.split(X, y):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = get_baseline_model_clf(model_name)
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            scores.append(balanced_accuracy_score(y_test, pred))

        results[model_name] = (
            np.mean(scores),
            np.std(scores)
        )

    return results