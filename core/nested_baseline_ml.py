import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, balanced_accuracy_score

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingRegressor, BaggingClassifier,
    RandomForestRegressor, RandomForestClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Model Factory
# Regression
def get_reg_model(model_name):

    if model_name == "DT":
        return DecisionTreeRegressor(random_state=42)

    elif model_name == "BaggingDT":
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(),
            random_state=42
        )

    elif model_name == "RF":
        return RandomForestRegressor(random_state=42, n_jobs=-1)

    elif model_name == "AdaBoost":
        return AdaBoostRegressor(random_state=42)

    elif model_name == "GradBoosting":
        return GradientBoostingRegressor(random_state=42)

    elif model_name == "XGB":
        return XGBRegressor(
            random_state=42,
            n_estimators=100,
            verbosity=0
        )

    elif model_name == "LGBM":
        return LGBMRegressor(
            random_state=42,
            n_estimators=100
        )

    else:
        raise ValueError("Unknown model")

#classification
def get_clf_model(model_name):

    if model_name == "DT":
        return DecisionTreeClassifier(random_state=42)

    elif model_name == "BaggingDT":
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            random_state=42
        )

    elif model_name == "RF":
        return RandomForestClassifier(random_state=42, n_jobs=-1)

    elif model_name == "AdaBoost":
        return AdaBoostClassifier(random_state=42)

    elif model_name == "GradBoosting":
        return GradientBoostingClassifier(random_state=42)

    elif model_name == "XGB":
        return XGBClassifier(
            random_state=42,
            n_estimators=100,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0
        )

    elif model_name == "LGBM":
        return LGBMClassifier(
            random_state=42,
            n_estimators=100
        )

    else:
        raise ValueError("Unknown model")

#Nested Baseline Regression
def run_nested_baseline_regression(X, y_reg, outer_folds=5, inner_folds=5):

    models = ["DT", "BaggingDT", "RF", "AdaBoost",
              "GradBoosting", "XGB", "LGBM"]

    results = {}

    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

    for model_name in models:

        outer_scores = []

        for train_idx, test_idx in outer_cv.split(X):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_reg[train_idx], y_reg[test_idx]

            model = get_reg_model(model_name)
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            score = r2_score(y_test, pred)

            outer_scores.append(score)

        results[model_name] = (
            np.mean(outer_scores),
            np.std(outer_scores)
        )

    return results

# Classification
def run_nested_baseline_classification(X, y_class,
                                       outer_folds=5):

    models = ["DT", "BaggingDT", "RF", "AdaBoost",
              "GradBoosting", "XGB", "LGBM"]

    results = {}

    outer_cv = StratifiedKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=42
    )

    for model_name in models:

        outer_scores = []

        for train_idx, test_idx in outer_cv.split(X, y_class):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_class[train_idx], y_class[test_idx]

            model = get_clf_model(model_name)
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            score = balanced_accuracy_score(y_test, pred)

            outer_scores.append(score)

        results[model_name] = (
            np.mean(outer_scores),
            np.std(outer_scores)
        )

    return results