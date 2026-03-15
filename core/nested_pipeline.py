import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, r2_score
import random

def random_search(space, n_trials=5):
    keys = list(space.keys())
    trials = []
    for _ in range(n_trials):
        param = {k: random.choice(space[k]) for k in keys}
        trials.append(param)
    return trials


def tune_regression_dt(X, y, inner_cv):

    reg_space = {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10]
    }

    candidates = random_search(reg_space, n_trials=6)

    best_score = -np.inf
    best_param = None

    for param in candidates:
        scores = []
        for train_idx, val_idx in inner_cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = DecisionTreeRegressor(**param)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            score = r2_score(y_val, pred)
            scores.append(score)

        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_param = param

    return best_param


def tune_classification_dt(X, y, inner_cv):

    clf_space = {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10]
    }

    candidates = random_search(clf_space, n_trials=6)

    best_score = -np.inf
    best_param = None

    for param in candidates:
        scores = []
        for train_idx, val_idx in inner_cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = DecisionTreeClassifier(**param)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            score = balanced_accuracy_score(y_val, pred)
            scores.append(score)

        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_param = param

    return best_param

def tune_pref(
    X_train_outer,
    y_class_train_outer,
    y_reg_train_outer,
    best_reg_param,
    best_clf_param,
    inner_cv_clf,
    pref_grid,
    reg_threshold
):
    best_score = -np.inf
    best_p = None

    for p in pref_grid:
        fold_scores = []

        for train_idx, val_idx in inner_cv_clf.split(X_train_outer, y_class_train_outer):

            X_tr = X_train_outer[train_idx]
            X_val = X_train_outer[val_idx]

            y_class_tr = y_class_train_outer[train_idx]
            y_class_val = y_class_train_outer[val_idx]

            y_reg_tr = y_reg_train_outer[train_idx]
            y_reg_val = y_reg_train_outer[val_idx]

            RM = DecisionTreeRegressor(**best_reg_param)
            CM = DecisionTreeClassifier(**best_clf_param)

            RM.fit(X_tr, y_reg_tr)
            CM.fit(X_tr, y_class_tr)

            reg_val = RM.predict(X_val)
            proba_val = CM.predict_proba(X_val)[:, 1]

            class_val = (proba_val >= 0.5).astype(int)
            reg_binary = (reg_val >= reg_threshold).astype(int)

            final_val = np.where(
                proba_val < p,
                reg_binary,
                class_val
            )

            score = balanced_accuracy_score(y_class_val, final_val)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)

        if mean_score > best_score:
            best_score = mean_score
            best_p = p

    return best_p

def run_nested(CONFIG, X, y_class, y_reg):

    outer_cv = StratifiedKFold(
        n_splits=CONFIG["outer_folds"],
        shuffle=True,
        random_state=CONFIG["random_state"]
    )

    outer_scores = []

    for train_outer_idx, test_outer_idx in outer_cv.split(X, y_class):

        X_train_outer = X[train_outer_idx]
        X_test_outer = X[test_outer_idx]

        y_class_train_outer = y_class[train_outer_idx]
        y_class_test_outer = y_class[test_outer_idx]

        y_reg_train_outer = y_reg[train_outer_idx]
        y_reg_test_outer = y_reg[test_outer_idx]

        inner_cv_clf = StratifiedKFold(
            n_splits=CONFIG["inner_folds"],
            shuffle=True,
            random_state=CONFIG["random_state"]
        )

        
        inner_cv_reg = KFold(
            n_splits=CONFIG["inner_folds"],
            shuffle=True,
            random_state=CONFIG["random_state"]
        )

        best_reg_param = tune_regression_dt(
            X_train_outer, y_reg_train_outer, inner_cv_reg
        )

        best_clf_param = tune_classification_dt(
            X_train_outer, y_class_train_outer, inner_cv_clf
        )

        best_p = tune_pref(
            X_train_outer,
            y_class_train_outer,
            y_reg_train_outer,
            best_reg_param,
            best_clf_param,
            inner_cv_clf,
            CONFIG["pref_grid"],
            CONFIG["reg_threshold"]
        )

        RM_final = DecisionTreeRegressor(**best_reg_param)
        CM_final = DecisionTreeClassifier(**best_clf_param)

        RM_final.fit(X_train_outer, y_reg_train_outer)
        CM_final.fit(X_train_outer, y_class_train_outer)

        reg_test = RM_final.predict(X_test_outer)
        proba_test = CM_final.predict_proba(X_test_outer)[:, 1]

        class_test = (proba_test >= 0.5).astype(int)
        reg_binary_test = (reg_test >= CONFIG["reg_threshold"]).astype(int)

        final_test = np.where(
            proba_test < best_p,
            reg_binary_test,
            class_test
        )

        bacc = balanced_accuracy_score(y_class_test_outer, final_test)
        outer_scores.append(bacc)

    return np.mean(outer_scores), np.std(outer_scores)