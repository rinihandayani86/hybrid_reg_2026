import time
import numpy as np
import pyswarms as ps
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def build_model(model_name, params):

    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=int(params[0]),
            max_depth=int(params[1]),
            min_samples_split=int(params[2]),
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "XGB":
        return XGBClassifier(
            n_estimators=int(params[0]),
            max_depth=int(params[1]),
            learning_rate=params[2],
            subsample=params[3],
            colsample_bytree=params[4],
            objective="binary:logistic",
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False
        )

    elif model_name == "LGBM":
        return LGBMClassifier(
            n_estimators=int(params[0]),
            num_leaves=int(params[1]),
            learning_rate=params[2],
            subsample=params[3],
            random_state=42
        )


def get_bounds(model_name):

    if model_name == "RF":
        lower = [100, 5, 2]
        upper = [300, 30, 10]

    elif model_name == "XGB":
        lower = [100, 3, 0.03, 0.7, 0.7]
        upper = [300, 10, 0.2, 1.0, 1.0]

    elif model_name == "LGBM":
        lower = [100, 20, 0.03, 0.7]
        upper = [400, 100, 0.2, 1.0]

    return np.array(lower), np.array(upper)


def pso_optimize_classifier(model_name, X_train, y_train,
                            particles=12, iterations=30):

    lower, upper = get_bounds(model_name)
    dim = len(lower)

    def objective(params):

        scores = []

        for particle in params:

            model = build_model(model_name, particle)

            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores = []

            for tr_idx, val_idx in cv.split(X_train):

                X_tr = X_train[tr_idx]
                X_va = X_train[val_idx]

                y_tr = y_train[tr_idx]
                y_va = y_train[val_idx]

                model.fit(X_tr, y_tr)
                pred = model.predict(X_va)

                fold_scores.append(
                    balanced_accuracy_score(y_va, pred)
                )

            scores.append(-np.mean(fold_scores))

        return np.array(scores)


    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=dim,
        options={'c1':0.5, 'c2':0.3, 'w':0.9},
        bounds=(lower, upper)
    )

    start_time = time.time()

    best_cost, best_pos = optimizer.optimize(
        objective,
        iters=iterations,
        verbose=True
    )

    total_time = time.time() - start_time

    best_model = build_model(model_name, best_pos)
    best_model.fit(X_train, y_train)

    return {
        "Best_Model": best_model,
        "Best_Params": best_pos,
        "Time_sec": total_time
    }