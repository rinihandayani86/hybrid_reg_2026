import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import pyswarms as ps
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous


# =====================================================
# MODEL FACTORY
# =====================================================

def get_model(model_name, params=None):

    if params is None:
        params = {}

    if model_name == "DT":
        return DecisionTreeRegressor(random_state=42, **params)

    elif model_name == "BaggingDT":
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(),
            random_state=42,
            n_jobs=-1,
            **params
        )

    elif model_name == "RF":
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )

    elif model_name == "XGB":
        return XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **params
        )

    elif model_name == "LGBM":
        return LGBMRegressor(
            random_state=42,
            **params
        )

    else:
        raise ValueError("Unknown model")


# =====================================================
# GA SEARCH SPACE
# =====================================================

def get_ga_space(model_name):

    if model_name == "DT":
        return {
            'max_depth': Integer(3, 15),
            'min_samples_split': Integer(2, 20)
        }

    elif model_name == "BaggingDT":
        return {
            'n_estimators': Integer(10, 200),
            'max_samples': Continuous(0.5, 1.0)
        }

    elif model_name == "RF":
        return {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 30),
            'min_samples_split': Integer(2, 20)
        }

    elif model_name == "XGB":
        return {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 10),
            'learning_rate': Continuous(0.01, 0.3),
            'subsample': Continuous(0.5, 1.0),
            'colsample_bytree': Continuous(0.5, 1.0)
        }

    elif model_name == "LGBM":
        return {
            'n_estimators': Integer(50, 300),
            'num_leaves': Integer(15, 63),
            'learning_rate': Continuous(0.01, 0.3)
        }


# =====================================================
# GA OPTIMIZATION
# =====================================================

def ga_optimize(model_name, X, y):

    param_grid = get_ga_space(model_name)
    model = get_model(model_name)

    ga = GASearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="r2",
        cv=5,
        population_size=20,
        generations=30,
        n_jobs=-1,
        verbose=True
    )

    ga.fit(X, y)

    return ga.best_params_


# =====================================================
# PSO OPTIMIZATION
# =====================================================

def get_pso_bounds(model_name):

    if model_name == "RF":
        lb = [50, 3, 2]
        ub = [300, 30, 20]

    elif model_name == "XGB":
        lb = [50, 3, 0.01, 0.5, 0.5]
        ub = [300, 10, 0.3, 1.0, 1.0]

    else:
        raise ValueError("PSO bounds not defined for this model")

    return (np.array(lb), np.array(ub))


def decode_params(model_name, position):

    if model_name == "RF":
        return {
            "n_estimators": int(position[0]),
            "max_depth": int(position[1]),
            "min_samples_split": int(position[2])
        }

    elif model_name == "XGB":
        return {
            "n_estimators": int(position[0]),
            "max_depth": int(position[1]),
            "learning_rate": float(position[2]),
            "subsample": float(position[3]),
            "colsample_bytree": float(position[4])
        }


def pso_optimize(model_name, X, y):

    bounds = get_pso_bounds(model_name)

    def objective(params):

        results = []

        for particle in params:

            param_dict = decode_params(model_name, particle)
            model = get_model(model_name, param_dict)

            from sklearn.model_selection import KFold
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

            fold_scores = []

            for tr_idx, val_idx in inner_cv.split(X):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                fold_scores.append(r2_score(y_val, pred))

            results.append(-np.mean(fold_scores))

        return np.array(results)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=10,
        dimensions=len(bounds[0]),
        options={'c1':0.5,'c2':0.3,'w':0.9},
        bounds=bounds
    )

    best_cost, best_pos = optimizer.optimize(objective, iters=30)

    return decode_params(model_name, best_pos)


# =====================================================
# REPEATED CV EVALUATION
# =====================================================

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