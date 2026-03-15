import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import pyswarms as ps
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous

# Model Factory
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

# Parameter Space - GA
def get_ga_space(model_name):

    if model_name == "DT":
        return {
            'max_depth': Integer(3, 15),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10)
        }

    elif model_name == "BaggingDT":
        return {
            'n_estimators': Integer(10, 200),
            'max_samples': Continuous(0.5, 1.0)
        }

    elif model_name == "RF":
        return {
            'n_estimators': Integer(100, 300),
            'max_depth': Integer(8, 20),
            'min_samples_split': Integer(2, 8)
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

# Parameter Space - PSO
def get_pso_bounds(model_name):

    if model_name == "DT":
        lb = [3, 2, 1]
        ub = [15, 20, 10]

    elif model_name == "BaggingDT":
        lb = [10, 0.5]
        ub = [200, 1.0]

    elif model_name == "RF":
        lb = [50, 3, 2]
        ub = [300, 20, 15]

    elif model_name == "XGB":
        lb = [50, 3, 0.01, 0.5, 0.5]
        ub = [300, 10, 0.3, 1.0, 1.0]

    elif model_name == "LGBM":
        lb = [50, 15, 0.01]
        ub = [300, 63, 0.3]

    return (np.array(lb), np.array(ub))

# Decode PSO Parameter Vector
def decode_params(model_name, position):

    if model_name == "DT":
        return {
            "max_depth": int(position[0]),
            "min_samples_split": int(position[1]),
            "min_samples_leaf": int(position[2])
        }

    elif model_name == "BaggingDT":
        return {
            "n_estimators": int(position[0]),
            "max_samples": float(position[1])
        }

    elif model_name == "RF":
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

    elif model_name == "LGBM":
        return {
            "n_estimators": int(position[0]),
            "num_leaves": int(position[1]),
            "learning_rate": float(position[2])
        }

# PSO Optimization
def pso_optimize(model_name, X, y):

    def objective(params):

        n_particles = params.shape[0]
        results = []

        for i in range(n_particles):

            param_dict = decode_params(model_name, params[i])
            model = get_model(model_name, param_dict)

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

    bounds = get_pso_bounds(model_name)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=10,
        dimensions=len(bounds[0]),
        options={'c1':0.5,'c2':0.3,'w':0.9},
        bounds=bounds
    )

    best_cost, best_pos = optimizer.optimize(objective, iters=30)

    return decode_params(model_name, best_pos)

# GA Optimization
def ga_optimize(model_name, X, y):

    param_grid = get_ga_space(model_name)
    model = get_model(model_name)

    ga = GASearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="r2",
        cv=3,
        population_size=20,
        generations=30,
        n_jobs=-1,
        verbose=True
    )

    ga.fit(X, y)

    return ga.best_params_

# Nested Runner
def run_nested_hpo_regression(model_name, optimizer_name, X, y):

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_scores = []

    for train_idx, test_idx in outer_cv.split(X):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_fold = X_train
        X_test_fold = X_test

        if optimizer_name == "PSO":
            best_params = pso_optimize(model_name, X_train, y_train)

        elif optimizer_name == "GA":
            best_params = ga_optimize(model_name, X_train, y_train)

        model = get_model(model_name, best_params)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        outer_scores.append(r2_score(y_test, pred))

    return np.mean(outer_scores), np.std(outer_scores)