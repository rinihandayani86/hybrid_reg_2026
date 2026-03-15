import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# =============================================
# MODEL FACTORY
# =============================================

def get_reg_model(model_name):

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

    elif model_name == "XGB":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

    elif model_name == "LGBM":
        return LGBMRegressor(random_state=42)

    else:
        raise ValueError("Unknown model name")


# =============================================
# SEARCH SPACE
# =============================================

def get_param_grid(model_name):

    if model_name == "DT":
        return {
            "model__max_depth": Integer(3, 30),
            "model__min_samples_split": Integer(2, 20)
        }

    elif model_name == "BaggingDT":
        return {
            "model__n_estimators": Integer(50, 300),
            "model__max_samples": Continuous(0.5, 1.0)
        }

    elif model_name == "RF":
        return {
            "model__n_estimators": Integer(100, 300),
            "model__max_depth": Integer(5, 30),
            "model__min_samples_split": Integer(2, 10)
        }

    elif model_name == "XGB":
        return {
            "model__n_estimators": Integer(100, 300),
            "model__max_depth": Integer(3, 10),
            "model__learning_rate": Continuous(0.03, 0.2),
            "model__subsample": Continuous(0.7, 1.0),
            "model__colsample_bytree": Continuous(0.7, 1.0)
        }

    elif model_name == "LGBM":
        return {
            "model__n_estimators": Integer(100, 400),
            "model__num_leaves": Integer(20, 100),
            "model__learning_rate": Continuous(0.03, 0.2),
            "model__subsample": Continuous(0.7, 1.0)
        }


# =============================================
# GA OPTIMIZATION PIPELINE
# =============================================

def run_ga_regression(
        model_name,
        X_train,
        y_train,
        X_val,
        y_val,
        cv_splits=5,
        population_size=15,
        generations=15
):

    pipe = Pipeline([
        ("model", get_reg_model(model_name))
    ])

    inner_cv = KFold(n_splits=cv_splits,
                     shuffle=True,
                     random_state=42)

    ga = GASearchCV(
        estimator=pipe,
        param_grid=get_param_grid(model_name),
        scoring="r2",
        cv=inner_cv,
        population_size=population_size,
        generations=generations,
        n_jobs=-1,
        verbose=True
    )

    start_time = time.time()
    ga.fit(X_train, y_train)
    total_time = time.time() - start_time

    best_model = ga.best_estimator_
    best_params = ga.best_params_

    # Validation evaluation
    pred_val = best_model.predict(X_val)
    r2 = r2_score(y_val, pred_val)
    mse = mean_squared_error(y_val, pred_val)

    return {
        "Best_Params": best_params,
        "R2_val": r2,
        "MSE_val": mse,
        "Time_sec": total_time
    }