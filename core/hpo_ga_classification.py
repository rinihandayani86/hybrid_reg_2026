import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_model(model_name, random_state=42):

    if model_name == "RF":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)

    elif model_name == "XGB":
        return XGBClassifier(
            objective="binary:logistic",
            random_state=random_state,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss"
        )

    elif model_name == "LGBM":
        return LGBMClassifier(random_state=random_state)

    else:
        raise ValueError("Unknown model name")


def get_param_grid(model_name):

    if model_name == "RF":
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


def ga_optimize_classifier(model_name, X_train, y_train,
                           cv_splits=5,
                           population_size=15,
                           generations=15):

    pipe = Pipeline([
        ("model", get_model(model_name))
    ])

    inner_cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    ga_search = GASearchCV(
        estimator=pipe,
        param_grid=get_param_grid(model_name),
        scoring="balanced_accuracy",
        cv=inner_cv,
        population_size=population_size,
        generations=generations,
        n_jobs=-1,
        verbose=True
    )

    start_time = time.time()
    ga_search.fit(X_train, y_train)
    total_time = time.time() - start_time

    best_model = ga_search.best_estimator_
    best_params = ga_search.best_params_

    return {
        "Best_Model": best_model,
        "Best_Params": best_params,
        "Time_sec": total_time
    }