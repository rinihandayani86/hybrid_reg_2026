import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous


def ga_optimize_xgb(X_train, y_train,
                    cv_splits=5,
                    population_size=15,
                    generations=15,
                    random_state=42):

    param_grid = {
        "model__n_estimators": Integer(100, 300),
        "model__max_depth": Integer(3, 10),
        "model__learning_rate": Continuous(0.03, 0.2),
        "model__subsample": Continuous(0.7, 1.0),
        "model__colsample_bytree": Continuous(0.7, 1.0)
    }

    pipe = Pipeline([
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        ))
    ])

    inner_cv = KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    ga_search = GASearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="r2",
        cv=inner_cv,
        population_size=population_size,
        generations=generations,
        n_jobs=-1,
        verbose=True
    )

    ga_search.fit(X_train, y_train)

    return ga_search.best_estimator_, ga_search.best_params_