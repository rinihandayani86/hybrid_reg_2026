import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous


def ga_optimize_baggingdt(X_train, y_train,
                          cv_splits=5,
                          population_size=15,
                          generations=15,
                          random_state=42):

    base_tree = DecisionTreeRegressor(random_state=random_state)

    param_grid = {
        "model__n_estimators": Integer(50, 300),
        "model__max_samples": Continuous(0.6, 1.0),
        "model__estimator__max_depth": Integer(5, 30)
    }

    pipe = Pipeline([
        ("model", BaggingRegressor(
            estimator=base_tree,
            random_state=random_state,
            n_jobs=-1
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