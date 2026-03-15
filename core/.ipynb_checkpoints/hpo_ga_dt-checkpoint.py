import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer

def ga_optimize_dt(X_train, y_train,
                   cv_splits=5,
                   population_size=15,
                   generations=15,
                   random_state=42):

    param_grid = {
        "model__max_depth": Integer(5, 30),
        "model__min_samples_split": Integer(2, 15),
        "model__min_samples_leaf": Integer(1, 5)
    }

    pipe = Pipeline([
        ("model", DecisionTreeRegressor(
            random_state=random_state
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