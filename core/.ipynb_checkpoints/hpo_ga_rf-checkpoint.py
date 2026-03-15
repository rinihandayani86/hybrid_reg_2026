import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer
from sklearn.pipeline import Pipeline


def ga_optimize_rf(X_train, y_train,
                   cv_splits=5,
                   population_size=15,
                   generations=15,
                   random_state=42):

    # Search space (masuk akal, tidak terlalu lebar)
    param_grid = {
        "model__n_estimators": Integer(100, 300),
        "model__max_depth": Integer(5, 30),
        "model__min_samples_split": Integer(2, 10)
    }

    # Pipeline (lebih rapi untuk param naming)
    pipe = Pipeline([
        ("model", RandomForestRegressor(
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