import pandas as pd

def train_classifier(model, X_train, y_train, X_val, y_val):

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_val)[:,1]
    pred = model.predict(X_val)

    df_classifier = pd.DataFrame({
        "y_true": y_val,
        "prob_classifier": prob,
        "pred_classifier": pred
    })

    return df_classifier