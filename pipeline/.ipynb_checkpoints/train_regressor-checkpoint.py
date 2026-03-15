import pandas as pd

def train_regressor(model, X_train, y_train_reg, X_val):

    model.fit(X_train, y_train_reg)

    reg_score = model.predict(X_val)

    reg_class = (reg_score >= 2.8).astype(int)

    df_reg = pd.DataFrame({
        "reg_score": reg_score,
        "reg_class": reg_class
    })

    return df_reg