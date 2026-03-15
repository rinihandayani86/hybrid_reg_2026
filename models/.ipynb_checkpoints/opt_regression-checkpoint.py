from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def get_ml_model_reg(name, dataset):

    # ====================================
    # RANDOM FOREST REGRESSOR
    # ====================================

    if name == "RF":

        if dataset == "Crop_1":
            return RandomForestRegressor(
                n_estimators=235,
                max_depth=24,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            )

        elif dataset == "Crop_2":
            return RandomForestRegressor(
                n_estimators=232,
                max_depth=24,
                min_samples_split=3,
                random_state=42,
                n_jobs=-1
            )

    # ====================================
    # XGBOOST REGRESSOR
    # ====================================

    elif name == "XGB":

        if dataset == "Crop_1":
            return XGBRegressor(
                n_estimators=246,
                max_depth=10,
                learning_rate=0.06280248314535586,
                subsample=0.8014052751840177,
                colsample_bytree=0.8365920989802558,
                random_state=42,
                n_jobs=-1
            )

        elif dataset == "Crop_2":
            return XGBRegressor(
                n_estimators=165,
                max_depth=10,
                learning_rate=0.05647328743891381,
                subsample=0.871801729007803,
                colsample_bytree=0.8586879457170303,
                random_state=42,
                n_jobs=-1
            )

    else:
        raise ValueError("Unknown model")