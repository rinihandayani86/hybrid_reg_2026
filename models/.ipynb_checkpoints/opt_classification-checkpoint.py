from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =====================================================
# MODEL FACTORY - ML
# =====================================================

def get_ml_model_clf(name, dataset):

    # ======================
    # RANDOM FOREST
    # ======================
    if name == "RF":

        if dataset == "Crop_1":
            return RandomForestClassifier(
                n_estimators=117,
                max_depth=24,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            )

        elif dataset == "Crop_2":
            return RandomForestClassifier(
                n_estimators=110,
                max_depth=30,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            )

    # ======================
    # XGBOOST
    # ======================
    elif name == "XGB":

        if dataset == "Crop_1":
            return XGBClassifier(
                n_estimators=236,
                max_depth=9,
                learning_rate=0.13189988653377488,
                subsample=0.8901402469684817,
                colsample_bytree=0.9236816313877201,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                use_label_encoder=False
            )

        elif dataset == "Crop_2":
            return XGBClassifier(
                n_estimators=210,
                max_depth=8,
                learning_rate=0.12,
                subsample=0.85,
                colsample_bytree=0.90,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                use_label_encoder=False
            )

    else:
        raise ValueError("Unknown model")