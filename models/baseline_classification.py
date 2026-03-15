from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_baseline_model(name):

    if name == "RF":
        return RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

    elif name == "XGB":
        return XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )