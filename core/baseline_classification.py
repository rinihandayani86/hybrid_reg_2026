import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session


# =====================================================
# MODEL FACTORY - ML
# =====================================================

def get_ml_model_clf(name):

    if name == "DT":
        return DecisionTreeClassifier(random_state=42)

    elif name == "BaggingDT":
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            random_state=42,
            n_jobs=-1
        )

    elif name == "RF":
        return RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

    elif name == "AdaBoost":
        return AdaBoostClassifier(random_state=42)

    elif name == "GradBoost":
        return GradientBoostingClassifier(random_state=42)

    elif name == "XGB":
        return XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False
        )

    elif name == "LGBM":
        return LGBMClassifier(random_state=42)

    else:
        raise ValueError("Unknown model name")


# =====================================================
# DL MODELS
# =====================================================

def build_mlp_clf(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def build_cnn_clf(input_dim):
    model = Sequential([
        InputLayer(input_shape=(input_dim, 1)),
        Conv1D(32, kernel_size=2, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


# =====================================================
# BASELINE CLASSIFICATION PIPELINE
# =====================================================

def run_baseline_classification(X_train, y_train,
                                X_val, y_val):

    ml_models = [
        "DT",
        "BaggingDT",
        "RF",
        "AdaBoost",
        "GradBoost",
        "XGB",
        "LGBM"
    ]

    results = {}

    # ======================
    # ML MODELS
    # ======================
    for name in ml_models:

        model = get_ml_model_clf(name)
        model.fit(X_train, y_train)

        pred = model.predict(X_val)

        bacc = balanced_accuracy_score(y_val, pred)
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()

        results[name] = {
            "Balanced_Accuracy": bacc,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }

    # ======================
    # DL MODELS
    # ======================

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    # ===== MLP =====
    clear_session()
    mlp = build_mlp_clf(X_train.shape[1])
    mlp.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        verbose=0,
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stop]
    )

    proba_mlp = mlp.predict(X_val_scaled, verbose=0).flatten()
    pred_mlp = (proba_mlp >= 0.5).astype(int)

    bacc = balanced_accuracy_score(y_val, pred_mlp)
    tn, fp, fn, tp = confusion_matrix(y_val, pred_mlp).ravel()

    results["MLP"] = {
        "Balanced_Accuracy": bacc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }

    # ===== CNN =====
    clear_session()
    X_train_cnn = X_train_scaled.reshape(-1, X_train.shape[1], 1)
    X_val_cnn = X_val_scaled.reshape(-1, X_val.shape[1], 1)

    cnn = build_cnn_clf(X_train.shape[1])
    cnn.fit(
        X_train_cnn, y_train,
        epochs=50,
        batch_size=32,
        verbose=0,
        validation_data=(X_val_cnn, y_val),
        callbacks=[early_stop]
    )

    proba_cnn = cnn.predict(X_val_cnn, verbose=0).flatten()
    pred_cnn = (proba_cnn >= 0.5).astype(int)

    bacc = balanced_accuracy_score(y_val, pred_cnn)
    tn, fp, fn, tp = confusion_matrix(y_val, pred_cnn).ravel()

    results["CNN"] = {
        "Balanced_Accuracy": bacc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }

    return results