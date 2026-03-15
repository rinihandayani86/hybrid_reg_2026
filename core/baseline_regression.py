import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, InputLayer
from tensorflow.keras.callbacks import EarlyStopping

import os
import matplotlib.pyplot as plt

def plot_predicted_vs_actual(y_true, y_pred,
                             model_name,
                             dataset_name,
                             save_dir="figures"):

    os.makedirs(save_dir, exist_ok=True)

    r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=(6,6))

    plt.scatter(y_true, y_pred, alpha=0.4)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.plot([min_val, max_val],
             [min_val, max_val],
             'r--', linewidth=2)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    #plt.title(f"{model_name} - Validation\nR² = {r2:.4f}")
    
    plt.grid(True)
    plt.text(0.95, 0.05, f"R² = {r2:.4f}",
         ha='right', va='bottom', transform=plt.gca().transAxes,
         fontsize=10, bbox=dict(facecolor='white', edgecolor='gray'))
    
    plt.tight_layout()

    file_path = os.path.join(
        save_dir,
        f"{dataset_name}_{model_name}_val_pred_vs_actual.pdf"
    )

    plt.savefig(file_path, format="pdf", dpi=300)
    plt.close()

    print(f"Saved: {file_path}")

# ==============================
# MODEL FACTORY - ML
# ==============================

def get_ml_model(name):

    if name == "DT":
        return DecisionTreeRegressor(random_state=42)

    elif name == "BaggingDT":
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(),
            random_state=42,
            n_jobs=-1
        )

    elif name == "RF":
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )

    elif name == "AdaBoost":
        return AdaBoostRegressor(random_state=42)

    elif name == "GradBoost":
        return GradientBoostingRegressor(random_state=42)

    elif name == "XGB":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

    elif name == "LGBM":
        return LGBMRegressor(random_state=42)

    else:
        raise ValueError("Unknown ML model")


# ==============================
# DL MODELS
# ==============================

def build_mlp(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def build_cnn(input_dim):
    model = Sequential([
        InputLayer(input_shape=(input_dim, 1)),
        Conv1D(32, kernel_size=2, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ==============================
# BASELINE REGRESSION PIPELINE
# ==============================

def run_baseline_regression(X_train, y_train,
                            X_val, y_val,
                            dataset_name):

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

        model = get_ml_model(name)
        model.fit(X_train, y_train)

        pred = model.predict(X_val)

        r2 = r2_score(y_val, pred)
        mse = mean_squared_error(y_val, pred)

        results[name] = {
            "R2": r2,
            "MSE": mse
        }

        # Plot hanya untuk RF dan XGB
        if name in ["RF", "XGB"]:
            plot_predicted_vs_actual(
                y_val,
                pred,
                model_name=name,
                dataset_name=dataset_name
            )

    # ======================
    # DL MODELS
    # ======================

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    # ----- MLP -----
    mlp = build_mlp(X_train.shape[1])
    mlp.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        verbose=0,
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stop]
    )

    pred_mlp = mlp.predict(X_val_scaled, verbose=0).flatten()

    results["MLP"] = {
        "R2": r2_score(y_val, pred_mlp),
        "MSE": mean_squared_error(y_val, pred_mlp)
    }

    # ----- CNN -----
    X_train_cnn = X_train_scaled.reshape(-1, X_train.shape[1], 1)
    X_val_cnn = X_val_scaled.reshape(-1, X_val.shape[1], 1)

    cnn = build_cnn(X_train.shape[1])
    cnn.fit(
        X_train_cnn, y_train,
        epochs=50,
        batch_size=32,
        verbose=0,
        validation_data=(X_val_cnn, y_val),
        callbacks=[early_stop]
    )

    pred_cnn = cnn.predict(X_val_cnn, verbose=0).flatten()

    results["CNN"] = {
        "R2": r2_score(y_val, pred_cnn),
        "MSE": mean_squared_error(y_val, pred_cnn)
    }

    return results