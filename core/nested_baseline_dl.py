import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, InputLayer
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# ===== MODEL BUILDERS ====
# =========================

def build_mlp_reg(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def build_mlp_clf(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def build_cnn_reg(input_dim):
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


# ============================================
# ===== Nested Baseline Regression (DL) =====
# ============================================

def run_nested_baseline_regression_dl(X, y_reg, outer_folds=5):

    models = ["MLP", "CNN"]
    results = {}

    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

    for model_name in models:

        outer_scores = []

        for train_idx, test_idx in outer_cv.split(X):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_reg[train_idx], y_reg[test_idx]

            # ===== Scaling (ONLY ONCE) =====
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            early_stop = EarlyStopping(
                patience=15,
                restore_best_weights=True
            )

            # ===== MLP =====
            if model_name == "MLP":

                model = build_mlp_reg(X.shape[1])

                model.fit(
                    X_train, y_train,
                    epochs=150,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.1,
                    callbacks=[early_stop]
                )

                pred = model.predict(X_test, verbose=0).flatten()

            # ===== CNN =====
            elif model_name == "CNN":

                X_train_cnn = X_train.reshape(-1, X.shape[1], 1)
                X_test_cnn = X_test.reshape(-1, X.shape[1], 1)

                model = build_cnn_reg(X.shape[1])

                model.fit(
                    X_train_cnn, y_train,
                    epochs=150,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.1,
                    callbacks=[early_stop]
                )

                pred = model.predict(X_test_cnn, verbose=0).flatten()

            score = r2_score(y_test, pred)
            outer_scores.append(score)

        results[model_name] = (
            np.mean(outer_scores),
            np.std(outer_scores)
        )

    return results


# ===============================================
# ===== Nested Baseline Classification (DL) ====
# ===============================================

def run_nested_baseline_classification_dl(X, y_class, outer_folds=5):

    models = ["MLP", "CNN"]
    results = {}

    outer_cv = StratifiedKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=42
    )

    for model_name in models:

        outer_scores = []

        for train_idx, test_idx in outer_cv.split(X, y_class):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_class[train_idx], y_class[test_idx]

            # ===== Scaling (ONLY ONCE) =====
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            early_stop = EarlyStopping(
                patience=15,
                restore_best_weights=True
            )

            # ===== MLP =====
            if model_name == "MLP":

                model = build_mlp_clf(X.shape[1])

                model.fit(
                    X_train, y_train,
                    epochs=150,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.1,
                    callbacks=[early_stop]
                )

                proba = model.predict(X_test, verbose=0).flatten()
                pred = (proba >= 0.5).astype(int)

            # ===== CNN =====
            elif model_name == "CNN":

                X_train_cnn = X_train.reshape(-1, X.shape[1], 1)
                X_test_cnn = X_test.reshape(-1, X.shape[1], 1)

                model = build_cnn_clf(X.shape[1])

                model.fit(
                    X_train_cnn, y_train,
                    epochs=150,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.1,
                    callbacks=[early_stop]
                )

                proba = model.predict(X_test_cnn, verbose=0).flatten()
                pred = (proba >= 0.5).astype(int)

            score = balanced_accuracy_score(y_test, pred)
            outer_scores.append(score)

        results[model_name] = (
            np.mean(outer_scores),
            np.std(outer_scores)
        )

    return results