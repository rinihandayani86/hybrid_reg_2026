import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session


# =====================================================
# MODEL BUILDERS
# =====================================================

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


# =====================================================
# REPEATED CV BASELINE DL
# =====================================================

def run_baseline_repeated_dl(X, y):

    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

    results = {
        "MLP": [],
        "CNN": []
    }

    for train_idx, test_idx in rkf.split(X):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scaling (DL wajib)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        early_stop = EarlyStopping(
            patience=15,
            restore_best_weights=True
        )

        # ======================
        # MLP
        # ======================
        clear_session()
        mlp = build_mlp(X.shape[1])

        mlp.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            verbose=0,
            validation_split=0.1,
            callbacks=[early_stop]
        )

        pred_mlp = mlp.predict(X_test, verbose=0).flatten()
        results["MLP"].append(r2_score(y_test, pred_mlp))

        # ======================
        # CNN
        # ======================
        clear_session()
        X_train_cnn = X_train.reshape(-1, X.shape[1], 1)
        X_test_cnn = X_test.reshape(-1, X.shape[1], 1)

        cnn = build_cnn(X.shape[1])

        cnn.fit(
            X_train_cnn, y_train,
            epochs=150,
            batch_size=32,
            verbose=0,
            validation_split=0.1,
            callbacks=[early_stop]
        )

        pred_cnn = cnn.predict(X_test_cnn, verbose=0).flatten()
        results["CNN"].append(r2_score(y_test, pred_cnn))

    final_results = {}

    for model_name in results:
        final_results[model_name] = (
            np.mean(results[model_name]),
            np.std(results[model_name])
        )

    return final_results