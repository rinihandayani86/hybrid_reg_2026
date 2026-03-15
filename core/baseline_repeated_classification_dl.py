import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session


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


def run_baseline_repeated_classification_dl(X, y):

    rskf = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

    results = {"MLP": [], "CNN": []}

    for train_idx, test_idx in rskf.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        early_stop = EarlyStopping(patience=15, restore_best_weights=True)

        # ===== MLP =====
        clear_session()
        mlp = build_mlp_clf(X.shape[1])
        mlp.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            verbose=0,
            validation_split=0.1,
            callbacks=[early_stop]
        )

        proba = mlp.predict(X_test, verbose=0).flatten()
        pred = (proba >= 0.5).astype(int)
        results["MLP"].append(balanced_accuracy_score(y_test, pred))

        # ===== CNN =====
        clear_session()
        X_train_cnn = X_train.reshape(-1, X.shape[1], 1)
        X_test_cnn = X_test.reshape(-1, X.shape[1], 1)

        cnn = build_cnn_clf(X.shape[1])
        cnn.fit(
            X_train_cnn, y_train,
            epochs=150,
            batch_size=32,
            verbose=0,
            validation_split=0.1,
            callbacks=[early_stop]
        )

        proba = cnn.predict(X_test_cnn, verbose=0).flatten()
        pred = (proba >= 0.5).astype(int)
        results["CNN"].append(balanced_accuracy_score(y_test, pred))

    final_results = {}

    for model in results:
        final_results[model] = (
            np.mean(results[model]),
            np.std(results[model])
        )

    return final_results