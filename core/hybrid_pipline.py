import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def run_hybrid_validation(
        reg_model,
        clf_model,
        X_val,
        y_val_reg,
        threshold_reg=2.8,
        p_min=0.50,
        p_max=0.90,
        step=0.01
):
    """
    reg_model  : trained regression model
    clf_model  : trained classification model
    X_val      : validation features
    y_val_reg  : continuous aroma score
    """

    # ---------------------------------------
    # 1. Prepare validation ground truth
    # ---------------------------------------
    y_val_binary = (y_val_reg >= threshold_reg).astype(int)

    # ---------------------------------------
    # 2. Regression prediction
    # ---------------------------------------
    reg_pred = reg_model.predict(X_val)
    reg_binary = (reg_pred >= threshold_reg).astype(int)

    # ---------------------------------------
    # 3. Classification prediction
    # ---------------------------------------
    proba_val = clf_model.predict_proba(X_val)[:, 1]
    clf_binary = (proba_val >= 0.5).astype(int)

    # ---------------------------------------
    # 4. Grid search pRef
    # ---------------------------------------
    p_values = np.arange(p_min, p_max + step, step)

    best_p = None
    best_score = -np.inf
    best_cm = None

    grid_results = []

    for p in p_values:

        final_pred = np.where(
            proba_val < p,
            reg_binary,
            clf_binary
        )

        score = balanced_accuracy_score(y_val_binary, final_pred)
        tn, fp, fn, tp = confusion_matrix(y_val_binary, final_pred).ravel()

        grid_results.append({
            "pRef": p,
            "BAcc": score,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        })

        if score > best_score:
            best_score = score
            best_p = p
            best_cm = (tp, tn, fp, fn)

    return best_p, best_score, best_cm, grid_results