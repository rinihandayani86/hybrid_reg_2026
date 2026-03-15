from sklearn.metrics import balanced_accuracy_score, confusion_matrix

def evaluate_results(y_true, y_pred):

    bacc = balanced_accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "BalancedAccuracy": bacc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }