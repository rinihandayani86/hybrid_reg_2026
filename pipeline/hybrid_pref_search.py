import numpy as np
from sklearn.metrics import balanced_accuracy_score


def search_best_pref(df_classifier, df_reg):

    pref_candidates = np.arange(0.5, 0.96, 0.01)

    best_pref = None
    best_score = -1

    # tambahkan ini
    history = []

    for pRef in pref_candidates:

        final_pred = df_classifier["pred_classifier"].copy()

#        mask = df_classifier["prob_classifier"] < pRef
        mask = (
            (df_classifier["prob_classifier"] > 0.5) &
            (df_classifier["prob_classifier"] < pRef)
        )

        final_pred[mask] = df_reg["reg_class"][mask]

        score = balanced_accuracy_score(
            df_classifier["y_true"],
            final_pred
        )

        history.append((pRef, score))

        if score > best_score:
            best_score = score
            best_pref = pRef

    return best_pref, history