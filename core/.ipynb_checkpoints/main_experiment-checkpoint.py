import sys
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from core.data_loader2 import load_and_split_dataset
from models.opt_classification import get_ml_model_clf
from models.opt_regression import get_ml_model_reg

from pipeline.train_classifier import train_classifier
from pipeline.train_regressor import train_regressor
from pipeline.hybrid_pref_search import search_best_pref
from pipeline.hybrid_decision import apply_hybrid

from utils.evaluation import evaluate_results
from utils.experiment_logger import ExperimentLogger


# ======================
# MAIN EXPERIMENT
# ======================

def run_experiment(datasets):

    logger = ExperimentLogger()

    classifiers = ["RF","XGB"]
    regressors = ["RF","XGB"]

    for ds in datasets:

        data = load_and_split_dataset(ds)

        X_train = data["train"]["X"]
        y_train_class = data["train"]["y_class"]
        y_train_reg = data["train"]["y_reg"]

        X_val = data["val"]["X"]
        y_val = data["val"]["y_class"]

        X_test = data["test"]["X"]
        y_test_class = data["test"]["y_class"]
        y_test_reg = data["test"]["y_reg"]

        for clf_name in classifiers:

            clf_model = get_ml_model_clf(clf_name, ds)

            # ======================
            # VALIDATION CLASSIFIER
            # ======================

            df_classifier_val = train_classifier(
                clf_model,
                X_train, y_train_class,
                X_val, y_val
            )

            for reg_name in regressors:

                print(f"Running: {ds} | {clf_name} | {reg_name}")

                reg_model = get_ml_model_reg(reg_name, ds)

                # ======================
                # VALIDATION REGRESSOR
                # ======================

                df_reg_val = train_regressor(
                    reg_model,
                    X_train, y_train_reg,
                    X_val
                )

                # ======================
                # SEARCH BEST pRef
                # ======================

                best_pref, history = search_best_pref(
                    df_classifier_val,
                    df_reg_val
                )

                # ======================
                # TEST PREDICTIONS
                # ======================

                prob_test = clf_model.predict_proba(X_test)[:,1]
                pred_test_clf = clf_model.predict(X_test)

                reg_score_test = reg_model.predict(X_test)
                reg_class_test = (reg_score_test >= 2.8).astype(int)

                df_classifier_test = pd.DataFrame({
                    "y_true": y_test_class,
                    "prob_classifier": prob_test,
                    "pred_classifier": pred_test_clf
                })

                df_reg_test = pd.DataFrame({
                    "reg_class": reg_class_test
                })

                df_hybrid_test = apply_hybrid(
                    df_classifier_test,
                    df_reg_test,
                    best_pref
                )

                # ======================
                # EVALUATION (TEST)
                # ======================

                metrics_test = evaluate_results(
                    df_hybrid_test["y_true"],
                    df_hybrid_test["final_pred"]
                )

                metrics_test["Dataset"] = ds
                metrics_test["Classifier"] = clf_name
                metrics_test["Regressor"] = reg_name
                metrics_test["pRef"] = best_pref
                metrics_test["Split"] = "Test"

                logger.log_results(metrics_test)
                df_hybrid_test["Dataset"] = ds
                df_hybrid_test["Classifier"] = clf_name
                df_hybrid_test["Regressor"] = reg_name
                df_hybrid_test["pRef"] = best_pref
                #df_hybrid_test["reg_score"] = reg_score_test
                
                logger.log_predictions(df_hybrid_test)
                logger.log_pref_history(ds, clf_name, reg_name, history)

    logger.save()


# ======================
# ENTRY POINT
# ======================

if __name__ == "__main__":

    print("Starting Hybrid Experiment...")

    datasets = ["Crop_1","Crop_2"]

    run_experiment(datasets)

    print("Experiment finished.")