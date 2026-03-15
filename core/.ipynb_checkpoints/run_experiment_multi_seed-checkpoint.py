import sys
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from core.data_loader2 import load_and_split_dataset

from models.baseline_classification import get_baseline_model
from models.opt_classification import get_ml_model_clf
from models.opt_regression import get_ml_model_reg

from pipeline.train_classifier import train_classifier
from pipeline.train_regressor import train_regressor
from pipeline.hybrid_pref_search import search_best_pref
from pipeline.hybrid_decision import apply_hybrid

from utils.evaluation import evaluate_results


def run_experiment_multi_seed(datasets, seeds=[0,1,2,3,4]):

    classifiers = ["RF","XGB"]
    regressors = ["RF","XGB"]

    results = []

    for seed in seeds:

        print(f"\n===== SEED {seed} =====")

        for ds in datasets:

            data = load_and_split_dataset(
                ds,
                random_state=seed
            )

            X_train = data["train"]["X"]
            y_train_class = data["train"]["y_class"]
            y_train_reg = data["train"]["y_reg"]

            X_val = data["val"]["X"]
            y_val = data["val"]["y_class"]

            X_test = data["test"]["X"]
            y_test_class = data["test"]["y_class"]

            for clf_name in classifiers:

                # ======================================
                # BASELINE CLASSIFIER (default params)
                # ======================================

                baseline_model = get_baseline_model(clf_name)

                baseline_model.fit(X_train, y_train_class)

                pred_base = baseline_model.predict(X_test)

                metrics_base = evaluate_results(
                    y_test_class,
                    pred_base
                )

                metrics_base["Dataset"] = ds
                metrics_base["Classifier"] = clf_name
                metrics_base["Regressor"] = "-"
                metrics_base["Seed"] = seed
                metrics_base["ModelType"] = "Baseline"

                results.append(metrics_base)

                # ======================================
                # GA OPTIMIZED CLASSIFIER
                # ======================================

                ga_model = get_ml_model_clf(clf_name, ds)

                ga_model.fit(X_train, y_train_class)

                pred_ga = ga_model.predict(X_test)

                metrics_ga = evaluate_results(
                    y_test_class,
                    pred_ga
                )

                metrics_ga["Dataset"] = ds
                metrics_ga["Classifier"] = clf_name
                metrics_ga["Regressor"] = "-"
                metrics_ga["Seed"] = seed
                metrics_ga["ModelType"] = "GA"

                results.append(metrics_ga)

                # ======================================
                # VALIDATION PHASE (pRef search)
                # ======================================

                clf_val_model = get_ml_model_clf(clf_name, ds)

                df_classifier_val = train_classifier(
                    clf_val_model,
                    X_train, y_train_class,
                    X_val, y_val
                )

                for reg_name in regressors:

                    print(f"{ds} | {clf_name} | {reg_name}")

                    reg_model = get_ml_model_reg(reg_name, ds)

                    df_reg_val = train_regressor(
                        reg_model,
                        X_train, y_train_reg,
                        X_val
                    )

                    best_pref, _ = search_best_pref(
                        df_classifier_val,
                        df_reg_val
                    )

                    # ======================================
                    # TEST PHASE
                    # ======================================

                    clf_test_model = get_ml_model_clf(clf_name, ds)

                    clf_test_model.fit(X_train, y_train_class)

                    prob_test = clf_test_model.predict_proba(X_test)[:,1]
                    pred_test = clf_test_model.predict(X_test)

                    reg_test_model = get_ml_model_reg(reg_name, ds)

                    reg_test_model.fit(X_train, y_train_reg)

                    reg_score = reg_test_model.predict(X_test)

                    reg_class = (reg_score >= 2.8).astype(int)

                    df_classifier_test = pd.DataFrame({
                        "y_true": y_test_class,
                        "prob_classifier": prob_test,
                        "pred_classifier": pred_test
                    })

                    df_reg_test = pd.DataFrame({
                        "reg_class": reg_class
                    })

                    df_hybrid = apply_hybrid(
                        df_classifier_test,
                        df_reg_test,
                        best_pref
                    )

                    metrics_hybrid = evaluate_results(
                        df_hybrid["y_true"],
                        df_hybrid["final_pred"]
                    )

                    metrics_hybrid["Dataset"] = ds
                    metrics_hybrid["Classifier"] = clf_name
                    metrics_hybrid["Regressor"] = reg_name
                    metrics_hybrid["Seed"] = seed
                    metrics_hybrid["ModelType"] = "Hybrid"

                    results.append(metrics_hybrid)

    df_results = pd.DataFrame(results)

    os.makedirs("experiment_logs", exist_ok=True)

    df_results.to_csv(
        "experiment_logs/results_multi_seed.csv",
        index=False
    )

    print("\nResults saved to experiment_logs/results_multi_seed2.csv")

    return df_results