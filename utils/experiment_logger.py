import pandas as pd
from pathlib import Path


class ExperimentLogger:

    def __init__(self, output_dir="experiment_logs"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results = []
        self.predictions = []
        self.pref_history = []

    # =========================
    # LOG METRICS
    # =========================

    def log_results(self, result_dict):

        self.results.append(result_dict)

    # =========================
    # LOG PREDICTIONS
    # =========================

    def log_predictions(self, df_predictions):

        self.predictions.append(df_predictions)

    # =========================
    # LOG pRef SEARCH
    # =========================

    def log_pref_history(self, dataset, classifier, regressor, history):

        for pRef, score in history:

            self.pref_history.append({
                "Dataset": dataset,
                "Classifier": classifier,
                "Regressor": regressor,
                "pRef": pRef,
                "BalancedAccuracy": score
            })

    # =========================
    # SAVE FILES
    # =========================

    def save(self):

        if self.results:
            df_results = pd.DataFrame(self.results)
            df_results.to_csv(
                self.output_dir / "experiment_results.csv",
                index=False
            )

        if self.predictions:
            df_pred = pd.concat(self.predictions, ignore_index=True)
            df_pred.to_csv(
                self.output_dir / "prediction_details.csv",
                index=False
            )

        if self.pref_history:
            df_pref = pd.DataFrame(self.pref_history)
            df_pref.to_csv(
                self.output_dir / "pref_search_history.csv",
                index=False
            )