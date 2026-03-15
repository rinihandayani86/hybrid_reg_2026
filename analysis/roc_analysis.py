import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

df = pd.read_csv("experiment_logs/prediction_details.csv")

datasets = df["Dataset"].unique()

for dataset in datasets:

    subset_ds = df[df["Dataset"] == dataset]

    models = subset_ds[["Classifier","Regressor"]].drop_duplicates()

    for _, row in models.iterrows():

        clf = row["Classifier"]
        reg = row["Regressor"]

        subset = subset_ds[
            (subset_ds["Classifier"] == clf) &
            (subset_ds["Regressor"] == reg)
        ]

        y_true = subset["y_true"]

        baseline_score = subset["prob_classifier"]
        hybrid_score = subset["final_pred"]

        fpr_base, tpr_base, _ = roc_curve(y_true, baseline_score)
        fpr_hyb, tpr_hyb, _ = roc_curve(y_true, hybrid_score)

        auc_base = auc(fpr_base, tpr_base)
        auc_hyb = auc(fpr_hyb, tpr_hyb)

        plt.figure()

        plt.plot(fpr_base, tpr_base, label=f"Baseline AUC = {auc_base:.3f}")
        plt.plot(fpr_hyb, tpr_hyb, label=f"Hybrid AUC = {auc_hyb:.3f}")

        plt.plot([0,1],[0,1],'k--')

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.title(f"ROC Curve ({dataset}) - {clf}+{reg}")

        plt.legend()

        plt.savefig(
            f"experiment_logs/roc_{dataset}_{clf}_{reg}.pdf",
            dpi=300
        )

        plt.close()

        print(
            f"{dataset} | {clf}+{reg} → "
            f"AUC baseline={auc_base:.4f}, "
            f"AUC hybrid={auc_hyb:.4f}"
        )

print("ROC analysis completed.")