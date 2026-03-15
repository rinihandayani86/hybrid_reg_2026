import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================
# LOAD DATA
# ======================

df = pd.read_csv("experiment_logs/prediction_details.csv")

# ======================
# ERROR FLAG
# ======================

df["clf_error"] = (df["y_true"] != df["pred_classifier"]).astype(int)
df["hyb_error"] = (df["y_true"] != df["final_pred"]).astype(int)

datasets = df["Dataset"].unique()

# ======================
# HISTOGRAM ERROR DISTRIBUTION
# ======================

for dataset in datasets:

    subset = df[df["Dataset"] == dataset]

    clf_error = subset[subset["clf_error"] == 1]["prob_classifier"]
    hyb_error = subset[subset["hyb_error"] == 1]["prob_classifier"]

    plt.figure()

    plt.hist(
        clf_error,
        bins=20,
        alpha=0.6,
        label="Classifier error"
    )

    plt.hist(
        hyb_error,
        bins=20,
        alpha=0.6,
        label="Hybrid error"
    )

    plt.xlabel("Classifier probability")
    plt.ylabel("Number of samples")
    plt.title(f"Misclassification Distribution ({dataset})")

    plt.legend()

    plt.savefig(
        f"experiment_logs/error_hist_{dataset}.pdf",
        dpi=300
    )

    plt.close()

print("Histogram analysis completed.")

# ======================
# ERROR RATE vs CONFIDENCE
# ======================

for dataset in datasets:

    subset = df[df["Dataset"] == dataset].copy()

    bins = np.linspace(0,1,11)

    subset["conf_bin"] = pd.cut(
        subset["prob_classifier"],
        bins
    )

    clf_error_rate = subset.groupby("conf_bin")["clf_error"].mean()
    hyb_error_rate = subset.groupby("conf_bin")["hyb_error"].mean()

    plt.figure()

    clf_error_rate.plot(marker='o', label="Classifier")
    hyb_error_rate.plot(marker='o', label="Hybrid")

    plt.ylabel("Error Rate")
    plt.xlabel("Confidence Bin")
    plt.title(f"Error Rate vs Confidence ({dataset})")

    plt.legend()

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"experiment_logs/error_rate_{dataset}.pdf",
        dpi=300
    )

    plt.close()

print("Error rate analysis completed.")