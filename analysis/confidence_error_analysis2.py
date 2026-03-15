import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":14
})

df = pd.read_csv("experiment_logs/prediction_details.csv")

df["clf_error"] = (df["y_true"] != df["pred_classifier"]).astype(int)
df["hyb_error"] = (df["y_true"] != df["final_pred"]).astype(int)

datasets = df["Dataset"].unique()

# ======================
# HISTOGRAM ERROR
# ======================

for dataset in datasets:

    subset = df[df["Dataset"] == dataset]

    clf_error = subset[subset["clf_error"] == 1]["prob_classifier"]
    hyb_error = subset[subset["hyb_error"] == 1]["prob_classifier"]

    bins = np.linspace(0,1,11)

    clf_counts, _ = np.histogram(clf_error, bins=bins)
    hyb_counts, _ = np.histogram(hyb_error, bins=bins)

    x = np.arange(len(clf_counts))

    width = 0.4

    plt.figure(figsize=(7,5))

    bars1 = plt.bar(
        x - width/2,
        clf_counts,
        width,
        label="Classifier"
    )

    bars2 = plt.bar(
        x + width/2,
        hyb_counts,
        width,
        label="Hybrid"
    )

    # ======================
    # LABEL BAR
    # ======================

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.2,
                f"{int(height)}",
                ha='center',
                va='bottom',
                fontsize=13,
                rotation=35
            )

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.2,
                f"{int(height)}",
                ha='center',
                va='bottom',
                fontsize=13,
                rotation=35
            )

    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Classifier Probability")
    plt.ylabel("Number of Errors")
    
    plt.grid(axis='y', alpha=0.3)

    #plt.title(f"Misclassification Distribution ({dataset})")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"experiment_logs/error3_hist_{dataset}.pdf",
        dpi=600
    )

    plt.close()

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

    x = np.arange(len(clf_error_rate))

    plt.figure(figsize=(7,5))

    plt.plot(
        x,
        clf_error_rate,
        marker='o',
        label="Classifier"
    )

    plt.plot(
        x,
        hyb_error_rate,
        marker='o',
        label="Hybrid"
    )

    # ======================
    # LABEL TITIK
    # ======================

    for i, v in enumerate(clf_error_rate):
        if not np.isnan(v):
            plt.text(
                i,
                v,
                f"{v:.2f}",
                ha='center',
                va='bottom',
                fontsize=13,
                rotation=35
            )

    for i, v in enumerate(hyb_error_rate):
        if not np.isnan(v):
            plt.text(
                i,
                v,
                f"{v:.2f}",
                ha='center',
                va='bottom',
                fontsize=13,
                rotation=35
            )

    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

    plt.xticks(x, labels, rotation=45, ha="right")

    plt.ylabel("Error Rate")
    plt.xlabel("Classifier Prob")

    #plt.title(f"Error Rate vs Confidence ({dataset})")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"experiment_logs/error3_rate_{dataset}.pdf",
        dpi=600
    )

    plt.close()

print("Confidence analysis completed.")