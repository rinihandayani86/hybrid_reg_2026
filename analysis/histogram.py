import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":14
})

df = pd.read_csv("experiment_logs/prediction_details.csv")

df["clf_error"] = (df["y_true"] != df["pred_classifier"]).astype(int)
df["hyb_error"] = (df["y_true"] != df["final_pred"]).astype(int)

datasets = sorted(df["Dataset"].unique())

bins = np.linspace(0,1,11)

fig, axes = plt.subplots(1,2, figsize=(12,6))

for idx, dataset in enumerate(datasets):

    subset = df[df["Dataset"] == dataset]

    clf_error = subset[subset["clf_error"] == 1]["prob_classifier"]
    hyb_error = subset[subset["hyb_error"] == 1]["prob_classifier"]

    clf_counts, _ = np.histogram(clf_error, bins=bins)
    hyb_counts, _ = np.histogram(hyb_error, bins=bins)

    x = np.arange(len(clf_counts))
    width = 0.4

    ax = axes[idx]

    bars1 = ax.bar(x-width/2, clf_counts, width, label="Classifier")
    bars2 = ax.bar(x+width/2, hyb_counts, width, label="Hybrid")

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x()+bar.get_width()/2,
                height+0.2,
                f"{int(height)}",
                ha='center',
                fontsize=12,
                rotation=45
            )

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x()+bar.get_width()/2,
                height+0.2,
                f"{int(height)}",
                ha='center',
                fontsize=12,
                rotation=45
            )

    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel("Classifier Probability")
    ax.set_ylabel("Number of Errors")

    ax.grid(axis='y', alpha=0.3)

    #ax.set_title(f"({chr(97+idx)}) {dataset}")
    ax.text(
        0.5,
        -0.30,
        f"({chr(97+idx)})",
        transform=ax.transAxes,
        ha="center",
        fontsize=14
    )

    ax.legend()

plt.tight_layout()

plt.savefig(
    "experiment_logs/error_histogram_panel.pdf",
    dpi=600
)

plt.close()