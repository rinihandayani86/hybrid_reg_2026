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

    subset = df[df["Dataset"] == dataset].copy()

    subset["conf_bin"] = pd.cut(subset["prob_classifier"], bins)

    clf_error_rate = subset.groupby("conf_bin")["clf_error"].mean()
    hyb_error_rate = subset.groupby("conf_bin")["hyb_error"].mean()

    x = np.arange(len(clf_error_rate))

    ax = axes[idx]

    ax.plot(x, clf_error_rate, marker='o', label="Classifier")
    ax.plot(x, hyb_error_rate, marker='o', label="Hybrid")

    for i,v in enumerate(clf_error_rate):
        if not np.isnan(v):
            ax.text(
                i-0.05,
                v+0.01,
                f"{v:.2f}",
                fontsize=12,
                rotation=45
            )

    for i,v in enumerate(hyb_error_rate):
        if not np.isnan(v):
            ax.text(
                i+0.05,
                v+0.01,
                f"{v:.2f}",
                fontsize=12,
                rotation=45
            )

    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylabel("Error Rate")
    #ax.set_xlabel("Confidence Bin")
    ax.set_xlabel("Classifier Prob")

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

    ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "experiment_logs/error_rate_panel.pdf",
    dpi=600
)

plt.close()

print("Panel figures generated.")