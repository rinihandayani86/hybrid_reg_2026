import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
from core.data_loader2 import load_and_split_dataset


plt.rcParams.update({
    "font.size":14
})

df = pd.read_csv("experiment_logs/prediction_details.csv")

df["clf_error"] = (df["y_true"] != df["pred_classifier"]).astype(int)
df["hyb_error"] = (df["y_true"] != df["final_pred"]).astype(int)

# ambil daftar dataset
datasets = sorted(df["Dataset"].unique())

for dataset in datasets:

    subset = df[df["Dataset"] == dataset]

    clf_error = subset[subset["clf_error"] == 1]["prob_classifier"]
    hyb_error = subset[subset["hyb_error"] == 1]["prob_classifier"]

    plt.figure(figsize=(8,5))

    sns.kdeplot(
        clf_error,
        label="Classifier Error Density",
        fill=True,
        alpha=0.3
    )

    sns.kdeplot(
        hyb_error,
        label="Hybrid Error Density",
        fill=True,
        alpha=0.3
    )

    # pRef example line
    plt.axvline(
        0.65,
        linestyle="--",
        linewidth=2,
        label="$p_{Ref}$ example"
    )

    plt.xlabel("Classifier Probability")
    plt.ylabel("Error Density")

    plt.title(f"Misclassification Density ({dataset})")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"experiment_logs/error_density_{dataset}.pdf",
        dpi=600
    )

    plt.close()

print("Density plots generated.")