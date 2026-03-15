from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    clf_model,
    X_train,
    y_train_class,
    cv=5,
    scoring="balanced_accuracy",
    train_sizes=np.linspace(0.1,1.0,10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(10,5))

plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training score"
)

plt.plot(
    train_sizes,
    val_mean,
    marker="o",
    label="Validation score"
)

plt.xlabel("Training samples")
plt.ylabel("Balanced Accuracy")

plt.title("Learning Curve (Random Forest Classifier)")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "experiment_logs/learning_curve_rf.pdf",
    dpi=600
)