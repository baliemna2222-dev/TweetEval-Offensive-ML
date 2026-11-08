from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df, output_path=None):
    ax = df["label"].value_counts().sort_index().plot(kind="bar")
    ax.set(title="Class distribution", xlabel="Label", ylabel="Tweet count")
    ax.set_xticklabels(["0", "1"], rotation=0)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.show()

def plot_confusion_matrix(matrix, output_path=None, title="Confusion matrix"):
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"])
    plt.title(title); plt.xlabel("Prediction"); plt.ylabel("Actual")
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.show()
