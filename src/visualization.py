import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

LABELS = ["non-offensive", "offensive"]
COLORS = {"non-offensive": "#4C78A8", "offensive": "#E45756"}


def _finish(fig, save_path):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_class_distribution(df, title="Class distribution", save_path=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="label", data=df, ax=ax, palette=[COLORS[l] for l in LABELS])
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABELS)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title(title); ax.set_xlabel("")
    return _finish(fig, save_path)


def plot_length_histogram(df, title="Tweet length (words)", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for lab, name in enumerate(LABELS):
        ax.hist(df.loc[df.label == lab, "text"].str.split().str.len(), bins=30, alpha=0.55,
                density=True, color=COLORS[name], label=name)
    ax.set_xlabel("words per tweet"); ax.set_title(title); ax.legend()
    return _finish(fig, save_path)


def plot_wordcloud(texts, title="WordCloud", save_path=None, stopwords=None, colormap="viridis"):
    from wordcloud import WordCloud
    wc = WordCloud(width=800, height=400, background_color="white", stopwords=stopwords,
                   colormap=colormap, random_state=42).generate(" ".join(texts))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); ax.set_title(title)
    return _finish(fig, save_path)


def plot_confusion_matrix(cm, title="Confusion matrix", ax=None, normalize=False):
    """Heatmap with counts (and row percentages in brackets)."""
    cm = np.asarray(cm)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
    pct = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i, j]}\n({100 * pct[i, j]:.0f}%)" for j in range(2)] for i in range(2)])
    sns.heatmap(pct if normalize else cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=LABELS, yticklabels=LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    return ax


def plot_confusion_matrices(cms, ncols=3, figsize_per=(4.2, 3.6)):
    """cms: dict name -> 2x2 matrix."""
    n = len(cms); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, cm) in zip(axes, cms.items()):
        plot_confusion_matrix(cm, name, ax=ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_metric_bars(table, metrics=("accuracy", "precision", "recall", "f1"), title="Model comparison",
                     figsize=(10, 4.5)):
    """table: DataFrame (index = names, columns include the metrics)."""
    ax = table[list(metrics)].plot(kind="bar", figsize=figsize, width=0.8, rot=20,
                                   color=["#9AA5B1", "#4C78A8", "#F58518", "#E45756"][:len(metrics)])
    ax.set_ylim(0, 1); ax.set_title(title); ax.set_ylabel("score"); ax.legend(loc="lower right", ncol=len(metrics))
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=1)
    return ax.get_figure()
