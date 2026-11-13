"""Paths and small helpers shared by all notebooks."""
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_paths(notebook_name=None):
    """Return the project paths. With a notebook name, outputs get one sub-folder per notebook:
    results/{figures,metrics,predictions}/<notebook_name>/ (created if missing)."""
    root = PROJECT_ROOT
    p = SimpleNamespace(
        root=root, raw=root / "data" / "raw", processed=root / "data" / "processed",
        models=root / "models", figures=root / "results" / "figures",
        metrics=root / "results" / "metrics", predictions=root / "results" / "predictions",
    )
    if notebook_name:
        for attr in ("figures", "metrics", "predictions"):
            setattr(p, attr, getattr(p, attr) / notebook_name)
    for d in (p.raw, p.processed, p.models, p.figures, p.metrics, p.predictions):
        d.mkdir(parents=True, exist_ok=True)
    return p


def savefig(fig, path, show=True):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
