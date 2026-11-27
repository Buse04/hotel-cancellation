"""Figures for Buse's models (Logistic Regression, XGBoost, MLP).

Every function returns the matplotlib figure and saves it as a PNG under
results/<folder>/ when a folder is given, so the notebooks keep the plots
and the repository keeps the image files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# One color per meaning, used in every figure
BLUE = "#2a78d6"      # unweighted model / not canceled / lowers cancellation risk
ORANGE = "#eb6834"    # class-weighted model / canceled / raises cancellation risk
GRAY = "#8a8a85"      # reference lines (random guess, base rate)
INK = "#2b2b28"       # text

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c1",
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "grid.color": "#e6e5df",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def save_figure(fig, folder, name):
    """Save a figure as results/<folder>/<name>.png and return the path."""
    if folder is None:
        return None
    output_dir = RESULTS_DIR / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    return path


# ---------------------------------------------------------------- tuning

def load_validation_results(results_file="results_buse.csv"):
    """Validation rows from the results CSV; repeated runs keep the latest row."""
    results = pd.read_csv(RESULTS_DIR / results_file)
    results = results[results["split"] == "validation"]
    return results.drop_duplicates(subset="model", keep="last")


def plot_logreg_tuning(results_file="results_buse.csv", folder=None):
    """F1 and ROC-AUC for each C, with and without class weighting."""
    results = load_validation_results(results_file)
    pattern = r"LogReg C=(?P<C>[\d.]+) cw=(?P<cw>\w+)"
    parsed = results["model"].str.extract(pattern).dropna()
    data = results.loc[parsed.index].join(parsed)
    data["C"] = data["C"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in [(axes[0], "f1", "F1"), (axes[1], "roc_auc", "ROC-AUC")]:
        for cw, color, name in [("None", BLUE, "no class weight"), ("balanced", ORANGE, "class_weight='balanced'")]:
            rows = data[data["cw"] == cw].sort_values("C")
            ax.plot(rows["C"], rows[metric], marker="o", markersize=7, linewidth=2, color=color, label=name)
            last = rows.iloc[-1]
            # When the two lines end at almost the same value, push the labels apart
            close = abs(data[data["C"] == last["C"]][metric].diff().iloc[-1]) < 0.01
            dy = (7 if cw == "balanced" else -7) if close else 0
            ax.annotate(f"{last[metric]:.3f}", (last["C"], last[metric]), textcoords="offset points",
                        xytext=(8, dy), va="center", fontsize=9, color=color)
        ax.set_xscale("log")
        ax.set_xticks(sorted(data["C"].unique()))
        ax.set_xticklabels([f"{c:g}" for c in sorted(data["C"].unique())])
        ax.set_xlabel("C (regularization strength; smaller = stronger)")
        ax.set_ylabel(f"Validation {label}")
        ax.set_title(f"Validation {label}")
        ax.grid(axis="y")
    # Same 0.15-wide window on both panels, so differences are shown at their true size
    axes[0].set_ylim(0.55, 0.70)
    axes[1].set_ylim(0.80, 0.95)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Logistic Regression tuning: class weighting matters, C does not", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, folder, "logreg_tuning")
    return fig


def plot_xgboost_tuning(results_file="results_buse.csv", folder=None):
    """Validation F1 for every depth / learning-rate setting, with and without scale_pos_weight."""
    results = load_validation_results(results_file)
    pattern = r"XGBoost depth=(?P<depth>\d+) lr=(?P<lr>[\d.]+) spw=(?P<spw>[\d.]+)"
    parsed = results["model"].str.extract(pattern).dropna()
    data = results.loc[parsed.index].join(parsed)
    data["spw"] = data["spw"].astype(float)
    data["setting"] = "depth " + data["depth"] + ", lr " + data["lr"]
    settings = (data[["depth", "lr", "setting"]].drop_duplicates()
                .sort_values(["depth", "lr"], key=lambda s: s.astype(float))["setting"].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(settings))
    for ax, metric, label, limits in [(axes[0], "f1", "F1", (0.65, 0.75)), (axes[1], "roc_auc", "ROC-AUC", (0.87, 0.97))]:
        for weighted, color, name in [(False, BLUE, "scale_pos_weight = 1"),
                                      (True, ORANGE, "scale_pos_weight ≈ 2.62 (balanced)")]:
            rows = data[(data["spw"] > 1) == weighted].set_index("setting").reindex(settings)
            ax.plot(x, rows[metric], marker="o", markersize=7, linewidth=2, color=color, label=name)
            ends = data[data["setting"] == settings[-1]][metric]
            dy = (7 if weighted else -7) if ends.max() - ends.min() < 0.01 else 0
            ax.annotate(f"{rows[metric].iloc[-1]:.3f}", (x[-1], rows[metric].iloc[-1]),
                        textcoords="offset points", xytext=(8, dy), va="center", fontsize=9, color=color)
        ax.set_ylim(*limits)
        ax.set_xlim(-0.3, len(settings) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(settings, rotation=30, ha="right")
        ax.set_xlabel("Setting (simpler → more complex)")
        ax.set_ylabel(f"Validation {label}")
        ax.set_title(f"Validation {label}")
        ax.grid(axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("XGBoost tuning: deeper trees help; class weighting raises F1 but not ROC-AUC",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, folder, "xgboost_tuning")
    return fig


# ---------------------------------------------------------------- final model

def plot_confusion_matrix(y_true, y_prob, title, folder=None, name="confusion_matrix", threshold=0.5):
    """Confusion matrix with counts and row percentages at the given threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_share = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    labels = ["Not canceled", "Canceled"]
    for i in range(2):
        for j in range(2):
            color = "white" if row_share[i, j] > 0.55 else INK
            ax.text(j, i, f"{matrix[i, j]:,}\n({row_share[i, j]:.0%})", ha="center", va="center",
                    fontsize=12, color=color)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save_figure(fig, folder, name)
    return fig


def plot_roc_pr(y_true, y_prob, model_name, folder=None, name="roc_pr_curves"):
    """ROC curve and precision-recall curve of one model, with reference lines."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    base_rate = np.mean(y_true)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color=BLUE, linewidth=2, label=f"{model_name} (AUC = {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="Random guess (AUC = 0.5)")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve",
                xlim=(0, 1), ylim=(0, 1.01))
    axes[1].plot(recall, precision, color=BLUE, linewidth=2, label=f"{model_name} (AP = {ap:.3f})")
    axes[1].axhline(base_rate, color=GRAY, linestyle="--", linewidth=1,
                    label=f"Base rate ({base_rate:.1%} canceled)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve",
                xlim=(0, 1), ylim=(0, 1.01))
    for ax in axes:
        ax.grid(True)
        ax.legend(frameon=False, loc="lower right" if ax is axes[0] else "lower left", fontsize=9)
    fig.suptitle(f"{model_name} on the test set", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, folder, name)
    return fig


# ---------------------------------------------------------------- feature importance

def original_column(encoded_name, columns):
    """Map an encoded feature name (e.g. 'cat__country_prt') to its original column."""
    name = encoded_name.split("__", 1)[-1]
    matches = [c for c in columns if name == c or name.startswith(c + "_")]
    return max(matches, key=len) if matches else name


def logreg_coefficients(pipeline):
    """Coefficients of a fitted Logistic Regression pipeline, indexed by encoded feature name."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    return pd.Series(coefficients, index=[n.split("__", 1)[-1] for n in names])


def plot_logreg_coefficients(pipeline, folder=None, top_n=20, name="logreg_coefficients"):
    """Largest positive and negative coefficients (on standardized / one-hot features)."""
    coefficients = logreg_coefficients(pipeline)
    top = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index[:top_n])
    top = top.sort_values()

    fig, ax = plt.subplots(figsize=(9, 0.33 * top_n + 1.5))
    colors = [ORANGE if v > 0 else BLUE for v in top]
    ax.barh(top.index, top.values, color=colors, height=0.7)
    for y, value in enumerate(top.values):
        ax.text(value + (0.05 if value >= 0 else -0.05), y, f"{value:+.2f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=GRAY, linewidth=1)
    span = top.abs().max() * 1.25
    ax.set_xlim(-span, span)
    ax.set_xlabel("Coefficient (log-odds)   ←  lowers cancellation risk  |  raises cancellation risk  →")
    ax.set_title(f"Logistic Regression: top {top_n} coefficients")
    ax.grid(axis="x")
    fig.tight_layout()
    save_figure(fig, folder, name)
    return fig


def tree_importance_by_column(pipeline, original_columns):
    """Model feature importances summed over the one-hot columns of each original feature."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    grouped = pd.Series(importances, index=[original_column(n, original_columns) for n in names])
    grouped = grouped.groupby(level=0).sum()
    return (grouped / grouped.sum()).sort_values(ascending=False)


def plot_tree_importance(pipeline, original_columns, model_name, folder=None, top_n=15,
                         name="feature_importance"):
    """Horizontal bar chart of the most important original features."""
    importance = tree_importance_by_column(pipeline, original_columns).head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8.5, 0.36 * top_n + 1.4))
    ax.barh(importance.index, importance.values, color=BLUE, height=0.7)
    for y, value in enumerate(importance.values):
        ax.text(value + 0.003, y, f"{value:.1%}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, importance.max() * 1.18)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of total importance (one-hot columns summed per feature)")
    ax.set_title(f"{model_name}: top {top_n} features")
    ax.grid(axis="x")
    fig.tight_layout()
    save_figure(fig, folder, name)
    return fig


# ---------------------------------------------------------------- comparison

def plot_test_comparison(results_file="results_buse.csv", folder=None, name="test_comparison"):
    """Test-set F1, ROC-AUC, precision and recall of every final model in the results CSV."""
    results = pd.read_csv(RESULTS_DIR / results_file)
    test = results[results["split"] == "random_test"].drop_duplicates(subset="model", keep="last")
    metrics = ["f1", "roc_auc", "precision", "recall"]
    labels = ["F1", "ROC-AUC", "Precision", "Recall"]
    palette = [BLUE, ORANGE, "#1baf7a", "#6250d6"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metrics))
    width = 0.8 / len(test)
    for i, (_, row) in enumerate(test.iterrows()):
        values = row[metrics].astype(float).values
        bars = ax.bar(x + (i - (len(test) - 1) / 2) * width, values, width - 0.03,
                      color=palette[i % len(palette)], label=row["model"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score on the test set")
    ax.set_title("Final models on the random test set")
    ax.legend(frameon=False, ncol=len(test), loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y")
    fig.tight_layout()
    save_figure(fig, folder, name)
    return fig
