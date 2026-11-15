"""Export Random Forest comparison figures without changing experiment splits."""

from pathlib import Path
import re
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


BLUE = "#31577D"
TEAL = "#168780"
ORANGE = "#C36B32"
METRICS = ["pr_auc", "f1", "precision", "recall", "roc_auc"]


def select_one_se_model(comparison, n_splits=5):
    """Rank train-CV AP, then apply the best model's one-standard-error rule.

    The caller supplies sample standard deviations across folds (ddof=1).
    Validation/test scores are deliberately not consulted. None depth is infinite.
    Remaining simplicity ties use CV AP, CV ROC-AUC, then numeric model number.
    """
    if n_splits < 2 or comparison.empty or comparison["model"].duplicated().any():
        raise ValueError("One-SE selection needs unique models and at least two CV folds.")
    ranked = comparison.copy().sort_values(
        ["cv_pr_auc_mean", "cv_roc_auc_mean"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)
    scores = ranked[["cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean"]].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ranked["cv_pr_auc_std"].lt(0).any():
        raise ValueError("CV means and standard deviations must be finite and valid.")
    ranked["cv_pr_auc_se"] = ranked["cv_pr_auc_std"] / np.sqrt(n_splits)
    cutoff = float(ranked.iloc[0]["cv_pr_auc_mean"] - ranked.iloc[0]["cv_pr_auc_se"])
    ranked["one_se_eligible"] = ranked["cv_pr_auc_mean"] >= cutoff
    candidates = ranked.loc[ranked["one_se_eligible"]].assign(
        depth_order=lambda frame: pd.to_numeric(frame["max_depth"]).fillna(np.inf),
        model_order=lambda frame: frame["model"].map(_model_number),
    ).sort_values(
        ["depth_order", "min_samples_leaf", "cv_pr_auc_mean", "cv_roc_auc_mean", "model_order"],
        ascending=[True, False, False, False, True], kind="stable",
    )
    chosen = candidates.iloc[0]["model"]
    ranked["selected_by_cv"] = ranked["model"] == chosen
    return ranked, chosen, cutoff


def aggregate_feature_importances(model) -> pd.Series:
    """Sum encoded impurity importances by original column, using encoder metadata.

    This matches the notebook's categorical one-hot/numeric passthrough pipeline.
    Category names are never parsed, so underscores in names cannot mix columns.
    """
    transformer = model.named_steps["preprocessing"]
    weights = np.asarray(model.named_steps["classifier"].feature_importances_)
    parts = []
    for name, encoder, columns in transformer.transformers_:
        if name == "remainder" or len(columns) == 0:
            continue
        block = weights[transformer.output_indices_[name]]
        if name == "categorical":
            if encoder.drop is not None or encoder.min_frequency is not None or encoder.max_categories is not None:
                raise ValueError("Importance aggregation expects the notebook's full one-hot encoding.")
            labels = np.repeat(list(columns), [len(categories) for categories in encoder.categories_])
        elif name == "numeric":
            labels = list(columns)
        else:
            raise ValueError(f"Unsupported preprocessing branch: {name}")
        if len(labels) != len(block):
            raise ValueError("Encoded feature counts do not match forest importances.")
        parts.append(pd.Series(block, index=labels, dtype=float))
    importance = pd.concat(parts).groupby(level=0, sort=False).sum()
    if not np.isclose(importance.sum(), weights.sum()):
        raise ValueError("Importance aggregation lost encoded feature contributions.")
    return importance.sort_values(ascending=False).rename("importance")


def _score_rows(results, experiment, partition):
    rows = results.loc[
        (results["experiment"] == experiment) & (results["partition"] == partition)
    ].copy()
    if rows.empty or rows["model"].duplicated().any():
        raise ValueError(f"Expected one {experiment}/{partition} row per model.")
    if not np.allclose(rows["threshold"], 0.5):
        raise ValueError(f"The {experiment}/{partition} comparison must use threshold 0.5.")
    return rows.sort_values(["pr_auc", "roc_auc"], ascending=[False, False], kind="stable")


def _axes_style(axis):
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#CBD5DF")
    axis.tick_params(axis="both", length=0, labelsize=9)
    axis.set_axisbelow(True)


def _model_number(name):
    return int(name.rsplit("_", 1)[-1])


def export_result_plots(results, feature_importances, output_directory,
                        run_name, *, selected_model_name, show=False, f1_gap=0.05) -> dict[str, Path]:
    """Write seven PNGs; random uses validation, temporal uses fixed test results.

    ``feature_importances`` maps experiment -> model name -> full, aggregated
    importance Series. Figures never select temporal settings or tune thresholds.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name must be a filename-safe identifier.")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    random = _score_rows(results, "random", "validation")
    temporal = _score_rows(results, "temporal", "test")
    chosen = selected_model_name
    if chosen not in set(random["model"]):
        raise ValueError("The explicitly CV-selected model must have validation diagnostics.")
    if chosen not in feature_importances["random"]:
        raise ValueError("Full importances for the train-CV-selected random model are required.")
    if set(feature_importances["temporal"]) != set(temporal["model"]):
        raise ValueError("Full importances for every predefined temporal model are required.")
    paths = {}

    def save(figure, name):
        path = directory / f"{name}_{run_name}.png"
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        paths[name] = path
        if show:
            from IPython.display import display
            display(figure)
        plt.close(figure)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        for experiment, rows, title, note in [
            ("random", random, "Random experiment | Validation F1",
             "Ordered by validation AP; F1 shown at 0.5. Teal marks the train-CV / 1-SE selected model."),
            ("temporal", temporal, "Temporal experiment | 2017 test F1",
             "Ordered by test AP. Train: 2015–2016; threshold 0.5. Descriptive results; no model selection."),
        ]:
            fig, ax = plt.subplots(figsize=(11, max(6, len(rows) * 0.38)))
            colors = [TEAL if experiment == "random" and name == chosen else BLUE for name in rows["model"]]
            bars = ax.barh(rows["model"], rows["f1"], color=colors, height=0.65)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_xlabel("F1 score")
            ax.set_title(title, loc="left", fontsize=14, pad=17, fontweight="bold")
            ax.grid(axis="x", color="#E4EAF0", linewidth=0.8)
            _axes_style(ax)
            for bar, value in zip(bars, rows["f1"]):
                ax.text(min(value + 0.012, 0.97), bar.get_y() + bar.get_height() / 2,
                        f"{value:.3f}", va="center", fontsize=9)
            fig.text(0.02, 0.01, note, fontsize=9, color="#465464")
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            save(fig, f"{experiment}_{'validation' if experiment == 'random' else 'test'}_f1")

        train = _score_rows(results, "random", "selection_train")
        pairs = train[["model", "f1"]].rename(columns={"f1": "train_f1"}).merge(
            random[["model", "f1"]].rename(columns={"f1": "validation_f1"}),
            on="model", validate="one_to_one")
        if len(pairs) != len(random):
            raise ValueError("Every random model needs matching train and validation F1.")
        pairs["gap"] = pairs["train_f1"] - pairs["validation_f1"]
        pairs = pairs.sort_values("gap", ascending=False, kind="stable")
        fig, ax = plt.subplots(figsize=(12, max(6, len(pairs) * 0.4)))
        y = np.arange(len(pairs))
        for position, row in zip(y, pairs.itertuples()):
            ax.plot([row.validation_f1, row.train_f1], [position, position],
                    color=ORANGE if row.gap >= f1_gap else "#C4CFD8", linewidth=3, zorder=1)
            ax.text(1.02, position, f"Δ {row.gap:+.3f}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=9, color=ORANGE if row.gap >= f1_gap else "#465464")
        ax.scatter(pairs["train_f1"], y, color=BLUE, s=42, label="Train", zorder=2)
        ax.scatter(pairs["validation_f1"], y, color=TEAL, s=42, label="Validation", zorder=3)
        ax.set_yticks(y, pairs["model"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1 score | same fitted model and threshold 0.5")
        ax.set_title("Random experiment | Train–validation F1 gap", loc="left", fontsize=14, pad=40, fontweight="bold")
        ax.grid(axis="x", color="#E4EAF0")
        _axes_style(ax)
        handles, _ = ax.get_legend_handles_labels()
        handles.append(Patch(color=ORANGE, label=f"Gap ≥ {f1_gap:.2f}"))
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
        fig.text(0.02, 0.01, "Large gaps are diagnostic flags, not proof of overfitting. Temporal has no validation set.",
                 fontsize=9, color="#465464")
        fig.tight_layout(rect=(0, 0.04, 0.96, 1))
        save(fig, "random_train_validation_f1_gap")

        for experiment, rows, partition in [("random", random, "validation"), ("temporal", temporal, "test")]:
            fig, ax = plt.subplots(figsize=(10, max(6, len(rows) * 0.42)))
            values = rows[METRICS].to_numpy(dtype=float)
            heatmap = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(5), ["PR-AUC (AP)", "F1", "Precision", "Recall", "ROC-AUC"])
            ax.set_yticks(range(len(rows)), rows["model"])
            ax.tick_params(length=0, labelsize=9)
            ax.set_title(f"{experiment.title()} experiment | {partition.title()} metrics",
                         loc="left", fontsize=14, pad=20, fontweight="bold")
            for (row, column), value in np.ndenumerate(values):
                ax.text(column, row, f"{value:.3f}", ha="center", va="center",
                        color="white" if value >= 0.55 else "#172B3A", fontsize=9)
            fig.colorbar(heatmap, ax=ax, fraction=0.04, pad=0.04, label="Score (0–1)")
            note = "Validation diagnostics ordered by AP; model selection uses train CV and the 1-SE rule." if experiment == "random" else "2017 test only, ordered by AP; threshold 0.5. Descriptive comparison, not model selection."
            fig.text(0.02, 0.01, note, fontsize=9, color="#465464")
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            save(fig, f"{experiment}_{partition}_metrics_heatmap")

        importance = feature_importances["random"][chosen].sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(importance.index, importance.values, color=TEAL, height=0.65)
        ax.invert_yaxis()
        upper = max(float(importance.max()) * 1.22, 0.05)
        ax.set_xlim(0, upper)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Share of training impurity importance (summed across one-hot categories)")
        ax.set_title(f"Random selected model | {chosen}\nTop 10 original features", loc="left", fontsize=14, pad=16, fontweight="bold")
        for position, value in enumerate(importance.values):
            ax.text(value + upper * 0.015, position, f"{value:.1%}", va="center", fontsize=9)
        ax.grid(axis="x", color="#E4EAF0")
        _axes_style(ax)
        fig.text(0.02, 0.01, "Relative importance within this fitted model; not a causal effect. No test data used.", fontsize=9, color="#465464")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        save(fig, "random_selected_top10_feature_importance")

        feature_table = pd.DataFrame.from_dict(feature_importances["temporal"], orient="index").fillna(0)
        feature_table = feature_table.loc[sorted(feature_table.index, key=_model_number)]
        top_features = feature_table.mean().sort_values(ascending=False).head(10).index
        feature_table = feature_table[top_features]
        fig, ax = plt.subplots(figsize=(14, max(8, len(feature_table) * 0.42)))
        values = feature_table.to_numpy(dtype=float)
        upper = max(float(values.max()), 0.01)
        heatmap = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=upper, aspect="auto")
        ax.set_xticks(range(len(top_features)), [fill(name.replace("_", " "), 16) for name in top_features])
        ax.set_yticks(range(len(feature_table)), feature_table.index)
        ax.tick_params(length=0, labelsize=8)
        ax.set_title("Temporal experiment | Original-feature importance across all models",
                     loc="left", fontsize=14, pad=20, fontweight="bold")
        for (row, column), value in np.ndenumerate(values):
            ax.text(column, row, f"{value:.1%}", ha="center", va="center", fontsize=8,
                    color="white" if value >= upper * 0.55 else "#172B3A")
        fig.colorbar(heatmap, ax=ax, fraction=0.03, pad=0.03, format=PercentFormatter(1), label="Training importance")
        fig.text(0.02, 0.01, "Columns: top 10 by mean training importance. Rows: model number. All models train on 2015–2016; no test-based selection.",
                 fontsize=9, color="#465464")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        save(fig, "temporal_feature_importance_heatmap")
    return paths
