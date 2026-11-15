"""LightGBM experiments using the project's unchanged booking partitions.

Random selection uses training-only CV; validation tunes only the selected
model's threshold. Temporal configurations are fixed before seeing 2017 scores.
No model, split summary, or intermediate score files are written automatically.
"""

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data_split import split_features_target
from .evaluation import evaluate


DEFAULT_PARAMETER_SPACE = {
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 500, 800],
    "num_leaves": [15, 31, 63],
    "max_depth": [5, 7, 10],
    "min_child_samples": [20, 50, 100],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
}
BASELINE_PARAMETERS = {
    "learning_rate": 0.1, "n_estimators": 200, "num_leaves": 31,
    "max_depth": 7, "min_child_samples": 20,
    "colsample_bytree": 1.0, "reg_lambda": 0.0,
}
CV_COLUMNS = [
    "cv_pr_auc_mean", "cv_pr_auc_std", "cv_pr_auc_se", "cv_roc_auc_mean",
    "cv_fit_seconds_mean", "one_se_eligible", "selected_by_cv",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Execution settings; sequential CV avoids nested CPU parallelism."""

    random_seed: int = 42
    model_count: int = 15
    cv_folds: int = 5
    model_jobs: int = 2
    thresholds: tuple[float, ...] = tuple(np.round(np.arange(0.2, 0.801, 0.025), 3))

    def __post_init__(self):
        if self.model_count < 1 or self.cv_folds < 2:
            raise ValueError("Use at least one model and two CV folds.")
        if not self.thresholds or not all(0 <= value <= 1 for value in self.thresholds):
            raise ValueError("Thresholds must be a nonempty sequence within [0, 1].")


@dataclass
class RandomExperimentResult:
    """Selected train-fitted pipeline plus its auditable selection tables."""

    selected_model_name: str
    selected_parameters: dict[str, Any]
    selected_pipeline: Pipeline
    selected_threshold: float
    cv_comparison: pd.DataFrame
    threshold_trials: pd.DataFrame
    evaluation_rows: list[dict[str, Any]]
    feature_importance: pd.Series
    fit_seconds: float
    test_evaluated: bool = field(default=False, init=False)


def sample_model_parameters(config: ExperimentConfig) -> dict[str, dict[str, Any]]:
    """Include a baseline, then sample unique depth-compatible configurations."""
    compatible_spaces = [
        {**DEFAULT_PARAMETER_SPACE, "max_depth": [depth], "num_leaves": [leaves]}
        for depth in DEFAULT_PARAMETER_SPACE["max_depth"]
        for leaves in DEFAULT_PARAMETER_SPACE["num_leaves"]
        if leaves <= 2 ** depth
    ]
    available_count = sum(
        int(np.prod([len(values) for values in space.values()]))
        for space in compatible_spaces
    )
    if config.model_count > available_count:
        raise ValueError(f"Only {available_count} unique configurations are available.")
    sampled = ParameterSampler(
        compatible_spaces, n_iter=min(config.model_count, available_count),
        random_state=config.random_seed,
    )
    parameters = [BASELINE_PARAMETERS.copy()]
    for candidate in sampled:
        if candidate not in parameters:
            parameters.append(candidate)
        if len(parameters) >= config.model_count:
            break
    return {f"lightgbm_{index}": values for index, values in enumerate(parameters[:config.model_count], 1)}


def parameter_summary(model_parameters, config: ExperimentConfig) -> pd.DataFrame:
    """Show the identical predefined configurations used in both experiments."""
    return pd.DataFrame([
        {"experiment": experiment, "model": name, **parameters,
         "random_state": config.random_seed, "objective": "binary"}
        for experiment in ("random", "temporal")
        for name, parameters in model_parameters.items()
    ])


def summarize_splits(data_splits) -> pd.DataFrame:
    """Check within-experiment disjointness and show sizes without writing files."""
    expected = {"random": {"train", "validation", "test"}, "temporal": {"train", "test"}}
    if set(data_splits) != set(expected):
        raise ValueError("Both random and temporal experiments are required.")
    rows = []
    for experiment, partitions in data_splits.items():
        if set(partitions) != expected[experiment]:
            raise ValueError(f"Unexpected partitions for {experiment}.")
        seen_indices = set()
        for partition_name, bookings in partitions.items():
            if not bookings.index.is_unique or seen_indices.intersection(bookings.index):
                raise ValueError(f"Overlapping or duplicate row indices in {experiment}.")
            if set(bookings["is_canceled"].unique()) != {0, 1}:
                raise ValueError("Each partition must contain both target classes.")
            seen_indices.update(bookings.index)
            rows.append({"experiment": experiment, "partition": partition_name,
                         "rows": len(bookings), "canceled_rate": bookings["is_canceled"].mean()})
    return pd.DataFrame(rows)


def build_lightgbm_pipeline(training_features, parameters, config: ExperimentConfig) -> Pipeline:
    """Build a fresh fold-local one-hot encoder and binary LightGBM classifier."""
    try:
        from lightgbm import LGBMClassifier
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Install requirements.txt in the notebook's Python environment: "
            "python -m pip install -r requirements.txt"
        ) from error
    numeric_columns = training_features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in training_features if column not in numeric_columns]
    encoder = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_columns),
        ("numeric", "passthrough", numeric_columns),
    ], sparse_threshold=1.0)
    classifier = LGBMClassifier(
        objective="binary", random_state=config.random_seed,
        n_jobs=config.model_jobs, importance_type="gain", verbosity=-1,
        deterministic=True, force_col_wise=True, **parameters,
    )
    return Pipeline([("preprocessing", encoder), ("classifier", classifier)])


def cancellation_probabilities(pipeline, features) -> np.ndarray:
    """Return probabilities for canceled class 1, regardless of class ordering."""
    positive_index = list(pipeline.named_steps["classifier"].classes_).index(1)
    return pipeline.predict_proba(features)[:, positive_index]


def original_feature_importance(pipeline) -> pd.Series:
    """Aggregate one-hot gain contributions using fitted encoder metadata."""
    transformer = pipeline.named_steps["preprocessing"]
    gains = pipeline.named_steps["classifier"].feature_importances_
    contributions = []
    for branch, encoder, columns in transformer.transformers_:
        if branch == "remainder" or not len(columns):
            continue
        branch_gains = gains[transformer.output_indices_[branch]]
        labels = (np.repeat(list(columns), [len(values) for values in encoder.categories_])
                  if branch == "categorical" else list(columns))
        contributions.append(pd.Series(branch_gains, index=labels, dtype=float))
    importance = pd.concat(contributions).groupby(level=0, sort=False).sum()
    total_gain = importance.sum()
    if total_gain > 0:
        importance /= total_gain
    return importance.sort_values(ascending=False).rename("gain_importance_share")


def select_one_se_model(cv_results: pd.DataFrame, cv_folds: int) -> tuple[pd.DataFrame, str]:
    """Prefer fewer leaves, shallower depth, larger leaves, then fewer trees.

    Eligibility uses the best mean AP minus its standard error. This ordering
    is predefined; validation/test scores never enter selection.
    """
    if cv_folds < 2 or cv_results.empty or cv_results["model"].duplicated().any():
        raise ValueError("Selection requires unique models and at least two folds.")
    comparison = cv_results.sort_values(
        ["cv_pr_auc_mean", "cv_roc_auc_mean", "model"],
        ascending=[False, False, True], kind="stable",
    ).reset_index(drop=True).copy()
    values = comparison[["cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean"]]
    if not np.isfinite(values.to_numpy()).all() or comparison["cv_pr_auc_std"].lt(0).any():
        raise ValueError("CV scores and standard deviations must be finite and valid.")
    comparison["cv_pr_auc_se"] = comparison["cv_pr_auc_std"] / np.sqrt(cv_folds)
    best = comparison.iloc[0]
    comparison["one_se_eligible"] = comparison["cv_pr_auc_mean"] >= best["cv_pr_auc_mean"] - best["cv_pr_auc_se"]
    eligible = comparison.loc[comparison["one_se_eligible"]].sort_values(
        ["num_leaves", "max_depth", "min_child_samples", "n_estimators",
         "cv_pr_auc_mean", "cv_roc_auc_mean", "model"],
        ascending=[True, True, False, True, False, False, True], kind="stable",
    )
    selected_name = str(eligible.iloc[0]["model"])
    comparison["selected_by_cv"] = comparison["model"] == selected_name
    return comparison, selected_name


def select_f1_threshold(target, probabilities, thresholds) -> tuple[float, pd.DataFrame]:
    """Tune on random validation only; ties favor proximity to 0.5, then lower."""
    trials = pd.DataFrame([
        {"threshold": float(value), "f1": f1_score(target, probabilities >= value, zero_division=0),
         "distance_from_half": abs(float(value) - 0.5)}
        for value in thresholds
    ]).sort_values(["f1", "distance_from_half", "threshold"], ascending=[False, True, True])
    return float(trials.iloc[0]["threshold"]), trials.reset_index(drop=True)


def score_partition(pipeline, bookings, *, model_name, parameters, experiment,
                    partition, threshold, training_rows, fit_seconds, config) -> dict:
    """Collect shared metrics and provenance without writing intermediate CSVs."""
    features, target = split_features_target(bookings)
    probabilities = cancellation_probabilities(pipeline, features)
    predictions = (probabilities >= threshold).astype(int)
    scores = evaluate(model_name, target, probabilities, results_file=None,
                      split=f"{experiment}_{partition}", threshold=threshold)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        target, predictions, labels=[0, 1]).ravel()
    return {**scores, "experiment": experiment, "partition": partition, **parameters,
            "random_state": config.random_seed, "threshold": threshold,
            "training_rows": training_rows, "evaluated_rows": len(target),
            "fit_seconds": fit_seconds, "pr_auc": average_precision_score(target, probabilities),
            "accuracy": accuracy_score(target, predictions), "tn": int(true_negative),
            "fp": int(false_positive), "fn": int(false_negative), "tp": int(true_positive)}


def fit_random_experiment(training_bookings, validation_bookings, model_parameters,
                          config: ExperimentConfig) -> RandomExperimentResult:
    """Select using training CV, collect diagnostics, then tune validation F1.

    Test data is deliberately absent from this interface. No train+validation
    refit occurs: the threshold belongs to the selected train-fitted pipeline.
    """
    train_features, train_target = split_features_target(training_bookings)
    if train_target.value_counts().min() < config.cv_folds:
        raise ValueError("Each training class needs at least cv_folds observations.")
    cross_validation = StratifiedKFold(config.cv_folds, shuffle=True, random_state=config.random_seed)
    cv_rows = []
    for model_name, parameters in model_parameters.items():
        pipeline = build_lightgbm_pipeline(train_features, parameters, config)
        scores = cross_validate(
            pipeline, train_features, train_target, cv=cross_validation,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=1, error_score="raise",
        )
        cv_rows.append({"model": model_name, **parameters,
                        "cv_pr_auc_mean": np.mean(scores["test_pr_auc"]),
                        "cv_pr_auc_std": np.std(scores["test_pr_auc"], ddof=1),
                        "cv_roc_auc_mean": np.mean(scores["test_roc_auc"]),
                        "cv_fit_seconds_mean": np.mean(scores["fit_time"])})
        print(f"{model_name}: train CV AP={cv_rows[-1]['cv_pr_auc_mean']:.4f}", flush=True)
    comparison, selected_name = select_one_se_model(pd.DataFrame(cv_rows), config.cv_folds)
    metadata = comparison.set_index("model")[CV_COLUMNS].to_dict("index")
    evaluation_rows = []
    for model_name, parameters in model_parameters.items():
        pipeline = build_lightgbm_pipeline(train_features, parameters, config)
        started = perf_counter()
        pipeline.fit(train_features, train_target)
        fit_seconds = perf_counter() - started
        for partition_name, bookings in [("selection_train", training_bookings), ("validation", validation_bookings)]:
            row = score_partition(
                pipeline, bookings, model_name=model_name, parameters=parameters,
                experiment="random", partition=partition_name, threshold=0.5,
                training_rows=len(training_bookings), fit_seconds=fit_seconds, config=config)
            evaluation_rows.append({**row, **metadata[model_name]})
        if model_name == selected_name:
            selected_pipeline, selected_fit_seconds = pipeline, fit_seconds
    validation_features, validation_target = split_features_target(validation_bookings)
    threshold, trials = select_f1_threshold(
        validation_target, cancellation_probabilities(selected_pipeline, validation_features), config.thresholds)
    selected_parameters = model_parameters[selected_name].copy()
    tuned_row = score_partition(
        selected_pipeline, validation_bookings, model_name=selected_name,
        parameters=selected_parameters, experiment="random", partition="validation_selected",
        threshold=threshold, training_rows=len(training_bookings),
        fit_seconds=selected_fit_seconds, config=config)
    evaluation_rows.append({**tuned_row, **metadata[selected_name]})
    return RandomExperimentResult(
        selected_name, selected_parameters, selected_pipeline, threshold, comparison,
        trials, evaluation_rows, original_feature_importance(selected_pipeline), selected_fit_seconds)


def evaluate_random_test(selection: RandomExperimentResult, training_bookings,
                         test_bookings, config: ExperimentConfig) -> list[dict]:
    """Evaluate the frozen model once, guarding against accidental cell reruns."""
    if selection.test_evaluated:
        raise RuntimeError("Test was already evaluated. Restart the experiment for a fresh run.")
    metadata = selection.cv_comparison.set_index("model").loc[selection.selected_model_name, CV_COLUMNS].to_dict()
    rows = []
    for partition_name, bookings in [("final_train", training_bookings), ("test", test_bookings)]:
        row = score_partition(
            selection.selected_pipeline, bookings, model_name=selection.selected_model_name,
            parameters=selection.selected_parameters, experiment="random", partition=partition_name,
            threshold=selection.selected_threshold, training_rows=len(training_bookings),
            fit_seconds=selection.fit_seconds, config=config)
        rows.append({**row, **metadata})
    selection.test_evaluated = True
    return rows


def run_temporal_experiment(training_bookings, test_bookings, model_parameters,
                            config: ExperimentConfig) -> tuple[list[dict], pd.DataFrame]:
    """Train all fixed settings on 2015-2016; report 2017 at threshold 0.5.

    No winner, validation set, early stopping, or threshold tuning is introduced.
    """
    if not training_bookings["arrival_date_year"].isin([2015, 2016]).all() or not test_bookings["arrival_date_year"].eq(2017).all():
        raise ValueError("Temporal training must use 2015-2016 and test must use 2017.")
    features, target = split_features_target(training_bookings)
    rows, importances = [], {}
    for model_name, parameters in model_parameters.items():
        pipeline = build_lightgbm_pipeline(features, parameters, config)
        started = perf_counter()
        pipeline.fit(features, target)
        fit_seconds = perf_counter() - started
        importances[model_name] = original_feature_importance(pipeline)
        for partition_name, bookings in [("train", training_bookings), ("test", test_bookings)]:
            rows.append(score_partition(
                pipeline, bookings, model_name=model_name, parameters=parameters,
                experiment="temporal", partition=partition_name, threshold=0.5,
                training_rows=len(training_bookings), fit_seconds=fit_seconds, config=config))
    return rows, pd.DataFrame(importances).T.rename_axis("model")


def combine_results(*evaluation_groups) -> pd.DataFrame:
    """Combine all partitions into one table; CV metadata is random-only."""
    results = pd.DataFrame([row for group in evaluation_groups for row in group])
    return results.sort_values(["experiment", "partition", "pr_auc"],
                               ascending=[True, True, False], kind="stable").reset_index(drop=True)


def random_overfit_summary(results: pd.DataFrame, f1_gap: float = 0.05) -> pd.DataFrame:
    """Compare random train/validation at 0.5; a gap is a diagnostic, not proof."""
    comparable = results.loc[(results["experiment"] == "random")
                             & results["partition"].isin(["selection_train", "validation"])
                             & np.isclose(results["threshold"], 0.5)]
    summary = comparable.pivot(index="model", columns="partition", values="f1").rename(
        columns={"selection_train": "train_f1", "validation": "validation_f1"})
    if summary[["train_f1", "validation_f1"]].isna().any().any():
        raise ValueError("Every model requires both train and validation diagnostics.")
    summary["f1_gap"] = summary["train_f1"] - summary["validation_f1"]
    summary["may_be_overfit"] = summary["f1_gap"] >= f1_gap
    return summary.sort_values("f1_gap", ascending=False).reset_index().rename_axis(columns=None)


def save_numbered_results(results: pd.DataFrame, directory: Path) -> Path:
    """Create one combined CSV without overwriting an existing run."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run_number = 1
    while True:
        path = directory / f"model_results_{run_number}.csv"
        try:
            output = path.open("x", encoding="utf-8", newline="")
        except FileExistsError:
            run_number += 1
            continue
        with output:
            results.to_csv(output, index=False)
        return path


def export_comparison_plots(results, selected_importance, results_path: Path,
                            *, show: bool = True) -> list[Path]:
    """Export partition-specific score charts and selected-model gain importance."""
    paths = []
    for experiment, partition in [("random", "validation"), ("temporal", "test")]:
        rows = results.loc[(results["experiment"] == experiment) & (results["partition"] == partition)]
        rows = rows.sort_values("pr_auc")
        figure, axis = plt.subplots(figsize=(10, max(4, len(rows) * 0.4)))
        axis.barh(rows["model"], rows["pr_auc"], color="#168780")
        axis.set(xlim=(0, 1), xlabel="Average precision (PR-AUC)",
                 title=f"LightGBM | {experiment} {partition} | descriptive comparison")
        paths.append(_save_figure(figure, results_path, f"{experiment}_{partition}_ap", show))
    figure, axis = plt.subplots(figsize=(10, 6))
    top_features = selected_importance.head(15).sort_values()
    axis.barh(top_features.index, top_features.values, color="#31577D")
    axis.set(xlabel="Share of training split gain (one-hot contributions summed)",
             title="Selected random LightGBM | top original features (not causal effects)")
    paths.append(_save_figure(figure, results_path, "random_feature_importance", show))
    return paths


def _save_figure(figure, results_path: Path, name: str, show: bool) -> Path:
    path = results_path.with_name(f"{name}_{results_path.stem}.png")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        from IPython.display import display
        display(figure)
    plt.close(figure)
    return path
