"""Evaluate models with the same metrics and save the results.

Every model is scored the same way, so all results can be compared in one table.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def evaluate(
    model_name: str,
    y_true: pd.Series,
    y_prob,
    results_file: str,
    split: str = "random_test",
    threshold: float = 0.5,
) -> dict:
    """Compute F1, ROC-AUC, precision and recall, print them and append them to a CSV."""
    y_pred = (y_prob >= threshold).astype(int)

    scores = {
        "model": model_name,
        "split": split,
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
    print(f"{model_name} ({split}): F1={scores['f1']:.3f}, ROC-AUC={scores['roc_auc']:.3f}")
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / results_file
    pd.DataFrame([scores]).to_csv(
        path, mode="a", header=not path.is_file(), index=False
    )
    return scores
