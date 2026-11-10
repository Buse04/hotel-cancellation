"""Create independent random and temporal splits of preprocessed bookings.

Each returned dataframe retains ``is_canceled`` as the target column and its
original row indices. Drop the target from model inputs before training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

if __package__:
    from .preprocessing import main as preprocess_bookings
else:
    from preprocessing import main as preprocess_bookings


def random_split(
    bookings: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/validation/test (70/15/15), stratified by is_canceled.

    Integer row counts may cause small rounding differences in proportions.
    """
    train, remaining = train_test_split(
        bookings,
        test_size=0.30,
        random_state=random_state,
        stratify=bookings["is_canceled"],
    )
    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=random_state,
        stratify=remaining["is_canceled"],
    )
    return train.copy(), validation.copy(), test.copy()


def temporal_split(
    bookings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return arrival-year train (2015–2016) and test (2017), without validation.

    Reject unexpected or missing years rather than silently dropping rows.
    """
    years = bookings["arrival_date_year"]
    if not years.isin([2015, 2016, 2017]).all():
        raise ValueError("Temporal split requires arrival years 2015, 2016 or 2017.")

    train = bookings.loc[years.isin([2015, 2016])].copy()
    test = bookings.loc[years == 2017].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal train and test sets must both contain bookings.")
    return train, test


def main() -> dict[str, dict[str, pd.DataFrame]]:
    """Preprocess once and create both experiments from the same booking copy."""
    bookings = preprocess_bookings()
    random_train, random_validation, random_test = random_split(bookings)
    temporal_train, temporal_test = temporal_split(bookings)
    return {
        "random": {
            "train": random_train,
            "validation": random_validation,
            "test": random_test,
        },
        "temporal": {
            "train": temporal_train,
            "test": temporal_test,
        },
    }


def print_split_summary(splits: dict[str, dict[str, pd.DataFrame]]) -> None:
    """Print the size and cancellation rate of each experiment's datasets."""
    for experiment, datasets in splits.items():
        print(f"{experiment.capitalize()} split:")
        for name, dataset in datasets.items():
            canceled_rate = dataset["is_canceled"].mean()
            print(f"  {name}: {len(dataset):,} bookings, canceled={canceled_rate:.2%}")


if __name__ == "__main__":
    print_split_summary(main())
