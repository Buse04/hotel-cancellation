"""Run the hotel cancellation project's workflow."""

import pandas as pd

if __package__:
    from .data_split import main as create_data_splits, print_split_summary
else:
    from data_split import main as create_data_splits, print_split_summary


def main() -> dict[str, dict[str, pd.DataFrame]]:
    """Load and preprocess bookings, create both splits, and report their sizes."""
    splits = create_data_splits()
    print_split_summary(splits)
    return splits


if __name__ == "__main__":
    main()
