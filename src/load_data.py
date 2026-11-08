"""Load the raw hotel bookings data."""

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "hotel_bookings.csv"
)

def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the hotel bookings CSV and return it as a Pandas DataFrame."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}"
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Data file is empty: {path}") from error
    return df