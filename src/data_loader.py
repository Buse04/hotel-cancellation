"""Load raw hotel bookings without cleaning or transforming their values."""

from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "hotel_bookings.csv"
)

def load_hotel_bookings(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Bookings CSV file not found: {data_path}")

    try:
        raw_bookings = pd.read_csv(
            data_path,
            encoding="utf-8-sig",
            dtype="string",
            keep_default_na=False,
            skip_blank_lines=False,
        )
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Bookings CSV is empty: {data_path}") from error

    return raw_bookings
