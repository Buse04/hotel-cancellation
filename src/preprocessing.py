"""Clean booking copies and impute selected numeric and categorical missing values."""

import pandas as pd

if __package__:
    from .data_loader import load_hotel_bookings
else:
    from data_loader import load_hotel_bookings

NUMERIC_COLUMNS = (
    "is_canceled", "lead_time", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
    "adr", "required_car_parking_spaces", "total_of_special_requests",
)

MISSING_TOKENS = ["", "null", "na", "n/a", "nan"]

def remove_leakage_columns(bookings: pd.DataFrame) -> pd.DataFrame:
    """Exclude reservation outcome and outcome-date columns."""
    return bookings.drop(columns=["reservation_status", "reservation_status_date"])


def remove_duplicate_rows(bookings: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of each identical row."""
    return bookings.drop_duplicates(keep="first")


def normalize_text_columns(bookings: pd.DataFrame) -> pd.DataFrame:
    """Lowercase text and collapse/trim whitespace in the working dataframe."""
    text_columns = bookings.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        bookings[column] = (
            bookings[column]
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"^ +| +$", "", regex=True)
        )
    return bookings

def convert_types(bookings: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns from text to numbers."""
    for column in NUMERIC_COLUMNS:
        values = bookings[column].replace(MISSING_TOKENS, pd.NA)
        bookings[column] = pd.to_numeric(values, errors="raise")
    return bookings

def remove_invalid_rows(bookings: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows from bookings."""
    total_nights = bookings["stays_in_weekend_nights"] + bookings["stays_in_week_nights"]

    invalid = (
        (bookings["adults"] == 0)
        | (total_nights == 0)
        | (bookings["adr"] <= 0)
        | (bookings["adr"] > 1000)
        | (bookings["babies"] >= 9)
        | (bookings["children"] >= 10)
        | (bookings["required_car_parking_spaces"] >= 8)
    )
    invalid = invalid.fillna(False)
    return bookings.loc[~invalid]


def handle_num_missing_values(bookings: pd.DataFrame) -> pd.DataFrame:
    """Fill the few missing children counts with 0 (no children)."""
    bookings["children"] = bookings["children"].fillna(0)
    return bookings


def handle_cat_missing_values(bookings: pd.DataFrame) -> pd.DataFrame:
    """Fill missing categorical values."""
    for column in ("agent", "company"):
        bookings[column] = (
            bookings[column].replace(MISSING_TOKENS, pd.NA).fillna("none")
        )
    bookings["country"] = (
        bookings["country"].replace(MISSING_TOKENS, pd.NA).fillna("unknown")
    )
    return bookings

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12
}

def add_features(bookings: pd.DataFrame) -> pd.DataFrame:
    """Add new features derived from existing columns."""
    bookings["total_nights"] = bookings["stays_in_weekend_nights"] + bookings["stays_in_week_nights"]
    bookings["total_guests"] = bookings["adults"] + bookings["children"] + bookings["babies"]

    arrival_date = pd.to_datetime(dict(
        year=bookings["arrival_date_year"],
        month=bookings["arrival_date_month"].map(MONTHS),
        day=bookings["arrival_date_day_of_month"],
    ))
    bookings["arrival_weekday"] = arrival_date.dt.day_name().str.lower()
    return bookings

UNUSED_COLUMNS = ["arrival_date_week_number", "arrival_date_day_of_month"]

def drop_unused_columns(bookings: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns."""
    return bookings.drop(columns=UNUSED_COLUMNS)

NONNEGATIVE_COLUMNS = ["stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "days_in_waiting_list",
    "required_car_parking_spaces", "total_of_special_requests", "adr",]

def validate_bookings(bookings: pd.DataFrame) -> pd.DataFrame:
    """Check that the cleaned data meets the expected rules; raise an error if not.

    Runs before drop_unused_columns: removing the date columns makes some
    different bookings look identical, which would fail the duplicate check.
    """
    assert bookings.isna().sum().sum() == 0, "Missing values remain"
    assert not bookings.duplicated().any(), "Duplicate rows remain"
    assert (bookings[NONNEGATIVE_COLUMNS] >= 0).all().all(), "Negative values found"
    assert (bookings["adults"] > 0).all(), "Bookings without adults remain"
    assert bookings["is_canceled"].isin([0, 1]).all(), "Target must be 0 or 1"
    return bookings


def main() -> pd.DataFrame:
    """Load raw bookings and apply the ordered steps to an independent copy."""
    raw_bookings = load_hotel_bookings()
    booking_copy = raw_bookings.copy(deep=True)
    booking_copy = remove_leakage_columns(booking_copy)
    booking_copy = remove_duplicate_rows(booking_copy)
    booking_copy = normalize_text_columns(booking_copy)
    booking_copy = convert_types(booking_copy)
    booking_copy = remove_invalid_rows(booking_copy)
    booking_copy = handle_num_missing_values(booking_copy)
    booking_copy = handle_cat_missing_values(booking_copy)
    booking_copy = add_features(booking_copy)
    booking_copy = validate_bookings(booking_copy)
    booking_copy = drop_unused_columns(booking_copy)
    return booking_copy


if __name__ == "__main__":
    main()
