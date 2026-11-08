"""Clean booking copies and impute selected numeric and categorical missing values."""

import pandas as pd

if __package__:
    from .data_loader import load_hotel_bookings
else:
    from data_loader import load_hotel_bookings


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


def check_negative_values(bookings: pd.DataFrame) -> pd.DataFrame:
    """Replace negative values in selected columns with pd.NA, preserving rows and types."""
    nonnegative_columns = (
        "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
        "babies", "is_repeated_guest", "previous_cancellations",
        "previous_bookings_not_canceled", "days_in_waiting_list",
        "required_car_parking_spaces", "total_of_special_requests", "adr",
    )
    for column in nonnegative_columns:
        numeric_values = pd.to_numeric(bookings[column], errors="coerce")
        negative_values = numeric_values.lt(0).fillna(False)
        bookings[column] = bookings[column].mask(negative_values, pd.NA)
    return bookings


def handle_num_missing_values(bookings: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ADR and children values with column medians, preserving string types."""
    for column in ("adr", "children"):
        numeric_values = pd.to_numeric(bookings[column], errors="coerce")
        median_value = numeric_values.median()
        if pd.isna(median_value):
            continue
        # The loader returns strings; convert only the fill value back to that type.
        fill_value = (
            str(median_value)
            if isinstance(bookings[column].dtype, pd.StringDtype)
            else median_value
        )
        bookings[column] = bookings[column].mask(numeric_values.isna(), fill_value)
    return bookings


def handle_cat_missing_values(bookings: pd.DataFrame) -> pd.DataFrame:
    """Fill missing company, country and agent values with the unknown category."""
    missing_tokens = ["", "null", "na", "n/a", "nan"]
    for column in ("company", "country", "agent"):
        bookings[column] = bookings[column].replace(missing_tokens, pd.NA).fillna("unknown")
    return bookings


def main() -> pd.DataFrame:
    """Load raw bookings and apply the ordered steps to an independent copy."""
    raw_bookings = load_hotel_bookings()
    booking_copy = raw_bookings.copy(deep=True)
    booking_copy = remove_leakage_columns(booking_copy)
    booking_copy = remove_duplicate_rows(booking_copy)
    booking_copy = normalize_text_columns(booking_copy)
    booking_copy = check_negative_values(booking_copy)
    booking_copy = handle_num_missing_values(booking_copy)
    booking_copy = handle_cat_missing_values(booking_copy)
    return booking_copy


if __name__ == "__main__":
    main()
