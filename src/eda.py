"""Reusable, non-destructive checks and plots for hotel booking exploration."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEAKAGE_COLUMNS = ("reservation_status", "reservation_status_date")
VALUE_COUNT_COLUMNS = (
    "hotel", "is_canceled", "arrival_date_year", "meal", "market_segment",
    "distribution_channel", "reserved_room_type", "assigned_room_type",
    "deposit_type", "customer_type",
)
GUEST_COLUMNS = ("adults", "children", "babies")
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
NUMERIC_COLUMNS = (
    "is_canceled", "lead_time", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
    "adr", "required_car_parking_spaces", "total_of_special_requests",
)
# Exclude binary labels, calendar codes and identifiers from numeric outlier checks.
OUTLIER_COLUMNS = tuple(column for column in NUMERIC_COLUMNS if column not in {
    "is_canceled", "is_repeated_guest", "arrival_date_year",
    "arrival_date_week_number", "arrival_date_day_of_month",
})
SEASON_MONTHS = {
    "Winter": ("December", "January", "February"),
    "Spring": ("March", "April", "May"),
    "Summer": ("June", "July", "August"),
    "Autumn": ("September", "October", "November"),
}


def feature_value_counts(frame: pd.DataFrame, columns=None) -> pd.DataFrame:
    """List all observed values for the requested features, including missing entries."""
    parts = []
    for column in frame.columns if columns is None else columns:
        counts = frame[column].value_counts(dropna=False)
        parts.append(pd.DataFrame({
            "feature": column,
            "value": ["<MISSING>" if pd.isna(value) else str(value) for value in counts.index],
            "is_missing": [bool(pd.isna(value)) for value in counts.index],
            "count": counts.to_numpy(),
            "percent": 100 * counts.to_numpy() / len(frame) if len(frame) else 0,
        }))
    return pd.concat(parts, ignore_index=True)


def duplicate_summary(frame: pd.DataFrame) -> pd.Series:
    """Distinguish removable copies from all members of duplicated groups."""
    extra = frame.duplicated()
    members = frame.duplicated(keep=False)
    return pd.Series({
        "rows": len(frame), "additional_duplicate_rows": int(extra.sum()),
        "rows_in_duplicate_groups": int(members.sum()),
        "duplicate_groups": len(frame.loc[members].drop_duplicates()),
        "additional_duplicate_percent": 100 * extra.mean() if len(frame) else 0.0,
    })


def iqr_outliers(
    frame: pd.DataFrame, columns: tuple[str, ...], multiplier: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return bounds and every observation strictly outside the IQR bounds."""
    if multiplier < 0:
        raise ValueError("The IQR multiplier must be nonnegative.")
    summaries, records = [], []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        finite = values.dropna().astype(float)
        if not np.isfinite(finite).all():
            raise ValueError(f"Nonfinite values found in {column}; inspect them before IQR analysis.")
        q1, q3 = finite.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        below = values.lt(lower).fillna(False)
        above = values.gt(upper).fillna(False)
        mask = below | above
        summaries.append({
            "feature": column, "nonmissing_count": len(finite), "q1": q1, "q3": q3,
            "iqr": iqr, "lower_bound": lower, "upper_bound": upper,
            "below_lower_count": int(below.sum()), "above_upper_count": int(above.sum()),
            "outlier_count": int(mask.sum()),
            "outlier_percent": 100 * mask.sum() / len(finite) if len(finite) else np.nan,
            "zero_iqr": bool(iqr == 0),
        })
        records.append(pd.DataFrame({
            "source_row_index": frame.index[mask], "feature": column,
            "value": values.loc[mask].to_numpy(), "lower_bound": lower, "upper_bound": upper,
            "direction": np.where(below.loc[mask].to_numpy(dtype=bool), "below_lower", "above_upper"),
        }))
    return pd.DataFrame(summaries).set_index("feature"), pd.concat(records, ignore_index=True)


def plot_iqr_checks(frame, bounds, multiplier: float = 1.5):
    """Plot each numeric feature independently, using all nonmissing observations."""
    columns = bounds.index.tolist()
    ncols = 3
    nrows = (len(columns) + ncols - 1) // ncols
    box_figure, box_axes = plt.subplots(nrows, ncols, figsize=(16, 3.4 * nrows), squeeze=False)
    scatter_figure, scatter_axes = plt.subplots(nrows, ncols, figsize=(16, 3.4 * nrows), squeeze=False)
    for column, box_axis, scatter_axis in zip(columns, box_axes.flat, scatter_axes.flat):
        values = frame[column].dropna().astype(float)
        lower, upper = bounds.loc[column, ["lower_bound", "upper_bound"]]
        outlier = (values < lower) | (values > upper)
        box_axis.boxplot(values, whis=multiplier, flierprops={"marker": ".", "markersize": 3, "markeredgecolor": "#b73135"})
        box_axis.set_title(column)
        box_axis.set_ylabel("Observed value")
        scatter_axis.scatter(values.index[~outlier], values[~outlier], s=3, alpha=0.25, color="#3479b7", rasterized=True)
        scatter_axis.scatter(values.index[outlier], values[outlier], s=5, alpha=0.6, color="#b73135", label="IQR outlier", rasterized=True)
        scatter_axis.axhline(lower, color="#a15e00", linestyle="--", label="IQR bounds")
        scatter_axis.axhline(upper, color="#a15e00", linestyle="--")
        scatter_axis.set(title=column, xlabel="Source row index", ylabel="Observed value")
        scatter_axis.legend(fontsize=7)
    for axes in (box_axes, scatter_axes):
        for axis in list(axes.flat)[len(columns):]:
            axis.set_visible(False)
    for figure, title in (
        (box_figure, f"Numeric feature box plots ({multiplier:g} x IQR)"),
        (scatter_figure, "Numeric values by source row: IQR outliers in red"),
    ):
        figure.suptitle(title)
        figure.tight_layout(rect=(0, 0, 1, 0.98))
    return box_figure, scatter_figure


def seasonal_room_prices(frame: pd.DataFrame, room_type: str) -> pd.DataFrame:
    """Mean ADR by calendar season and hotel for one reserved room type."""
    month_to_season = {month: season for season, months in SEASON_MONTHS.items() for month in months}
    selected = frame.loc[frame["reserved_room_type"].eq(room_type), ["hotel", "arrival_date_month", "adr"]].copy()
    if selected.empty:
        raise ValueError(f"No bookings found for reserved room type {room_type!r}.")
    selected["season"] = selected["arrival_date_month"].map(month_to_season)
    if selected["season"].isna().any():
        raise ValueError("Unexpected arrival month; update the season mapping.")
    hotels = sorted(frame["hotel"].dropna().unique())
    if len(hotels) != 2:
        raise ValueError("The seasonal comparison expects exactly two hotel types.")
    summary = selected.groupby(["season", "hotel"])["adr"].agg(mean_adr="mean", median_adr="median", booking_count="size", priced_booking_count="count")
    full_index = pd.MultiIndex.from_product([SEASON_MONTHS, hotels], names=["season", "hotel"])
    return summary.reindex(full_index).reset_index()


def plot_seasonal_prices(summary, room_type: str):
    """Keep missing season/hotel prices as gaps rather than imputing zeros."""
    figure, axis = plt.subplots(figsize=(10, 5))
    for hotel in sorted(summary["hotel"].unique()):
        subset = summary.loc[summary["hotel"].eq(hotel)]
        axis.plot(subset["season"], subset["mean_adr"].astype(float), marker="o", label=hotel)
    axis.set(title=f"Mean seasonal ADR for reserved room type {room_type}", xlabel="Calendar season (all years pooled)", ylabel="Mean ADR (dataset units)")
    axis.legend(title="Hotel type")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def zero_guest_mask(frame: pd.DataFrame) -> pd.Series:
    """Match only rows where all three guest counts are explicitly zero."""
    return frame.loc[:, list(GUEST_COLUMNS)].eq(0).fillna(False).all(axis=1)


def yearly_room_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize booking-level ADR across all room types, by year and hotel."""
    return frame.groupby(["arrival_date_year", "hotel"], sort=True)["adr"].agg(
        mean_adr="mean", median_adr="median", booking_count="size", priced_booking_count="count"
    ).reset_index()


def guest_counts(frame: pd.DataFrame, period: str, include_canceled: bool = True) -> pd.DataFrame:
    """Sum booked guests, not distinct people; do not interpret unknown counts as zero."""
    if period not in {"arrival_date_month", "arrival_date_year"}:
        raise ValueError("Guest-count period must be arrival_date_month or arrival_date_year.")
    selected = frame if include_canceled else frame.loc[frame["is_canceled"].eq(0)]
    selected = selected.loc[:, [period, "hotel", *GUEST_COLUMNS]].copy()
    selected["guests"] = selected.loc[:, list(GUEST_COLUMNS)].sum(axis=1, min_count=3)
    summary = selected.groupby([period, "hotel"])["guests"].agg(
        guest_count=lambda values: values.sum(min_count=1),
        booking_count="size", complete_guest_count_bookings="count",
    )
    periods = MONTHS if period == "arrival_date_month" else sorted(frame[period].dropna().unique())
    hotels = sorted(frame["hotel"].dropna().unique())
    # Unobserved periods stay missing rather than being silently represented as zeros.
    index = pd.MultiIndex.from_product([periods, hotels], names=[period, "hotel"])
    return summary.reindex(index).reset_index()


def plot_hotel_trends(summary, period: str, metric: str, title: str, ylabel: str):
    """Return an in-memory figure with one line per hotel; never export files."""
    figure, axis = plt.subplots(figsize=(11, 5))
    periods = summary[period].drop_duplicates().tolist()
    positions = np.arange(len(periods))
    for hotel in sorted(summary["hotel"].unique()):
        values = summary.loc[summary["hotel"].eq(hotel)].set_index(period)[metric].reindex(periods)
        axis.plot(positions, values.to_numpy(dtype=float, na_value=np.nan), marker="o", label=hotel)
    labels = [str(int(value)) if period == "arrival_date_year" else str(value) for value in periods]
    axis.set_xticks(positions, labels, rotation=45 if period == "arrival_date_month" else 0)
    axis.set(title=title, xlabel="Arrival year" if period == "arrival_date_year" else "Arrival month (all years pooled)", ylabel=ylabel)
    axis.legend(title="Hotel type")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure
