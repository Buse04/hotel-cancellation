"""Run the hotel cancellation project's workflow."""

if __package__:
    from .data_loader import load_hotel_bookings
else:
    from data_loader import load_hotel_bookings


def main() -> None:
    raw_bookings = load_hotel_bookings()
    row_count, column_count = raw_bookings.shape
    print(f"Loaded {row_count:,} bookings with {column_count} columns.")


if __name__ == "__main__":
    main()
