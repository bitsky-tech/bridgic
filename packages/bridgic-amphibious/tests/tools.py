"""
Mock tools for travel planning testing.

These tools are simple and reliable, designed to test the cognitive worker architecture.
They return mock data that simulates real travel planning operations.
"""
from typing import List, Optional
from bridgic.core.agentic.tool_specs import FunctionToolSpec


async def search_flights(
    origin: str,
    destination: str,
    date: str,
    passengers: int = 1
) -> str:
    """
    Search for available flights between two cities.

    Parameters
    ----------
    origin : str
        The departure city (e.g., "Beijing", "Shanghai").
    destination : str
        The arrival city (e.g., "Tokyo", "Paris").
    date : str
        The departure date in YYYY-MM-DD format.
    passengers : int, optional
        Number of passengers, by default 1.

    Returns
    -------
    str
        Natural language description of available flights.
    """
    return (
        f"Found 3 available flights from {origin} to {destination} on {date}:\n"
        f"1. Flight CA123, departs at 08:00, price: ¥3500\n"
        f"2. Flight MU456, departs at 14:20, price: ¥3200\n"
        f"3. Flight CZ789, departs at 10:15, price: ¥3800"
    )


async def search_hotels(
    city: str,
    check_in: str,
    check_out: str,
    guests: int = 1,
    stars: Optional[int] = None
) -> str:
    """
    Search for available hotels in a city.

    Parameters
    ----------
    city : str
        The city name where to search for hotels.
    check_in : str
        Check-in date in YYYY-MM-DD format.
    check_out : str
        Check-out date in YYYY-MM-DD format.
    guests : int, optional
        Number of guests, by default 1.
    stars : Optional[int], optional
        Preferred hotel star rating (1-5), by default None (all ratings).

    Returns
    -------
    str
        Natural language description of available hotels.
    """
    star_filter = f" ({stars}-star)" if stars is not None else ""
    return (
        f"Found 3 available hotels in {city}{star_filter} for {check_in} to {check_out}:\n"
        f"1. Grand {city} Hotel, 5 stars, ¥800 per night\n"
        f"2. {city} Central Plaza, 4 stars, ¥500 per night\n"
        f"3. Comfort Inn {city}, 3 stars, ¥300 per night"
    )


async def book_flight(
    flight_number: str,
    passengers: int = 1
) -> str:
    """
    Book a flight with the given flight number.

    Parameters
    ----------
    flight_number : str
        The flight number to book (e.g., "CA123").
    passengers : int, optional
        Number of passengers, by default 1.

    Returns
    -------
    str
        Natural language booking confirmation message.
    """
    booking_id = f"BK{hash(flight_number) % 10000:04d}"
    return f"Flight booking confirmed! Flight {flight_number} for {passengers} passenger(s). Booking ID: {booking_id}"


async def book_hotel(
    hotel_name: str,
    check_in: str,
    check_out: str,
    guests: int = 1
) -> str:
    """
    Book a hotel room.

    Parameters
    ----------
    hotel_name : str
        The name of the hotel to book.
    check_in : str
        Check-in date in YYYY-MM-DD format.
    check_out : str
        Check-out date in YYYY-MM-DD format.
    guests : int, optional
        Number of guests, by default 1.

    Returns
    -------
    str
        Natural language booking confirmation message.
    """
    reservation_id = f"HT{hash(f'{hotel_name}{check_in}') % 10000:04d}"
    return (
        f"Hotel booking confirmed! {hotel_name} from {check_in} to {check_out} "
        f"for {guests} guest(s). Reservation ID: {reservation_id}"
    )


# Create ToolSpec instances for all tools
def get_travel_planning_tools() -> List[FunctionToolSpec]:
    """
    Get a list of all travel planning mock tools.

    Returns
    -------
    List[FunctionToolSpec]
        A list of FunctionToolSpec instances for all travel planning tools.
    """
    return [
        FunctionToolSpec.from_raw(search_flights),
        FunctionToolSpec.from_raw(search_hotels),
        FunctionToolSpec.from_raw(book_flight),
        FunctionToolSpec.from_raw(book_hotel),
    ]
