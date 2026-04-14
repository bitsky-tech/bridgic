"""
Mock tools for travel planning testing.

These tools are simple and reliable, designed to test the cognitive worker architecture.
They return mock data that simulates real travel planning operations.
"""
from typing import List, Dict, Any, Optional
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


async def search_attractions(
    city: str,
    category: Optional[str] = None
) -> str:
    """
    Search for tourist attractions in a city.

    Parameters
    ----------
    city : str
        The city name where to search for attractions.
    category : Optional[str], optional
        Category of attractions (e.g., "museum", "park", "landmark", "restaurant"),
        by default None (all categories).

    Returns
    -------
    str
        Natural language description of tourist attractions.
    """
    category_filter = f" in {category} category" if category else ""
    return (
        f"Found 4 tourist attractions in {city}{category_filter}:\n"
        f"1. {city} Historical Museum (museum), ticket: ¥50\n"
        f"2. {city} Central Park (park), free entry\n"
        f"3. {city} Tower (landmark), ticket: ¥120\n"
        f"4. Traditional {city} Restaurant (restaurant)"
    )


async def create_itinerary(
    destination: str,
    start_date: str,
    end_date: str,
    activities: List[str]
) -> str:
    """
    Create a travel itinerary for a trip.

    Parameters
    ----------
    destination : str
        The destination city or location.
    start_date : str
        Trip start date in YYYY-MM-DD format.
    end_date : str
        Trip end date in YYYY-MM-DD format.
    activities : List[str]
        List of planned activities or places to visit.

    Returns
    -------
    str
        Natural language description of the created itinerary.
    """
    activities_str = "\n".join(f"  - {activity}" for activity in activities)
    return (
        f"Travel itinerary created for {destination}:\n"
        f"Dates: {start_date} to {end_date}\n"
        f"Planned activities:\n{activities_str}\n"
        f"Itinerary status: Ready"
    )


async def get_weather(
    city: str,
    date: str
) -> str:
    """
    Get weather forecast for a city on a specific date.

    Parameters
    ----------
    city : str
        The city name to get weather for.
    date : str
        The date in YYYY-MM-DD format.

    Returns
    -------
    str
        Natural language description of the weather forecast.
    """
    city_hash = hash(city) % 100
    temp = 20 + (city_hash % 15)
    condition = ["sunny", "cloudy", "rainy", "partly cloudy"][city_hash % 4]
    return f"Weather forecast for {city} on {date}: {condition}, temperature around {temp}°C"


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


async def book_room(
    hotel_name: str,
    room_type: str = "standard",
    check_in: str = None,
    check_out: str = None,
    guests: int = 1
) -> str:
    """
    Book a room at a hotel.

    Parameters
    ----------
    hotel_name : str
        The name of the hotel where to book a room.
    room_type : str, optional
        Type of room (e.g., "standard", "deluxe", "suite"), by default "standard".
    check_in : str, optional
        Check-in date in YYYY-MM-DD format. If not provided, defaults to tomorrow.
    check_out : str, optional
        Check-out date in YYYY-MM-DD format. If not provided, defaults to check_in + 1 day.
    guests : int, optional
        Number of guests, by default 1.

    Returns
    -------
    str
        Natural language booking confirmation message.
    """
    from datetime import datetime, timedelta

    # Default dates if not provided
    if check_in is None:
        check_in = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    if check_out is None:
        check_in_date = datetime.strptime(check_in, "%Y-%m-%d")
        check_out = (check_in_date + timedelta(days=1)).strftime("%Y-%m-%d")

    reservation_id = f"RM{hash(f'{hotel_name}{check_in}') % 10000:04d}"
    return (
        f"Room booking confirmed! {room_type} room at {hotel_name} from {check_in} to {check_out} "
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
    tools = [
        FunctionToolSpec.from_raw(search_flights),
        FunctionToolSpec.from_raw(search_hotels),
        FunctionToolSpec.from_raw(book_flight),
        FunctionToolSpec.from_raw(book_hotel),
        FunctionToolSpec.from_raw(book_room),
    ]
    return tools
