"""
Amadeus flight search integration.
Returns None gracefully when credentials are missing or search fails —
the itinerary generator continues without flight data in that case.
"""

import os
from datetime import datetime
from typing import Optional


def search_flights(
    origin_city: str,
    destination: str,
    departure_date: str,
    adults: int = 1,
    max_results: int = 4,
) -> Optional[str]:
    """
    Search Amadeus for real outbound flights.
    Returns a formatted block to inject into the Claude prompt, or None.
    Requires AMADEUS_API_KEY and AMADEUS_API_SECRET env vars.
    """
    api_key = os.getenv("AMADEUS_API_KEY", "").strip()
    api_secret = os.getenv("AMADEUS_API_SECRET", "").strip()
    if not api_key or not api_secret:
        return None

    try:
        from amadeus import Client, ResponseError, Location
    except ImportError:
        return None

    hostname = os.getenv("AMADEUS_HOSTNAME", "production")
    client = Client(client_id=api_key, client_secret=api_secret, hostname=hostname)

    origin_code = _city_iata(client, origin_city, Location)
    dest_code = _city_iata(client, destination, Location)
    if not origin_code or not dest_code:
        return None

    try:
        resp = client.shopping.flight_offers_search.get(
            originLocationCode=origin_code,
            destinationLocationCode=dest_code,
            departureDate=departure_date,
            adults=adults,
            max=max_results,
            currencyCode="USD",
        )
    except Exception:
        return None

    if not resp.data:
        return None

    # Carrier name lookup dictionary from response metadata
    carriers: dict = {}
    try:
        carriers = resp.result.get("dictionaries", {}).get("carriers", {})
    except Exception:
        pass

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"REAL FLIGHT DATA — {origin_city.split(',')[0].strip()} ({origin_code}) → "
        f"{destination.split(',')[0].strip()} ({dest_code})  |  Date: {departure_date}",
        "Use the options below verbatim in the ✈️ Getting There section.",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    parsed = 0
    for i, offer in enumerate(resp.data[:max_results], 1):
        try:
            price = offer["price"]["grandTotal"]
            currency = offer["price"]["currency"]
            itin = offer["itineraries"][0]
            segs = itin["segments"]

            dep = segs[0]["departure"]
            arr = segs[-1]["arrival"]
            duration = _fmt_duration(itin["duration"])
            stops = len(segs) - 1
            stop_label = "Non-stop ✅" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"

            # Carrier name(s)
            carrier_codes = list(dict.fromkeys(s["carrierCode"] for s in segs))
            carrier_str = " + ".join(
                f"{carriers.get(c, c)} ({c})" for c in carrier_codes
            )

            # Via airports for connecting flights
            via = ""
            if stops > 0:
                via_codes = [s["arrival"]["iataCode"] for s in segs[:-1]]
                via = f" via {', '.join(via_codes)}"

            # Departure / arrival times with +Nd for overnight flights
            dep_time = dep["at"][11:16]
            arr_time = arr["at"][11:16]
            date_offset = _day_offset(dep["at"], arr["at"])
            arr_label = f"{arr_time}{date_offset}"

            # Flight numbers
            flight_nums = " → ".join(
                f"{s['carrierCode']}{s['number']}" for s in segs
            )

            lines.append(
                f"\nOption {i}: {carrier_str}{via}\n"
                f"  Flights : {flight_nums}\n"
                f"  Departs : {dep['iataCode']} at {dep_time}\n"
                f"  Arrives : {arr['iataCode']} at {arr_label}\n"
                f"  Duration: {duration}  |  {stop_label}\n"
                f"  Price   : {currency} {price} per person"
            )
            parsed += 1
        except (KeyError, IndexError, TypeError):
            continue

    if parsed == 0:
        return None

    lines += [
        "",
        "→ Recommend Option 1 as the primary flight.",
        "→ Mention alternatives so the traveller can choose based on budget/time preference.",
        "→ Include exact flight numbers, times, and duration in the itinerary.",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _city_iata(client, city: str, Location) -> Optional[str]:
    """Return the primary IATA city code for a city name string."""
    city_name = city.split(",")[0].strip()
    try:
        resp = client.reference_data.locations.get(
            keyword=city_name,
            subType=Location.CITY,
        )
        for loc in resp.data or []:
            code = loc.get("iataCode")
            if code:
                return code
    except Exception:
        pass
    return None


def _fmt_duration(pt: str) -> str:
    """Convert ISO 8601 duration 'PT12H45M' → '12h 45m'."""
    pt = pt.replace("PT", "")
    hours, mins = "", ""
    if "H" in pt:
        h, pt = pt.split("H")
        hours = f"{h}h "
    if "M" in pt:
        m = pt.replace("M", "")
        mins = f"{m}m"
    return (hours + mins).strip() or pt


def _day_offset(dep_iso: str, arr_iso: str) -> str:
    """Return '+1d', '+2d', etc. if arrival is on a later calendar day."""
    try:
        dep_date = datetime.fromisoformat(dep_iso).date()
        arr_date = datetime.fromisoformat(arr_iso).date()
        diff = (arr_date - dep_date).days
        return f" (+{diff}d)" if diff > 0 else ""
    except Exception:
        return ""
