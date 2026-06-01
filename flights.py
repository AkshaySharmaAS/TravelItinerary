"""
Live flight search via SerpApi — returns real Google Flights data:
actual prices, flight numbers, departure/arrival times, layovers, aircraft.

Free tier: 100 searches / month, no credit card.
Sign up at https://serpapi.com  →  copy your API key  →  set SERPAPI_KEY in .env

Falls back gracefully to None when the key is missing or a search fails,
so the itinerary is still generated without flight data.
"""

import csv
import json
import math
import os
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

# ── OpenFlights airport data (for city → IATA code lookup only) ───────────────
_DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openflights_data")
_AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
_ROUTES_URL   = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"

_airports_cache: Optional[Dict[str, dict]] = None
_routes_cache:   Optional[Dict[Tuple[str, str], int]] = None   # (src,dst) → count (unused here, but used for ranking)
_route_counts:   Optional[Dict[str, int]] = None               # iata → total routes


def _fetch(fname: str, url: str) -> str:
    os.makedirs(_DATA_DIR, exist_ok=True)
    path = os.path.join(_DATA_DIR, fname)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def _airports_db() -> Dict[str, dict]:
    global _airports_cache
    if _airports_cache is not None:
        return _airports_cache
    result: Dict[str, dict] = {}
    with open(_fetch("airports.dat", _AIRPORTS_URL), encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if len(row) < 9:
                continue
            iata = row[4].strip().strip('"')
            if not iata or iata == r"\N" or len(iata) != 3:
                continue
            try:
                lat, lon = float(row[6]), float(row[7])
            except ValueError:
                continue
            result[iata] = {
                "name":    row[1].strip('"'),
                "city":    row[2].strip('"'),
                "country": row[3].strip('"'),
                "lat": lat, "lon": lon,
            }
    _airports_cache = result
    return result


def _route_count_db() -> Dict[str, int]:
    """Returns {iata: number_of_routes} — used to rank airports for a city."""
    global _route_counts
    if _route_counts is not None:
        return _route_counts
    counts: Dict[str, int] = {}
    try:
        with open(_fetch("routes.dat", _ROUTES_URL), encoding="utf-8", errors="ignore") as f:
            for row in csv.reader(f):
                if len(row) < 5:
                    continue
                src, dst = row[2].strip(), row[4].strip()
                if src and src != r"\N" and len(src) == 3:
                    counts[src] = counts.get(src, 0) + 1
                if dst and dst != r"\N" and len(dst) == 3:
                    counts[dst] = counts.get(dst, 0) + 1
    except Exception:
        pass
    _route_counts = counts
    return counts


def _city_iata(city_str: str) -> Optional[str]:
    """
    Returns the IATA code of the busiest airport serving a city.
    Busiest = most routes in OpenFlights data → almost always the main international hub.
    """
    city_name = city_str.split(",")[0].strip().lower()
    airports   = _airports_db()
    rcounts    = _route_count_db()

    exact, partial = [], []
    for iata, info in airports.items():
        c = info["city"].lower()
        if c == city_name:
            exact.append(iata)
        elif city_name in c or c in city_name:
            partial.append(iata)

    candidates = exact or partial
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    return max(candidates, key=lambda x: rcounts.get(x, 0))


# ── SerpApi call ──────────────────────────────────────────────────────────────

def _serpapi_flights(orig: str, dest: str, date: str, adults: int) -> Optional[dict]:
    api_key = os.getenv("SERPAPI_KEY", "").strip()
    if not api_key:
        return None

    params = {
        "engine":        "google_flights",
        "departure_id":  orig,
        "arrival_id":    dest,
        "outbound_date": date,
        "currency":      "USD",
        "hl":            "en",
        "type":          "2",          # one-way
        "adults":        str(adults),
        "api_key":       api_key,
    }
    url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TravelAgentCRM/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"SerpApi request failed: {exc}")
        return None


# ── Formatting helpers ────────────────────────────────────────────────────────

def _mins(m: int) -> str:
    h, mn = divmod(int(m), 60)
    return f"{h}h {mn:02d}m"


def _time(iso: str) -> str:
    """'2025-08-01 14:35' → '14:35'"""
    return iso[11:] if len(iso) > 10 else iso


def _day_offset(dep: str, arr: str) -> str:
    try:
        from datetime import datetime
        d = (datetime.fromisoformat(arr).date() - datetime.fromisoformat(dep).date()).days
        return f" (+{d}d)" if d > 0 else ""
    except Exception:
        return ""


# ── Public entry point ────────────────────────────────────────────────────────

def search_flights(
    origin_city:    str,
    destination:    str,
    departure_date: Optional[str] = None,
    adults:         int = 1,
    max_results:    int = 4,
) -> Optional[str]:
    """
    Returns a formatted flight-data block for the Claude prompt, or None.
    Requires SERPAPI_KEY env var and a departure_date.
    """
    if not departure_date:
        return None
    if not os.getenv("SERPAPI_KEY", "").strip():
        return None

    try:
        orig_iata = _city_iata(origin_city)
        dest_iata = _city_iata(destination)
    except Exception:
        return None

    if not orig_iata or not dest_iata:
        return None

    data = _serpapi_flights(orig_iata, dest_iata, departure_date, adults)
    if not data:
        return None

    all_offers: List[dict] = data.get("best_flights", []) + data.get("other_flights", [])
    if not all_offers:
        return None

    airports   = _airports_db()
    orig_city  = airports.get(orig_iata, {}).get("city", origin_city.split(",")[0])
    dest_city  = airports.get(dest_iata, {}).get("city", destination.split(",")[0])

    sep = "=" * 56
    lines = [
        sep,
        "LIVE FLIGHT DATA  (Google Flights via SerpApi)",
        f"{orig_city} ({orig_iata})  →  {dest_city} ({dest_iata})  |  {departure_date}",
        "All prices are per person, economy class, from Google Flights.",
        "Use these exact details verbatim in the ✈️ Getting There section.",
        sep,
    ]

    parsed = 0
    for offer in all_offers[:max_results]:
        try:
            segs      = offer.get("flights", [])
            layovers  = offer.get("layovers", [])
            price     = offer.get("price")
            total_dur = offer.get("total_duration", 0)

            if not segs:
                continue

            first_seg = segs[0]
            last_seg  = segs[-1]
            dep_ap    = first_seg["departure_airport"]
            arr_ap    = last_seg["arrival_airport"]

            dep_time  = _time(dep_ap.get("time", ""))
            arr_time  = _time(arr_ap.get("time", ""))
            offset    = _day_offset(dep_ap.get("time", ""), arr_ap.get("time", ""))

            # Deduplicated airlines in order
            airlines    = list(dict.fromkeys(s.get("airline", "") for s in segs))
            airline_str = " + ".join(a for a in airlines if a)

            # Flight numbers
            flight_nums = "  →  ".join(s.get("flight_number", "?") for s in segs)

            stops      = len(segs) - 1
            stop_label = "Non-stop ✅" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"

            # Aircraft (first segment)
            aircraft   = first_seg.get("airplane", "")
            aircraft_s = f"\n  Aircraft: {aircraft}" if aircraft else ""

            # Layover detail
            layover_s  = ""
            for lv in layovers:
                lv_name = lv.get("name", lv.get("id", ""))
                lv_dur  = _mins(lv.get("duration", 0))
                overnight = "  ⚠️ overnight" if lv.get("overnight") else ""
                layover_s += f"\n  Layover : {lv_name} — {lv_dur}{overnight}"

            price_s = f"USD {price:,}" if price else "check airline site"

            # Carbon emissions vs typical
            co2 = offer.get("carbon_emissions", {})
            co2_diff = co2.get("difference_percent")
            co2_s = ""
            if co2_diff is not None:
                label = "above" if co2_diff > 0 else "below"
                co2_s = f"\n  Carbon  : {abs(co2_diff)}% {label} average for this route"

            lines.append(
                f"\nOption {parsed + 1}:  {airline_str}  |  {stop_label}\n"
                f"  Flights : {flight_nums}\n"
                f"  Departs : {dep_ap.get('name','?')} ({dep_ap.get('id','?')})  at  {dep_time}\n"
                f"  Arrives : {arr_ap.get('name','?')} ({arr_ap.get('id','?')})  at  {arr_time}{offset}\n"
                f"  Duration: {_mins(total_dur)}"
                f"{aircraft_s}"
                f"{layover_s}"
                f"{co2_s}\n"
                f"  💰 Price : {price_s} per person"
            )
            parsed += 1
        except (KeyError, TypeError, IndexError):
            continue

    if parsed == 0:
        return None

    # Price insight if available
    insight = data.get("price_insights", {})
    if insight.get("typical_price_range"):
        lo, hi = insight["typical_price_range"]
        lines.append(f"\n📊 Typical price range for this route: USD {lo:,} – {hi:,}")

    lines += [
        "",
        "→ Recommend the best-value option (balance of price, duration, stops).",
        "→ Quote exact flight numbers, departure/arrival times, and airport full names.",
        "→ Mention baggage allowance and check-in tips for the recommended airline.",
        sep,
    ]
    return "\n".join(lines)
