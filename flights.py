"""
Flight route data using OpenFlights open dataset — no API key required.
Data: https://openflights.org/data  (CC BY 4.0)

On first use, downloads three small CSV files (~3 MB total) from the
OpenFlights GitHub mirror and caches them to disk.  Subsequent starts
load instantly from the cache.  Completely offline after first fetch.
"""

import csv
import math
import os
import urllib.request
from typing import Dict, List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openflights_data")

_AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
_AIRLINES_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"
_ROUTES_URL   = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"

# In-memory caches — loaded once per process
_airports: Optional[Dict[str, dict]] = None
_airlines: Optional[Dict[str, str]] = None
_routes:   Optional[Dict[Tuple[str, str], List[str]]] = None


# ── Data loading ──────────────────────────────────────────────────────────────

def _fetch(fname: str, url: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def _airports_db() -> Dict[str, dict]:
    global _airports
    if _airports is not None:
        return _airports
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
    _airports = result
    return result


def _airlines_db() -> Dict[str, str]:
    global _airlines
    if _airlines is not None:
        return _airlines
    result: Dict[str, str] = {}
    with open(_fetch("airlines.dat", _AIRLINES_URL), encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if len(row) < 8:
                continue
            iata   = row[3].strip().strip('"')
            name   = row[1].strip('"')
            active = row[7].strip().strip('"')
            if iata and iata != r"\N" and len(iata) <= 3 and active == "Y":
                result[iata] = name
    _airlines = result
    return result


def _routes_db() -> Dict[Tuple[str, str], List[str]]:
    """{ (src_iata, dst_iata): [airline_iata, ...] }  — direct flights only."""
    global _routes
    if _routes is not None:
        return _routes
    result: Dict[Tuple[str, str], List[str]] = {}
    with open(_fetch("routes.dat", _ROUTES_URL), encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if len(row) < 8:
                continue
            airline = row[0].strip()
            src     = row[2].strip()
            dst     = row[4].strip()
            stops   = row[7].strip()
            if (not airline or airline == r"\N" or
                    not src or src == r"\N" or len(src) != 3 or
                    not dst or dst == r"\N" or len(dst) != 3 or
                    stops != "0"):
                continue
            key = (src, dst)
            if airline not in result.get(key, []):
                result.setdefault(key, []).append(airline)
    _routes = result
    return result


# ── City → IATA lookup ────────────────────────────────────────────────────────

def _city_iata(city_str: str, airports: Dict, routes: Dict) -> Optional[str]:
    """Return the IATA code of the busiest airport in the given city."""
    city_name = city_str.split(",")[0].strip().lower()

    # Exact city match first, then partial
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

    # Pick the busiest airport by total route count
    def _route_count(iata: str) -> int:
        return sum(1 for s, d in routes if s == iata or d == iata)

    return max(candidates, key=_route_count)


# ── Geo / time helpers ────────────────────────────────────────────────────────

def _km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _fmt_dur(hours: float) -> str:
    h, m = int(hours), int((hours % 1) * 60)
    return f"{h}h {m:02d}m"


def _price(km: float) -> Tuple[int, int]:
    """Rough economy price range in USD."""
    base = 80 + km * 0.11
    return int(base), int(base * 1.45)


# ── Main entry point ──────────────────────────────────────────────────────────

def search_flights(
    origin_city: str,
    destination: str,
    departure_date: Optional[str] = None,
    adults: int = 1,
    max_results: int = 4,
) -> Optional[str]:
    """
    Look up real airline routes from the OpenFlights open dataset.
    Returns a formatted text block for the Claude prompt, or None if no data.
    """
    try:
        airports = _airports_db()
        airlines = _airlines_db()
        routes   = _routes_db()
    except Exception as exc:
        print(f"OpenFlights data error: {exc}")
        return None

    orig_iata = _city_iata(origin_city, airports, routes)
    dest_iata = _city_iata(destination, airports, routes)
    if not orig_iata or not dest_iata:
        return None

    orig = airports[orig_iata]
    dest = airports[dest_iata]
    dist = _km(orig["lat"], orig["lon"], dest["lat"], dest["lon"])
    if dist < 1:
        return None  # same airport

    direct_hours = dist / 850 + 0.75   # 850 km/h cruise + 45 min ground
    lo_p, hi_p   = _price(dist)

    # ── Direct routes ─────────────────────────────────────────────────────────
    direct_airlines = routes.get((orig_iata, dest_iata), [])

    # ── 1-stop options ────────────────────────────────────────────────────────
    one_stop: List[dict] = []
    if len(direct_airlines) < max_results:
        outbound_hubs = {d for (s, d) in routes if s == orig_iata}
        inbound_hubs  = {s for (s, d) in routes if d == dest_iata}
        shared_hubs   = outbound_hubs & inbound_hubs

        for hub in list(shared_hubs)[:20]:
            a1 = routes.get((orig_iata, hub), [])
            a2 = routes.get((hub, dest_iata), [])
            if not a1 or not a2 or hub not in airports:
                continue
            h = airports[hub]
            d1 = _km(orig["lat"], orig["lon"], h["lat"], h["lon"])
            d2 = _km(h["lat"], h["lon"], dest["lat"], dest["lon"])
            via_hrs = (d1 + d2) / 850 + 0.75 + 2.0  # 2h layover
            one_stop.append({
                "hub": hub, "hub_city": h["city"],
                "a1": a1[0], "a2": a2[0],
                "hours": via_hrs,
            })
        one_stop.sort(key=lambda x: x["hours"])

    if not direct_airlines and not one_stop:
        return None

    date_line = f"Departure date: {departure_date}" if departure_date else "Date: not specified — suggest typical schedules"

    lines = [
        "━" * 52,
        f"REAL ROUTE DATA (OpenFlights open dataset)",
        f"{origin_city.split(',')[0].strip()} ({orig_iata}) → {destination.split(',')[0].strip()} ({dest_iata})",
        f"Great-circle distance: ~{int(dist):,} km  |  Est. flight time: {_fmt_dur(direct_hours)}",
        date_line,
        "Incorporate these real airlines and route details verbatim into the ✈️ Getting There section.",
        "━" * 52,
    ]

    opt = 1
    for code in direct_airlines[:max_results]:
        name = airlines.get(code, code)
        lines.append(
            f"\nOption {opt} — Non-stop ✅\n"
            f"  Airline : {name} ({code})\n"
            f"  Route   : {orig_iata} → {dest_iata}\n"
            f"  Duration: ~{_fmt_dur(direct_hours)}\n"
            f"  Price   : USD {lo_p}–{hi_p} per person (economy, estimated)\n"
            f"  Tip     : Suggest realistic departure/arrival times for this route"
        )
        opt += 1

    needed = max(0, max_results - len(direct_airlines))
    for s in one_stop[:needed]:
        a1n = airlines.get(s["a1"], s["a1"])
        a2n = airlines.get(s["a2"], s["a2"])
        lines.append(
            f"\nOption {opt} — 1 Stop\n"
            f"  Airlines: {a1n} ({s['a1']}) + {a2n} ({s['a2']})\n"
            f"  Route   : {orig_iata} → {s['hub']} ({s['hub_city']}) → {dest_iata}\n"
            f"  Duration: ~{_fmt_dur(s['hours'])} incl. layover\n"
            f"  Price   : USD {max(50, lo_p - 80)}–{lo_p + 60} per person (economy, estimated)\n"
            f"  Tip     : Suggest a 1.5–2.5h layover at {s['hub_city']}"
        )
        opt += 1

    lines += [
        "",
        "→ Recommend Option 1 as the primary choice; mention others as budget/time alternatives.",
        "→ Include specific realistic departure and arrival times, and bag allowance notes.",
        "━" * 52,
    ]
    return "\n".join(lines)
