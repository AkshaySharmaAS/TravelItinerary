import json
import os
from datetime import datetime
from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import Customer, Itinerary, SessionLocal, Trip, create_tables, get_db, migrate_db

load_dotenv()
create_tables()
migrate_db()

app = FastAPI(title="Travel Agent CRM")

SYSTEM_PROMPT = """You are an expert travel agent with decades of experience planning international trips. \
You create detailed, practical, inspiring, and beautifully structured travel itineraries.

Use the following structure exactly:

# [Destination] — [N]-Day [Theme] Adventure

## ✈️ Getting There
How to travel from departure to destination (flights, trains, etc.)

## 🌤️ Weather & Climate
Seasonal weather analysis for the destination during the travel period. Be specific about temperatures, rainfall, what to pack.

## 🏨 Where to Stay
3 accommodation options at different price points with approximate nightly rates.

## Day 1: [Evocative Day Title]
**🌤️ Weather:** [emoji] [Condition], [temp range] — [one-line description]

### 🌅 Morning
- Activities, attractions with practical details

### ☀️ Afternoon
- Activities, attractions with practical details

### 🌙 Evening
- Dinner recommendation with cuisine type and price range (💰 budget / 💰💰 mid / 💰💰💰 upscale)
- Evening activities or nightlife

[Repeat Day structure for all days]

## 🍽️ Must-Try Restaurants
Top 5–6 restaurants across the trip with cuisine, signature dish, and price range.

## 💡 Essential Travel Tips
Local customs, safety, currency, language, transport apps, SIM cards, etc.

## 💰 Estimated Budget Breakdown
Table or list of major costs: flights, accommodation per night, meals/day, activities, transport.

Guidelines:
- If REAL FLIGHT DATA is provided above the day breakdown, use those exact flight numbers, times, carriers, and prices in the ✈️ Getting There section. Do NOT invent alternative flights when real data is given.
- Weather emoji: ☀️ sunny · 🌤️ partly cloudy · ⛅ mixed · 🌥️ overcast · 🌦️ light rain · 🌧️ rainy · 🌩️ storms · ❄️ cold/snow · 🌫️ fog · 🌬️ windy
- Include hidden gems alongside famous attractions
- Realistic travel times between locations
- Bold key names, prices, and important tips"""


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class CustomerCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    country: Optional[str] = None
    notes: Optional[str] = None


class CustomerUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    country: Optional[str] = None
    notes: Optional[str] = None


class TripCreate(BaseModel):
    customer_id: int
    destination: str
    departure_city: str
    departure_country: str
    num_days: int
    start_date: Optional[str] = None
    status: Optional[str] = "planning"
    budget: Optional[str] = None
    budget_type: Optional[str] = "overall"
    notes: Optional[str] = None


class TripUpdate(BaseModel):
    destination: Optional[str] = None
    departure_city: Optional[str] = None
    departure_country: Optional[str] = None
    num_days: Optional[int] = None
    start_date: Optional[str] = None
    status: Optional[str] = None
    budget: Optional[str] = None
    budget_type: Optional[str] = None
    notes: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _customer_dict(c: Customer) -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "email": c.email,
        "phone": c.phone,
        "country": c.country,
        "notes": c.notes,
        "trip_count": len(c.trips),
        "created_at": c.created_at.isoformat(),
    }


def _trip_dict(t: Trip, include_customer: bool = True) -> dict:
    d = {
        "id": t.id,
        "customer_id": t.customer_id,
        "destination": t.destination,
        "departure_city": t.departure_city,
        "departure_country": t.departure_country,
        "num_days": t.num_days,
        "start_date": t.start_date,
        "status": t.status,
        "budget": t.budget,
        "budget_type": t.budget_type or "overall",
        "notes": t.notes,
        "has_itinerary": t.itinerary is not None,
        "created_at": t.created_at.isoformat(),
    }
    if include_customer and t.customer:
        d["customer_name"] = t.customer.name
        d["customer_email"] = t.customer.email
    return d


# ── SSE streaming generator ───────────────────────────────────────────────────

def _stream_itinerary(trip_data: dict, api_key: str):
    """Synchronous generator that streams an itinerary from Claude via SSE."""
    from flights import search_flights

    client = anthropic.Anthropic(api_key=api_key)
    full_content = ""

    # ── Step 1: fetch real flight data ───────────────────────────────────────
    flight_block = ""
    if trip_data.get("start_date"):
        yield f"data: {json.dumps({'type': 'status', 'message': '✈️ Searching for real flights...'})}\n\n"
        try:
            flight_block = search_flights(
                origin_city=trip_data["departure_city"],
                destination=trip_data["destination"],
                departure_date=trip_data["start_date"],
            ) or ""
        except Exception:
            flight_block = ""
        if flight_block:
            yield f"data: {json.dumps({'type': 'status', 'message': '✈️ Live flight data retrieved! Building itinerary...'})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Connecting to Claude AI...'})}\n\n"
    else:
        yield f"data: {json.dumps({'type': 'status', 'message': 'Connecting to Claude AI...'})}\n\n"

    # ── Step 2: build prompt ──────────────────────────────────────────────────
    start_date_line = (
        f"**Travel Dates:** Starting {trip_data['start_date']}\n"
        if trip_data.get("start_date") else ""
    )
    budget_type_label = "per person" if trip_data.get("budget_type") == "per_person" else "total overall"
    budget_line = (
        f"**Budget:** {trip_data['budget']} ({budget_type_label})\n"
        if trip_data.get("budget") else ""
    )
    notes_line = (
        f"**Special Requests:** {trip_data['notes']}\n"
        if trip_data.get("notes") else ""
    )
    flight_section = f"\n\n{flight_block}\n\n" if flight_block else ""

    no_date_note = (
        "\nNote: No start date provided — use typical seasonal weather patterns.\n"
        if not trip_data.get("start_date") else ""
    )

    user_prompt = (
        f"Please create a detailed {trip_data['num_days']}-day travel itinerary for:\n\n"
        f"**Departure:** {trip_data['departure_city']}, {trip_data['departure_country']}\n"
        f"**Destination:** {trip_data['destination']}\n"
        f"**Duration:** {trip_data['num_days']} days\n"
        f"{start_date_line}"
        f"{budget_line}"
        f"{notes_line}"
        f"{no_date_note}"
        f"{flight_section}"
        f"Follow the exact structure from your instructions. Cover all {trip_data['num_days']} days with "
        f"Morning / Afternoon / Evening sections. "
        f"{'Use the real flight data above exactly as provided in the Getting There section.' if flight_block else 'Include realistic flight options in the Getting There section.'} "
        f"Include weather analysis for {trip_data['destination']} "
        f"{'in ' + trip_data['start_date'] if trip_data.get('start_date') else 'based on typical seasonal patterns'}. "
        "Make it inspiring, detailed, and completely practical!"
    )

    try:
        with client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "thinking":
                        yield f"data: {json.dumps({'type': 'thinking', 'message': 'Planning your perfect itinerary...'})}\n\n"
                    elif event.content_block.type == "text":
                        yield f"data: {json.dumps({'type': 'text_start'})}\n\n"

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        text = event.delta.text
                        full_content += text
                        yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

        # Persist the itinerary
        db = SessionLocal()
        try:
            existing = (
                db.query(Itinerary)
                .filter(Itinerary.trip_id == trip_data["id"])
                .first()
            )
            if existing:
                existing.content = full_content
                existing.generated_at = datetime.utcnow()
            else:
                db.add(Itinerary(trip_id=trip_data["id"], content=full_content))
            db.commit()
            yield f"data: {json.dumps({'type': 'done', 'message': 'Itinerary generated and saved!'})}\n\n"
        except Exception as db_err:
            yield (
                f"data: {json.dumps({'type': 'done', 'message': f'Generated but could not save: {db_err}'})}\n\n"
            )
        finally:
            db.close()

    except anthropic.AuthenticationError:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid API key. Check your ANTHROPIC_API_KEY.'})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"


# ── Customer routes ───────────────────────────────────────────────────────────

@app.get("/api/customers")
def list_customers(db: Session = Depends(get_db)):
    customers = db.query(Customer).order_by(Customer.created_at.desc()).all()
    return [_customer_dict(c) for c in customers]


@app.post("/api/customers", status_code=201)
def create_customer(payload: CustomerCreate, db: Session = Depends(get_db)):
    if db.query(Customer).filter(Customer.email == payload.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    customer = Customer(**payload.model_dump())
    db.add(customer)
    db.commit()
    db.refresh(customer)
    return _customer_dict(customer)


@app.get("/api/customers/{customer_id}")
def get_customer(customer_id: int, db: Session = Depends(get_db)):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    data = _customer_dict(customer)
    data["trips"] = [_trip_dict(t, include_customer=False) for t in customer.trips]
    return data


@app.put("/api/customers/{customer_id}")
def update_customer(
    customer_id: int, payload: CustomerUpdate, db: Session = Depends(get_db)
):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(customer, key, value)
    customer.updated_at = datetime.utcnow()
    db.commit()
    return _customer_dict(customer)


@app.delete("/api/customers/{customer_id}", status_code=204)
def delete_customer(customer_id: int, db: Session = Depends(get_db)):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    db.delete(customer)
    db.commit()


# ── Trip routes ───────────────────────────────────────────────────────────────

@app.get("/api/trips")
def list_trips(db: Session = Depends(get_db)):
    trips = db.query(Trip).order_by(Trip.created_at.desc()).all()
    return [_trip_dict(t) for t in trips]


@app.post("/api/trips", status_code=201)
def create_trip(payload: TripCreate, db: Session = Depends(get_db)):
    if not db.query(Customer).filter(Customer.id == payload.customer_id).first():
        raise HTTPException(status_code=404, detail="Customer not found")
    trip = Trip(**payload.model_dump())
    db.add(trip)
    db.commit()
    db.refresh(trip)
    return _trip_dict(trip)


@app.get("/api/trips/{trip_id}")
def get_trip(trip_id: int, db: Session = Depends(get_db)):
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    data = _trip_dict(trip)
    data["itinerary"] = trip.itinerary.content if trip.itinerary else None
    return data


@app.put("/api/trips/{trip_id}")
def update_trip(trip_id: int, payload: TripUpdate, db: Session = Depends(get_db)):
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(trip, key, value)
    trip.updated_at = datetime.utcnow()
    db.commit()
    return _trip_dict(trip)


@app.delete("/api/trips/{trip_id}", status_code=204)
def delete_trip(trip_id: int, db: Session = Depends(get_db)):
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    db.delete(trip)
    db.commit()


# ── Itinerary generation ──────────────────────────────────────────────────────

@app.get("/api/trips/{trip_id}/generate-itinerary")
def generate_itinerary(trip_id: int, db: Session = Depends(get_db)):
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="ANTHROPIC_API_KEY environment variable not set"
        )

    trip_data = {
        "id": trip.id,
        "destination": trip.destination,
        "departure_city": trip.departure_city,
        "departure_country": trip.departure_country,
        "num_days": trip.num_days,
        "start_date": trip.start_date,
        "budget": trip.budget,
        "budget_type": trip.budget_type or "overall",
        "notes": trip.notes,
    }

    return StreamingResponse(
        _stream_itinerary(trip_data, api_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/api/dashboard/stats")
def dashboard_stats(db: Session = Depends(get_db)):
    statuses = ["planning", "confirmed", "completed", "cancelled"]
    status_counts = {
        s: db.query(Trip).filter(Trip.status == s).count() for s in statuses
    }
    recent_trips = db.query(Trip).order_by(Trip.created_at.desc()).limit(6).all()
    return {
        "total_customers": db.query(Customer).count(),
        "total_trips": db.query(Trip).count(),
        "total_itineraries": db.query(Itinerary).count(),
        "status_counts": status_counts,
        "recent_trips": [_trip_dict(t) for t in recent_trips],
    }


# ── Static files & SPA fallback ───────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
