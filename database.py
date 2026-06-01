import os
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, create_engine, inspect as sa_inspect, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./travel_crm.db")

# Render (and some other hosts) provide postgres:// — SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_is_sqlite = DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(200), nullable=False)
    role = Column(String(20), nullable=False, default="travel_agent")  # admin | travel_agent
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True)
    phone = Column(String(30))
    specialty = Column(String(200))
    languages = Column(String(200))
    experience_years = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    reviews = relationship("ItineraryReview", back_populates="agent", cascade="all, delete-orphan")


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(30))
    country = Column(String(100))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    trips = relationship("Trip", back_populates="customer", cascade="all, delete-orphan")


class Trip(Base):
    __tablename__ = "trips"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    destination = Column(String(200), nullable=False)
    departure_city = Column(String(100), nullable=False)
    departure_country = Column(String(100), nullable=False)
    num_days = Column(Integer, nullable=False)
    start_date = Column(String(20))
    status = Column(String(50), default="planning")
    budget = Column(String(50))
    budget_type = Column(String(20), default="overall")
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    customer = relationship("Customer", back_populates="trips")
    itinerary = relationship(
        "Itinerary", back_populates="trip", uselist=False, cascade="all, delete-orphan"
    )
    review = relationship(
        "ItineraryReview", back_populates="trip", uselist=False, cascade="all, delete-orphan"
    )


class Itinerary(Base):
    __tablename__ = "itineraries"

    id = Column(Integer, primary_key=True, index=True)
    trip_id = Column(Integer, ForeignKey("trips.id"), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)

    trip = relationship("Trip", back_populates="itinerary")


class ItineraryReview(Base):
    __tablename__ = "itinerary_reviews"

    id = Column(Integer, primary_key=True, index=True)
    trip_id = Column(Integer, ForeignKey("trips.id"), nullable=False, unique=True)
    agent_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String(30), nullable=False)  # accepted | changes_suggested
    comment = Column(Text)
    reviewed_at = Column(DateTime, default=datetime.utcnow)

    trip = relationship("Trip", back_populates="review")
    agent = relationship("User", back_populates="reviews")


def create_tables():
    Base.metadata.create_all(bind=engine)


def migrate_db():
    """Add new columns to existing tables without recreating them. Works on SQLite and PostgreSQL."""
    try:
        inspector = sa_inspect(engine)
        tables = inspector.get_table_names()

        with engine.connect() as conn:
            if "trips" in tables:
                existing = {c["name"] for c in inspector.get_columns("trips")}
                if "budget_type" not in existing:
                    conn.execute(text("ALTER TABLE trips ADD COLUMN budget_type VARCHAR(20) DEFAULT 'overall'"))
                    conn.commit()

            # New tables are created by create_tables(); only need column migrations here
    except Exception as exc:
        print(f"Migration note: {exc}")


def seed_admin():
    """Ensure admin user exists on startup."""
    from auth import hash_password
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.username == "admin").first():
            admin = User(
                username="admin",
                password_hash=hash_password("admin123"),
                role="admin",
                name="Administrator",
                email="admin@travelcrm.local",
                is_active=True,
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
