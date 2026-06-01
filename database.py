import os
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, inspect as sa_inspect, text
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


class Itinerary(Base):
    __tablename__ = "itineraries"

    id = Column(Integer, primary_key=True, index=True)
    trip_id = Column(Integer, ForeignKey("trips.id"), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)

    trip = relationship("Trip", back_populates="itinerary")


def create_tables():
    Base.metadata.create_all(bind=engine)


def migrate_db():
    """Add new columns to existing tables without recreating them. Works on SQLite and PostgreSQL."""
    try:
        inspector = sa_inspect(engine)
        if "trips" in inspector.get_table_names():
            existing = {c["name"] for c in inspector.get_columns("trips")}
            with engine.connect() as conn:
                if "budget_type" not in existing:
                    conn.execute(text("ALTER TABLE trips ADD COLUMN budget_type VARCHAR(20) DEFAULT 'overall'"))
                    conn.commit()
    except Exception as exc:
        print(f"Migration note: {exc}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
