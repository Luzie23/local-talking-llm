
# EMSAG: 03.11.2026
# Load the project-root environment configuration and initialize SQLAlchemy.

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        f"DATABASE_URL is not configured. Check the environment file: {ENV_FILE}"
    )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def test_connection() -> bool:
    """Check whether the PostgreSQL database is reachable."""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return True
