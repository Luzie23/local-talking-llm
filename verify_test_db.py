# EMSAG: 03.11.2026
# Verify test database isolation and confirm that test transactions were rolled back.

from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parent
env_values = dotenv_values(PROJECT_ROOT / ".env")

database_url = env_values.get("DATABASE_URL")

if not database_url:
    raise RuntimeError("DATABASE_URL is missing.")

test_url = make_url(database_url).set(
    database="local_talking_llm_test"
)

if test_url.database != "local_talking_llm_test":
    raise RuntimeError("Unsafe database target.")

engine = create_engine(test_url)

try:
    with engine.connect() as connection:
        database_name = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

        participant_count = connection.execute(
            text(
                "SELECT count(*) FROM participants "
                "WHERE pseudonym LIKE 'TEST-%'"
            )
        ).scalar_one()

        print("Database:", database_name)
        print("Test participants:", participant_count)

        if database_name != "local_talking_llm_test":
            raise RuntimeError("Connected to the wrong database.")

        if participant_count != 0:
            print("WARNING: Test participants remain in the database.")
        else:
            print("Rollback verification: PASSED")

finally:
    engine.dispose()