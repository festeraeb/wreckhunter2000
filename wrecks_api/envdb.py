import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager

_HERE = Path(__file__).parent
ENV_DB_PATH = os.environ.get("ENV_DB_PATH", str(_HERE.parent / "db" / "env_data.db"))


@contextmanager
def get_env_db():
    """Context manager for the environment sqlite DB.

    This is a lightweight stub. The DB may not exist yet; callers should
    handle missing DBs gracefully.
    """
    p = Path(ENV_DB_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Environment DB not found: {ENV_DB_PATH}")
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def db_exists() -> bool:
    return Path(ENV_DB_PATH).exists()
