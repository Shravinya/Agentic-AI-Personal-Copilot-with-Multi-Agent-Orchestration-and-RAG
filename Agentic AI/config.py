"""Load settings from environment (.env). No secrets hardcoded in code."""
import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL: str = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
)


def require_api_key() -> str:
    if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in .env in the project root."
        )
    return GEMINI_API_KEY.strip()
