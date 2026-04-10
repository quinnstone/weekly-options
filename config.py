"""
Configuration module for the Weekly Options Trading Analysis System.

Loads environment variables from .env and exposes them through a Config class.
Defines directory paths and pipeline stage constants.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CANDIDATES_DIR = DATA_DIR / "candidates"
REPORTS_DIR = DATA_DIR / "reports"
PERFORMANCE_DIR = DATA_DIR / "performance"

# Ensure data directories exist
for _dir in (DATA_DIR, CANDIDATES_DIR, REPORTS_DIR, PERFORMANCE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Portfolio size — 3 high-conviction picks rather than 5 to reduce leverage
PORTFOLIO_SIZE = 3

# Pipeline stage ordering (Wed/Fri/Mon cadence)
PIPELINE_STAGES = [
    "wednesday_scan",
    "friday_refresh",
    "monday_picks",
]


class Config:
    """Centralized access to all configuration values and API keys."""

    # --- API Keys ---

    @property
    def finnhub_api_key(self) -> str:
        return os.getenv("FINNHUB_API_KEY", "")

    @property
    def fred_api_key(self) -> str:
        return os.getenv("FRED_API_KEY", "")

    @property
    def discord_webhook_url(self) -> str:
        return os.getenv("DISCORD_WEBHOOK_URL", "")

    @property
    def news_api_key(self) -> str:
        return os.getenv("NEWS_API_KEY", "")

    @property
    def tradier_api_key(self) -> str:
        return os.getenv("TRADIER_API_KEY", "")

    @property
    def anthropic_api_key(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def tradier_base_url(self) -> str:
        """Tradier API base URL. Sandbox by default; switch to production when ready."""
        return os.getenv("TRADIER_BASE_URL", "https://sandbox.tradier.com")

    # --- Directory Paths ---

    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    @property
    def candidates_dir(self) -> Path:
        return CANDIDATES_DIR

    @property
    def reports_dir(self) -> Path:
        return REPORTS_DIR

    @property
    def performance_dir(self) -> Path:
        return PERFORMANCE_DIR

    # --- Pipeline ---

    @property
    def pipeline_stages(self) -> list:
        return list(PIPELINE_STAGES)

    # --- Convenience helpers ---

    def has_finnhub(self) -> bool:
        return bool(self.finnhub_api_key and self.finnhub_api_key != "your_finnhub_key")

    def has_fred(self) -> bool:
        return bool(self.fred_api_key and self.fred_api_key != "your_fred_key")

    def has_discord(self) -> bool:
        return bool(
            self.discord_webhook_url
            and self.discord_webhook_url != "your_webhook_url"
        )

    def has_news_api(self) -> bool:
        return bool(self.news_api_key and self.news_api_key != "your_newsapi_key")

    def has_tradier(self) -> bool:
        return bool(self.tradier_api_key and self.tradier_api_key != "your_tradier_key")

    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key != "your_anthropic_key")


# Module-level singleton
config = Config()
