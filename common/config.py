import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class CityConfig:
    name: str = "Lahore"
    country: str = "Pakistan"
    latitude: float = 31.5204
    longitude: float = 74.3587
    timezone: str = "Asia/Karachi"


CITY = CityConfig()

# The six metrics the brief requires forecasts for.
METRICS = ["aqi", "temperature", "precipitation", "humidity", "uv_index", "wind"]

FORECAST_HORIZON_HOURS = 24


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Set it in your local .env file (see .env.example) or as a GitHub Actions secret."
        )
    return value


def optional_env(name: str) -> str | None:
    return os.environ.get(name)
