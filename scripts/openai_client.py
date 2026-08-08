"""Shared OpenAI client and model configuration."""

from functools import lru_cache
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_EMBEDDING_DIMENSIONS = 768


def _setting(name: str, default: str) -> str:
    """Return a non-empty environment setting or its default value."""
    value = os.getenv(name, "").strip()
    return value or default


def _integer_setting(name: str, default: int) -> int:
    """Read a positive integer environment setting."""
    raw_value = _setting(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc

    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


OPENAI_TEXT_MODEL = _setting(
    "OPENAI_TEXT_MODEL",
    _setting("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
)
OPENAI_VISION_MODEL = _setting("OPENAI_VISION_MODEL", OPENAI_TEXT_MODEL)
OPENAI_EMBEDDING_MODEL = _setting(
    "OPENAI_EMBEDDING_MODEL",
    DEFAULT_OPENAI_EMBEDDING_MODEL,
)
OPENAI_EMBEDDING_DIMENSIONS = _integer_setting(
    "OPENAI_EMBEDDING_DIMENSIONS",
    DEFAULT_OPENAI_EMBEDDING_DIMENSIONS,
)
OPENAI_REASONING_EFFORT = _setting("OPENAI_REASONING_EFFORT", "none")


def is_reasoning_model(model: str) -> bool:
    """Return whether a model needs reasoning-specific sampling options."""
    return model.startswith(("gpt-5.6", "o1", "o3", "o4"))


def sampling_parameters(
    model: str,
    *,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
) -> dict[str, object]:
    """Build sampling parameters compatible with the selected model family."""
    parameters: dict[str, object] = {}
    if max_completion_tokens is not None:
        parameters["max_completion_tokens"] = max_completion_tokens

    if is_reasoning_model(model):
        parameters["reasoning_effort"] = OPENAI_REASONING_EFFORT
    elif temperature is not None:
        parameters["temperature"] = temperature

    return parameters


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Create and cache the OpenAI client after validating its API key."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to the environment before "
            "calling OpenAI."
        )

    return OpenAI(api_key=api_key)
