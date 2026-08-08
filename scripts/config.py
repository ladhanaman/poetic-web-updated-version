"""Runtime configuration for local and Streamlit deployments."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _as_text(value: Any) -> str:
    """Convert a configuration value to a trimmed string."""
    return str(value).strip() if value is not None else ""


def _streamlit_secret(name: str) -> str:
    """Read a secret from Streamlit's top-level or grouped secret settings."""
    try:
        import streamlit as st

        secrets = st.secrets
        direct_value = _as_text(secrets.get(name, ""))
        if direct_value:
            return direct_value

        prefix, suffix = name.split("_", maxsplit=1)
        section = secrets.get(prefix.lower(), {})
        if isinstance(section, Mapping):
            for key in (suffix.lower(), suffix, name, name.lower()):
                grouped_value = _as_text(section.get(key, ""))
                if grouped_value:
                    return grouped_value
    except Exception:
        # Streamlit is optional for CLI scripts, and secrets may be unavailable
        # outside a configured Streamlit runtime.
        return ""

    return ""


def get_setting(name: str, default: str = "") -> str:
    """Resolve a setting from the process environment, then Streamlit secrets."""
    environment_value = _as_text(os.getenv(name, ""))
    if environment_value:
        return environment_value

    return _streamlit_secret(name) or default
