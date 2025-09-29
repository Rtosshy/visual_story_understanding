import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.utils.paths import DOTENV_PATH

if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)


def env(key: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(key, default)
    if required and value is None:
        raise RuntimeError(f"Required environment variable '{key}' not set.")
    return value


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)
