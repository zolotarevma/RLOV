"""Загрузка сценария из JSON."""

import json
from pathlib import Path


def load_scenario(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
