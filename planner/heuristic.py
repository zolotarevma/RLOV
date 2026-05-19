"""Эвристический планировщик."""

import random
from .base import BasePlanner


class HeuristicPlanner(BasePlanner):
    """
    Выбирает следующую веху на основе эвристики.
    Предпочитает вехи типа 'ending' (завершающие сюжет),
    иначе выбирает случайную доступную с равной вероятностью.
    """

    def select_beacon(self, state: "GameState", available_beacons: list[dict]) -> str:
        endings = [b for b in available_beacons if b.get("type") == "ending"]
        if endings:
            return random.choice(endings)["id"]
        return random.choice(available_beacons)["id"]
