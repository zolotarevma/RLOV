"""Абстрактный класс планировщика."""

from abc import ABC, abstractmethod


class BasePlanner(ABC):
    """
    Метод select_beacon принимает состояние игры и список доступных вех,
    возвращает ID выбранной вехи.
    """

    @abstractmethod
    def select_beacon(self, state: "GameState", available_beacons: list[dict]) -> str:
        ...
