"""Класс игрового состояния."""

from typing import Any


class GameState:
    """
    Хранит текущее состояние игрового мира.
    Атрибуты:
        flags: dict[str, Any] — флаги состояния
        history: list[dict] — история событий (вехи, действия игрока)
        current_beacon: str | None — текущая активная веха
        step: int — номер текущего шага
    """

    def __init__(self) -> None:
        self.flags: dict[str, Any] = {}
        self.history: list[dict] = []
        self.current_beacon: str | None = None
        self.step: int = 0

    def set_flag(self, name: str, value: Any = True) -> None:
        self.flags[name] = value

    def has_flag(self, name: str) -> bool:
        return name in self.flags

    def get_flag(self, name: str, default: Any = None) -> Any:
        return self.flags.get(name, default)

    def add_event(self, event: dict) -> None:
        self.history.append(event)

    def get_history(self, n: int | None = None) -> list[dict]:
        if n is None:
            return self.history
        return self.history[-n:]

    def advance_step(self) -> None:
        self.step += 1

    def to_dict(self) -> dict:
        return {
            "flags": self.flags,
            "current_beacon": self.current_beacon,
            "step": self.step,
            "history": self.history,
        }
