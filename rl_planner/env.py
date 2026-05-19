"""Среда Gymnasium для обучения RL-планировщика."""
import random
import gymnasium
from gymnasium import spaces
import numpy as np
from game_state import GameState
from game_state.scenario import load_scenario
from llm_generator.generator import generate_scene


class StoryEnv(gymnasium.Env):
    def __init__(self, scenario_path: str, max_steps: int = 20, training: bool = False):
        super().__init__()
        self.scenario = load_scenario(scenario_path)
        self.beacons = self.scenario["beacons"]
        self.max_steps = max_steps
        self.state = None
        self.path = []
        self.done = False
        self.last_player_flag = None
        self.training = training

        self.action_space = spaces.Discrete(len(self.beacons))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self._get_all_flags()),), dtype=np.float32)

        self.beacon_to_idx = {b["id"]: i for i, b in enumerate(self.beacons)}
        self.idx_to_beacon = {i: b["id"] for i, b in enumerate(self.beacons)}

    def _get_all_flags(self):
        flags = set()
        for b in self.beacons:
            flags.update(b.get("preconditions", {}).keys())
            flags.update(b.get("effects", {}).keys())
        return sorted(flags)

    def _state_to_vector(self):
        vec = []
        for k in self._get_all_flags():
            vec.append(1.0 if self.state.has_flag(k) else 0.0)
        return np.array(vec, dtype=np.float32)

    def _check_preconditions(self, pre: dict) -> bool:
        if not pre:
            return True
        regular = {k: v for k, v in pre.items() if k != "or"}
        for k, v in regular.items():
            if v is True:
                if not self.state.has_flag(k) or self.state.get_flag(k) is not True:
                    return False
            elif v is False:
                if self.state.has_flag(k) and self.state.get_flag(k) is not False:
                    return False
            else:
                if not self.state.has_flag(k) or self.state.get_flag(k) != v:
                    return False
        if "or" in pre:
            or_conditions = pre["or"]
            if not isinstance(or_conditions, list):
                return False
            or_satisfied = any(
                all(
                    (lambda fv: (
                        (fv is True and self.state.has_flag(k) and self.state.get_flag(k) is True) or
                        (fv is False and (not self.state.has_flag(k) or self.state.get_flag(k) is False)) or
                        (fv not in (True, False) and self.state.has_flag(k) and self.state.get_flag(k) == fv)
                    ))(v)
                    for k, v in cond.items()
                )
                for cond in or_conditions
            )
            if not or_satisfied:
                return False
        return True

    def _get_available(self):
        if not self.path:
            return [b for b in self.beacons if b["type"] == "start"]

        current_id = self.path[-1]
        current_beacon = next(b for b in self.beacons if b["id"] == current_id)
        choices_ids = current_beacon.get("choices", [])
        available = []

        if current_beacon.get("narrative_effects") and self.last_player_flag:
            choices_ids = [
                cid for cid in choices_ids
                if next(b for b in self.beacons if b["id"] == cid).get("expected_player_flag") == self.last_player_flag
            ]

        for bid in choices_ids:
            b = next(bb for bb in self.beacons if bb["id"] == bid)
            if self._check_preconditions(b.get("preconditions", {})):
                if bid not in self.path:
                    available.append(b)
        return available

    def reset(self):
        self.state = GameState()
        self.path = []
        self.done = False
        self.last_player_flag = None
        start = next(b for b in self.beacons if b["type"] == "start")
        self._apply_beacon(start)
        self.path.append(start["id"])
        return self._state_to_vector()

    def _apply_beacon(self, beacon):
        for k, v in beacon.get("effects", {}).items():
            self.state.set_flag(k, v)
        self.state.add_event({"beacon_id": beacon["id"]})

    def step(self, action_idx):
        if self.done:
            raise RuntimeError("Episode already done")

        chosen_id = self.idx_to_beacon[action_idx]
        available = self._get_available()
        available_ids = [b["id"] for b in available]

        if chosen_id not in available_ids:
            reward = -10.0
            self.done = True
            return self._state_to_vector(), reward, self.done, {"error": "invalid_action"}

        chosen = next(b for b in self.beacons if b["id"] == chosen_id)

        expected = chosen.get("expected_player_flag")
        if self.last_player_flag is not None and expected is not None:
            if self.last_player_flag == expected:
                consistency_reward = 5.0
            else:
                consistency_reward = -1.0
        else:
            consistency_reward = 0.0

        scene = generate_scene(chosen, self.state.to_dict())
        player_options = scene.get("player_options", [])
        new_player_flag = None
        if player_options and chosen["type"] != "ending":
            if self.training:
                opt = player_options[0]
            else:
                opt = random.choice(player_options)
            effect = opt.get("effect", "") if isinstance(opt, dict) else ""
            if effect and effect != "none":
                self.state.set_flag(effect, True)
                new_player_flag = effect
            else:
                new_player_flag = None

        self._apply_beacon(chosen)
        self.path.append(chosen_id)

        if chosen["type"] == "ending":
            self.done = True
        elif len(self.path) >= self.max_steps:
            self.done = True

        if self.done:
            if chosen["type"] == "ending":
                unique_beacons = len(set(self.path))
                pre = chosen.get("preconditions", {})
                if pre:
                    total_conditions = 0
                    satisfied_conditions = 0

                    regular = {k: v for k, v in pre.items() if k != "or"}
                    for k, v in regular.items():
                        total_conditions += 1
                        if self.state.has_flag(k) and self.state.get_flag(k) == v:
                            satisfied_conditions += 1

                    if "or" in pre:
                        or_conditions = pre["or"]
                        total_conditions += 1
                        if any(
                                all(self.state.has_flag(k) and self.state.get_flag(k) == v
                                    for k, v in cond.items())
                                for cond in or_conditions
                        ):
                            satisfied_conditions += 1

                    match_ratio = satisfied_conditions / total_conditions
                else:
                    match_ratio = 1.0
                    satisfied_conditions = 0

                reward = float(unique_beacons) / 10 + match_ratio * 2.0 + 0.1 * satisfied_conditions
            else:
                reward = -10.0
        else:
            reward = consistency_reward

        self.last_player_flag = new_player_flag
        return self._state_to_vector(), reward, self.done, {}
