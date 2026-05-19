"""Консольный интерфейс игры с планировщиком и валидатором."""

from game_state import GameState
from game_state.scenario import load_scenario
from validator.graph_checker import validate_scenario, check_path
from llm_generator.generator import generate_scene
from config import SCENARIOS_DIR, MAX_STEPS, PLANNER, DQN_MODEL_PATH, DQN_FLAGS_ORDER_PATH
import json


def _check_preconditions(state: GameState, pre: dict) -> bool:
    if not pre:
        return True
    regular = {k: v for k, v in pre.items() if k != "or"}
    for k, v in regular.items():
        if v is True:
            if not state.has_flag(k) or state.get_flag(k) is not True:
                return False
        elif v is False:
            if state.has_flag(k) and state.get_flag(k) is not False:
                return False
        else:
            if not state.has_flag(k) or state.get_flag(k) != v:
                return False
    if "or" in pre:
        or_conditions = pre["or"]
        if not isinstance(or_conditions, list):
            return False
        or_satisfied = any(
            all(
                (lambda fv: (
                    (fv is True and state.has_flag(k) and state.get_flag(k) is True) or
                    (fv is False and (not state.has_flag(k) or state.get_flag(k) is False)) or
                    (fv not in (True, False) and state.has_flag(k) and state.get_flag(k) == fv)
                ))(v)
                for k, v in cond.items()
            )
            for cond in or_conditions
        )
        if not or_satisfied:
            return False
    return True


def get_available_beacons(scenario, state, restrict_to_choices=None):
    beacons = scenario["beacons"]
    if restrict_to_choices is not None:
        available = []
        for bid in restrict_to_choices:
            b = next(bb for bb in beacons if bb["id"] == bid)
            if _check_preconditions(state, b.get("preconditions", {})):
                if b["id"] not in [h["beacon_id"] for h in state.history]:
                    available.append(b)
        return available

    if not state.history:
        current_id = None
    else:
        current_id = state.history[-1]["beacon_id"]
    if current_id is None:
        choices_ids = [b["id"] for b in beacons if b["type"] == "start"]
    else:
        current_beacon = next(b for b in beacons if b["id"] == current_id)
        choices_ids = current_beacon.get("choices", [])
    available = []
    for bid in choices_ids:
        b = next(bb for bb in beacons if bb["id"] == bid)
        if _check_preconditions(state, b.get("preconditions", {})):
            if b["id"] not in [h["beacon_id"] for h in state.history]:
                available.append(b)
    return available


def apply_effects(state: GameState, beacon: dict) -> None:
    for k, v in beacon.get("effects", {}).items():
        state.set_flag(k, v)
    state.current_beacon = beacon["id"]
    state.add_event({"beacon_id": beacon["id"], "name": beacon["name"], "step": state.step})


def main() -> None:
    scenario = load_scenario(f"{SCENARIOS_DIR}/mayor_support.json")
    print(f"=== {scenario['title']} ===\n")

    if PLANNER == "dqn":
        from rl_planner.dqn_planner import DQNPlanner
        with open(DQN_FLAGS_ORDER_PATH, "r") as f:
            flags_order = json.load(f)
        state_dim = len(flags_order)
        action_dim = len(scenario["beacons"])
        planner = DQNPlanner(DQN_MODEL_PATH, DQN_FLAGS_ORDER_PATH, state_dim, action_dim)
        planner.set_mapping(scenario["beacons"])
    else:
        from planner.heuristic import HeuristicPlanner
        planner = HeuristicPlanner()

    issues = validate_scenario(scenario["beacons"])
    if issues:
        print("Ошибки в структуре сценария:")
        for iss in issues:
            print(f" - [{iss['type']}] {iss['description']}")

    state = GameState()
    start_beacon = next(b for b in scenario["beacons"] if b["type"] == "start")
    apply_effects(state, start_beacon)
    print(f"{start_beacon['description']}")

    forced_next_id = None

    for step in range(MAX_STEPS):
        if forced_next_id is not None:
            current_beacon = next(b for b in scenario["beacons"] if b["id"] == forced_next_id)
            forced_next_id = None
        else:
            last_id = state.history[-1]["beacon_id"] if state.history else start_beacon["id"]
            current_beacon = next(b for b in scenario["beacons"] if b["id"] == last_id)

        if step > 0:
            print(f"\n{current_beacon['name']} ({current_beacon['type']})")
            apply_effects(state, current_beacon)

        scene = generate_scene(current_beacon, state.to_dict())
        rejections = scene.get("_rejections", 0)
        if rejections > 0:
            print(f"(Потребовалось повторных генераций: {rejections})")
        print(f"\n{scene['intro']}")
        if scene.get("dialogues"):
            for d in scene["dialogues"]:
                speaker = d.get('speaker') or d.get('character') or d.get('name') or 'Unknown'
                text = d.get('text') or d.get('line') or d.get('content') or ''
                print(f"  {speaker}: {text}")

        player_options = scene.get("player_options", [])
        if player_options and current_beacon["type"] != "ending" and current_beacon.get("narrative_effects"):
            valid_indices = []
            for i, opt in enumerate(player_options):
                effect = opt.get("effect", "") if isinstance(opt, dict) else ""
                if not effect or effect == "none":
                    continue
                target = None
                for cid in current_beacon.get("choices", []):
                    b = next((bb for bb in scenario["beacons"] if bb["id"] == cid), None)
                    if b and b.get("expected_player_flag") == effect:
                        target = b
                        break
                if target and _check_preconditions(state, target.get("preconditions", {})):
                    valid_indices.append(i)

            if not valid_indices:
                print("Нет доступных действий.")
                state.advance_step()
                continue

            print("\nPossible actions:")
            for display_idx, orig_idx in enumerate(valid_indices, 1):
                opt = player_options[orig_idx]
                text = opt.get("text", str(opt)) if isinstance(opt, dict) else str(opt)
                print(f"  {display_idx}. {text}")

            choice = input("Выбери действие (номер): ").strip()
            try:
                local_choice = int(choice) - 1
                if local_choice < 0 or local_choice >= len(valid_indices):
                    raise ValueError
                orig_idx = valid_indices[local_choice]
                selected = player_options[orig_idx]
                effect = selected.get("effect", "") if isinstance(selected, dict) else ""

                if effect and effect != "none":
                    state.set_flag(effect, True)
                    print(f"(Установлен флаг: {effect})")
                    choices = current_beacon.get("choices", [])
                    target = None
                    for cid in choices:
                        b = next((bb for bb in scenario["beacons"] if bb["id"] == cid), None)
                        if b and b.get("expected_player_flag") == effect:
                            target = b
                            break
                    if target:
                        forced_next_id = target["id"]
                    else:
                        if len(choices) == 1:
                            forced_next_id = choices[0]
                else:
                    print("(Выбор учтён, флаг не изменён)")
            except (IndexError, ValueError):
                print("Некорректный выбор, продолжение с первым доступным вариантом.")
                orig_idx = valid_indices[0]
                selected = player_options[orig_idx]
                effect = selected.get("effect", "") if isinstance(selected, dict) else ""
                if effect and effect != "none":
                    state.set_flag(effect, True)
                    choices = current_beacon.get("choices", [])
                    target = next((b for cid in choices
                                   for b in [next(bb for bb in scenario["beacons"] if bb["id"] == cid)]
                                   if b.get("expected_player_flag") == effect), None)
                    if target:
                        forced_next_id = target["id"]
                    else:
                        if len(choices) == 1:
                            forced_next_id = choices[0]
                else:
                    print("(Выбор учтён, флаг не изменён)")
            except (IndexError, ValueError):
                print("Некорректный выбор, продолжение с первым вариантом.")
                if player_options:
                    opt = player_options[0]
                    effect = opt.get("effect", "") if isinstance(opt, dict) else ""
                    if effect and effect != "none":
                        state.set_flag(effect, True)
                        if current_beacon.get("narrative_effects"):
                            choices = current_beacon.get("choices", [])
                            target = next((b for cid in choices
                                           for b in [next(bb for bb in scenario["beacons"] if bb["id"] == cid)]
                                           if b.get("expected_player_flag") == effect), None)
                            if target:
                                forced_next_id = target["id"]
        else:
            if current_beacon["type"] == "ending":
                outcome = scene.get("outcome", "")
                if outcome:
                    print(outcome)
                print(f"\n[КОНЦОВКА] {current_beacon['description']}")
                break

            available = get_available_beacons(scenario, state, restrict_to_choices=current_beacon.get("choices", []))
            if available:
                chosen_id = planner.select_beacon(state, available)
                forced_next_id = chosen_id
            else:
                print("Нет доступных вех. История завершена.")
                break

        path = [h["beacon_id"] for h in state.history]
        issues = check_path(scenario["beacons"], path)
        if issues:
            print("Проблемы на текущем пути:")
            for iss in issues:
                print(f" - [{iss['type']}] {iss['description']}")
        else:
            print("Путь корректен.")

        if current_beacon["type"] == "ending":
            break
        state.advance_step()

    print(f"\nИтоговое состояние: {state.to_dict()}")


if __name__ == "__main__":
    main()
