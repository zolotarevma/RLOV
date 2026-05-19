"""Пакетный запуск прохождения сценария с метриками."""

import sys
import json
import csv
import os
from pathlib import Path
from collections import Counter
from game_state import GameState
from game_state.scenario import load_scenario
from llm_generator.generator import generate_scene
from metrics.structural import full_structure_report
from metrics.diversity import pairwise_diversity
from metrics.semantic import prompt_scene_similarity
from config import LLM_CLIENT, OLLAMA_MODEL, GEMINI_MODEL, OPENROUTER_MODEL
import logging
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)


# ──────────────────────── вспомогательные функции ────────────────────────


def save_results_to_csv(results: dict, filename: str):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'llm', 'planner', 'runs',
                'avg_steps', 'avg_unique', 'dead_ends', 'unreachable', 'reached',
                'diversity', 'avg_similarity', 'avg_rejections', 'fallback_rate',
                'avg_gen_time', 'consistency', 'top_errors'
            ])
        writer.writerow([
            results['llm'],
            results['planner'],
            results['runs'],
            results['avg_steps'],
            results['avg_unique'],
            results['dead_ends'],
            results['unreachable'],
            results['reached'],
            results['diversity'],
            results['avg_similarity'],
            results['avg_rejections'],
            results['fallback_rate'],
            results['avg_gen_time'],
            results['consistency'],
            results['top_errors']
        ])


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
                ((
                    (v is True and state.has_flag(k) and state.get_flag(k) is True) or
                    (v is False and (not state.has_flag(k) or state.get_flag(k) is False)) or
                    (v not in (True, False) and state.has_flag(k) and state.get_flag(k) == v)
                ))
                for k, v in cond.items()
            )
            for cond in or_conditions
        )
        if not or_satisfied:
            return False
    return True


def _get_available(beacons: list[dict], state: GameState, path: list[str],
                   last_player_flag: str | None = None) -> list[dict]:
    if not path:
        return [b for b in beacons if b["type"] == "start"]

    current_id = path[-1]
    current_beacon = next(b for b in beacons if b["id"] == current_id)
    choices_ids = current_beacon.get("choices", [])
    available = []

    if current_beacon.get("narrative_effects") and last_player_flag:
        choices_ids = [
            cid for cid in choices_ids
            if next(b for b in beacons if b["id"] == cid).get("expected_player_flag") == last_player_flag
        ]

    for cid in choices_ids:
        b = next(bb for bb in beacons if bb["id"] == cid)
        if _check_preconditions(state, b.get("preconditions", {})):
            if cid not in path:
                available.append(b)
    return available


def _apply_effects(state: GameState, beacon: dict) -> None:
    for k, v in beacon.get("effects", {}).items():
        state.set_flag(k, v)
    state.add_event({"beacon_id": beacon["id"]})


# ──────────────────────────── основной прогон ────────────────────────────

def run_single(scenario: dict, planner, collect_scenes: bool = True) -> dict:
    beacons = scenario["beacons"]
    state = GameState()
    path = []
    scene_intros = []
    sim_pairs = []
    total_rejections = 0
    total_fallbacks = 0
    total_gen_time = 0.0
    all_error_types = []
    consistency_hits = 0
    consistency_total = 0

    start = next(b for b in beacons if b["type"] == "start")
    _apply_effects(state, start)
    path.append(start["id"])

    dead_ends_hit = 0
    unreachable_hit = False
    reached_ending = False
    last_player_flag = None

    for _ in range(20):
        available = _get_available(beacons, state, path, last_player_flag)
        if not available:
            break

        chosen_id = planner.select_beacon(state, available)
        chosen = next(b for b in beacons if b["id"] == chosen_id)

        scene = generate_scene(chosen, state.to_dict())
        intro = scene.get("intro", "")
        if collect_scenes:
            scene_intros.append(intro)
            sim_pairs.append((chosen.get("description", ""), intro))

        total_rejections += scene.get("_rejections", 0)
        if scene.get("_fallback", False):
            total_fallbacks += 1
        total_gen_time += scene.get("_gen_time", 0.0)
        all_error_types.extend(scene.get("_error_types", []))

        player_options = scene.get("player_options", [])
        if player_options and chosen["type"] != "ending":
            opt = player_options[0]
            effect = opt.get("effect", "") if isinstance(opt, dict) else ""

            expected_flag = chosen.get("expected_player_flag")
            if expected_flag is not None:
                consistency_total += 1
                if effect == expected_flag:
                    consistency_hits += 1

            if effect and effect != "none":
                state.set_flag(effect, True)
                last_player_flag = effect
            else:
                last_player_flag = None
        else:
            last_player_flag = None

        _apply_effects(state, chosen)
        path.append(chosen_id)

        from validator.graph_checker import check_path
        issues = check_path(beacons, path)
        dead = any(i["type"] == "dead_end" for i in issues)
        unreachable = any(i["type"] == "unreachable_ending" for i in issues)
        if dead:
            dead_ends_hit += 1
        if unreachable:
            unreachable_hit = True
        if dead or unreachable:
            break

        if chosen["type"] == "ending":
            reached_ending = True
            break

    return {
        "path": path,
        "dead_ends": dead_ends_hit,
        "unreachable_endings": 1 if unreachable_hit else 0,
        "reached_ending": reached_ending,
        "scene_intros": scene_intros,
        "sim_pairs": sim_pairs,
        "total_rejections": total_rejections,
        "total_fallbacks": total_fallbacks,
        "total_gen_time": total_gen_time,
        "all_error_types": all_error_types,
        "consistency_hits": consistency_hits,
        "consistency_total": consistency_total,
    }


# ──────────────────────────── фабрика планировщиков ───────────────────────

def make_planner(planner_type: str, beacons: list[dict]):
    if planner_type == "dqn":
        from rl_planner.dqn_planner import DQNPlanner
        project_root = Path(__file__).resolve().parent.parent
        model_path = str(project_root / "rl_planner" / "dqn_model.pt")
        flags_order_path = str(project_root / "rl_planner" / "flags_order.json")
        if not Path(model_path).exists():
            print("DQN модель не найдена, переключаюсь на Heuristic.")
            from planner.heuristic import HeuristicPlanner
            return HeuristicPlanner()
        with open(flags_order_path, "r") as f:
            flags_order = json.load(f)
        state_dim = len(flags_order)
        action_dim = len(beacons)
        planner = DQNPlanner(model_path, flags_order_path, state_dim, action_dim)
        planner.set_mapping(beacons)
        planner.q_net.eval()
        return planner
    else:
        from planner.heuristic import HeuristicPlanner
        return HeuristicPlanner()


# ──────────────────────────── запуск ─────────────────────────────────────

def main():
    scenario_path = "scenarios/mayor_support.json"
    scenario = load_scenario(scenario_path)
    beacons = scenario["beacons"]

    planner_type = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    planner = make_planner(planner_type, beacons)

    all_paths = []
    all_intros = []
    all_sim_pairs = []
    total_rej = 0
    total_fallback = 0
    total_gen_time = 0.0
    all_error_types = []
    total_cons_hits = 0
    total_cons_total = 0

    for _ in range(n_runs):
        res = run_single(scenario, planner, collect_scenes=True)
        all_paths.append(res["path"])
        all_intros.extend(res["scene_intros"])
        all_sim_pairs.extend(res["sim_pairs"])
        total_rej += res["total_rejections"]
        total_fallback += res["total_fallbacks"]
        total_gen_time += res["total_gen_time"]
        all_error_types.extend(res["all_error_types"])
        total_cons_hits += res["consistency_hits"]
        total_cons_total += res["consistency_total"]

    structure = full_structure_report(beacons, all_paths)
    print("\n===== Структурный отчёт =====")
    print(f"Прогонов: {structure['num_runs']}")
    print(f"Средняя длина пути: {structure['avg_steps']:.1f}")
    print(f"Среднее тупиков: {structure['avg_dead_ends']:.2f}")
    print(f"Среднее недостижимых концовок: {structure['avg_unreachable']:.2f}")
    print(f"Среднее уникальных вех: {structure['avg_unique_beacons']:.1f}")
    print(f"Статических тупиков: {structure['static_dead_ends']}")

    if len(all_intros) > 1:
        diversity = pairwise_diversity(all_intros)
        print(f"\nРазнообразие сцен (ROUGE-L): {diversity:.3f}")
    else:
        diversity = None
        print("\nНедостаточно данных для метрики разнообразия.")

    if all_sim_pairs:
        sims = [prompt_scene_similarity({"description": desc}, {"intro": intro}) for desc, intro in all_sim_pairs]
        avg_sim = sum(sims) / len(sims)
        print(f"Среднее семантическое сходство (beacon vs scene): {avg_sim:.3f}")
    else:
        avg_sim = None
        print("Нет данных для семантической метрики.")

    endings = set(b["id"] for b in beacons if b["type"] == "ending")
    reached = sum(1 for p in all_paths if p and p[-1] in endings)
    print(f"\nДостигнуто концовок: {reached}/{n_runs}")

    num_scenes = len(all_intros)
    avg_rejections = total_rej / num_scenes if num_scenes else 0.0
    fallback_rate = total_fallback / num_scenes if num_scenes else 0.0
    avg_gen_time = total_gen_time / num_scenes if num_scenes else 0.0
    consistency = total_cons_hits / total_cons_total if total_cons_total else 0.0

    error_counts = Counter(all_error_types)
    most_common_errors = error_counts.most_common(3)

    print(f"\n--- Метрики генерации ---")
    print(f"Среднее число повторных генераций на сцену: {avg_rejections:.3f}")
    print(f"Fallback rate: {fallback_rate:.3f}")
    print(f"Среднее время генерации сцены (сек): {avg_gen_time:.3f}")
    print(f"Consistency rate: {consistency:.3f}")
    print(f"Топ ошибок валидатора: {most_common_errors}")

    if LLM_CLIENT == 'ollama':
        llm_name = f"ollama:{OLLAMA_MODEL}"
    elif LLM_CLIENT == 'gemini':
        llm_name = f"gemini:{GEMINI_MODEL}"
    elif LLM_CLIENT == 'openrouter':
        llm_name = f"openrouter:{OPENROUTER_MODEL}"
    else:
        llm_name = LLM_CLIENT

    results = {
        'llm': llm_name,
        'planner': planner_type,
        'runs': n_runs,
        'avg_steps': structure['avg_steps'],
        'avg_unique': structure['avg_unique_beacons'],
        'dead_ends': structure['avg_dead_ends'],
        'unreachable': structure['avg_unreachable'],
        'reached': reached,
        'diversity': diversity,
        'avg_similarity': avg_sim,
        'avg_rejections': avg_rejections,
        'fallback_rate': fallback_rate,
        'avg_gen_time': avg_gen_time,
        'consistency': consistency,
        'top_errors': str(most_common_errors) if 'most_common_errors' in dir() else "[]"
    }
    save_results_to_csv(results, "experiments_results.csv")


if __name__ == "__main__":
    main()
