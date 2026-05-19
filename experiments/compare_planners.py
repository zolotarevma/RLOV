"""Сравнение планировщиков (heuristic vs DQN) на заданном сценарии."""

import sys
import json
from pathlib import Path
from collections import Counter
from game_state import GameState
from game_state.scenario import load_scenario
from llm_generator.generator import generate_scene
from metrics.structural import full_structure_report
from metrics.diversity import pairwise_diversity
from metrics.semantic import prompt_scene_similarity
import logging
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)


# ──────────────────────── вспомогательные функции ────────────────────────

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
            if cid not in path:  # чтобы не зацикливаться
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
        expected_flag = chosen.get("expected_player_flag")

        if player_options and chosen["type"] != "ending":
            selected_opt = None
            if expected_flag is not None:
                for opt in player_options:
                    eff = opt.get("effect", "") if isinstance(opt, dict) else ""
                    if eff == expected_flag:
                        selected_opt = opt
                        break
            if selected_opt is None:
                selected_opt = player_options[0]

            effect = selected_opt.get("effect", "") if isinstance(selected_opt, dict) else ""
            if effect and effect != "none":
                state.set_flag(effect, True)
                last_player_flag = effect
            else:
                last_player_flag = None

            if expected_flag is not None:
                consistency_total += 1
                actual_effect = selected_opt.get("effect", "") if isinstance(selected_opt, dict) else ""
                if actual_effect == expected_flag:
                    consistency_hits += 1

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


# ──────────────────────── фабрика планировщиков ─────────────────────────

def make_planner(planner_type: str, beacons: list[dict]):
    if planner_type == "dqn":
        from rl_planner.dqn_planner import DQNPlanner
        project_root = Path(__file__).resolve().parent.parent
        model_path = str(project_root / "rl_planner" / "dqn_model.pt")
        flags_path = str(project_root / "rl_planner" / "flags_order.json")
        if not Path(model_path).exists():
            print("DQN модель не найдена, переключаюсь на Heuristic.")
            from planner.heuristic import HeuristicPlanner
            return HeuristicPlanner()
        with open(flags_path, "r") as f:
            flags_order = json.load(f)
        state_dim = len(flags_order)
        action_dim = len(beacons)
        planner = DQNPlanner(model_path, flags_path, state_dim, action_dim)
        planner.set_mapping(beacons)
        planner.q_net.eval()
        return planner
    else:
        from planner.heuristic import HeuristicPlanner
        return HeuristicPlanner()


# ──────────────────────── запуск сравнения ───────────────────────────────

def main():
    scenario_path = "scenarios/mayor_support.json"
    scenario = load_scenario(scenario_path)
    beacons = scenario["beacons"]

    planner_types = ["heuristic", "dqn"]
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    results = {}
    for ptype in planner_types:
        planner = make_planner(ptype, beacons)
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

        struct = full_structure_report(beacons, all_paths)
        diversity = pairwise_diversity(all_intros) if len(all_intros) > 1 else 0.0
        sims = [prompt_scene_similarity({"description": desc}, {"intro": intro})
                for desc, intro in all_sim_pairs]
        avg_sim = sum(sims) / len(sims) if sims else 0.0

        endings = set(b["id"] for b in beacons if b["type"] == "ending")
        reached = sum(1 for p in all_paths if p and p[-1] in endings)

        num_scenes = len(all_intros)
        avg_rejections = total_rej / num_scenes if num_scenes else 0.0
        fallback_rate = total_fallback / num_scenes if num_scenes else 0.0
        avg_gen_time = total_gen_time / num_scenes if num_scenes else 0.0
        consistency = total_cons_hits / total_cons_total if total_cons_total else 0.0

        error_counts = Counter(all_error_types)
        most_common_errors = error_counts.most_common(3)

        results[ptype] = {
            "avg_steps": struct["avg_steps"],
            "avg_unique": struct["avg_unique_beacons"],
            "dead_ends": struct["avg_dead_ends"],
            "unreachable": struct["avg_unreachable"],
            "reached": reached,
            "diversity": diversity,
            "avg_similarity": avg_sim,
            "avg_rejections": avg_rejections,
            "fallback_rate": fallback_rate,
            "avg_gen_time": avg_gen_time,
            "consistency": consistency,
            "top_errors": most_common_errors,
        }

    print("\n===== Сравнение планировщиков =====\n")
    header = f"{'Метрика':<30} {'Heuristic':>12} {'DQN':>12}"
    print(header)
    print("-" * len(header))
    for metric in ["avg_steps", "avg_unique", "dead_ends", "unreachable",
                   "reached", "diversity", "avg_similarity", "avg_rejections", "consistency"]:
        h_val = results["heuristic"][metric]
        d_val = results["dqn"][metric]
        if isinstance(h_val, float):
            print(f"{metric:<30} {h_val:>12.3f} {d_val:>12.3f}")
        else:
            print(f"{metric:<30} {h_val:>12} {d_val:>12}")
    print(
        f"{'avg_gen_time (s)':<30} {results['heuristic']['avg_gen_time']:>12.3f} {results['dqn']['avg_gen_time']:>12.3f}")
    print(
        f"{'fallback_rate':<30} {results['heuristic']['fallback_rate']:>12.3f} {results['dqn']['fallback_rate']:>12.3f}")
    print(f"\nТоп ошибок валидатора (heuristic): {results['heuristic']['top_errors']}")
    print(f"Топ ошибок валидатора (DQN): {results['dqn']['top_errors']}")


if __name__ == "__main__":
    main()
