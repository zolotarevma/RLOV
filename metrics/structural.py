"""Структурные метрики графа вех."""

from validator.graph_checker import check_path, validate_scenario


def evaluate_structure(beacons: list[dict], path: list[str]) -> dict:
    issues = check_path(beacons, path)
    dead_ends = sum(1 for i in issues if i["type"] == "dead_end")
    unreachable = sum(1 for i in issues if i["type"] == "unreachable_ending")
    return {
        "total_steps": len(path),
        "dead_ends": dead_ends,
        "unreachable_endings": unreachable,
    }


def compare_structure(
    beacons: list[dict],
    paths_with_validator: list[list[str]],
    paths_without_validator: list[list[str]],
) -> dict:
    def avg_metrics(paths):
        if not paths:
            return {"avg_dead_ends": 0, "avg_unreachable": 0, "avg_steps": 0}
        total_de, total_ur, total_steps = 0, 0, 0
        for p in paths:
            m = evaluate_structure(beacons, p)
            total_de += m["dead_ends"]
            total_ur += m["unreachable_endings"]
            total_steps += m["total_steps"]
        n = len(paths)
        return {
            "avg_dead_ends": total_de / n,
            "avg_unreachable": total_ur / n,
            "avg_steps": total_steps / n,
        }
    return {
        "with_validator": avg_metrics(paths_with_validator),
        "without_validator": avg_metrics(paths_without_validator),
    }


def full_structure_report(beacons: list[dict], paths: list[list[str]]) -> dict:
    n = len(paths)
    total_steps = 0
    total_de = 0
    total_ur = 0
    unique_all = 0
    for p in paths:
        m = evaluate_structure(beacons, p)
        total_de += m["dead_ends"]
        total_ur += m["unreachable_endings"]
        total_steps += m["total_steps"]
        unique_all += len(set(p))

    issues = validate_scenario(beacons)
    dead_end_count = sum(1 for i in issues if i["type"] == "dead_end")

    return {
        "num_runs": n,
        "avg_steps": total_steps / n if n else 0,
        "avg_dead_ends": total_de / n if n else 0,
        "avg_unreachable": total_ur / n if n else 0,
        "avg_unique_beacons": unique_all / n if n else 0,
        "static_dead_ends": dead_end_count,
    }
