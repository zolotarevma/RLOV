"""Формальный анализ графа вех."""


def validate_scenario(beacons: list[dict]) -> list[dict]:
    issues = []
    beacon_map = {b["id"]: b for b in beacons}

    for bid, b in beacon_map.items():
        if b.get("type") != "ending" and len(b.get("choices", [])) == 0:
            issues.append({
                "type": "dead_end",
                "description": f"Тупик: веха '{b['name']}' не концовка и без выходов.",
                "details": {"beacon_id": bid},
            })
    return issues


def check_path(beacons: list[dict], current_path: list[str]) -> list[dict]:
    issues = []
    beacon_map = {b["id"]: b for b in beacons}
    current_id = current_path[-1] if current_path else None
    if current_id is None:
        return issues

    cur = beacon_map.get(current_id)
    if cur and cur.get("type") != "ending" and len(cur.get("choices", [])) == 0:
        issues.append({
            "type": "dead_end",
            "description": f"Тупик: веха '{cur['name']}' не концовка и без выходов.",
            "details": {"beacon_id": current_id},
        })

    endings = [bid for bid, b in beacon_map.items() if b.get("type") == "ending"]
    if endings and not any(_is_reachable(beacon_map, current_id, end) for end in endings):
        issues.append({
            "type": "unreachable_ending",
            "description": "Из текущей позиции недостижима ни одна концовка.",
            "details": {"current_id": current_id},
        })
    return issues


def _is_reachable(beacon_map: dict, start_id: str, target_id: str) -> bool:
    visited = set()
    queue = [start_id]
    while queue:
        cur = queue.pop(0)
        if cur == target_id:
            return True
        visited.add(cur)
        for nxt in beacon_map.get(cur, {}).get("choices", []):
            if nxt not in visited:
                queue.append(nxt)
    return False
