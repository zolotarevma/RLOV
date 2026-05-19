"""Основная логика генерации сцены с валидацией."""

import json
import time
from .client import call_llm
from .prompts import SYSTEM_PROMPT, build_prompt
from validator.scene_checker import validate_scene_json

MAX_RETRIES = 5
RETRY_DELAY = 2


def generate_scene(beacon: dict, state: dict) -> dict:
    raw = None
    prompt = build_prompt(beacon, state)
    rejections = 0
    error_types = []
    fallback = False
    t_start = time.time()

    for attempt in range(MAX_RETRIES):
        raw = call_llm(SYSTEM_PROMPT, prompt)
        try:
            scene = json.loads(raw)
        except json.JSONDecodeError:
            rejections += 1
            error_types.append("Invalid JSON")
            time.sleep(RETRY_DELAY)
            continue

        errors = validate_scene_json(scene, beacon)
        if not errors:
            scene["_rejections"] = rejections
            scene["_fallback"] = False
            scene["_error_types"] = error_types
            scene["_gen_time"] = time.time() - t_start
            return scene

        if errors:
            first_msg = errors[0].split(":")[0].strip()
            error_types.append(first_msg)
        print(f"[WARN] Scene validation failed (attempt {attempt + 1}/{MAX_RETRIES}): {errors}")
        rejections += 1
        time.sleep(RETRY_DELAY)

    fallback = True
    print("[WARN] All retries exhausted, returning fallback scene.")
    return {
        "intro": raw if 'raw' in locals() else "The story continues...",
        "dialogues": [],
        "player_options": [],
        "outcome": "The scene fades to black.",
        "_rejections": rejections,
        "_fallback": True,
        "_error_types": error_types,
        "_gen_time": time.time() - t_start
    }
