"""Основная логика генерации сцены с валидацией."""

import json
import time
from config import LANGUAGE
from .client import call_llm
from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_RU, build_prompt
from validator.scene_checker import validate_scene_json
import re

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
        system_prompt = SYSTEM_PROMPT_RU if LANGUAGE == "ru" else SYSTEM_PROMPT
        raw = call_llm(system_prompt, prompt)

        cleaned = raw.strip()
        cleaned = re.sub(r'//.*', '', cleaned)
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx:end_idx+1]
        try:
            scene = json.loads(cleaned)
        except json.JSONDecodeError:
            rejections += 1
            error_types.append("Invalid JSON")
            print(f"[WARN] Invalid JSON")
            time.sleep(RETRY_DELAY)
            continue

        # Исправление лишних опций, когда narrative_effects пуст
        if not beacon.get("narrative_effects") and len(scene.get("player_options", [])) > 1:
            # Ищем опцию с effect "none"
            none_opts = [opt for opt in scene["player_options"] if isinstance(opt, dict) and opt.get("effect") == "none"]
            if none_opts:
                scene["player_options"] = [none_opts[0]]
            else:
                # Если нет, берём первую
                scene["player_options"] = [scene["player_options"][0]]

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
