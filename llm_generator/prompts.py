"""Шаблон промпта для генератора сцен LLM."""
from config import LANGUAGE

SYSTEM_PROMPT = (
    "You are a screenwriter for a fantasy RPG. Generate a scene description, dialogues, "
    "and player choices according to the instructions.\n\n"
    "CRITICAL RULES for the 'player_options' array:\n"
    "- If the prompt provides a non-empty list of valid narrative effects, you MUST use exactly those effects in the "
    "'effect' fields, one per option.\n"
    "- If the prompt says 'No narrative effects are needed' or the list is empty, you MUST generate exactly ONE "
    "player_option with 'effect' set to the literal string 'none' (lowercase)."
    "Do NOT invent any other effect names.\n"
    "Note: Each effect is a technical flag that determines the next scene."
    "Do not create additional choices beyond those corresponding to valid effects.\n\n"
    "Respond ONLY with a valid JSON object:\n"
    "Respond ONLY with a valid JSON object:\n"
    '{"intro": "...", "dialogues": [{"speaker": "Name", "text": "..."}], '
    '"player_options": [{"text": "...", "effect": "..."}], "outcome": "..."}'
)

SYSTEM_PROMPT_RU = (
    "Ты — сценарист фэнтезийной RPG. Сгенерируй описание сцены, диалоги "
    "и варианты действий игрока в соответствии с инструкциями.\n\n"
    "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА для массива 'player_options':\n"
    "- Если в промпте указан непустой список допустимых эффектов, ты ДОЛЖЕН использовать именно эти эффекты в полях "
    "'effect', по одному на вариант.\n"
    "- Если в промпте сказано 'No narrative effects are needed' или список пуст, ты ДОЛЖЕН сгенерировать РОВНО ОДИН "
    "player_option с 'effect', равным строке 'none' (нижний регистр)."
    "Не придумывай других эффектов.\n"
    "Пояснение: каждый эффект является техническим флагом, определяющим следующую сцену."
    "Не придумывай дополнительных действий, не указанных в списке эффектов.\n\n"
    "Ответь ТОЛЬКО валидным JSON-объектом:\n"
    '{"intro": "...", "dialogues": [{"speaker": "Имя", "text": "..."}], '
    '"player_options": [{"text": "...", "effect": "..."}], "outcome": "..."}'
)


def build_prompt(beacon: dict, state: dict) -> str:
    effects_list = beacon.get("narrative_effects", [])
    if effects_list:
        n = len(effects_list)
        if LANGUAGE == "ru":
            effect_instruction = (
                f"ДОПУСТИМЫЕ ЭФФЕКТЫ ДЛЯ ЭТОЙ СЦЕНЫ: {', '.join(effects_list)}\n"
                f"Ты ДОЛЖЕН сгенерировать ровно {n} player_option(ов). "
                f"Каждое поле 'effect' ДОЛЖНО быть одним из: {', '.join(effects_list)}\n"
            )
        else:
            effect_instruction = (
                f"VALID EFFECTS FOR THIS SCENE: {', '.join(effects_list)}\n"
                f"You MUST generate exactly {n} player_option(s). "
                f"Each option's 'effect' field MUST be one of: {', '.join(effects_list)}\n"
            )
    else:
        if LANGUAGE == "ru":
            effect_instruction = (
                "ОБЯЗАТЕЛЬНОЕ ТРЕБОВАНИЕ: Эффекты НЕ нужны для этой сцены.\n"
                "Сгенерируй РОВНО ОДИН player_option с 'effect', равным строке 'none' (нижний регистр).\n"
                "Не используй других эффектов, только 'none'.\n"
            )
        else:
            effect_instruction = (
                "ABSOLUTE REQUIREMENT: NO narrative effects are allowed for this scene.\n"
                "Generate EXACTLY ONE player_option with 'effect' set to the literal string 'none'.\n"
                "Do NOT create any other effect names.\n"
            )

    example_json = ""
    if not effects_list and LANGUAGE == "ru":
        example_json = (
            "ПРИМЕР ДЛЯ ЭТОЙ СЦЕНЫ (без эффектов):\n"
            '{"intro": "Вы вошли в зал...", "dialogues": [...], '
            '"player_options": [{"text": "Продолжить", "effect": "none"}], "outcome": "..."}\n\n'
        )
    elif not effects_list:
        example_json = (
            "CORRECT EXAMPLE FOR THIS SCENE (no effects):\n"
            '{"intro": "You enter the hall...", "dialogues": [...], '
            '"player_options": [{"text": "Continue", "effect": "none"}], "outcome": "..."}\n\n'
        )

    # Концовка промпта с описанием вехи
    if LANGUAGE == "ru":
        beacon_info = (
            f"Состояние игрового мира:\n{state}\n\n"
            f"Текущая веха:\n"
            f"  Название: {beacon['name']}\n"
            f"  Описание: {beacon['description']}\n"
            f"  Тип: {beacon['type']}\n\n"
            "НАПОМИНАНИЕ: Строго следуй правилам для эффектов. "
            "Невалидные эффекты приведут к отклонению ответа.\n"
            "Сгенерируй JSON сейчас:"
        )
    else:
        beacon_info = (
            f"Game world state:\n{state}\n\n"
            f"Current beacon:\n"
            f"  Name: {beacon['name']}\n"
            f"  Description: {beacon['description']}\n"
            f"  Type: {beacon['type']}\n\n"
            "REMINDER: Follow the effect rules strictly. "
            "Invalid effects will cause the response to be rejected.\n"
            "Generate the JSON now:"
        )

    return f"{effect_instruction}\n{example_json}{beacon_info}"
