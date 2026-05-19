"""Шаблон промпта для генератора сцен LLM."""

SYSTEM_PROMPT = (
    "You are a screenwriter for a fantasy RPG. Generate a scene description, dialogues, "
    "and player choices according to the instructions.\n\n"
    "CRITICAL RULES for the 'player_options' array:\n"
    "- If the prompt provides a non-empty list of valid narrative effects, you MUST use exactly those effects in the "
    "'effect' fields, one per option.\n"
    "- If the prompt says 'No narrative effects are needed' or the list is empty, you MUST generate exactly ONE "
    "player_option with 'effect' set to the literal string 'none' (lowercase)."
    "Do NOT invent any other effect names.\n\n"
    "Respond ONLY with a valid JSON object:\n"
    '{"intro": "...", "dialogues": [{"speaker": "Name", "text": "..."}], '
    '"player_options": [{"text": "...", "effect": "..."}], "outcome": "..."}'
)


def build_prompt(beacon: dict, state: dict) -> str:
    effects_list = beacon.get("narrative_effects", [])
    if effects_list:
        n = len(effects_list)
        effect_instruction = (
            f"VALID EFFECTS FOR THIS SCENE: {', '.join(effects_list)}\n"
            f"You MUST generate exactly {n} player_option(s). "
            f"Each option's 'effect' field MUST be one of: {', '.join(effects_list)}\n"
        )
    else:
        effect_instruction = (
            "ABSOLUTE REQUIREMENT: NO narrative effects are allowed for this scene.\n"
            "Generate EXACTLY ONE player_option with 'effect' set to the literal string 'none'.\n"
            "Do NOT create any other effect names like 'increase_relation', 'advance_step', etc. — "
            "the word must be exactly 'none'.\n"
        )

    example_json = ""
    if not effects_list:
        example_json = (
            "CORRECT EXAMPLE FOR THIS SCENE (no effects):\n"
            '{"intro": "You enter the hall...", "dialogues": [...], '
            '"player_options": [{"text": "Continue", "effect": "none"}], "outcome": "..."}\n\n'
        )

    return (
        f"{effect_instruction}\n"
        f"{example_json}"
        f"Game world state:\n{state}\n\n"
        f"Current beacon:\n"
        f"  Name: {beacon['name']}\n"
        f"  Description: {beacon['description']}\n"
        f"  Type: {beacon['type']}\n\n"
        "REMINDER: Follow the effect rules strictly. "
        "Invalid effects will cause the response to be rejected.\n"
        "Generate the JSON now:"
    )
