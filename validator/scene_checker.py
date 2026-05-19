"""Проверка корректности сгенерированной сцены."""


def validate_scene_json(scene: dict, beacon: dict) -> list[str]:
    errors = []

    if not isinstance(scene, dict):
        return ["Scene is not a dictionary"]

    if "intro" not in scene or not scene["intro"]:
        errors.append("Missing or empty 'intro' field")

    if "dialogues" in scene:
        if not isinstance(scene["dialogues"], list):
            errors.append("'dialogues' must be a list")
        else:
            for i, d in enumerate(scene["dialogues"]):
                if not isinstance(d, dict):
                    errors.append(f"Dialogue {i} is not a dictionary")
                    continue
                if not (d.get("speaker") or d.get("character") or d.get("name")):
                    errors.append(f"Dialogue {i} missing speaker field")
                if not (d.get("text") or d.get("line") or d.get("content")):
                    errors.append(f"Dialogue {i} missing text field")

    if "player_options" in scene:
        if not isinstance(scene["player_options"], list):
            errors.append("'player_options' must be a list")
        else:
            valid_effects = beacon.get("narrative_effects", [])
            for i, opt in enumerate(scene["player_options"]):
                if isinstance(opt, str):
                    errors.append(f"player_option {i} is a string, expected dict with 'text' and 'effect'")
                    continue
                if not isinstance(opt, dict):
                    errors.append(f"player_option {i} is not a dictionary")
                    continue
                if "text" not in opt or not opt["text"]:
                    errors.append(f"player_option {i} missing or empty 'text'")

                effect = opt.get("effect", "")

                if valid_effects:
                    if not effect:
                        errors.append(
                            f"player_option {i} missing 'effect' (required because narrative_effects is not empty)"
                        )
                    elif effect not in valid_effects:
                        errors.append(
                            f"player_option {i} has invalid effect '{effect}'. "
                            f"Valid: {valid_effects}"
                        )
                else:
                    if effect and effect != "none":
                        errors.append(
                            f"player_option {i} has unexpected effect '{effect}'. "
                            f"Use 'none' when narrative_effects is empty."
                        )

    return errors
