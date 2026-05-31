"""
Графический интерфейс Streamlit для демонстрации RLOV.
"""

import streamlit as st
import time
from pathlib import Path

# Локальные модули
from game_state import GameState
from game_state.scenario import load_scenario
from llm_generator.generator import generate_scene
from config import (
    SCENARIOS_DIR, PLANNER, DQN_MODEL_PATH, DQN_FLAGS_ORDER_PATH,
    LLM_CLIENT, LANGUAGE
)


# ----- Вспомогательные функции (дублированы для GUI) -----
def _check_preconditions(state, pre):
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


def _apply_effects(state, beacon):
    for k, v in beacon.get("effects", {}).items():
        state.set_flag(k, v)
    state.add_event({"beacon_id": beacon["id"]})


def _get_available(beacons, state, current_beacon):
    if not current_beacon.get("narrative_effects"):
        choices_ids = current_beacon.get("choices", [])
    else:
        choices_ids = current_beacon.get("choices", [])
    available_beacons = []
    for bid in choices_ids:
        b = next((bb for bb in beacons if bb["id"] == bid), None)
        if b and _check_preconditions(state, b.get("preconditions", {})):
            available_beacons.append(b)
    return available_beacons


# ----- Инициализация сессии -----
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'reached_ending' not in st.session_state:
    st.session_state.reached_ending = False

# ----- Рендеринг -----
st.set_page_config(page_title="RLOV: Адаптивный сюжет", page_icon="🎮", layout="centered")
st.title("🎮 RLOV: Адаптивный сюжет")

# Кастомный CSS (тот же, но добавляем стили для кнопки рестарта)
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
        color: #D0D0D0;
    }
    .stButton > button {
        background-color: #2E4A2E;
        color: #F0E68C;
        border: 2px solid #5B3A1A;
        border-radius: 8px;
        font-size: 18px;
        padding: 12px;
        width: 100%;
        margin: 8px 0;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #3A5C3A;
        border-color: #8B6914;
        color: #FFD700;
    }
    .rl-message {
        background-color: #2E2E2E;
        border-left: 5px solid #8B6914;
        padding: 15px;
        margin: 20px 0;
        font-style: italic;
        color: #FFD700;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ----- Стартовый экран -----
if not st.session_state.game_started:
    st.markdown("""
    ## Добро пожаловать в мир RLOV!

    Это демонстрация гибридной системы генерации адаптивных сюжетов.
    Вы — герой, а RL-агент управляет макро-событиями,
    подстраивая историю под ваши решения.

    **Сценарий:** Экспедиция к Забытому Храму  
    **Модель:** {}  
    **Планировщик:** {}
    """.format(LLM_CLIENT, PLANNER))

    if st.button("🚀 Начать игру"):
        # Инициализация игры
        scenario_path = Path(SCENARIOS_DIR) / "expedition_ru.json"
        st.session_state.scenario = load_scenario(str(scenario_path))
        st.session_state.beacons = st.session_state.scenario["beacons"]

        st.session_state.state = GameState()
        st.session_state.path = []

        # Планировщик
        import json

        if PLANNER == "dqn":
            from rl_planner.dqn_planner import DQNPlanner

            with open(DQN_FLAGS_ORDER_PATH, "r") as f:
                flags_order = json.load(f)
            state_dim = len(flags_order)
            action_dim = len(st.session_state.beacons)
            planner = DQNPlanner(DQN_MODEL_PATH, DQN_FLAGS_ORDER_PATH, state_dim, action_dim)
            planner.set_mapping(st.session_state.beacons)
            planner.q_net.eval()
        else:
            from planner.heuristic import HeuristicPlanner

            planner = HeuristicPlanner()
        st.session_state.planner = planner

        start = next(b for b in st.session_state.beacons if b["type"] == "start")
        _apply_effects(st.session_state.state, start)
        st.session_state.path.append(start["id"])
        st.session_state.current_beacon = start

        scene = generate_scene(st.session_state.current_beacon, st.session_state.state.to_dict())
        st.session_state.intro = scene.get("intro", "")
        st.session_state.dialogues = scene.get("dialogues", [])
        st.session_state.player_options = scene.get("player_options", [])

        st.session_state.game_started = True
        st.session_state.reached_ending = False
        st.rerun()

else:
    # ----- Основной игровой цикл -----
    if st.session_state.reached_ending:
        ending_beacon = st.session_state.current_beacon
        ending_type = ending_beacon.get("id", "")
        # Определяем цвет плашки в зависимости от типа концовки
        if "perfect" in ending_type:
            color = "#2E7D32"  # зелёный
            emoji = "🏆"
        elif "good" in ending_type:
            color = "#F57F17"  # жёлтый
            emoji = "⚖️"
        else:
            color = "#B71C1C"  # красный
            emoji = "💀"

        st.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; margin:10px 0;">
            <h2 style="color:white; margin:0;">{emoji} {ending_beacon['name']}</h2>
            <p style="color:white; font-size:18px;">{ending_beacon['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.write(st.session_state.intro)
    else:
        game_container = st.empty()
        with game_container.container():
            # Отображение сцены
            st.markdown(f"### 📜 {st.session_state.current_beacon['name']}")
            st.write(st.session_state.intro)
            if st.session_state.dialogues:
                for d in st.session_state.dialogues:
                    speaker = d.get('speaker') or d.get('character') or d.get('name') or 'Неизвестный'
                    text = d.get('text') or d.get('line') or d.get('content') or ''
                    st.markdown(f"**{speaker}:** {text}")

            current = st.session_state.current_beacon
            options = st.session_state.player_options

            if current["type"] == "ending":
                st.session_state.reached_ending = True
                st.rerun()

            if current.get("narrative_effects") and options:
                # ---- Фильтрация доступных опций ----
                valid_options = []
                for opt in options:
                    effect = opt.get("effect", "") if isinstance(opt, dict) else ""
                    if not effect or effect == "none":
                        continue
                    # Ищем целевую веху по expected_player_flag
                    target = None
                    for cid in current.get("choices", []):
                        b = next((bb for bb in st.session_state.beacons if bb["id"] == cid), None)
                        if b and b.get("expected_player_flag") == effect:
                            target = b
                            break
                    if target and _check_preconditions(st.session_state.state, target.get("preconditions", {})):
                        valid_options.append(opt)

                if not valid_options:
                    st.warning("Нет доступных действий. Возможно, все квесты выполнены.")
                    # Автоматически выбираем первый возможный переход (например, leave_city)
                    # Ищем любой доступный target
                    for cid in current.get("choices", []):
                        b = next((bb for bb in st.session_state.beacons if bb["id"] == cid), None)
                        if b and _check_preconditions(st.session_state.state, b.get("preconditions", {})):
                            target = b
                            effect = target.get("expected_player_flag", "")
                            if effect:
                                st.session_state.state.set_flag(effect, True)
                            _apply_effects(st.session_state.state, target)
                            st.session_state.path.append(target["id"])
                            st.session_state.current_beacon = target
                            scene = generate_scene(target, st.session_state.state.to_dict())
                            st.session_state.intro = scene.get("intro", "")
                            st.session_state.dialogues = scene.get("dialogues", [])
                            st.session_state.player_options = scene.get("player_options", [])
                            st.rerun()
                    st.stop()
                else:
                    # Отображаем только доступные опции
                    st.markdown("---")
                    cols = st.columns(len(valid_options))
                    for i, opt in enumerate(valid_options):
                        with cols[i]:
                            if st.button(opt.get("text", str(opt)), key=f"opt_{i}"):
                                effect = opt.get("effect", "") if isinstance(opt, dict) else ""
                                if effect and effect != "none":
                                    st.session_state.state.set_flag(effect, True)
                                # Найти целевую веху (она точно будет, т.к. мы её уже проверили)
                                target = None
                                for cid in current.get("choices", []):
                                    b = next((bb for bb in st.session_state.beacons if bb["id"] == cid), None)
                                    if b and b.get("expected_player_flag") == effect:
                                        target = b
                                        break
                                # На всякий случай fallback
                                if target is None:
                                    if len(current.get("choices", [])) == 1:
                                        target = next(
                                            bb for bb in st.session_state.beacons if bb["id"] == current["choices"][0])
                                    else:
                                        st.error("Ошибка перехода!")
                                        st.stop()

                                _apply_effects(st.session_state.state, target)
                                st.session_state.path.append(target["id"])
                                st.session_state.current_beacon = target

                                scene = generate_scene(target, st.session_state.state.to_dict())
                                st.session_state.intro = scene.get("intro", "")
                                st.session_state.dialogues = scene.get("dialogues", [])
                                st.session_state.player_options = scene.get("player_options", [])
                                game_container.empty()
                                time.sleep(0.001)
                                st.rerun()
            else:  # макро-выбор (RL)
                with st.spinner("🧠 RL-агент выбирает оптимальный маршрут..."):
                    time.sleep(3)  # можно уменьшить или убрать
                    available = _get_available(st.session_state.beacons, st.session_state.state, current)
                    if available:
                        chosen_id = st.session_state.planner.select_beacon(st.session_state.state, available)
                        chosen = next(b for b in st.session_state.beacons if b["id"] == chosen_id)
                        _apply_effects(st.session_state.state, chosen)
                        st.session_state.path.append(chosen_id)
                        st.session_state.current_beacon = chosen

                        scene = generate_scene(chosen, st.session_state.state.to_dict())
                        st.session_state.intro = scene.get("intro", "")
                        st.session_state.dialogues = scene.get("dialogues", [])
                        st.session_state.player_options = scene.get("player_options", [])
                    else:
                        st.error("Нет доступных вех.")
                        st.session_state.reached_ending = True
                    game_container.empty()
                    time.sleep(0.001)
                    st.rerun()


# Боковая панель (всегда доступна)
with st.sidebar:
    st.header("⚙️ Управление")
    if st.session_state.game_started:
        st.write(f"Сценарий: Expedition")
        st.write(f"Язык: {LANGUAGE}")
        st.write(f"Модель: {LLM_CLIENT}")
        st.write(f"Планировщик: {PLANNER}")
        if st.button("🔄 Начать заново"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    else:
        st.write("Демонстрация RLOV")
