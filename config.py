"""Настройки проекта."""
import os

# Путь к файлам сценариев
SCENARIOS_DIR = "experiments/scenarios"

# LLM-клиент: "stub", "ollama", "gemini", "openrouter"
LLM_CLIENT = "stub"

# Параметры для Ollama "llama3.1:8b", "mistral:7b", "gemma2:9b", "qwen2.5:7b"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://localhost:11434"

# Параметры для OpenRouter "openai/gpt-oss-120b:free"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")  # получите ключ на https://openrouter.ai/keys
OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# Google Gemini
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # получите ключ в Google AI Studio

# Максимальное количество шагов в сценарии
MAX_STEPS = 20

PLANNER = "dqn"  # "heuristic", "dqn"
DQN_MODEL_PATH = "rl_planner/dqn_model.pt"
DQN_FLAGS_ORDER_PATH = "rl_planner/flags_order.json"
