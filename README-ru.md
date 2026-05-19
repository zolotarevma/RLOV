# RLOV: RL-LLM Orchestrator with Validation

Гибридная система генерации адаптивных сюжетов для видеоигр жанра квест/RPG с пошаговым геймплеем. Прототип разработан в рамках выпускной квалификационной работы.

## Основные компоненты

- **RL-планировщик** (DQN) — обучаемое макроуправление нарративными вехами.
- **LLM-генератор** — создание сцен и диалогов с помощью больших языковых моделей (локальные Ollama, Gemini, OpenRouter).
- **Модуль валидации** — формальный контроль графа сценария и проверка JSON-ответов LLM.

## Установка

```bash
git clone https://github.com/zolotarevma/RLOV
cd RLOV
pip install -r requirements.txt
```

## Настройка API-ключей

Для работы с облачными LLM необходимо установить переменные окружения:

- `OPENROUTER_API_KEY` — ключ API OpenRouter (https://openrouter.ai/keys)
- `GEMINI_API_KEY` — ключ Google AI Studio (https://makersuite.google.com/app/apikey)

Локальные модели запускаются через [Ollama](https://ollama.com).

## Конфигурация

Следующие параметры задаются в `config.py`:

- `LLM_CLIENT` — `"stub"`, `"ollama"`, `"gemini"` или `"openrouter"`
- `OLLAMA_MODEL` — название модели (например, `"llama3.1:8b"`)
- `PLANNER` — `"heuristic"` или `"dqn"`

## Запуск

Интерактивный режим (консольная игра):
```bash
python main.py
```

## Пакетный эксперимент

```bash
python experiments/run_experiment.py dqn 20
```

## Сравнение планировщиков

```bash
python experiments/compare_planners.py 30
```

## Сценарий

В проекте используется нелинейный сценарий «Mayor's Support» (`experiments/scenarios/mayor_support.json`) с 16 сценами и тремя концовками.

## Автор

Максим Золотарев, СПбГУ, 2026.
