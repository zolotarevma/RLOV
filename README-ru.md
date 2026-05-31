# RLOV: RL-LLM Orchestrator with Validation

Гибридная система генерации адаптивных сюжетов для видеоигр жанра квест/RPG с пошаговым геймплеем. Прототип разработан в рамках выпускной квалификационной работы.

## Основные компоненты

- **RL-планировщик** (DQN) — обучаемое макроуправление нарративными вехами.
- **LLM-генератор** — создание сцен и диалогов с помощью больших языковых моделей (локальные Ollama, Gemini, OpenRouter, GigaChat).
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
- `GIGACHAT_AUTHORIZATION_KEY` — ключ авторизации GigaChat ([личный кабинет GigaChat API](https://developers.sber.ru/studio))

Локальные модели запускаются через [Ollama](https://ollama.com).

## Конфигурация

Следующие параметры задаются в `config.py`:

- `LLM_CLIENT` — `"stub"`, `"ollama"`, `"gemini"`, `"openrouter"` или `gigachat`
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

## Готовые RL-модели

В папке `.backups` лежат заранее обученные модели для двух сценариев:
- `dqn_model_mayor.pt` / `flags_order_mayor.json` — для сценария "Mayor's Support" (mayor_support.json)
- `dqn_model_expedition.pt` / `flags_order_expedition.json` — для сценария "Expedition to the Forgotten Temple" (expedition.json)

Чтобы использовать готовую модель, скопируйте нужную пару в корень проекта и переименуйте:
```bash
cp .backups/dqn_model_mayor.pt dqn_model.pt
cp .backups/flags_order_mayor.json flags_order.json
```

## Сценарии

В проекте используются два нелинейных сценария:

- **Mayor's Support** (`experiments/scenarios/mayor_support.json`) — основной сценарий, использованный в ВКР, с 16 вехами и тремя концовками.
- **Expedition to the Forgotten Temple** (`experiments/scenarios/expedition.json`) — более сложный сценарий с несколькими точками макро-выбора для RL-агента, применявшийся в расширенных экспериментах.

Оба сценария доступны на английском и русском языках (`*_ru.json`).

## Графический интерфейс

В проекте реализован веб-интерфейс на базе **Streamlit** (`app_gui.py`), который превращает консольный прототип в интерактивную демонстрацию с тёмной RPG-темой.

**Возможности интерфейса:**
- Наглядное отображение сцены, диалогов и вариантов действий.
- Фильтрация уже выполненных заданий.
- Автоматический макро-выбор RL-агента с анимацией ожидания.
- Цветовое оформление концовок (🏆 идеальная, ⚖️ хорошая, 💀 плохая).
- Кнопка «Начать заново» на боковой панели.

**Запуск интерфейса:**
```bash
streamlit run app_gui.py
```

## Автор

Максим Золотарев, СПбГУ, 2026.
