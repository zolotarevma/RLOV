# RLOV: RL-LLM Orchestrator with Validation

A hybrid system for generating adaptive storylines in turn-based quest/RPG video games. Prototype developed as part of a bachelor's thesis.

## Core Components

- **RL Planner** (DQN) — learnable macro-management of narrative beacons.
- **LLM Generator** — creates scenes and dialogues using large language models (local Ollama, Gemini, OpenRouter).
- **Validation Module** — formal graph verification and JSON response checking for LLM output.

## Installation

```bash
git clone https://github.com/zolotarevma/RLOV
cd RLOV
pip install -r requirements.txt
```

## API Key Setup

For cloud-based LLMs, set the following environment variables:

- `OPENROUTER_API_KEY` — OpenRouter API key (https://openrouter.ai/keys)
- `GEMINI_API_KEY` — Google AI Studio API key (https://makersuite.google.com/app/apikey)

Local models run via [Ollama](https://ollama.com).

## Configuration

All parameters are in `config.py`:

- `LLM_CLIENT` — `"stub"`, `"ollama"`, `"gemini"` or `"openrouter"`
- `OLLAMA_MODEL` — model name (e.g., `"llama3.1:8b"`)
- `PLANNER` — `"heuristic"` or `"dqn"`

## Usage

Interactive mode (console game):
```bash
python main.py
```

## Batch experiment

```bash
python experiments/run_experiment.py dqn 20
```

## Planner comparison

```bash
python experiments/compare_planners.py 30
```

## Pre-trained RL Models

The `.backups` folder contains pre-trained models for two scenarios:
- `dqn_model_mayor.pt` / `flags_order_mayor.json` – for the "Mayor's Support" scenario (mayor_support.json)
- `dqn_model_expedition.pt` / `flags_order_expedition.json` – for the "Expedition" scenario (expedition.json)

To use a pre-trained model, copy the required pair to the project root and rename them:
```bash
cp .backups/dqn_model_mayor.pt dqn_model.pt
cp .backups/flags_order_mayor.json flags_order.json
```

## Scenarios

The project includes two non-linear scenarios:

- **Mayor's Support** (`experiments/scenarios/mayor_support.json`) – the main scenario used in the thesis, with 16 beacons and 3 endings.
- **Expedition to the Forgotten Temple** (`experiments/scenarios/expedition.json`) – a more complex scenario with multiple macro-choice points for the RL agent, used in advanced experiments.

Both scenarios are available in English and Russian (`*_ru.json`).

## Author

Maxim Zolotarev, St Petersburg University, 2026.
