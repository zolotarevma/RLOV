"""Клиенты для вызова LLM (заглушка, Ollama, OpenRouter, Gemini)."""

import json
import time
import requests
from config import (
    LLM_CLIENT,
    OLLAMA_MODEL,
    OLLAMA_HOST,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_API_BASE,
    GIGACHAT_MODEL,
    GIGACHAT_TEMPERATURE,
    GIGACHAT_MAX_TOKENS
)
from abc import ABC, abstractmethod
from typing import Dict, Type
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LLMClient(ABC):

    @abstractmethod
    def generate(self, system: str, prompt: str) -> str:
        pass


class StubClient(LLMClient):

    def generate(self, system: str, prompt: str) -> str:
        import re

        name_match = re.search(r"Name:\s*(.*)", prompt)
        beacon_name = name_match.group(1).strip() if name_match else "unknown beacon"

        effects_match = re.search(
            r"VALID EFFECTS FOR THIS SCENE:\s*(.*)", prompt
        )
        if effects_match:
            effects_str = effects_match.group(1)
            effects = [e.strip() for e in effects_str.split(",") if e.strip()]
            if effects == ["continue"]:
                player_options = [{"text": "Continue", "effect": "continue"}]
            else:
                player_options = [{"text": f"Choose {eff}", "effect": eff} for eff in effects]
        else:
            player_options = []

        dummy = {
            "intro": f"You are in '{beacon_name}'. The atmosphere is tense.",
            "dialogues": [
                {"speaker": "Narrator", "text": f"Welcome to '{beacon_name}'."}
            ],
            "player_options": player_options,
            "outcome": f"You have made your choice in '{beacon_name}'.",
        }
        return json.dumps(dummy, ensure_ascii=False)


class OllamaClient(LLMClient):

    def generate(self, system: str, prompt: str) -> str:
        url = f"{OLLAMA_HOST}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "system": system,
            "prompt": prompt,
            "stream": False,
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")


class GeminiClient(LLMClient):

    def generate(self, system: str, prompt: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system,
        )
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = model.generate_content(prompt)
                text = resp.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                return text
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""


class OpenRouterClient(LLMClient):

    def generate(self, system: str, prompt: str) -> str:
        import openai
        client = openai.OpenAI(
            base_url=OPENROUTER_API_BASE,
            api_key=OPENROUTER_API_KEY,
        )
        headers = {
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "RLOV Prototype",
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=500,
                    extra_headers=headers,
                )
                return resp.choices[0].message.content
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""


class GigaChatClient(LLMClient):

    def generate(self, system: str, prompt: str) -> str:
        from llm_generator.gigachat_auth import get_access_token
        token = get_access_token()
        if not token:
            return '{"intro": "GigaChat token error.", "dialogues": [], "player_options": []}'

        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GIGACHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": GIGACHAT_TEMPERATURE,
            "max_tokens": GIGACHAT_MAX_TOKENS
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, verify=False)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            return text.strip()
        except Exception as e:
            print(f"[GigaChat] Error: {e}")
            return '{"intro": "GigaChat failed to generate.", "dialogues": [], "player_options": []}'


_CLIENT_MAP: Dict[str, Type[LLMClient]] = {
    "stub": StubClient,
    "ollama": OllamaClient,
    "gemini": GeminiClient,
    "openrouter": OpenRouterClient,
    "gigachat": GigaChatClient,
}


def call_llm(system: str, prompt: str) -> str:
    client_class = _CLIENT_MAP.get(LLM_CLIENT)
    if client_class is None:
        raise ValueError(f"Unsupported LLM_CLIENT: {LLM_CLIENT}")
    return client_class().generate(system, prompt)
