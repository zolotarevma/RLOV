"""Семантические метрики на основе Sentence-BERT."""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def semantic_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    model = get_model()
    emb = model.encode([text_a, text_b])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])


def prompt_scene_similarity(beacon: dict, scene: dict) -> float:
    prompt_text = beacon.get("description", "")
    intro_text = scene.get("intro", "")
    return semantic_similarity(prompt_text, intro_text)
