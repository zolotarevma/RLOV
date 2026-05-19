"""Метрики разнообразия через ROUGE-L."""

from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def rouge_l_distance(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    scores = _scorer.score(text_a, text_b)
    return 1.0 - scores['rougeL'].fmeasure


def pairwise_diversity(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    dists = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            dists.append(rouge_l_distance(texts[i], texts[j]))
    return sum(dists) / len(dists)
