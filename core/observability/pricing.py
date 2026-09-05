from typing import Dict

# Тарифы за 1 миллион токенов (в долларах США)
MODEL_RATES: Dict[str, Dict[str, float]] = {
    # Alibaba Qwen 2.5 7B (ориентир: DeepInfra / Together AI cloud rates)
    "qwen2.5:7b": {
        "input_per_1m": 0.30,
        "output_per_1m": 1.20,
        "cached_per_1m": 0.075,
    },
    "qwen2.5:1.5b": {
        "input_per_1m": 0.10,
        "output_per_1m": 0.40,
        "cached_per_1m": 0.025,
    },
    # Meta Llama 3.2 (1B/3B)
    "llama3.2:latest": {
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
        "cached_per_1m": 0.0375,
    },
    "llama3.2": {
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
        "cached_per_1m": 0.0375,
    },
    "tinyllama": {
        "input_per_1m": 0.05,
        "output_per_1m": 0.20,
        "cached_per_1m": 0.01,
    },
    # Дефолтный тариф
    "default": {
        "input_per_1m": 0.30,
        "output_per_1m": 1.20,
        "cached_per_1m": 0.075,
    },
}


def get_model_rates(model: str) -> Dict[str, float]:
    """Возвращает тарифную сетку для указанной модели с fallback на default."""
    normalized = model.lower().strip()
    if normalized in MODEL_RATES:
        return MODEL_RATES[normalized]

    for key, rates in MODEL_RATES.items():
        if key in normalized or normalized in key:
            return rates

    return MODEL_RATES["default"]


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Рассчитывает ориентировочную стоимость вызова модели в долларах."""
    rates = get_model_rates(model)

    # Некешированные входные токены
    uncached_inputs = max(0, input_tokens - cached_tokens)

    input_cost = (uncached_inputs / 1_000_000.0) * rates["input_per_1m"]
    cached_cost = (cached_tokens / 1_000_000.0) * rates.get("cached_per_1m", rates["input_per_1m"] * 0.25)
    output_cost = (output_tokens / 1_000_000.0) * rates["output_per_1m"]

    return round(input_cost + cached_cost + output_cost, 6)
