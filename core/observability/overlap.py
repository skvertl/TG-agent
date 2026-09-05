import math
from typing import List, Tuple


def estimate_tokens(text: str) -> int:
    """
    Быстрая и надежная оценка количества токенов без внешних тяжелых зависимостей.
    Для смешанного (русский/английский/код) текста: ~3.8 символа на токен или ~1.3 токена на слово.
    """
    if not text or not text.strip():
        return 0

    stripped = text.strip()
    words = stripped.split()
    chars = len(stripped)

    # Комбинированная эвристика: учитывает длину и количество слов
    by_chars = chars / 3.8
    by_words = len(words) * 1.3

    estimated = int(math.ceil((by_chars + by_words) / 2.0))
    return max(1, estimated)


def calculate_context_overlap(
    previous_messages: List[str],
    current_messages: List[str],
) -> Tuple[int, float]:
    """
    Вычисляет объем и долю повторного контекста (токенов) между предыдущим и текущим ходом.
    Возвращает: (repeated_tokens, repeated_percentage [0.0..100.0]).
    """
    if not previous_messages or not current_messages:
        return 0, 0.0

    # Оцениваем токены каждого блока в текущем контексте
    current_tokens = [estimate_tokens(msg) for msg in current_messages]
    total_current_tokens = sum(current_tokens)
    if total_current_tokens == 0:
        return 0, 0.0

    prev_set = set(previous_messages)
    repeated_tokens = 0

    for msg, tokens in zip(current_messages, current_tokens):
        if msg in prev_set:
            repeated_tokens += tokens

    pct = round((repeated_tokens / total_current_tokens) * 100.0, 1)
    return repeated_tokens, pct
