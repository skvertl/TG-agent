import re
from typing import List


def split_message(text: str, max_chunk_size: int = 4000) -> List[str]:
    """
    Разбивает длинное сообщение на чанки размером <= max_chunk_size.
    Сохраняет целостность блоков кода Markdown (```lang ... ```), закрывая
    и открывая их заново на стыках чанков.
    """
    if not text:
        return []

    if len(text) <= max_chunk_size:
        return [text]

    chunks: List[str] = []
    remaining = text
    in_code_block = False
    code_lang = ""

    while remaining:
        prefix = f"```{code_lang}\n" if in_code_block else ""

        # Если остаток полностью помещается с учетом префикса
        if len(prefix) + len(remaining) <= max_chunk_size:
            chunks.append(prefix + remaining)
            break

        # Сначала пробуем взять максимальную длину без учета суффикса
        raw_available = max_chunk_size - len(prefix)
        if raw_available <= 0:
            raw_available = 1

        candidate = remaining[:raw_available]

        # Ищем наилучшую точку разделения
        split_pos = -1
        cut_advance = -1

        # 1. Разрыв абзаца
        p_pos = candidate.rfind("\n\n")
        if p_pos > 0:
            split_pos = p_pos
            cut_advance = p_pos + 2
        else:
            # 2. Перенос строки
            n_pos = candidate.rfind("\n")
            if n_pos > 0:
                split_pos = n_pos
                cut_advance = n_pos + 1
            else:
                # 3. Пробел
                s_pos = candidate.rfind(" ")
                if s_pos > 0:
                    split_pos = s_pos
                    cut_advance = s_pos + 1
                else:
                    # 4. Жесткий срез
                    split_pos = len(candidate)
                    cut_advance = len(candidate)

        raw_chunk = candidate[:split_pos]

        # Вычисляем состояние Markdown-блоков кода внутри raw_chunk
        fences = list(re.finditer(r"```([a-zA-Z0-9_-]*)", raw_chunk))
        chunk_in_code = in_code_block
        chunk_lang = code_lang

        for f in fences:
            if chunk_in_code:
                chunk_in_code = False
                chunk_lang = ""
            else:
                chunk_in_code = True
                chunk_lang = f.group(1)

        # Если в конце чанка остался открытый блок кода, нужно добавить закрывающие ```
        suffix = "```" if chunk_in_code else ""
        extra_len = len("\n") + len(suffix) if suffix and not raw_chunk.endswith("\n") else len(suffix)

        # Если добавление суффикса превышает max_chunk_size, ужимаем raw_chunk
        if len(prefix) + len(raw_chunk) + extra_len > max_chunk_size:
            overflow = (len(prefix) + len(raw_chunk) + extra_len) - max_chunk_size
            trim_pos = max(1, split_pos - overflow)
            # Пересчитываем cut_advance и raw_chunk
            raw_chunk = candidate[:trim_pos]
            cut_advance = trim_pos

            # Пересчитываем fences для урезанного куска
            fences = list(re.finditer(r"```([a-zA-Z0-9_-]*)", raw_chunk))
            chunk_in_code = in_code_block
            chunk_lang = code_lang
            for f in fences:
                if chunk_in_code:
                    chunk_in_code = False
                    chunk_lang = ""
                else:
                    chunk_in_code = True
                    chunk_lang = f.group(1)
            suffix = "```" if chunk_in_code else ""

        # Формируем итоговый чанк
        if chunk_in_code:
            final_chunk = prefix + raw_chunk
            if not final_chunk.endswith("\n"):
                final_chunk += "\n"
            final_chunk += suffix
        else:
            final_chunk = prefix + raw_chunk

        chunks.append(final_chunk)

        # Переходим к следующей итерации
        in_code_block = chunk_in_code
        code_lang = chunk_lang
        remaining = remaining[cut_advance:]
        if not in_code_block:
            remaining = remaining.lstrip("\r\n")

    return chunks
