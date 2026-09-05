# ==========================================
# Stage 1: Build & Dependencies
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Установка системных зависимостей для сборки колес, если требуются
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Установка зависимостей в пользовательскую директорию
RUN pip install --no-cache-dir --user -r requirements.txt


# ==========================================
# Stage 2: Runtime Image (Non-root user)
# ==========================================
FROM python:3.11-slim AS runner

# Создание непривилегированного пользователя appuser (UID 10001)
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

WORKDIR /app

# Копирование установленных библиотек из builder в профиль appuser
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Настройка переменных окружения Python и PATH
ENV PATH="/home/appuser/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Копирование исходного кода приложения
COPY --chown=appuser:appuser core/ /app/core/
COPY --chown=appuser:appuser adapters/ /app/adapters/
COPY --chown=appuser:appuser config.py /app/config.py
COPY --chown=appuser:appuser main.py /app/main.py

# Создание директории для персистентной БД телеметрии
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

# Переключение на непривилегированного пользователя
USER appuser

# Точка входа
CMD ["python", "main.py"]
