# NeuroAnalyst Backend

Flask API для анализа сайтов с помощью GPT.

## Endpoints

- `POST /analyze` - Анализ сайта
- `POST /followup` - Корректировки анализа
- `POST /clear-chat` - Очистка истории чата
- `GET /ping` - Health check

## Deploy на Render

1. Подключи репозиторий
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `gunicorn app:app`
4. Добавь Environment Variable: `OPENAI_API_KEY`
