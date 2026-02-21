# Изучение RAG — Retrieval Augmented Generation


## Требования
- Python 3.11+
- [Ollama](https://ollama.com) — Локальная генерация ответов


## Установка

### 1. Клонировать репозиторий

```bash
git clone <repo-url>
cd rag
```

### 2. Создать и активировать виртуальное окружение

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux
```

### 3. Установить зависимости
```bash
pip install -r requirements.txt
```

### 4. Настроить переменные окружения

Скопировать `.env.example` в `.env` и задать значения:
```bash
cp .env.example .env
```

Содержимое `.env`:
```
BASE_URL=http://localhost:11434/v1/
API_KEY=ollama
MODEL=qwen2.5-coder:7b
TEMPERATURE=0.1
```

### 5. Запустить модель в Ollama

```bash
ollama pull qwen2.5-coder:7b
ollama serve
```


## Запуск

```bash
python src/main.py "text for query"
```


## Тесты

```bash
pip install -r requirements-dev.txt
python -m pytest
```