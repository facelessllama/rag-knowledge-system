#!/usr/bin/env bash
# Поднимает всё окружение RAG-системы, кроме Docker daemon (запускается отдельно вручную).
# Идемпотентен: если сервис уже запущен, повторно не поднимает.
set -euo pipefail

PROJECT_DIR="/home/serg/rag-knowledge-system"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

wait_for() {
    # wait_for <url> <label> <timeout_sec>
    local url="$1" label="$2" timeout="$3"
    for ((i = 0; i < timeout; i++)); do
        if curl -sf "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "  ПРЕДУПРЕЖДЕНИЕ: $label не ответил за ${timeout}с"
    return 1
}

echo "=== RAG Knowledge System: запуск окружения ==="

echo "[1/4] Проверка Docker daemon..."
if ! docker info > /dev/null 2>&1; then
    echo "ОШИБКА: Docker daemon недоступен. Запусти Docker Desktop и повтори запуск."
    exit 1
fi
echo "  OK"

echo "[2/4] Qdrant / Postgres / Langfuse..."
cd "$PROJECT_DIR"
docker compose -f docker/docker-compose.yml --env-file .env up -d
wait_for "http://localhost:6333/" "Qdrant" 30
echo "  OK"

echo "[3/4] Ollama (порт 11435)..."
if curl -sf http://localhost:11435/api/tags > /dev/null 2>&1; then
    echo "  уже запущена"
else
    OLLAMA_HOST=0.0.0.0:11435 nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    disown
    wait_for "http://localhost:11435/api/tags" "Ollama" 30
fi
echo "  OK"

echo "[4/4] API (порт 8000)..."
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "  уже запущен"
else
    source venv/bin/activate
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 \
        > "$LOG_DIR/api.log" 2>&1 &
    disown
    echo "  прогрев моделей, может занять ~30-60 сек..."
    wait_for "http://localhost:8000/health" "API" 90
fi

echo ""
echo "=== Статус ==="
curl -s http://localhost:8000/health 2>&1 || echo "API не отвечает, смотри $LOG_DIR/api.log"
echo ""
echo "UI:   http://localhost:8000/app"
echo "Docs: http://localhost:8000/docs"
echo "Логи: $LOG_DIR/{ollama,api}.log"
