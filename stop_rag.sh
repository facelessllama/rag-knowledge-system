#!/usr/bin/env bash
# Останавливает всё, что поднимает start_rag.sh, кроме Docker daemon (его гасить руками).
set -uo pipefail

PROJECT_DIR="/home/serg/rag-knowledge-system"

kill_port() {
    # kill_port <port> <label>
    local port="$1" label="$2"
    local pids
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        echo "  $label: не запущен"
        return
    fi
    kill $pids 2>/dev/null
    for i in $(seq 1 10); do
        pids=$(lsof -ti ":$port" 2>/dev/null || true)
        [ -z "$pids" ] && break
        sleep 1
    done
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        kill -9 $pids 2>/dev/null || true
    fi
    echo "  $label: остановлен"
}

echo "=== RAG Knowledge System: остановка окружения ==="

echo "[1/3] API (порт 8000)..."
kill_port 8000 "API"

echo "[2/3] Ollama (порт 11435)..."
kill_port 11435 "Ollama"

echo "[3/3] Qdrant / Postgres / Langfuse..."
cd "$PROJECT_DIR"
if docker info > /dev/null 2>&1; then
    docker compose -f docker/docker-compose.yml --env-file .env stop
    echo "  контейнеры остановлены (не удалены, следующий start_rag.sh их просто запустит заново)"
else
    echo "  Docker daemon недоступен — контейнеры и так не работают, пропускаю"
fi

echo ""
echo "=== Готово. Всё остановлено (Docker daemon трогать не стал). ==="
