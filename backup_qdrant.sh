#!/bin/bash
# Создаёт снапшот Qdrant и сохраняет его локально.
# Использование: ./backup_qdrant.sh
# Cron (ежедневно в 3:00): 0 3 * * * /home/serg/rag-knowledge-system/backup_qdrant.sh

set -e

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
COLLECTION="${QDRANT_COLLECTION:-knowledge_base}"
BACKUP_DIR="$(dirname "$0")/backups/qdrant"
KEEP_DAYS=7

mkdir -p "$BACKUP_DIR"

echo "Creating Qdrant snapshot for '$COLLECTION'..."
RESPONSE=$(curl -s -X POST "${QDRANT_URL}/collections/${COLLECTION}/snapshots")
SNAPSHOT_NAME=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['name'])" 2>/dev/null)

if [ -z "$SNAPSHOT_NAME" ]; then
    echo "ERROR: Failed to create snapshot. Response: $RESPONSE"
    exit 1
fi

echo "Downloading snapshot: $SNAPSHOT_NAME"
DEST="$BACKUP_DIR/${SNAPSHOT_NAME}"
curl -s -o "$DEST" "${QDRANT_URL}/collections/${COLLECTION}/snapshots/${SNAPSHOT_NAME}"

echo "Saved to: $DEST ($(du -sh "$DEST" | cut -f1))"

# Удаляем снапшоты старше KEEP_DAYS дней
find "$BACKUP_DIR" -name "*.snapshot" -mtime +$KEEP_DAYS -delete
echo "Cleaned up snapshots older than $KEEP_DAYS days."
echo "Done."
