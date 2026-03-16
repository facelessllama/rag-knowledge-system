#!/bin/bash
while true; do
    echo "Запуск cloudflared..."
    cloudflared tunnel --url http://localhost:8000 2>&1 | tee /tmp/cf.log &
    CF_PID=$!
    sleep 10
    NEW_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cf.log | head -1)
    
    if [ -n "$NEW_URL" ]; then
        echo "URL: $NEW_URL"
        curl -s -X POST "https://b24-typ264.bitrix24.ru/rest/1/1zjpjhcsq6ddbodn/imbot.update.json" \
          -H "Content-Type: application/json" \
          -d "{\"BOT_ID\": 16, \"CLIENT_ID\": \"local.69b7864a22f254.89803254\", \"FIELDS\": {\"EVENT_MESSAGE_ADD\": \"${NEW_URL}/bitrix/webhook\", \"EVENT_WELCOME_MESSAGE\": \"${NEW_URL}/bitrix/webhook\", \"EVENT_BOT_DELETE\": \"${NEW_URL}/bitrix/webhook\"}}"
        echo "Бот обновлён!"
    fi
    
    wait $CF_PID
    echo "Туннель упал, перезапускаем через 5 сек..."
    sleep 5
done
