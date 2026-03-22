#!/bin/bash
# Запускает ngrok-туннель и регистрирует webhook в Telegram.
# Требует: ngrok authtoken (ngrok config add-authtoken <TOKEN>)
# Использование: ./start_tunnel.sh

BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-$(grep TELEGRAM_BOT_TOKEN .env 2>/dev/null | cut -d= -f2)}"

echo "Запуск ngrok..."
ngrok http 8000 --log=stdout > /tmp/ngrok.log 2>&1 &
NGROK_PID=$!

# Ждём пока ngrok поднимется
sleep 3

# Получаем публичный URL через ngrok API
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
tunnels = json.load(sys.stdin).get('tunnels', [])
for t in tunnels:
    if t.get('proto') == 'https':
        print(t['public_url'])
        break
")

if [ -z "$NGROK_URL" ]; then
    echo "Не удалось получить URL от ngrok. Проверьте: http://localhost:4040"
    kill $NGROK_PID
    exit 1
fi

echo "Туннель: $NGROK_URL"

# Регистрируем webhook в Telegram
if [ -n "$BOT_TOKEN" ]; then
    RESULT=$(curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/setWebhook" \
        -d "url=${NGROK_URL}/telegram/webhook")
    echo "Telegram webhook: $RESULT"
else
    echo "TELEGRAM_BOT_TOKEN не задан — webhook не зарегистрирован"
    echo "Зарегистрируйте вручную:"
    echo "  curl -X POST 'https://api.telegram.org/bot<TOKEN>/setWebhook' -d 'url=${NGROK_URL}/telegram/webhook'"
fi

echo ""
echo "Готово! Туннель работает. Для остановки: kill $NGROK_PID"
wait $NGROK_PID
