#!/bin/bash
# Starts an ngrok tunnel and registers the webhook with Telegram.
# Requires: ngrok authtoken (ngrok config add-authtoken <TOKEN>)
# Usage: ./start_tunnel.sh

BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-$(grep TELEGRAM_BOT_TOKEN .env 2>/dev/null | cut -d= -f2)}"

echo "Starting ngrok..."
ngrok http 8000 --log=stdout > /tmp/ngrok.log 2>&1 &
NGROK_PID=$!

# Wait for ngrok to come up
sleep 3

# Fetch the public URL via the ngrok API
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
tunnels = json.load(sys.stdin).get('tunnels', [])
for t in tunnels:
    if t.get('proto') == 'https':
        print(t['public_url'])
        break
")

if [ -z "$NGROK_URL" ]; then
    echo "Could not get a URL from ngrok. Check: http://localhost:4040"
    kill $NGROK_PID
    exit 1
fi

echo "Tunnel: $NGROK_URL"

# Register the webhook with Telegram
if [ -n "$BOT_TOKEN" ]; then
    RESULT=$(curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/setWebhook" \
        -d "url=${NGROK_URL}/telegram/webhook")
    echo "Telegram webhook: $RESULT"
else
    echo "TELEGRAM_BOT_TOKEN not set — webhook not registered"
    echo "Register it manually:"
    echo "  curl -X POST 'https://api.telegram.org/bot<TOKEN>/setWebhook' -d 'url=${NGROK_URL}/telegram/webhook'"
fi

echo ""
echo "Done! Tunnel is running. To stop: kill $NGROK_PID"
wait $NGROK_PID
