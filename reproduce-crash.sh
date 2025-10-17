#!/bin/bash
# Script to reproduce DeltaNet crash and capture logs (Linux version)
# This runs BEFORE rebuilding to verify the crash location

set -e

SERVER_LOG="/mnt/d/Projects/llama.cpp/logs/reproduce-crash.log"
TEST_LOG="/mnt/d/Projects/llama.cpp/logs/test-output.log"
SERVER_BIN="/mnt/d/Projects/llama.cpp/build/bin/llama-server"
MODEL_PATH="/mnt/d/Projects/qwen_quant/converted/q3/qwen3_80b-Q3_K_M.gguf"

# Create logs directory if it doesn't exist
mkdir -p "/mnt/d/Projects/llama.cpp/logs"

echo "====================================================="
echo "DeltaNet Crash Reproduction Script (Linux)"
echo "====================================================="
echo ""
echo "This script will:"
echo "1. Kill any existing llama-server processes"
echo "2. Start the server with CUDA_LAUNCH_BLOCKING=1"
echo "3. Send a test request to trigger token generation"
echo "4. Capture the crash location with detailed logs"
echo ""
echo "Log files:"
echo "  Server: $SERVER_LOG"
echo "  Test:   $TEST_LOG"
echo ""

# Kill existing server instances
echo "[1/4] Killing existing llama-server processes..."
pkill -9 llama-server || true
sleep 2

# Start the server with CUDA debugging enabled
echo "[2/4] Starting llama-server with CUDA_LAUNCH_BLOCKING=1..."
echo "  (This enables synchronous CUDA execution for detailed error messages)"
echo ""

# Export CUDA debugging environment variable
export CUDA_LAUNCH_BLOCKING=1

# Start server in background, redirecting output to log
nohup "$SERVER_BIN" \
  -m "$MODEL_PATH" \
  -ngl 35 \
  -c 2048 \
  -b 512 \
  --host 0.0.0.0 \
  --port 8081 \
  > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"
echo "[3/4] Waiting for server to load model (up to 600 seconds / 10 minutes)..."
echo "  (Q3_K_M model on external drive may take longer to load)"

# Wait for server to be ready
MAX_WAIT=600
WAITED=0
SERVER_READY=false

while [ $WAITED -lt $MAX_WAIT ]; do
  sleep 10
  WAITED=$((WAITED + 10))

  # Show last line of log for progress indication
  if [ -f "$SERVER_LOG" ]; then
    LAST_LOG=$(tail -1 "$SERVER_LOG" 2>/dev/null || echo "")
    echo "  [$WAITED/${MAX_WAIT}s] Last log: ${LAST_LOG:0:80}"
  else
    echo "  Checking server status ($WAITED/$MAX_WAIT seconds)..."
  fi

  if curl -s -f http://localhost:8081/health > /dev/null 2>&1; then
    SERVER_READY=true
    echo "  Server is ready!"
    break
  fi
done

if [ "$SERVER_READY" = false ]; then
  echo ""
  echo "WARNING: Server may not be fully ready, but proceeding with test..."
  echo ""
fi

# Send test request
echo "[4/4] Sending test request to trigger token generation..."
echo ""

cat > /tmp/test-payload.json <<'EOF'
{
  "messages": [
    {
      "role": "user",
      "content": "Hi"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 10
}
EOF

if curl -s -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/test-payload.json \
  --max-time 30 \
  -o "$TEST_LOG" 2>&1; then

  echo "SUCCESS: Response received!"
  echo ""
  echo "Response:"
  cat "$TEST_LOG"
  echo ""
  echo "Test completed successfully - NO CRASH DETECTED"
else
  echo "EXPECTED RESULT: Server crashed or connection failed"
  echo ""
  echo "This is expected if the crash occurs during token generation."
fi

echo ""
echo "====================================================="
echo "Crash Analysis"
echo "====================================================="
echo ""
echo "Server logs (last 50 lines):"
echo "-----------------------------------------------------"

if [ -f "$SERVER_LOG" ]; then
  tail -50 "$SERVER_LOG"
else
  echo "No server log found at $SERVER_LOG"
fi

echo ""
echo "====================================================="
echo "Key Log Patterns to Look For:"
echo "====================================================="
echo ""
echo "Look for the LAST occurrence of these patterns:"
echo "  [delta_net_recurrent] Before v_diff"
echo "  [delta_net_recurrent] v_diff created"
echo "  [delta_net_recurrent] Before delta_mul"
echo "  [delta_net_recurrent] delta created"
echo "  [delta_net_recurrent] Before transpose"
echo "  [delta_net_recurrent] After transpose"
echo "  [delta_net_recurrent] delta_t_broadcast created"
echo "  [delta_net_recurrent] k_t_broadcast created"
echo "  [delta_net_recurrent] Before k_delta_mul"
echo "  [delta_net_recurrent] k_delta_product created"
echo "  [delta_net_recurrent] Before state_add"
echo "  [delta_net_recurrent] updated_state created"
echo "  GGML_ASSERT"
echo "  CUDA"
echo ""
echo "The crash occurs immediately AFTER the last logged message."
echo ""
echo "Full logs available at:"
echo "  Server: $SERVER_LOG"
echo "  Test:   $TEST_LOG"
echo ""

# Automatically kill server after showing logs
echo "Killing server (PID: $SERVER_PID)..."
kill -9 $SERVER_PID 2>/dev/null || true
echo "Done."
