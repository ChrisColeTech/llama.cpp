# Script to reproduce DeltaNet crash and capture logs
# This runs BEFORE rebuilding to verify the crash location

$ErrorActionPreference = "Stop"
$ServerLog = "C:\Projects\llama.cpp-v2\logs\reproduce-crash.log"
$TestLog = "C:\Projects\llama.cpp-v2\logs\test-output.log"

# Create logs directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "C:\Projects\llama.cpp-v2\logs" | Out-Null

Write-Host "====================================================="
Write-Host "DeltaNet Crash Reproduction Script"
Write-Host "====================================================="
Write-Host ""
Write-Host "This script will:"
Write-Host "1. Kill any existing llama-server processes"
Write-Host "2. Start the server and wait for it to load"
Write-Host "3. Send a test request to trigger token generation"
Write-Host "4. Capture the crash location with detailed logs"
Write-Host ""
Write-Host "Log files:"
Write-Host "  Server: $ServerLog"
Write-Host "  Test:   $TestLog"
Write-Host ""

# Kill existing server instances
Write-Host "[1/4] Killing existing llama-server processes..."
try {
    taskkill /F /IM llama-server.exe 2>$null | Out-Null
} catch {
    # Ignore error if no process found
}
Start-Sleep -Seconds 2

# Start the server with CUDA debugging enabled
Write-Host "[2/4] Starting llama-server with CUDA_LAUNCH_BLOCKING=1..."
Write-Host "  (This enables synchronous CUDA execution for detailed error messages)"

# Set CUDA debugging environment variable (will be inherited by child process)
$env:CUDA_LAUNCH_BLOCKING = "1"

$serverProcess = Start-Process -FilePath "C:\Projects\llama.cpp-v2\bin\bin\Release\llama-server.exe" `
    -ArgumentList @(
        "-m", "D:\Projects\qwen_quant\converted\q2\qwen3_80b-Q2_K.gguf",
        "-ngl", "35",
        "-c", "2048",
        "-b", "512",
        "--host", "0.0.0.0",
        "--port", "8081"
    ) `
    -RedirectStandardOutput $ServerLog `
    -PassThru `
    -NoNewWindow

Write-Host "Server started (PID: $($serverProcess.Id))"
Write-Host "[3/4] Waiting for server to load model (60 seconds)..."

# Wait for server to be ready
$maxWaitTime = 60
$waited = 0
$serverReady = $false

while ($waited -lt $maxWaitTime) {
    Start-Sleep -Seconds 5
    $waited += 5

    Write-Host "  Checking server status ($waited/$maxWaitTime seconds)..."

    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8081/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $serverReady = $true
            Write-Host "  Server is ready!"
            break
        }
    } catch {
        # Server not ready yet, continue waiting
    }
}

if (-not $serverReady) {
    Write-Host ""
    Write-Host "WARNING: Server may not be fully ready, but proceeding with test..."
    Write-Host ""
}

# Send test request
Write-Host "[4/4] Sending test request to trigger token generation..."
Write-Host ""

$testPayload = @{
    messages = @(
        @{
            role = "user"
            content = "Hi"
        }
    )
    temperature = 0.7
    max_tokens = 10
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8081/v1/chat/completions" `
        -Method Post `
        -ContentType "application/json" `
        -Body $testPayload `
        -TimeoutSec 30

    Write-Host "SUCCESS: Response received!"
    Write-Host ""
    Write-Host "Response:"
    $response | ConvertTo-Json -Depth 10 | Out-File $TestLog
    $response | ConvertTo-Json -Depth 10
    Write-Host ""
    Write-Host "Test completed successfully - NO CRASH DETECTED"

} catch {
    Write-Host "EXPECTED RESULT: Server crashed or connection failed"
    Write-Host ""
    Write-Host "Error details:"
    Write-Host $_.Exception.Message
    Write-Host ""
    Write-Host "This is expected if the crash occurs during token generation."
    $_.Exception.Message | Out-File $TestLog
}

Write-Host ""
Write-Host "====================================================="
Write-Host "Crash Analysis"
Write-Host "====================================================="
Write-Host ""
Write-Host "Server logs (last 50 lines):"
Write-Host "-----------------------------------------------------"

if (Test-Path $ServerLog) {
    Get-Content $ServerLog -Tail 50 | Write-Host
} else {
    Write-Host "No server log found at $ServerLog"
}

Write-Host ""
Write-Host "====================================================="
Write-Host "Key Log Patterns to Look For:"
Write-Host "====================================================="
Write-Host ""
Write-Host "Look for the LAST occurrence of these patterns:"
Write-Host "  [delta_net_recurrent] Before v_diff"
Write-Host "  [delta_net_recurrent] v_diff created"
Write-Host "  [delta_net_recurrent] Before delta_mul"
Write-Host "  [delta_net_recurrent] delta created"
Write-Host "  [delta_net_recurrent] Before transpose"
Write-Host "  [delta_net_recurrent] After transpose"
Write-Host "  [delta_net_recurrent] delta_t_broadcast created"
Write-Host "  [delta_net_recurrent] k_t_broadcast created"
Write-Host "  [delta_net_recurrent] Before k_delta_mul"
Write-Host "  [delta_net_recurrent] k_delta_product created"
Write-Host "  [delta_net_recurrent] Before state_add"
Write-Host "  [delta_net_recurrent] updated_state created"
Write-Host "  GGML_ASSERT"
Write-Host ""
Write-Host "The crash occurs immediately AFTER the last logged message."
Write-Host ""
Write-Host "Full logs available at:"
Write-Host "  Server: $ServerLog"
Write-Host "  Test:   $TestLog"
Write-Host ""

# Keep server running for inspection if desired
Write-Host "Press any key to kill the server and exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
