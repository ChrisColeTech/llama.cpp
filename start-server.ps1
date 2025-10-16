# Start llama-server
# Logs output to server.log

$ErrorActionPreference = "Stop"
$LogFile = "C:\Projects\llama.cpp-v2\logs\server.log"
$ModelPath = "D:\Projects\qwen_quant\converted\q2\qwen3_80b-Q2_K.gguf"
$ServerExe = ".\bin\bin\Release\llama-server.exe"

# Create logs directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "C:\Projects\llama.cpp-v2\logs" | Out-Null

Write-Host "Starting llama-server..."
Write-Host "Log file: $LogFile"
Write-Host "Server will be available at http://localhost:8081"
Write-Host ""
Write-Host "Press Ctrl+C to stop the server"
Write-Host ""

& $ServerExe `
  -m $ModelPath `
  -ngl 35 `
  -c 2048 `
  -b 512 `
  --host 0.0.0.0 `
  --port 8081 `
  2>&1 | Out-File -FilePath $LogFile
