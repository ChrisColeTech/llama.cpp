# Test llama-server with a simple prompt
# Logs output to test.log

$ErrorActionPreference = "Stop"
$LogFile = "C:\Projects\llama.cpp-v2\logs\test.log"
$ServerUrl = "http://localhost:8081/v1/chat/completions"

# Create logs directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "C:\Projects\llama.cpp-v2\logs" | Out-Null

Write-Host "Testing llama-server at $ServerUrl"
Write-Host "Log file: $LogFile"
Write-Host ""

$Body = @{
    model = "qwen3"
    messages = @(
        @{
            role = "user"
            content = "Hi"
        }
    )
    max_tokens = 10
} | ConvertTo-Json

Write-Host "Sending request..."
$Response = curl $ServerUrl `
  -Method POST `
  -ContentType "application/json" `
  -Body $Body 2>&1 | Out-File -FilePath $LogFile

Write-Host ""
Write-Host "Response logged to $LogFile"
