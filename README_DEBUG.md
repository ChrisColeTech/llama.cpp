# Qwen3-Next DeltaNet Debugging

Build and test scripts for debugging Qwen3-Next-80B DeltaNet implementation.

## Quick Start

```bash
# Build llama-server and test program (logs to logs/build.log)
npm run build

# Start the server (logs to logs/server.log)
npm run start

# In another terminal, test the server (logs to logs/test.log)
npm run test

# Watch server logs in real-time
npm run logs

# Watch build logs in real-time
npm run logs:build
```

## Using PowerShell Directly

```powershell
# Build
.\build.ps1

# Start server
.\start-server.ps1

# Test server (in another terminal)
.\test-server.ps1

# Watch logs
Get-Content logs\server.log -Wait -Tail 50
```

## Log Files

All logs are in `C:\Projects\llama.cpp-v2\logs\`:

- `build.log` - CMake build output
- `server.log` - Server output and diagnostics
- `test.log` - Test request/response logs

## Configuration

### Server Settings

- **Model**: `D:\Projects\qwen_quant\converted\q3\qwen3_80b-Q3_K_M.gguf`
- **Port**: 8081
- **GPU Layers**: 35
- **Context**: 2048
- **Batch**: 512

Edit `start-server.ps1` to change settings.

## Debugging Workflow

1. Edit source files in `src/models/llm_build_qwen3next.cpp`
2. `npm run build`
3. `npm run start`
4. Wait ~60 seconds for model load
5. `npm run test`
6. Check `logs/server.log` for debug output

## Current Issue

Crash at `llm_build_qwen3next.cpp:536` during token generation:
```
GGML_ASSERT(ggml_can_repeat(b, a)) failed
```

See `D:\Projects\qwen_quant\docs\summaries\deltanet_gqa_debug_handoff.md` for details.

## Useful Commands

```powershell
# Kill stuck server
Get-Process llama-server | Stop-Process -Force

# Clear logs
Remove-Item logs\*.log

# Filter for DeltaNet debug messages
Get-Content logs\server.log -Wait -Tail 100 | Select-String "delta_net"

# Search for errors
Select-String -Path logs\server.log -Pattern "error|ASSERT|failed"
```
