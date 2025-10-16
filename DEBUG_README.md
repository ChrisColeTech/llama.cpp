# Qwen3-Next DeltaNet GQA Debugging Guide

## Overview

This document describes the test-driven debugging approach for fixing the DeltaNet GQA (Grouped Query Attention) crash in the Qwen3-Next-80B model implementation.

## Objective

**Primary Goal**: Fix the crash during token generation in DeltaNet recurrent mode and enable successful inference with the Qwen3-Next-80B model.

**What We're Testing**:
1. Verify the v_t reshape fix resolves the dimension mismatch at line 569
2. Identify any subsequent dimension mismatches in the pipeline
3. Test full token generation sequence until we get successful output

**Success Criteria**:
- Server starts without crashing during model load
- First token generation completes without GGML_ASSERT failures
- Multi-token generation works correctly
- Model produces coherent text output

## Problem Summary

- **Model**: Qwen3-Next-80B (79.67B parameters)
- **Issue**: Crash during token generation (recurrent mode) in DeltaNet linear attention layers
- **Root Cause**: Dimension mismatch in GQA where num_key_heads (H_k=16) != num_value_heads (H_v=32)
- **Crash Location**: `delta_net_recurrent` function during tensor operations
- **Known Issue**: v_t tensor is `[1, 128, 32, 1]` but needs to be `[128, 1, 32, 1]` to match kv_memory

## Test-Driven Debugging Methodology

### Philosophy

**Test First, Build Second**

Before making any code changes and rebuilding (which takes time), we:
1. Reproduce the crash with comprehensive logging
2. Capture the exact crash location
3. Verify our hypothesis
4. Make targeted fixes
5. Rebuild and test
6. Repeat if necessary

This approach saves time by ensuring we understand the problem before attempting fixes.

### Workflow

```
┌─────────────────────────────────────────┐
│ 1. Run Crash Reproduction Test         │
│    npm run test:crash                   │
│    ↓                                    │
│    Captures detailed logs showing       │
│    exactly where crash occurs           │
└─────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 2. Analyze Crash Logs                  │
│    npm run logs:crash                   │
│    ↓                                    │
│    Find last successful operation       │
│    before crash                         │
└─────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 3. Make Targeted Code Changes          │
│    Edit llm_build_qwen3next.cpp         │
│    ↓                                    │
│    Add fixes based on crash analysis   │
└─────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 4. Rebuild with New Changes            │
│    npm run build                        │
│    ↓                                    │
│    Compile with enhanced logging        │
└─────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 5. Test Again                          │
│    npm run test:crash                   │
│    ↓                                    │
│    Verify fix OR find next crash point  │
└─────────────────────────────────────────┘
             ↓
        Repeat if needed
```

## NPM Scripts

### Testing Commands

| Command | Purpose |
|---------|---------|
| `npm run test:crash` | Reproduce crash with detailed logging |
| `npm run test` | Quick server health test |
| `npm run logs:crash` | View crash reproduction logs |

### Build Commands

| Command | Purpose |
|---------|---------|
| `npm run build` | Build llama-server with CUDA support |
| `npm run logs:build` | View build logs |

### Runtime Commands

| Command | Purpose |
|---------|---------|
| `npm run start` | Start llama-server |
| `npm run logs` | View server logs (live tail) |

## Comprehensive Logging

### Log Patterns to Look For

The code contains extensive logging at every tensor operation in `delta_net_recurrent`:

```
[delta_net_recurrent] Entry: S_k=128, H_k=16, S_v=128, H_v=32, n_tokens=1, n_seqs=1
[delta_net_recurrent] q dims: [128, 16, 1, 1]
[delta_net_recurrent] Before v_diff - v_t: [1, 128, 32, 1], kv_memory: [128, 1, 32, 1]
[delta_net_recurrent] v_t_reshaped: [128, 1, 32, 1]
[delta_net_recurrent] v_diff created: [128, 1, 32, 1]
[delta_net_recurrent] Before delta_mul - v_diff: [128, 1, 32, 1], beta_t: [1, 1, 32, 1]
[delta_net_recurrent] delta created: [128, 1, 32, 1]
...
GGML_ASSERT(ggml_can_repeat(b, a)) failed  ← CRASH POINT
```

### Reading Crash Logs

1. **Find the last successful operation**: Look for the last `[delta_net_recurrent] ... created` message
2. **Identify the failing operation**: The crash occurs immediately after the last logged message
3. **Check dimensions**: Compare tensor dimensions before and after operations
4. **Verify broadcasting**: Ensure `ggml_can_repeat` compatibility for broadcast operations

## Common Dimension Mismatch Patterns

### GQA Head Dimension Mismatches

- **Problem**: H_k (16) != H_v (32)
- **Solution**: Use `ggml_repeat_4d` to expand key/query heads to match value heads

### Reshape vs Permute

- **Reshape**: Changes dimensions but preserves total elements
- **Permute**: Swaps dimensions without changing underlying data layout
- **Key Point**: After permute, always use `ggml_cont` to make tensor contiguous

### Broadcasting Requirements

GGML's `ggml_can_repeat(b, a)` requires:
- For each dimension i: `b->ne[i] == a->ne[i]` OR `b->ne[i] == 1`
- Use `ggml_repeat_4d` to broadcast dimension-1 tensors to target shape

## File Locations

### Source Code
- `C:\Projects\llama.cpp-v2\src\models\llm_build_qwen3next.cpp` - Main implementation
- `C:\Projects\llama.cpp-v2\src\models\llm_build_qwen3next.h` - Header file

### Scripts
- `build.ps1` - Build script with CMake/CUDA configuration
- `start-server.ps1` - Server startup script
- `test-server.ps1` - Quick health test
- `reproduce-crash.ps1` - Comprehensive crash reproduction test

### Logs
- `C:\Projects\llama.cpp-v2\logs\build.log` - Build output
- `C:\Projects\llama.cpp-v2\logs\server.log` - Server runtime logs
- `C:\Projects\llama.cpp-v2\logs\reproduce-crash.log` - Crash test logs
- `C:\Projects\llama.cpp-v2\logs\test.log` - Quick test output

## Model Configuration

- **Model Path**: `D:\Projects\qwen_quant\converted\q3\qwen3_80b-Q3_K_M.gguf`
- **Quantization**: Q3_K_M (3-bit quantization)
- **GPU Layers**: 35 (split across RTX 5090 + RTX 4070 Ti Super)
- **Context Size**: 2048 tokens
- **Batch Size**: 512

## Architecture Details

### DeltaNet Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| S_k (State Key Dim) | 128 | Key state dimension |
| S_v (State Value Dim) | 128 | Value state dimension |
| H_k (Key Heads) | 16 | Number of key heads |
| H_v (Value Heads) | 32 | Number of value heads |
| head_v_dim | 128 | Value head dimension (d_inner / num_v_heads) |

### Two Modes

1. **Batch/Prompt Mode** (`delta_net`): Processes multiple tokens with chunk-based computation
2. **Recurrent Mode** (`delta_net_recurrent`): Processes one token at a time during generation

## Current Implementation Status

### ✅ Completed

1. **Enhanced Logging** (`llm_build_qwen3next.cpp:563-646`)
   - Added comprehensive dimension logging after every tensor operation
   - Logs show before/after dimensions for all reshapes, broadcasts, and operations
   - Will pinpoint exact crash location down to the specific line

2. **Test Infrastructure**
   - Created `reproduce-crash.ps1` for automated crash reproduction
   - Added npm scripts: `test:crash`, `logs:crash`
   - Script handles server lifecycle, model loading, and test request

3. **v_t Reshape Fix** (`llm_build_qwen3next.cpp:569`)
   - Applied `ggml_reshape_4d` to transform v_t from `[1, 128, 32, 1]` to `[128, 1, 32, 1]`
   - Matches kv_memory dimensions for `ggml_sub` operation
   - Hypothesis: This should resolve the original `ggml_can_repeat` assertion failure

4. **GQA Head Expansion** (`llm_build_qwen3next.cpp:425-448`)
   - Already implemented: Repeats q and k from H_k=16 to H_v=32 at function entry
   - Ensures head count compatibility throughout the function

5. **Documentation**
   - `DEBUG_README.md` - Complete debugging guide with workflow and methodology
   - Test-driven approach documented with expected outcomes
   - Command reference and troubleshooting guide

### 🔄 Ready to Test

**What We're Testing Now**:
- Does the v_t reshape fix resolve the crash at line 572 (`ggml_sub`)?
- If yes: Does token generation complete successfully?
- If no: What's the next dimension mismatch in the pipeline?

**How to Test**:
```powershell
npm run test:crash
```

### ⏳ Pending Verification

- Full token generation sequence (not just graph building)
- Multi-token generation with state propagation
- Performance validation
- Cleanup of debug logging

## Example Debugging Session

```powershell
# 1. Reproduce the crash and capture logs
npm run test:crash

# 2. Analyze the crash location
npm run logs:crash

# Output shows:
# [delta_net_recurrent] Before v_diff - v_t: [1, 128, 32, 1], kv_memory: [128, 1, 32, 1]
# GGML_ASSERT(ggml_can_repeat(b, a)) failed
#
# Conclusion: v_t needs reshaping from [1, 128, 32, 1] to [128, 1, 32, 1]

# 3. Apply fix in llm_build_qwen3next.cpp
# (add ggml_reshape_4d operation)

# 4. Rebuild
npm run build

# 5. Test again
npm run test:crash

# 6. Check results - either:
#    - SUCCESS: No crash, token generation works
#    - NEW CRASH: Different location, repeat process
```

## Troubleshooting

### Build Fails

```powershell
# Check build logs
npm run logs:build

# Common issues:
# - CUDA version mismatch
# - Missing CMake configuration
# - Visual Studio not found
```

### Server Won't Start

```powershell
# Kill existing processes
taskkill /F /IM llama-server.exe

# Check if port 8081 is in use
netstat -ano | findstr :8081

# Start with logs
npm run start
npm run logs
```

### Crash Test Hangs

The crash test script waits 60 seconds for model loading. For this 80B model:
- First load: ~2-3 minutes (model file read + GPU upload)
- Subsequent loads: ~1-2 minutes (cached)

If hanging, check server logs for loading progress.

## References

- **Handoff Document**: `D:\Projects\qwen_quant\docs\summaries\deltanet_gqa_debug_handoff.md`
- **GGML Tensor Docs**: See llama.cpp repository for tensor operation details
- **DeltaNet Paper**: Linear attention mechanism used in Qwen3-Next

## Immediate Next Steps

### 1. Run Crash Reproduction Test
```powershell
npm run test:crash
```

**Expected Outcomes**:

**Scenario A - v_t Fix Worked**:
- Test completes successfully
- Server returns a response without crashing
- Logs show all tensor operations completed
- **Action**: Test multiple tokens, then proceed to cleanup

**Scenario B - New Crash Found**:
- Server crashes at a different location
- Logs show the last successful operation before crash
- Enhanced logging pinpoints exact tensor mismatch
- **Action**: Analyze logs, apply fix, rebuild, repeat test

**Scenario C - Same Crash Location**:
- Crash still occurs at v_diff/delta operations
- v_t reshape didn't solve the issue
- **Action**: Review logs, investigate alternate reshape approach

### 2. After Successful Token Generation

Once we achieve successful single-token generation:

1. **Test Multiple Tokens**: Increase `max_tokens` to 50-100 in test script
2. **Verify State Propagation**: Check logs for state dimension consistency across tokens
3. **Test Longer Contexts**: Try prompts with different lengths
4. **Performance Validation**: Compare generation speed with expected throughput
5. **Cleanup**: Remove verbose debug logging, keep only critical error logs

### 3. Long-Term Improvements

After stable operation:

- Add unit tests for DeltaNet tensor operations
- Document GQA dimension handling for future maintenance
- Consider upstreaming fixes to llama.cpp repository
- Test with other Qwen3-Next model sizes (if available)
