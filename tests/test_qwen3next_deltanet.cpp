#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

// Test the dimension handling in delta_net_recurrent with GQA (num_k_heads != num_v_heads)
// This reproduces the crash we're seeing with Qwen3-Next-80B

int main() {
    // Model parameters from Qwen3-Next-80B
    const int64_t S_k = 128;  // ssm_d_state (key/query state dimension)
    const int64_t S_v = 128;  // same for value
    const int64_t H_k_orig = 16;  // ssm_n_group (original num_k_heads)
    const int64_t H_v = 32;   // ssm_dt_rank (num_v_heads)
    const int64_t n_tokens = 1;  // Single token generation (recurrent mode)
    const int64_t n_seqs = 1;

    printf("Testing delta_net_recurrent with GQA dimensions:\n");
    printf("  S_k=%lld, S_v=%lld\n", S_k, S_v);
    printf("  H_k_orig=%lld, H_v=%lld (GQA: %lld != %lld)\n", H_k_orig, H_v, H_k_orig, H_v);
    printf("  n_tokens=%lld, n_seqs=%lld\n\n", n_tokens, n_seqs);

    // Initialize GGML context
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024 * 1024,  // 1GB
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize GGML context\n");
        return 1;
    }

    printf("Step 1: Create input tensors with original dimensions (before repeat)\n");
    // Input tensors with original H_k dimensions
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_k, H_k_orig, n_tokens, n_seqs);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_k, H_k_orig, n_tokens, n_seqs);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H_v, n_tokens, n_seqs);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, H_v, n_tokens, n_seqs, 1);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, H_v, 1, n_tokens, n_seqs);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v * H_v, 1, n_seqs);

    printf("  q: [%lld, %lld, %lld, %lld]\n", q->ne[0], q->ne[1], q->ne[2], q->ne[3]);
    printf("  k: [%lld, %lld, %lld, %lld]\n", k->ne[0], k->ne[1], k->ne[2], k->ne[3]);
    printf("  v: [%lld, %lld, %lld, %lld]\n", v->ne[0], v->ne[1], v->ne[2], v->ne[3]);

    // Simulate the GQA repeat that happens in delta_net_recurrent
    printf("\nStep 2: Apply GQA repeat (H_k: %lld -> %lld)\n", H_k_orig, H_v);
    int64_t H_k = H_k_orig;  // Start with original

    if (H_k != H_v) {
        assert(H_v % H_k == 0);
        printf("  Repeating q and k to match H_v...\n");
        q = ggml_repeat_4d(ctx, q, S_k, H_v, n_tokens, n_seqs);
        k = ggml_repeat_4d(ctx, k, S_k, H_v, n_tokens, n_seqs);
        H_k = H_v;  // Update H_k
        printf("  After repeat - H_k updated to: %lld\n", H_k);
    }

    printf("  q after repeat: [%lld, %lld, %lld, %lld]\n", q->ne[0], q->ne[1], q->ne[2], q->ne[3]);
    printf("  k after repeat: [%lld, %lld, %lld, %lld]\n", k->ne[0], k->ne[1], k->ne[2], k->ne[3]);

    // Now simulate the recurrent computation path
    printf("\nStep 3: Reshape for recurrent computation\n");

    // Permute: [S_k, H_k, n_tokens, n_seqs] -> [S_k, n_tokens, H_k, n_seqs]
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    printf("  After permute:\n");
    printf("    q: [%lld, %lld, %lld, %lld]\n", q->ne[0], q->ne[1], q->ne[2], q->ne[3]);
    printf("    k: [%lld, %lld, %lld, %lld]\n", k->ne[0], k->ne[1], k->ne[2], k->ne[3]);

    // Create token views - THIS IS WHERE THE BUG MANIFESTS
    printf("\nStep 4: Create views for token processing (using H_k=%lld)\n", H_k);

    ggml_tensor * q_tokens = ggml_cont_4d(ctx, q, n_tokens, S_k, H_k, n_seqs);
    ggml_tensor * k_tokens = ggml_cont_4d(ctx, k, n_tokens, S_k, H_k, n_seqs);
    ggml_tensor * v_tokens = ggml_cont_4d(ctx, v, n_tokens, S_v, H_k, n_seqs);

    printf("  q_tokens: [%lld, %lld, %lld, %lld]\n", q_tokens->ne[0], q_tokens->ne[1], q_tokens->ne[2], q_tokens->ne[3]);
    printf("  k_tokens: [%lld, %lld, %lld, %lld]\n", k_tokens->ne[0], k_tokens->ne[1], k_tokens->ne[2], k_tokens->ne[3]);
    printf("  v_tokens: [%lld, %lld, %lld, %lld]\n", v_tokens->ne[0], v_tokens->ne[1], v_tokens->ne[2], v_tokens->ne[3]);

    // Reshape state
    ggml_tensor * state_reshaped = ggml_cont_4d(ctx, state, S_v, S_v, H_k, n_seqs);
    printf("  state: [%lld, %lld, %lld, %lld]\n", state_reshaped->ne[0], state_reshaped->ne[1], state_reshaped->ne[2], state_reshaped->ne[3]);

    // Single token views (n_tokens == 1, so these are just assignments)
    ggml_tensor * q_t = q_tokens;
    ggml_tensor * k_t = k_tokens;
    ggml_tensor * v_t = v_tokens;

    printf("\nStep 5: Test the critical matmul operations\n");

    // Simulate gated_state (for simplicity, just use state_reshaped)
    ggml_tensor * gated_state_reshaped = state_reshaped;
    printf("  gated_state_reshaped: [%lld, %lld, %lld, %lld]\n",
           gated_state_reshaped->ne[0], gated_state_reshaped->ne[1],
           gated_state_reshaped->ne[2], gated_state_reshaped->ne[3]);

    // THIS IS THE OPERATION THAT CRASHES
    printf("\n  Testing: k_t_reshaped = permute(k_t, 1, 0, 2, 3)\n");
    printf("    k_t: [%lld, %lld, %lld, %lld]\n", k_t->ne[0], k_t->ne[1], k_t->ne[2], k_t->ne[3]);

    ggml_tensor * k_t_reshaped = ggml_cont(ctx, ggml_permute(ctx, k_t, 1, 0, 2, 3));
    printf("    k_t_reshaped: [%lld, %lld, %lld, %lld]\n",
           k_t_reshaped->ne[0], k_t_reshaped->ne[1], k_t_reshaped->ne[2], k_t_reshaped->ne[3]);

    printf("\n  Testing: kv_memory = mul_mat(gated_state_reshaped, k_t_reshaped)\n");
    printf("    A: [%lld, %lld, %lld, %lld]\n",
           gated_state_reshaped->ne[0], gated_state_reshaped->ne[1],
           gated_state_reshaped->ne[2], gated_state_reshaped->ne[3]);
    printf("    B: [%lld, %lld, %lld, %lld]\n",
           k_t_reshaped->ne[0], k_t_reshaped->ne[1], k_t_reshaped->ne[2], k_t_reshaped->ne[3]);

    // This should work if dimensions are correct
    ggml_tensor * kv_memory = ggml_mul_mat(ctx, gated_state_reshaped, k_t_reshaped);
    printf("    Result: [%lld, %lld, %lld, %lld]\n",
           kv_memory->ne[0], kv_memory->ne[1], kv_memory->ne[2], kv_memory->ne[3]);

    printf("\n✓ Test passed! Dimensions are compatible.\n");

    ggml_free(ctx);
    return 0;
}
