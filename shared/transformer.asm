; ============================================================================
; transformer.asm — Full Qwen3-0.6B Transformer Forward Pass & Model Loader
;
; Implements exact Qwen3 architecture:
;   - Q8_0 tied embeddings + LM head
;   - 28 layers: RMSNorm -> Q/K/V (shared Q8_0 activation quant) -> per-head
;     QK-Norm -> split-half NeoX RoPE -> KV-cache -> GQA Attention (16Q/8KV)
;     -> O-proj + residual -> RMSNorm -> SwiGLU MLP + residual
;   - Final RMSNorm -> tied LM head logits
; ============================================================================

bits 64
default rel

%include "constants.inc"

%ifndef FAST_Q8
%define FAST_Q8 1
%endif

%ifndef BARE_METAL
extern sys_openat_ro
extern sys_close
extern sys_fstat
extern sys_mmap_file
%endif
extern sys_mmap_anon
extern sys_munmap

extern init_rope_freqs
extern compute_rope_cs
extern rope_apply_avx2
extern rmsnorm_avx2
extern softmax_avx2
extern silu_mul_avx2
extern vec_add_avx2

extern matmul_set_fp16
extern dequant_row_q8_0
extern dot_f32_avx2
extern saxpy_f32_avx2

extern tp_matmul_q8_0
%if FAST_Q8
extern tp_gqa_attn
%ifndef BARE_METAL
extern tp_matmul_batch
extern g_mm_xstride
%endif
%endif

section .text

global g_ctx
global g_use_hugepages
global model_init
global model_free
global transformer_forward
%if FAST_Q8
global gqa_attn_range
%ifndef BARE_METAL
global transformer_forward_batch
%endif
%endif

; Helper macro: align reg up to 64 bytes
%macro ALIGN64 1
    add     %1, 63
    and     %1, -64
%endmacro

; ----------------------------------------------------------------------------
; Linux:      int64 model_init(const char *path, uint64 max_seq_override)
; Bare-Metal: int64 model_init(const void *base, uint64 size, uint64 max_seq_override)
; Returns 0 on success, negative error code on failure.
; ----------------------------------------------------------------------------
model_init:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, STAT_SIZE + 8          ; 6 pushes (48) + 152 + 8(ret) = 208 (16B aligned)

    lea     rbx, [rel g_ctx]
%ifdef BARE_METAL
    mov     r15, rdi                    ; base ptr in physical RAM
    mov     r14, rsi                    ; size
    mov     r12, rdx                    ; max_seq_override
%else
    mov     r12, rsi                    ; max_seq_override

    ; Open .asmbin read-only
    call    sys_openat_ro
    test    rax, rax
    js      .err
    mov     r13, rax                    ; fd

    ; fstat to get file size
    mov     rdi, r13
    mov     rsi, rsp
    call    sys_fstat
    test    rax, rax
    js      .err_close
    mov     r14, [rsp + STAT_ST_SIZE]   ; file size

    cmp     dword [rel g_use_hugepages], 0
    jne     .huge_copy

    ; mmap entire .asmbin read-only
    mov     rdi, r14
    mov     rsi, r13
    call    sys_mmap_file
    mov     r15, rax                    ; mmap_base

.close_fd:
    ; Close fd (mmap keeps mapping alive)
    mov     rdi, r13
    call    sys_close

    test    r15, r15
    js      .err_mmap
%endif

    ; Verify header magic and version (v1 fp32 scales or v2 fp16 scales)
    cmp     dword [r15 + HDR_MAGIC], ASMBIN_MAGIC
    jne     .err_bad_hdr
    mov     eax, [r15 + HDR_VERSION]
    cmp     eax, ASMBIN_VERSION
    je      .ver_ok
    cmp     eax, ASMBIN_VERSION_V2
    jne     .err_bad_hdr
.ver_ok:
    xor     ecx, ecx
    test    dword [r15 + HDR_FLAGS], HDR_FLAG_FP16_SCALES
    setnz   cl
    cmp     eax, ASMBIN_VERSION_V2
    jb      .store_scale_mode
    mov     ecx, 1
.store_scale_mode:
    mov     [rbx + CTX_FP16_SCALES], ecx
    mov     edx, 3
    add     edx, ecx                    ; 3 for fp32 (K>>3), 4 for fp16 (K>>4)
    mov     [rbx + CTX_SCALE_SHIFT], edx
    mov     edi, ecx
    call    matmul_set_fp16

    mov     [rbx + CTX_MMAP_BASE], r15
    mov     [rbx + CTX_MMAP_SIZE], r14

    ; Load scalar dimensions from header
    mov     eax, [r15 + HDR_DIM]
    mov     [rbx + CTX_DIM], rax
    mov     eax, [r15 + HDR_HIDDEN_DIM]
    mov     [rbx + CTX_HIDDEN_DIM], rax
    mov     eax, [r15 + HDR_N_LAYERS]
    mov     [rbx + CTX_N_LAYERS], rax
    mov     eax, [r15 + HDR_N_HEADS]
    mov     [rbx + CTX_N_HEADS], rax
    mov     eax, [r15 + HDR_N_KV_HEADS]
    mov     [rbx + CTX_N_KV_HEADS], rax
    mov     eax, [r15 + HDR_HEAD_DIM]
    mov     [rbx + CTX_HEAD_DIM], rax

    ; q_dim = n_heads * head_dim; kv_dim = n_kv_heads * head_dim
    mov     rax, [rbx + CTX_N_HEADS]
    imul    rax, [rbx + CTX_HEAD_DIM]
    mov     [rbx + CTX_Q_DIM], rax
    mov     rax, [rbx + CTX_N_KV_HEADS]
    imul    rax, [rbx + CTX_HEAD_DIM]
    mov     [rbx + CTX_KV_DIM], rax

    mov     eax, [r15 + HDR_VOCAB_SIZE]
    mov     [rbx + CTX_VOCAB_SIZE], rax
    mov     eax, [r15 + HDR_MAX_SEQ_LEN]
    test    r12, r12
    jz      .use_hdr_seq
    cmp     r12, rax
    ja      .use_hdr_seq
    mov     rax, r12
.use_hdr_seq:
    mov     [rbx + CTX_MAX_SEQ], rax

    mov     rax, [r15 + HDR_LAYER_STRIDE]
    mov     [rbx + CTX_LAYER_STRIDE], rax
    mov     eax, [r15 + HDR_HT_MASK]
    mov     [rbx + CTX_HT_MASK], rax
    mov     eax, [r15 + HDR_EOS_1]
    mov     [rbx + CTX_EOS_1], rax
    mov     eax, [r15 + HDR_EOS_2]
    mov     [rbx + CTX_EOS_2], rax
    mov     eax, [r15 + HDR_RMS_EPS]
    mov     [rbx + CTX_RMS_EPS], eax
    mov     eax, [r15 + HDR_ROPE_THETA]
    mov     [rbx + CTX_ROPE_THETA], eax

    ; Resolve pointers into mmap'd .asmbin
    mov     rax, [r15 + HDR_OFF_TOKENIZER]
    add     rax, r15                    ; tok_base
    mov     rcx, [rax + TOKH_OFF_B2T]
    add     rcx, rax
    mov     [rbx + CTX_B2T_PTR], rcx
    mov     rcx, [rax + TOKH_OFF_OFFSETS]
    add     rcx, rax
    mov     [rbx + CTX_TOK_OFFS_PTR], rcx
    mov     rcx, [rax + TOKH_OFF_BYTES]
    add     rcx, rax
    mov     [rbx + CTX_TOK_BYTES_PTR], rcx
    mov     rcx, [rax + TOKH_OFF_HT]
    add     rcx, rax
    mov     [rbx + CTX_HT_PTR], rcx

    mov     rax, [r15 + HDR_OFF_EMBED]
    add     rax, r15
    mov     [rbx + CTX_EMB_PTR], rax
    mov     rax, [r15 + HDR_OFF_LAYERS]
    add     rax, r15
    mov     [rbx + CTX_LAYERS_PTR], rax
    mov     rax, [r15 + HDR_OFF_FINAL_NORM]
    add     rax, r15
    mov     [rbx + CTX_FINAL_NORM_PTR], rax

    ; Compute single anonymous mmap arena size for all activation buffers + KV cache
    ; First pass: compute exact aligned offsets (stored temporarily as relative offsets!)
    xor     rbp, rbp                    ; running offset = 0

    %macro ALLOC_BUF 2
        ALIGN64 rbp
        mov     [rbx + %1], rbp
        add     rbp, %2
    %endmacro

    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2                      ; dim * 4
    ALLOC_BUF CTX_BUF_X, rax
    ALLOC_BUF CTX_BUF_XB, rax

    ; Place buf_q, buf_k, buf_v CONTIGUOUSLY so Q+K+V execute in ONE fused matmul!
    mov     rax, [rbx + CTX_Q_DIM]
    shl     rax, 2                      ; q_dim * 4
    ALLOC_BUF CTX_BUF_Q, rax
    mov     rax, [rbx + CTX_KV_DIM]
    shl     rax, 2                      ; kv_dim * 4
    ALLOC_BUF CTX_BUF_K, rax
    ALLOC_BUF CTX_BUF_V, rax

    mov     rax, [rbx + CTX_Q_DIM]
    shl     rax, 2                      ; q_dim * 4
    ALLOC_BUF CTX_BUF_ATT_OUT, rax

    mov     rax, [rbx + CTX_N_HEADS]
    imul    rax, [rbx + CTX_MAX_SEQ]
    shl     rax, 2                      ; n_heads * max_seq * 4 (private per-head scores)
    ALLOC_BUF CTX_BUF_ATT_SCR, rax

    ; Place buf_gate and buf_up CONTIGUOUSLY so gate+up execute in ONE fused matmul!
    mov     rax, [rbx + CTX_HIDDEN_DIM]
    shl     rax, 2                      ; hidden_dim * 4
    ALLOC_BUF CTX_BUF_GATE, rax
    ALLOC_BUF CTX_BUF_UP, rax

    mov     rax, [rbx + CTX_VOCAB_SIZE]
    shl     rax, 2                      ; vocab_size * 4
    ALLOC_BUF CTX_BUF_LOGITS, rax
    ALLOC_BUF CTX_BUF_PROBS, rax

    ; Activation Q8_0 buffers (max K = max(dim, q_dim, hidden_dim) <= hidden_dim + 4096)
    mov     rax, [rbx + CTX_HIDDEN_DIM]
    cmp     rax, [rbx + CTX_Q_DIM]
    jae     .kmax_ok
    mov     rax, [rbx + CTX_Q_DIM]
.kmax_ok:
    ALLOC_BUF CTX_BUF_XQ, rax           ; max_K int8
    mov     rcx, rax
    shr     rcx, 3                      ; (max_K / 32) * 4
    ALLOC_BUF CTX_BUF_XS, rcx

    ; RoPE buffers: inv_freq (head_dim/2 * 8), cos (head_dim/2 * 4), sin (head_dim/2 * 4)
    mov     rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2                      ; (head_dim/2) * 8 = head_dim * 4
    ALLOC_BUF CTX_BUF_INV_FREQ, rax
    shr     rax, 1                      ; (head_dim/2) * 4
    ALLOC_BUF CTX_BUF_COS, rax
    ALLOC_BUF CTX_BUF_SIN, rax

    ; Prompt & gen token buffers
    ALLOC_BUF CTX_BUF_PROMPT, MAX_PROMPT_TOKENS * 4
    ALLOC_BUF CTX_BUF_GEN,    MAX_GEN_TOKENS * 4

    ; KV-cache: each is n_layers * max_seq * kv_dim * 4 bytes
    mov     rax, [rbx + CTX_N_LAYERS]
    imul    rax, [rbx + CTX_MAX_SEQ]
    imul    rax, [rbx + CTX_KV_DIM]
    shl     rax, 2
    ALLOC_BUF CTX_BUF_K_CACHE, rax
    ALLOC_BUF CTX_BUF_V_CACHE, rax

    ; Batched prefill buffers (MAX_BATCH tokens)
    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2
    imul    rax, rax, MAX_BATCH
    ALLOC_BUF CTX_BUF_BX, rax
    ALLOC_BUF CTX_BUF_BXB, rax
    mov     rax, [rbx + CTX_KV_DIM]
    add     rax, rax
    add     rax, [rbx + CTX_Q_DIM]
    shl     rax, 2
    imul    rax, rax, MAX_BATCH
    ALLOC_BUF CTX_BUF_BQKV, rax
    mov     rax, [rbx + CTX_Q_DIM]
    shl     rax, 2
    imul    rax, rax, MAX_BATCH
    ALLOC_BUF CTX_BUF_BATT, rax
    mov     rax, [rbx + CTX_HIDDEN_DIM]
    shl     rax, 3                      ; [gate | up] per token
    imul    rax, rax, MAX_BATCH
    ALLOC_BUF CTX_BUF_BGU, rax
    mov     rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 1                      ; (head_dim / 2) floats per token
    imul    rax, rax, MAX_BATCH
    ALLOC_BUF CTX_BUF_BCOS, rax
    ALLOC_BUF CTX_BUF_BSIN, rax

    ALIGN64 rbp
    mov     [rbx + CTX_ARENA_SIZE], rbp

    ; Allocate arena via anonymous mmap
    mov     rdi, rbp
    call    sys_mmap_anon
    test    rax, rax
    js      .err_arena
    mov     [rbx + CTX_ARENA_BASE], rax

    ; Relocate all 20 CTX_BUF_* offsets -> absolute pointers
    lea     rcx, [rbx + CTX_BUF_X]
    mov     edx, 20
.reloc_lp:
    add     [rcx], rax
    add     rcx, 8
    dec     edx
    jnz     .reloc_lp
    lea     rcx, [rbx + CTX_BUF_BX]     ; and the 7 batched prefill buffers
    mov     edx, 7
.reloc2_lp:
    add     [rcx], rax
    add     rcx, 8
    dec     edx
    jnz     .reloc2_lp

    ; Precompute exact FP64 RoPE inverse frequencies
    mov     rdi, [rbx + CTX_BUF_INV_FREQ]
    mov     rsi, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_ROPE_THETA]
    call    init_rope_freqs

    xor     eax, eax                    ; success!
    jmp     .ret

.err_arena:
    mov     rdi, r15
    mov     rsi, r14
    call    sys_munmap
    mov     rax, -4
    jmp     .ret
.err_bad_hdr:
    mov     rdi, r15
    mov     rsi, r14
    call    sys_munmap
    mov     rax, -3
    jmp     .ret
%ifndef BARE_METAL
; Copy the weights into anonymous memory aligned to 2 MiB and marked
; MADV_HUGEPAGE: transparent huge pages cut TLB misses while the 614 MiB of
; weights stream through every token (a file mapping stays on 4 KiB pages).
.huge_copy:
    lea     rdi, [r14 + HUGE_2M]
    call    sys_mmap_anon
    test    rax, rax
    js      .huge_fail
    lea     r15, [rax + HUGE_2M - 1]
    and     r15, -HUGE_2M               ; 2 MiB aligned base
    mov     rdi, r15
    lea     rsi, [r14 + HUGE_2M - 1]
    and     rsi, -HUGE_2M
    mov     edx, MADV_HUGEPAGE
    mov     eax, SYS_madvise
    syscall
    xor     ebp, ebp                    ; bytes read
.huge_read:
    cmp     rbp, r14
    jae     .close_fd
    mov     edi, r13d
    lea     rsi, [r15 + rbp]
    mov     rdx, r14
    sub     rdx, rbp
    mov     eax, SYS_read
    syscall
    cmp     rax, -4                     ; EINTR
    je      .huge_read
    test    rax, rax
    jle     .huge_read_fail
    add     rbp, rax
    jmp     .huge_read
.huge_read_fail:
    mov     r15, -5
    jmp     .close_fd
.huge_fail:
    mov     r15, rax
    jmp     .close_fd
.err_mmap:
    mov     rax, -2
    jmp     .ret
.err_close:
    mov     rdi, r13
    call    sys_close
.err:
    mov     rax, -1
%endif
.ret:
    add     rsp, STAT_SIZE + 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void model_free(void)
; Unmaps arena and .asmbin file cleanly.
; ----------------------------------------------------------------------------
model_free:
    push    rbx
    lea     rbx, [rel g_ctx]
    mov     rdi, [rbx + CTX_ARENA_BASE]
    test    rdi, rdi
    jz      .unmap_model
    mov     rsi, [rbx + CTX_ARENA_SIZE]
    call    sys_munmap
.unmap_model:
    mov     rdi, [rbx + CTX_MMAP_BASE]
    test    rdi, rdi
    jz      .done
    mov     rsi, [rbx + CTX_MMAP_SIZE]
    call    sys_munmap
.done:
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void transformer_forward(uint32 token_id, uint64 pos, int compute_logits)
; rdi = token_id, rsi = pos, rdx = compute_logits (0 during non-final prefill)
; ----------------------------------------------------------------------------
transformer_forward:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    ; Reserve 128 bytes of stack locals (6 pushes=48 + 128 + 8(ret)=184 -> use 136 for 16B align!)
    sub     rsp, 136
    ; Stack locals:
    ;   [rsp + 0]  = pos
    ;   [rsp + 8]  = compute_logits
    ;   [rsp + 16] = layer_idx l
    ;   [rsp + 24] = p_in_norm
    ;   [rsp + 32] = p_q_norm
    ;   [rsp + 40] = p_k_norm
    ;   [rsp + 48] = p_post_norm
    ;   [rsp + 56] = p_q_proj
    ;   [rsp + 64] = p_k_proj
    ;   [rsp + 72] = p_v_proj
    ;   [rsp + 80] = p_o_proj
    ;   [rsp + 88] = p_gate_proj
    ;   [rsp + 96] = p_up_proj
    ;   [rsp + 104]= p_down_proj
    ;   [rsp + 112]= att_scale (float32)
    ;   [rsp + 116]= gqa_ratio (uint32)
    ;   [rsp + 120]= layer_kv_base_bytes (l * max_seq * kv_dim * 4)

    mov     [rsp + 0], rsi              ; pos
    mov     [rsp + 8], rdx              ; compute_logits
    lea     rbx, [rel g_ctx]

    ; Precompute att_scale = 1.0f / sqrt(head_dim) and gqa_ratio = n_heads / n_kv_heads
    mov     rax, [rbx + CTX_HEAD_DIM]
    vcvtsi2ss xmm0, xmm0, rax
    vsqrtss xmm0, xmm0, xmm0
    vmovss  xmm1, [rel t_one]
    vdivss  xmm1, xmm1, xmm0
    vmovss  [rsp + 112], xmm1

    mov     rax, [rbx + CTX_N_HEADS]
    xor     edx, edx
    div     qword [rbx + CTX_N_KV_HEADS]
    mov     [rsp + 116], eax

    ; 1. Embedding lookup: dequantize row token_id of embed_tokens into buf_x
    mov     rax, [rbx + CTX_DIM]
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    mov     rdx, rax
    shr     rdx, cl
    add     rdx, rax                    ; emb_row_stride = dim + (dim >> scale_shift)
    imul    rdi, rdx
    add     rdi, [rbx + CTX_EMB_PTR]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, rax                    ; dim
    call    dequant_row_q8_0

    ; 2. Compute RoPE cos/sin tables ONCE for this position (reused by all 28 layers!)
    mov     rdi, [rbx + CTX_BUF_COS]
    mov     rsi, [rbx + CTX_BUF_SIN]
    mov     rdx, [rbx + CTX_BUF_INV_FREQ]
    mov     rcx, [rbx + CTX_HEAD_DIM]
    shr     rcx, 1                      ; half_dim
    mov     r8,  [rsp + 0]              ; pos
    call    compute_rope_cs

    ; 3. Transformer Layer Loop (l = 0 .. n_layers - 1)
    mov     qword [rsp + 16], 0

.layer_lp:
    mov     rax, [rsp + 16]
    cmp     rax, [rbx + CTX_N_LAYERS]
    jae     .layers_done

    ; Compute all 11 tensor pointers for layer l (matching export_weights.py layout)
    imul    rax, [rbx + CTX_LAYER_STRIDE]
    add     rax, [rbx + CTX_LAYERS_PTR]

    mov     [rsp + 24], rax             ; in_norm
    mov     rcx, [rbx + CTX_DIM]
    shl     rcx, 2
    add     rax, rcx
    mov     [rsp + 32], rax             ; q_norm
    mov     rcx, [rbx + CTX_HEAD_DIM]
    shl     rcx, 2
    add     rax, rcx
    mov     [rsp + 40], rax             ; k_norm
    add     rax, rcx
    mov     [rsp + 48], rax             ; post_norm
    mov     rcx, [rbx + CTX_DIM]
    shl     rcx, 2
    add     rax, rcx

    ; Helper: row_stride(K) = K + (K >> scale_shift) [3 for v1 fp32, 4 for v2 fp16]
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    mov     r12, [rbx + CTX_DIM]
    mov     r13, r12
    shr     r13, cl
    add     r12, r13                    ; r12 = dim_stride (1152 v1 / 1088 v2)

    ALIGN64 rax
    mov     [rsp + 56], rax             ; q_proj [q_dim, dim]
    mov     rcx, [rbx + CTX_Q_DIM]
    imul    rcx, r12
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 64], rax             ; k_proj [kv_dim, dim]
    mov     rcx, [rbx + CTX_KV_DIM]
    imul    rcx, r12
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 72], rax             ; v_proj [kv_dim, dim]
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 80], rax             ; o_proj [dim, q_dim]
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    mov     r13, [rbx + CTX_Q_DIM]
    mov     rdx, r13
    shr     rdx, cl
    add     r13, rdx                    ; q_dim_stride (2304 v1 / 2176 v2)
    imul    r13, [rbx + CTX_DIM]
    add     rax, r13

    ALIGN64 rax
    mov     [rsp + 88], rax             ; gate_proj [hidden_dim, dim]
    mov     rcx, [rbx + CTX_HIDDEN_DIM]
    imul    rcx, r12
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 96], rax             ; up_proj [hidden_dim, dim]
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 104], rax            ; down_proj [dim, hidden_dim]

    ; --- 3a. Input RMSNorm: buf_xb = rmsnorm(buf_x, in_norm) ---
    mov     rdi, [rbx + CTX_BUF_XB]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rsp + 24]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2

    ; --- 3b. Fused Q + K + V projection in ONE parallel matmul (zero padding between q/k/v!) ---
    mov     rdi, [rsp + 56]             ; q_proj (contiguous with k_proj, v_proj)
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_Q]      ; buf_q (contiguous with buf_k, buf_v)
    mov     rcx, [rbx + CTX_Q_DIM]
    add     rcx, [rbx + CTX_KV_DIM]
    add     rcx, [rbx + CTX_KV_DIM]     ; M = q_dim + 2*kv_dim (4096)
    mov     r8,  [rbx + CTX_DIM]        ; K = dim (1024)
    call    tp_matmul_q8_0

    ; --- 3c. Qwen3 QK-Norm (RMSNorm per head over head_dim, in-place) ---
    mov     r12, [rbx + CTX_BUF_Q]
    mov     r13, [rbx + CTX_N_HEADS]
    mov     r14, [rbx + CTX_HEAD_DIM]
    shl     r14, 2                      ; head_bytes
.qnorm_lp:
    mov     rdi, r12
    mov     rsi, r12
    mov     rdx, [rsp + 32]             ; q_norm
    mov     rcx, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    add     r12, r14
    dec     r13
    jnz     .qnorm_lp

    mov     r12, [rbx + CTX_BUF_K]
    mov     r13, [rbx + CTX_N_KV_HEADS]
.knorm_lp:
    mov     rdi, r12
    mov     rsi, r12
    mov     rdx, [rsp + 40]             ; k_norm
    mov     rcx, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    add     r12, r14
    dec     r13
    jnz     .knorm_lp

    ; --- 3d. Split-half RoPE on Q and K ---
    mov     rdi, [rbx + CTX_BUF_Q]
    mov     rsi, [rbx + CTX_BUF_COS]
    mov     rdx, [rbx + CTX_BUF_SIN]
    mov     rcx, [rbx + CTX_N_HEADS]
    mov     r8,  [rbx + CTX_HEAD_DIM]
    call    rope_apply_avx2

    mov     rdi, [rbx + CTX_BUF_K]
    mov     rsi, [rbx + CTX_BUF_COS]
    mov     rdx, [rbx + CTX_BUF_SIN]
    mov     rcx, [rbx + CTX_N_KV_HEADS]
    mov     r8,  [rbx + CTX_HEAD_DIM]
    call    rope_apply_avx2

    ; --- 3e. Store K, V at contiguous cache[l, kv_h, pos, head_dim] ---
    ; layer_kv_base_bytes = l * max_seq * kv_dim * 4
    mov     rax, [rsp + 16]             ; l
    imul    rax, [rbx + CTX_MAX_SEQ]
    imul    rax, [rbx + CTX_KV_DIM]
    shl     rax, 2
    mov     [rsp + 120], rax

%if FAST_Q8
    ; Head-major contiguous KV layout: cache[l, kv_h, pos, head_dim]
    ; Eliminates 4KB L1D set aliasing & TLB thrashing across timesteps!
    mov     r14, [rbx + CTX_HEAD_DIM]
    shl     r14, 2                      ; head_bytes (512)
    mov     r15, [rbx + CTX_MAX_SEQ]
    imul    r15, r14                    ; kv_head_stride = max_seq * head_bytes
    mov     rcx, [rsp + 0]              ; pos
    imul    rcx, r14                    ; pos_off = pos * head_bytes
    add     rax, rcx                    ; layer_kv_base + pos_off

    mov     rdi, [rbx + CTX_BUF_K_CACHE]
    add     rdi, rax
    mov     rdx, [rbx + CTX_BUF_V_CACHE]
    add     rdx, rax
    mov     rsi, [rbx + CTX_BUF_K]
    mov     r8,  [rbx + CTX_BUF_V]
    mov     r12, [rbx + CTX_N_KV_HEADS]
.kv_head_lp:
    xor     r9d, r9d
.kv_copy_lp:
    cmp     r9, r14
    jae     .kv_head_next
    vmovups ymm0, [rsi + r9]
    vmovups ymm1, [r8 + r9]
    vmovups [rdi + r9], ymm0
    vmovups [rdx + r9], ymm1
    add     r9, 32
    jmp     .kv_copy_lp
.kv_head_next:
    add     rsi, r14
    add     r8,  r14
    add     rdi, r15
    add     rdx, r15
    dec     r12
    jnz     .kv_head_lp

    ; --- 3f. Parallel GQA Attention over heads h = 0 .. n_heads - 1 ---
    mov     rdi, [rsp + 120]            ; layer_kv_base
    mov     rsi, [rsp + 0]              ; pos
    mov     edx, [rsp + 116]            ; gqa_ratio (upper 32 bits)
    shl     rdx, 32
    mov     eax, [rsp + 112]            ; att_scale float32 bits (lower 32 bits)
    or      rdx, rax
    mov     rcx, [rbx + CTX_N_HEADS]
    call    tp_gqa_attn
%else
    ; Legacy layout: cache[l, pos, kv_h, head_dim]
    mov     rcx, [rsp + 0]              ; pos
    imul    rcx, [rbx + CTX_KV_DIM]
    shl     rcx, 2
    add     rax, rcx                    ; total byte offset into k_cache / v_cache

    mov     rdi, [rbx + CTX_BUF_K_CACHE]
    add     rdi, rax
    mov     rsi, [rbx + CTX_BUF_K]
    mov     rdx, [rbx + CTX_BUF_V_CACHE]
    add     rdx, rax
    mov     r8,  [rbx + CTX_BUF_V]
    mov     rcx, [rbx + CTX_KV_DIM]
    shl     rcx, 2                      ; kv_bytes
    xor     r9d, r9d
.kv_copy_lp:
    cmp     r9, rcx
    jae     .kv_copy_done
    vmovups ymm0, [rsi + r9]
    vmovups ymm1, [r8 + r9]
    vmovups [rdi + r9], ymm0
    vmovups [rdx + r9], ymm1
    add     r9, 32
    jmp     .kv_copy_lp
.kv_copy_done:

    ; --- 3f. GQA Attention over all heads h = 0 .. n_heads - 1 ---
    xor     r12d, r12d                  ; h = 0
.head_lp:
    cmp     r12, [rbx + CTX_N_HEADS]
    jae     .att_done

    ; kv_h = h / gqa_ratio
    mov     eax, r12d
    xor     edx, edx
    div     dword [rsp + 116]           ; eax = kv_h
    imul    rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2                      ; kv_h_bytes = kv_h * head_dim * 4
    mov     r13, rax                    ; r13 = kv_h_bytes

    ; q_head = buf_q + h * head_dim * 4
    mov     rax, r12
    imul    rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2
    mov     r14, [rbx + CTX_BUF_Q]
    add     r14, rax                    ; r14 = q_head

    ; Compute attention scores[0..pos]: dot(q_head, k_cache[l, t, kv_h]) * att_scale
    mov     r15, [rbx + CTX_BUF_K_CACHE]
    add     r15, [rsp + 120]            ; layer_k_base
    add     r15, r13                    ; k_ptr for t = 0
    mov     rbp, [rbx + CTX_KV_DIM]
    shl     rbp, 2                      ; step per timestep t = kv_dim * 4

    xor     ecx, ecx                    ; t = 0
    mov     r11, [rbx + CTX_BUF_ATT_SCR]
.score_lp:
    cmp     rcx, [rsp + 0]
    ja      .scores_done
    push    rcx
    push    r11                         ; 2 pushes -> 16B aligned
    mov     rdi, r14                    ; q_head
    mov     rsi, r15                    ; k_vec
    mov     rdx, [rbx + CTX_HEAD_DIM]
    call    dot_f32_avx2                ; xmm0 = dot
    pop     r11
    pop     rcx
    vmulss  xmm0, xmm0, [rsp + 112]     ; * att_scale
    vmovss  [r11 + rcx*4], xmm0
    add     r15, rbp
    inc     rcx
    jmp     .score_lp

.scores_done:
    ; Softmax scores[0..pos] (length = pos + 1)
    mov     rdi, [rbx + CTX_BUF_ATT_SCR]
    mov     rsi, [rsp + 0]
    inc     rsi
    call    softmax_avx2

    ; Zero att_head = buf_att_out + h * head_dim * 4
    mov     rax, r12
    imul    rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2
    mov     r14, [rbx + CTX_BUF_ATT_OUT]
    add     r14, rax                    ; r14 = att_head

    mov     rcx, [rbx + CTX_HEAD_DIM]
    shl     rcx, 2
    vxorps  ymm0, ymm0, ymm0
    xor     eax, eax
.zero_att:
    cmp     rax, rcx
    jae     .val_accum
    vmovups [r14 + rax], ymm0
    add     rax, 32
    jmp     .zero_att

.val_accum:
    ; Weighted sum of V vectors: att_head += probs[t] * v_cache[l, t, kv_h]
    mov     r15, [rbx + CTX_BUF_V_CACHE]
    add     r15, [rsp + 120]
    add     r15, r13                    ; v_ptr for t = 0
    xor     ecx, ecx
    mov     r11, [rbx + CTX_BUF_ATT_SCR]
.val_lp:
    cmp     rcx, [rsp + 0]
    ja      .head_next
    vmovss  xmm0, [r11 + rcx*4]         ; prob[t]
    push    rcx
    push    r11
    mov     rdi, r14                    ; att_head
    mov     rsi, r15                    ; v_vec
    mov     rdx, [rbx + CTX_HEAD_DIM]
    call    saxpy_f32_avx2
    pop     r11
    pop     rcx
    add     r15, rbp
    inc     rcx
    jmp     .val_lp

.head_next:
    inc     r12
    jmp     .head_lp
%endif

.att_done:
    ; --- 3g. Output projection + residual: buf_x += o_proj @ buf_att_out ---
    mov     rdi, [rsp + 80]             ; o_proj
    mov     rsi, [rbx + CTX_BUF_ATT_OUT]
    mov     rdx, [rbx + CTX_BUF_XB]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8,  [rbx + CTX_Q_DIM]
    call    tp_matmul_q8_0

    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2

    ; --- 3h. Post-attention RMSNorm + SwiGLU MLP + residual ---
    mov     rdi, [rbx + CTX_BUF_XB]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rsp + 48]             ; post_norm
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2

    ; Fused gate_proj + up_proj in ONE parallel matmul (contiguous in weights & arena!)
    mov     rdi, [rsp + 88]             ; gate_proj (contiguous with up_proj)
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_GATE]   ; buf_gate (contiguous with buf_up)
    mov     rcx, [rbx + CTX_HIDDEN_DIM]
    add     rcx, rcx                    ; M = 2 * hidden_dim (6144)
    mov     r8,  [rbx + CTX_DIM]        ; K = dim (1024)
    call    tp_matmul_q8_0

    ; SwiGLU: buf_gate = silu(buf_gate) * buf_up
    mov     rdi, [rbx + CTX_BUF_GATE]
    mov     rsi, [rbx + CTX_BUF_UP]
    mov     rdx, [rbx + CTX_HIDDEN_DIM]
    call    silu_mul_avx2

    ; down_proj + residual
    mov     rdi, [rsp + 104]            ; down_proj
    mov     rsi, [rbx + CTX_BUF_GATE]
    mov     rdx, [rbx + CTX_BUF_XB]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8,  [rbx + CTX_HIDDEN_DIM]
    call    tp_matmul_q8_0

    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2

    inc     qword [rsp + 16]
    jmp     .layer_lp

.layers_done:
    cmp     qword [rsp + 8], 0
    je      .done

    ; 4. Final RMSNorm + tied LM-head projection -> buf_logits
    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rbx + CTX_FINAL_NORM_PTR]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2

    mov     rdi, [rbx + CTX_EMB_PTR]    ; tied embeddings!
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rbx + CTX_BUF_LOGITS]
    mov     rcx, [rbx + CTX_VOCAB_SIZE]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0

.done:
    add     rsp, 136
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

%if FAST_Q8
; ----------------------------------------------------------------------------
; void gqa_attn_range(uint64 layer_kv_base, uint64 pos, uint64 scale_gqa_packed,
;                     uint64 h0, uint64 h1)
; rdi = layer_kv_base, rsi = pos, rdx = (gqa_ratio << 32) | att_scale_bits,
; rcx = h0, r8 = h1
; Thread-safe parallel GQA attention over contiguous cache[l, kv_h, pos, hd].
; ----------------------------------------------------------------------------
gqa_attn_range:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 40                     ; 6 pushes(48) + 40 + ret(8) = 96 (16B aligned)
    ; [rsp + 0]  = layer_kv_base
    ; [rsp + 8]  = pos
    ; [rsp + 16] = att_scale (f32)
    ; [rsp + 20] = gqa_ratio (u32)
    ; [rsp + 24] = h1
    ; [rsp + 32] = kv_head_stride (max_seq * head_dim * 4)

    mov     [rsp + 0], rdi
    mov     [rsp + 8], rsi
    mov     [rsp + 16], edx             ; att_scale bits
    shr     rdx, 32
    mov     [rsp + 20], edx             ; gqa_ratio
    mov     [rsp + 24], r8              ; h1

    lea     rbx, [rel g_ctx]
    mov     rbp, [rbx + CTX_HEAD_DIM]
    shl     rbp, 2                      ; rbp = head_bytes (step per t = 512B contiguous!)
    mov     rax, [rbx + CTX_MAX_SEQ]
    imul    rax, rbp
    mov     [rsp + 32], rax             ; kv_head_stride

    mov     r12, rcx                    ; h = h0
.h_lp:
    cmp     r12, [rsp + 24]
    jae     .h_done

    ; kv_h = h / gqa_ratio; r13 = layer_kv_base + kv_h * kv_head_stride
    mov     eax, r12d
    xor     edx, edx
    div     dword [rsp + 20]
    imul    rax, [rsp + 32]
    add     rax, [rsp + 0]
    mov     r13, rax

    ; q_head = buf_q + h * head_bytes
    mov     rax, r12
    imul    rax, rbp
    mov     r14, [rbx + CTX_BUF_Q]
    add     r14, rax

    ; Private score buffer scr_h = buf_att_scr + h * max_seq * 4
    mov     r11, r12
    imul    r11, [rbx + CTX_MAX_SEQ]
    shl     r11, 2
    add     r11, [rbx + CTX_BUF_ATT_SCR]

    ; Scores[0..pos]: dot(q_head, k_cache[l, kv_h, t]) * att_scale
    mov     r15, [rbx + CTX_BUF_K_CACHE]
    add     r15, r13
    xor     ecx, ecx
.sc_lp:
    cmp     rcx, [rsp + 8]
    ja      .sc_done
    push    rcx
    push    r11
    mov     rdi, r14
    mov     rsi, r15
    mov     rdx, [rbx + CTX_HEAD_DIM]
    call    dot_f32_avx2
    pop     r11
    pop     rcx
    vmulss  xmm0, xmm0, [rsp + 16]
    vmovss  [r11 + rcx*4], xmm0
    add     r15, rbp                    ; contiguous +512B!
    inc     rcx
    jmp     .sc_lp

.sc_done:
    push    r11
    sub     rsp, 8                      ; 16B align
    mov     rdi, r11
    mov     rsi, [rsp + 24]             ; pos (rsp+8 + 16 = rsp+24)
    inc     rsi
    call    softmax_avx2
    add     rsp, 8
    pop     r11

    ; Zero att_head = buf_att_out + h * head_bytes
    mov     rax, r12
    imul    rax, rbp
    mov     r14, [rbx + CTX_BUF_ATT_OUT]
    add     r14, rax
    vxorps  ymm0, ymm0, ymm0
    xor     eax, eax
.z_lp:
    cmp     rax, rbp
    jae     .v_acc
    vmovups [r14 + rax], ymm0
    add     rax, 32
    jmp     .z_lp

.v_acc:
    mov     r15, [rbx + CTX_BUF_V_CACHE]
    add     r15, r13
    xor     ecx, ecx
.v_lp:
    cmp     rcx, [rsp + 8]
    ja      .h_next
    vmovss  xmm0, [r11 + rcx*4]
    push    rcx
    push    r11
    mov     rdi, r14
    mov     rsi, r15
    mov     rdx, [rbx + CTX_HEAD_DIM]
    call    saxpy_f32_avx2
    pop     r11
    pop     rcx
    add     r15, rbp                    ; contiguous +512B!
    inc     rcx
    jmp     .v_lp

.h_next:
    inc     r12
    jmp     .h_lp

.h_done:
    add     rsp, 40
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret
%endif

%if FAST_Q8
%ifndef BARE_METAL
; ----------------------------------------------------------------------------
; void transformer_forward_batch(const uint32 *toks, uint64 B, uint64 pos0,
;                                int compute_logits)
; Prompt prefill of B <= MAX_BATCH tokens at positions pos0 .. pos0 + B - 1.
; Each layer streams its weights once for the whole batch (tp_matmul_batch);
; every other step runs the single-token code per token, and token t attends
; to positions 0 .. pos0 + t after all K/V of the batch are in the cache.
; The KV cache and the final logits are bit-identical to B calls of
; transformer_forward.
; ----------------------------------------------------------------------------
%define FB_TOKS      0
%define FB_POS0      8
%define FB_LOGITS    16
%define FB_LAYER     24
%define FB_IN_NORM   32
%define FB_Q_NORM    40
%define FB_K_NORM    48
%define FB_POST_NORM 56
%define FB_QKV       64
%define FB_O         72
%define FB_GU        80
%define FB_DOWN      88
%define FB_SCALE     96                  ; f32 att_scale, gqa ratio at +4
%define FB_KVBASE    104
%define FB_QKVM      112                 ; q_dim + 2 * kv_dim
%define FB_SAVE_Q    120
%define FB_SAVE_ATT  128
%define FB_ROPE      136

; Per-token loop helper: rax = t * (bytes per token of field) + buffer
%macro TOK_OFS 3                         ; dst, per-token bytes (reg/mem), buffer field
    mov     %1, %2
    imul    %1, r13
    add     %1, [rbx + %3]
%endmacro

transformer_forward_batch:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 168
    mov     [rsp + FB_TOKS], rdi
    mov     r12, rsi                    ; B
    mov     [rsp + FB_POS0], rdx
    mov     [rsp + FB_LOGITS], rcx
    lea     rbx, [rel g_ctx]

    mov     rax, [rbx + CTX_HEAD_DIM]
    vcvtsi2ss xmm0, xmm0, rax
    vsqrtss xmm0, xmm0, xmm0
    vmovss  xmm1, [rel t_one]
    vdivss  xmm1, xmm1, xmm0
    vmovss  [rsp + FB_SCALE], xmm1
    mov     rax, [rbx + CTX_N_HEADS]
    xor     edx, edx
    div     qword [rbx + CTX_N_KV_HEADS]
    mov     [rsp + FB_SCALE + 4], eax
    mov     rax, [rbx + CTX_Q_DIM]
    add     rax, [rbx + CTX_KV_DIM]
    add     rax, [rbx + CTX_KV_DIM]
    mov     [rsp + FB_QKVM], rax

    ; 1. Embeddings and RoPE tables of every token
    xor     r13d, r13d
.fb_emb:
    cmp     r13, r12
    jae     .fb_emb_done
    mov     rax, [rsp + FB_TOKS]
    mov     edi, [rax + r13*4]
    mov     rax, [rbx + CTX_DIM]
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    mov     rdx, rax
    shr     rdx, cl
    add     rdx, rax                    ; embedding row stride
    imul    rdi, rdx
    add     rdi, [rbx + CTX_EMB_PTR]
    mov     rsi, rax
    shl     rsi, 2
    imul    rsi, r13
    add     rsi, [rbx + CTX_BUF_BX]
    mov     rdx, rax
    call    dequant_row_q8_0
    mov     rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 1                      ; (head_dim / 2) * 4 bytes
    imul    rax, r13
    mov     rdi, [rbx + CTX_BUF_BCOS]
    add     rdi, rax
    mov     rsi, [rbx + CTX_BUF_BSIN]
    add     rsi, rax
    mov     rdx, [rbx + CTX_BUF_INV_FREQ]
    mov     rcx, [rbx + CTX_HEAD_DIM]
    shr     rcx, 1
    mov     r8, [rsp + FB_POS0]
    add     r8, r13
    call    compute_rope_cs
    inc     r13
    jmp     .fb_emb
.fb_emb_done:

    mov     qword [rsp + FB_LAYER], 0
.fb_layer:
    mov     rax, [rsp + FB_LAYER]
    cmp     rax, [rbx + CTX_N_LAYERS]
    jae     .fb_layers_done

    ; Tensor pointers of layer l (same layout as transformer_forward)
    imul    rax, [rbx + CTX_LAYER_STRIDE]
    add     rax, [rbx + CTX_LAYERS_PTR]
    mov     [rsp + FB_IN_NORM], rax
    mov     rcx, [rbx + CTX_DIM]
    shl     rcx, 2
    add     rax, rcx
    mov     [rsp + FB_Q_NORM], rax
    mov     rcx, [rbx + CTX_HEAD_DIM]
    shl     rcx, 2
    add     rax, rcx
    mov     [rsp + FB_K_NORM], rax
    add     rax, rcx
    mov     [rsp + FB_POST_NORM], rax
    mov     rcx, [rbx + CTX_DIM]
    shl     rcx, 2
    add     rax, rcx
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    mov     r14, [rbx + CTX_DIM]
    mov     r15, r14
    shr     r15, cl
    add     r14, r15                    ; dim row stride
    ALIGN64 rax
    mov     [rsp + FB_QKV], rax         ; q, k, v rows are contiguous
    mov     rcx, [rsp + FB_QKVM]
    imul    rcx, r14
    add     rax, rcx
    ALIGN64 rax
    mov     [rsp + FB_O], rax
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    mov     r15, [rbx + CTX_Q_DIM]
    mov     rdx, r15
    shr     rdx, cl
    add     r15, rdx                    ; q_dim row stride
    imul    r15, [rbx + CTX_DIM]
    add     rax, r15
    ALIGN64 rax
    mov     [rsp + FB_GU], rax          ; gate, up rows are contiguous
    mov     rcx, [rbx + CTX_HIDDEN_DIM]
    add     rcx, rcx
    imul    rcx, r14
    add     rax, rcx
    ALIGN64 rax
    mov     [rsp + FB_DOWN], rax
    mov     rax, [rsp + FB_LAYER]       ; layer_kv_base = l * max_seq * kv_dim * 4
    imul    rax, [rbx + CTX_MAX_SEQ]
    imul    rax, [rbx + CTX_KV_DIM]
    shl     rax, 2
    mov     [rsp + FB_KVBASE], rax

    ; Input RMSNorm of every token
    xor     r13d, r13d
.fb_n1:
    cmp     r13, r12
    jae     .fb_n1_done
    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2
    imul    rax, r13
    mov     rdi, [rbx + CTX_BUF_BXB]
    add     rdi, rax
    mov     rsi, [rbx + CTX_BUF_BX]
    add     rsi, rax
    mov     rdx, [rsp + FB_IN_NORM]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    inc     r13
    jmp     .fb_n1
.fb_n1_done:

    ; Fused Q + K + V for the whole batch
    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2
    mov     [rel g_mm_xstride], rax
    mov     rdi, [rsp + FB_QKV]
    mov     rsi, [rbx + CTX_BUF_BXB]
    mov     rdx, [rbx + CTX_BUF_BQKV]
    mov     rcx, [rsp + FB_QKVM]
    mov     r8, [rbx + CTX_DIM]
    mov     r9, r12
    call    tp_matmul_batch

    ; Per token: QK-norm, RoPE, K/V into the cache at pos0 + t
    xor     r13d, r13d
.fb_tok:
    cmp     r13, r12
    jae     .fb_tok_done
    mov     r14, [rsp + FB_QKVM]
    shl     r14, 2
    imul    r14, r13
    add     r14, [rbx + CTX_BUF_BQKV]   ; q of token t (k, v follow)
    mov     r15, r14
    mov     rbp, [rbx + CTX_N_HEADS]
.fb_qn:
    mov     rdi, r15
    mov     rsi, r15
    mov     rdx, [rsp + FB_Q_NORM]
    mov     rcx, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    mov     rax, [rbx + CTX_HEAD_DIM]
    lea     r15, [r15 + rax*4]
    dec     rbp
    jnz     .fb_qn
    mov     rbp, [rbx + CTX_N_KV_HEADS] ; r15 = k of token t
.fb_kn:
    mov     rdi, r15
    mov     rsi, r15
    mov     rdx, [rsp + FB_K_NORM]
    mov     rcx, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    mov     rax, [rbx + CTX_HEAD_DIM]
    lea     r15, [r15 + rax*4]
    dec     rbp
    jnz     .fb_kn
    mov     rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 1
    imul    rax, r13
    mov     [rsp + FB_ROPE], rax
    mov     rdi, r14
    mov     rsi, [rbx + CTX_BUF_BCOS]
    add     rsi, rax
    mov     rdx, [rbx + CTX_BUF_BSIN]
    add     rdx, rax
    mov     rcx, [rbx + CTX_N_HEADS]
    mov     r8, [rbx + CTX_HEAD_DIM]
    call    rope_apply_avx2
    mov     rax, [rbx + CTX_Q_DIM]
    lea     rdi, [r14 + rax*4]
    mov     rax, [rsp + FB_ROPE]
    mov     rsi, [rbx + CTX_BUF_BCOS]
    add     rsi, rax
    mov     rdx, [rbx + CTX_BUF_BSIN]
    add     rdx, rax
    mov     rcx, [rbx + CTX_N_KV_HEADS]
    mov     r8, [rbx + CTX_HEAD_DIM]
    call    rope_apply_avx2
    mov     r8, [rbx + CTX_HEAD_DIM]
    shl     r8, 2                       ; head_bytes
    mov     r9, [rbx + CTX_MAX_SEQ]
    imul    r9, r8                      ; kv_head_stride
    mov     rcx, [rsp + FB_POS0]
    add     rcx, r13
    imul    rcx, r8
    add     rcx, [rsp + FB_KVBASE]
    mov     rdi, [rbx + CTX_BUF_K_CACHE]
    add     rdi, rcx
    mov     rdx, [rbx + CTX_BUF_V_CACHE]
    add     rdx, rcx
    mov     rax, [rbx + CTX_Q_DIM]
    lea     rsi, [r14 + rax*4]          ; k_t
    mov     rax, [rbx + CTX_KV_DIM]
    lea     r10, [rsi + rax*4]          ; v_t
    mov     r11, [rbx + CTX_N_KV_HEADS]
.fb_kvh:
    xor     eax, eax
.fb_kvc:
    cmp     rax, r8
    jae     .fb_kvn
    vmovups ymm0, [rsi + rax]
    vmovups ymm1, [r10 + rax]
    vmovups [rdi + rax], ymm0
    vmovups [rdx + rax], ymm1
    add     rax, 32
    jmp     .fb_kvc
.fb_kvn:
    add     rsi, r8
    add     r10, r8
    add     rdi, r9
    add     rdx, r9
    dec     r11
    jnz     .fb_kvh
    inc     r13
    jmp     .fb_tok
.fb_tok_done:

    ; Attention per token: point buf_q / buf_att_out at token t (no copies)
    mov     rax, [rbx + CTX_BUF_Q]
    mov     [rsp + FB_SAVE_Q], rax
    mov     rax, [rbx + CTX_BUF_ATT_OUT]
    mov     [rsp + FB_SAVE_ATT], rax
    xor     r13d, r13d
.fb_att:
    cmp     r13, r12
    jae     .fb_att_done
    mov     rax, [rsp + FB_QKVM]
    shl     rax, 2
    imul    rax, r13
    add     rax, [rbx + CTX_BUF_BQKV]
    mov     [rbx + CTX_BUF_Q], rax
    mov     rax, [rbx + CTX_Q_DIM]
    shl     rax, 2
    imul    rax, r13
    add     rax, [rbx + CTX_BUF_BATT]
    mov     [rbx + CTX_BUF_ATT_OUT], rax
    mov     rdi, [rsp + FB_KVBASE]
    mov     rsi, [rsp + FB_POS0]
    add     rsi, r13
    mov     edx, [rsp + FB_SCALE + 4]
    shl     rdx, 32
    mov     eax, [rsp + FB_SCALE]
    or      rdx, rax
    mov     rcx, [rbx + CTX_N_HEADS]
    call    tp_gqa_attn
    inc     r13
    jmp     .fb_att
.fb_att_done:
    mov     rax, [rsp + FB_SAVE_Q]
    mov     [rbx + CTX_BUF_Q], rax
    mov     rax, [rsp + FB_SAVE_ATT]
    mov     [rbx + CTX_BUF_ATT_OUT], rax

    ; Output projection for the batch + residual per token
    mov     rax, [rbx + CTX_Q_DIM]
    shl     rax, 2
    mov     [rel g_mm_xstride], rax
    mov     rdi, [rsp + FB_O]
    mov     rsi, [rbx + CTX_BUF_BATT]
    mov     rdx, [rbx + CTX_BUF_BXB]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8, [rbx + CTX_Q_DIM]
    mov     r9, r12
    call    tp_matmul_batch
    xor     r13d, r13d
.fb_r1:
    cmp     r13, r12
    jae     .fb_r1_done
    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2
    imul    rax, r13
    mov     rdi, [rbx + CTX_BUF_BX]
    add     rdi, rax
    mov     rsi, [rbx + CTX_BUF_BXB]
    add     rsi, rax
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2
    inc     r13
    jmp     .fb_r1
.fb_r1_done:

    ; Post-attention RMSNorm of every token
    xor     r13d, r13d
.fb_n2:
    cmp     r13, r12
    jae     .fb_n2_done
    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2
    imul    rax, r13
    mov     rdi, [rbx + CTX_BUF_BXB]
    add     rdi, rax
    mov     rsi, [rbx + CTX_BUF_BX]
    add     rsi, rax
    mov     rdx, [rsp + FB_POST_NORM]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    inc     r13
    jmp     .fb_n2
.fb_n2_done:

    ; Fused gate + up for the batch, SwiGLU per token
    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2
    mov     [rel g_mm_xstride], rax
    mov     rdi, [rsp + FB_GU]
    mov     rsi, [rbx + CTX_BUF_BXB]
    mov     rdx, [rbx + CTX_BUF_BGU]
    mov     rcx, [rbx + CTX_HIDDEN_DIM]
    add     rcx, rcx
    mov     r8, [rbx + CTX_DIM]
    mov     r9, r12
    call    tp_matmul_batch
    xor     r13d, r13d
.fb_silu:
    cmp     r13, r12
    jae     .fb_silu_done
    mov     rax, [rbx + CTX_HIDDEN_DIM]
    shl     rax, 3
    imul    rax, r13
    add     rax, [rbx + CTX_BUF_BGU]
    mov     rdi, rax
    mov     rsi, [rbx + CTX_HIDDEN_DIM]
    lea     rsi, [rax + rsi*4]
    mov     rdx, [rbx + CTX_HIDDEN_DIM]
    call    silu_mul_avx2
    inc     r13
    jmp     .fb_silu
.fb_silu_done:

    ; Down projection for the batch (input = gate half of each token) + residual
    mov     rax, [rbx + CTX_HIDDEN_DIM]
    shl     rax, 3
    mov     [rel g_mm_xstride], rax
    mov     rdi, [rsp + FB_DOWN]
    mov     rsi, [rbx + CTX_BUF_BGU]
    mov     rdx, [rbx + CTX_BUF_BXB]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8, [rbx + CTX_HIDDEN_DIM]
    mov     r9, r12
    call    tp_matmul_batch
    xor     r13d, r13d
.fb_r2:
    cmp     r13, r12
    jae     .fb_r2_done
    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2
    imul    rax, r13
    mov     rdi, [rbx + CTX_BUF_BX]
    add     rdi, rax
    mov     rsi, [rbx + CTX_BUF_BXB]
    add     rsi, rax
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2
    inc     r13
    jmp     .fb_r2
.fb_r2_done:

    inc     qword [rsp + FB_LAYER]
    jmp     .fb_layer

.fb_layers_done:
    cmp     qword [rsp + FB_LOGITS], 0
    je      .fb_done
    ; Final RMSNorm of the last token into buf_x + tied LM head
    lea     rax, [r12 - 1]
    imul    rax, [rbx + CTX_DIM]
    shl     rax, 2
    add     rax, [rbx + CTX_BUF_BX]
    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, rax
    mov     rdx, [rbx + CTX_FINAL_NORM_PTR]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    mov     rdi, [rbx + CTX_EMB_PTR]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rbx + CTX_BUF_LOGITS]
    mov     rcx, [rbx + CTX_VOCAB_SIZE]
    mov     r8, [rbx + CTX_DIM]
    call    tp_matmul_q8_0
.fb_done:
    add     rsp, 168
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret
%endif ; !BARE_METAL
%endif ; FAST_Q8

section .rodata
align 4
t_one: dd 1.0

section .bss
alignb 64
g_ctx: resb CTX_SIZE
g_use_hugepages: resd 1
