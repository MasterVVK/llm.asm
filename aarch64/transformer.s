// ============================================================================
// transformer.s — model loader + Qwen3-0.6B forward pass (AArch64 port)
// Same structure as shared/transformer.asm: fused QKV and gate+up matmuls,
// per-head QK-norm, split-half RoPE, head-major KV cache, parallel GQA.
// Only .asmbin v2 (fp16 block scales) is supported.
// ============================================================================
        .include "constants.s"

        .text
        .global g_ctx, model_init, transformer_forward, gqa_attn_head, transformer_forward_batch

        .macro ALIGN64 reg
        add     \reg, \reg, #63
        and     \reg, \reg, #~63
        .endm

// int64 model_init(const char *path x0, uint64 max_seq x1, int hugepages x2, int repack x3)
// Returns 0 or a negative error code.
model_init:
        stp     x29, x30, [sp, #-80]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        stp     x23, x24, [sp, #48]
        str     x25, [sp, #64]
        ADR_L   x19, g_ctx
        mov     x22, x1                 // max_seq override
        mov     x25, x2                 // hugepages
        str     x3, [sp, #72]           // repack
        bl      sys_open_ro
        cmp     x0, #0
        b.lt    .Lerr_open
        mov     x20, x0                 // fd
        bl      sys_fstat_size
        cmp     x0, #0
        b.le    .Lerr_close
        mov     x21, x0                 // file size
        ldr     x0, [sp, #72]           // repacking? then THP goes to the repacked copy
        cbnz    x0, 4f
        cbnz    x25, .Lhuge
4:
        mov     x0, x21
        mov     x1, x20
        bl      sys_mmap_file
        mov     x23, x0
        b       .Lmapped
.Lhuge: // anonymous copy on 2 MiB transparent huge pages
        mov     x24, #(2 << 20)
        add     x0, x21, x24
        bl      sys_mmap_anon
        cmn     x0, #4096
        b.hi    .Lmapped_err
        add     x23, x0, x24
        sub     x23, x23, #1
        and     x23, x23, #~((2 << 20) - 1)   // 2 MiB aligned base
        add     x1, x21, x24
        sub     x1, x1, #1
        and     x1, x1, #~((2 << 20) - 1)
        mov     x0, x23
        mov     x2, #MADV_HUGEPAGE
        bl      sys_madvise
        mov     x0, x20
        mov     x1, x23
        mov     x2, x21
        bl      sys_read_all
        cmp     x0, x21
        b.ne    .Lmapped_err
.Lmapped:
        mov     x0, x20
        bl      sys_close
        cmn     x23, #4096
        b.hi    .Lerr_mmap
        // header checks: magic, version 2 with fp16 scales
        ldr     w0, [x23, #HDR_MAGIC]
        ldr     w1, =ASMBIN_MAGIC
        cmp     w0, w1
        b.ne    .Lerr_hdr
        ldr     w0, [x23, #HDR_VERSION]
        cmp     w0, #2
        b.ne    .Lerr_hdr
        str     x23, [x19, #CTX_MMAP_BASE]
        str     x21, [x19, #CTX_MMAP_SIZE]
        .irp f, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM, VOCAB_SIZE, HT_MASK, EOS_1, EOS_2
        ldr     w0, [x23, #HDR_\f]
        str     x0, [x19, #CTX_\f]
        .endr
        ldr     w0, [x23, #HDR_RMS_EPS]
        str     w0, [x19, #CTX_RMS_EPS]
        ldr     w0, [x23, #HDR_ROPE_THETA]
        str     w0, [x19, #CTX_ROPE_THETA]
        ldr     x0, [x23, #HDR_LAYER_STRIDE]
        str     x0, [x19, #CTX_LAYER_STRIDE]
        ldr     w0, [x23, #HDR_MAX_SEQ_LEN]
        cbz     x22, 1f
        cmp     x22, x0
        csel    x0, x22, x0, ls
1:      str     x0, [x19, #CTX_MAX_SEQ]
        ldr     x0, [x19, #CTX_N_HEADS]
        ldr     x1, [x19, #CTX_HEAD_DIM]
        mul     x0, x0, x1
        str     x0, [x19, #CTX_Q_DIM]
        ldr     x0, [x19, #CTX_N_KV_HEADS]
        mul     x0, x0, x1
        str     x0, [x19, #CTX_KV_DIM]
        // tokenizer / weight pointers
        ldr     x0, [x23, #HDR_OFF_TOKENIZER]
        add     x0, x0, x23
        ldr     x1, [x0, #TOKH_OFF_B2T]
        add     x1, x1, x0
        str     x1, [x19, #CTX_B2T_PTR]
        ldr     x1, [x0, #TOKH_OFF_OFFSETS]
        add     x1, x1, x0
        str     x1, [x19, #CTX_TOK_OFFS_PTR]
        ldr     x1, [x0, #TOKH_OFF_BYTES]
        add     x1, x1, x0
        str     x1, [x19, #CTX_TOK_BYTES_PTR]
        ldr     x1, [x0, #TOKH_OFF_HT]
        add     x1, x1, x0
        str     x1, [x19, #CTX_HT_PTR]
        ldr     x0, [x23, #HDR_OFF_EMBED]
        add     x0, x0, x23
        str     x0, [x19, #CTX_EMB_PTR]
        ldr     x0, [x23, #HDR_OFF_LAYERS]
        add     x0, x0, x23
        str     x0, [x19, #CTX_LAYERS_PTR]
        ldr     x0, [x23, #HDR_OFF_FINAL_NORM]
        add     x0, x0, x23
        str     x0, [x19, #CTX_FINAL_NORM_PTR]
        // arena layout (relative offsets first)
        mov     x20, #0
        .macro ALLOC field, bytes
        ALIGN64 x20
        str     x20, [x19, #\field]
        add     x20, x20, \bytes
        .endm
        ldr     x1, [x19, #CTX_DIM]
        lsl     x1, x1, #2
        ALLOC   CTX_BUF_X, x1
        ALLOC   CTX_BUF_XB, x1
        ldr     x1, [x19, #CTX_Q_DIM]   // q, k, v contiguous for the fused matmul
        lsl     x1, x1, #2
        ALLOC   CTX_BUF_Q, x1
        ldr     x2, [x19, #CTX_KV_DIM]
        lsl     x2, x2, #2
        str     x20, [x19, #CTX_BUF_K]
        add     x20, x20, x2
        str     x20, [x19, #CTX_BUF_V]
        add     x20, x20, x2
        ALLOC   CTX_BUF_ATT_OUT, x1
        ldr     x1, [x19, #CTX_N_HEADS]
        ldr     x2, [x19, #CTX_MAX_SEQ]
        mul     x1, x1, x2
        lsl     x1, x1, #2
        ALLOC   CTX_BUF_ATT_SCR, x1
        ldr     x1, [x19, #CTX_HIDDEN_DIM]   // gate, up contiguous
        lsl     x1, x1, #2
        ALLOC   CTX_BUF_GATE, x1
        str     x20, [x19, #CTX_BUF_UP]
        add     x20, x20, x1
        ldr     x1, [x19, #CTX_VOCAB_SIZE]
        lsl     x1, x1, #2
        ALLOC   CTX_BUF_LOGITS, x1
        ldr     x1, [x19, #CTX_HEAD_DIM]
        lsl     x1, x1, #2              // (head_dim/2) doubles
        ALLOC   CTX_BUF_INV_FREQ, x1
        lsr     x1, x1, #1
        ALLOC   CTX_BUF_COS, x1
        ALLOC   CTX_BUF_SIN, x1
        ldr     x1, [x19, #CTX_N_LAYERS]
        ldr     x2, [x19, #CTX_MAX_SEQ]
        mul     x1, x1, x2
        ldr     x2, [x19, #CTX_KV_DIM]
        mul     x1, x1, x2
        lsl     x1, x1, #2
        ALLOC   CTX_BUF_K_CACHE, x1
        ALLOC   CTX_BUF_V_CACHE, x1
        mov     x1, #(MAX_PROMPT_TOKENS * 4)
        ALLOC   CTX_BUF_PROMPT, x1
        mov     x1, #(MAX_GEN_TOKENS * 4)
        ALLOC   CTX_BUF_GEN, x1
        // batched prefill buffers (MAX_BATCH tokens)
        mov     x2, #MAX_BATCH
        ldr     x1, [x19, #CTX_DIM]
        lsl     x1, x1, #2
        mul     x1, x1, x2
        ALLOC   CTX_BUF_BX, x1
        ALLOC   CTX_BUF_BXB, x1
        ldr     x1, [x19, #CTX_KV_DIM]
        lsl     x1, x1, #1
        ldr     x3, [x19, #CTX_Q_DIM]
        add     x1, x1, x3
        lsl     x1, x1, #2
        mul     x1, x1, x2
        ALLOC   CTX_BUF_BQKV, x1
        ldr     x1, [x19, #CTX_Q_DIM]
        lsl     x1, x1, #2
        mul     x1, x1, x2
        ALLOC   CTX_BUF_BATT, x1
        ldr     x1, [x19, #CTX_HIDDEN_DIM]
        lsl     x1, x1, #3              // [gate | up]
        mul     x1, x1, x2
        ALLOC   CTX_BUF_BGU, x1
        ldr     x1, [x19, #CTX_HEAD_DIM]
        lsl     x1, x1, #1              // (head_dim / 2) floats
        mul     x1, x1, x2
        ALLOC   CTX_BUF_BCOS, x1
        ALLOC   CTX_BUF_BSIN, x1
        ALIGN64 x20
        str     x20, [x19, #CTX_ARENA_SIZE]
        mov     x0, x20
        bl      sys_mmap_anon
        cmn     x0, #4096
        b.hi    .Lerr_arena
        str     x0, [x19, #CTX_ARENA_BASE]
        add     x1, x19, #CTX_BUF_X     // relocate offsets to pointers
        mov     x2, #CTX_N_BUFS
2:      ldr     x3, [x1]
        add     x3, x3, x0
        str     x3, [x1], #8
        subs    x2, x2, #1
        b.ne    2b
        ldr     x0, [x19, #CTX_BUF_INV_FREQ]
        ldr     x1, [x19, #CTX_HEAD_DIM]
        ldr     s0, [x19, #CTX_ROPE_THETA]
        bl      init_rope_freqs
        ldr     x0, [sp, #72]
        cbz     x0, 3f
        mov     x0, x25
        bl      repack_model
        cmp     x0, #0
        b.lt    .Lret
3:      mov     x0, #0
        b       .Lret
.Lmapped_err:
        mov     x0, x20
        bl      sys_close
.Lerr_mmap:
        mov     x0, #-2
        b       .Lret
.Lerr_close:
        mov     x0, x20
        bl      sys_close
.Lerr_open:
        mov     x0, #-1
        b       .Lret
.Lerr_hdr:
        mov     x0, #-3
        b       .Lret
.Lerr_arena:
        mov     x0, #-4
.Lret:  ldr     x25, [sp, #64]
        ldp     x23, x24, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #80
        ret

// int64 repack_model(int hugepages x0) — copy every Q8_0 matrix into a 4-row
// interleaved buffer at the same offsets as in the file; sets g_mm_delta.
repack_model:
        stp     x29, x30, [sp, #-96]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        stp     x23, x24, [sp, #48]
        stp     x25, x26, [sp, #64]
        str     x27, [sp, #80]
        ADR_L   x19, g_ctx
        mov     x27, x0
        mov     x24, #(2 << 20)
        ldr     x0, [x19, #CTX_MMAP_SIZE]
        add     x0, x0, x24
        bl      sys_mmap_anon
        cmn     x0, #4096
        b.hi    8f
        add     x20, x0, x24
        sub     x20, x20, #1
        and     x20, x20, #~((2 << 20) - 1)
        cbz     x27, 1f
        mov     x0, x20
        ldr     x1, [x19, #CTX_MMAP_SIZE]
        mov     x2, #MADV_HUGEPAGE
        bl      sys_madvise
1:      ldr     x0, [x19, #CTX_MMAP_BASE]
        sub     x20, x20, x0            // delta
        ldr     x21, [x19, #CTX_DIM]
        ldr     x22, [x19, #CTX_HEAD_DIM]
        add     x26, x21, x21, lsr #4   // dim row stride
        mov     x23, #0                 // layer
2:      ldr     x0, [x19, #CTX_N_LAYERS]
        cmp     x23, x0
        b.hs    3f
        ldr     x0, [x19, #CTX_LAYER_STRIDE]
        ldr     x1, [x19, #CTX_LAYERS_PTR]
        madd    x25, x23, x0, x1
        add     x0, x21, x22
        lsl     x0, x0, #3              // (dim + head_dim) * 2 floats * 4 bytes
        add     x25, x25, x0
        ALIGN64 x25                     // qkv
        ldr     x2, [x19, #CTX_Q_DIM]
        ldr     x3, [x19, #CTX_KV_DIM]
        add     x2, x2, x3, lsl #1
        str     x2, [sp, #88]
        mov     x0, x25
        add     x1, x25, x20
        mov     x3, x21
        bl      repack_q8_0_4x4
        ldr     x2, [sp, #88]
        madd    x25, x2, x26, x25
        ALIGN64 x25                     // o
        mov     x0, x25
        add     x1, x25, x20
        mov     x2, x21
        ldr     x3, [x19, #CTX_Q_DIM]
        bl      repack_q8_0_4x4
        ldr     x2, [x19, #CTX_Q_DIM]
        add     x2, x2, x2, lsr #4
        madd    x25, x21, x2, x25
        ALIGN64 x25                     // gate + up
        ldr     x2, [x19, #CTX_HIDDEN_DIM]
        lsl     x2, x2, #1
        mov     x0, x25
        add     x1, x25, x20
        mov     x3, x21
        bl      repack_q8_0_4x4
        ldr     x2, [x19, #CTX_HIDDEN_DIM]
        lsl     x2, x2, #1
        madd    x25, x2, x26, x25
        ALIGN64 x25                     // down
        mov     x0, x25
        add     x1, x25, x20
        mov     x2, x21
        ldr     x3, [x19, #CTX_HIDDEN_DIM]
        bl      repack_q8_0_4x4
        add     x23, x23, #1
        b       2b
3:      ldr     x0, [x19, #CTX_EMB_PTR] // tied embeddings / LM head
        add     x1, x0, x20
        ldr     x2, [x19, #CTX_VOCAB_SIZE]
        mov     x3, x21
        bl      repack_q8_0_4x4
        ADR_L   x0, g_mm_delta
        str     x20, [x0]
        mov     x0, #0
        b       9f
8:      mov     x0, #-5
9:      ldr     x27, [sp, #80]
        ldp     x25, x26, [sp, #64]
        ldp     x23, x24, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #96
        ret

// void transformer_forward(uint32 token w0, uint64 pos x1, int compute_logits x2)
        .equ F_LOGITS,    96
        .equ F_IN_NORM,   104
        .equ F_Q_NORM,    112
        .equ F_K_NORM,    120
        .equ F_POST_NORM, 128
        .equ F_QKV,       136
        .equ F_O,         144
        .equ F_GATE_UP,   152
        .equ F_DOWN,      160
        .equ F_SIZE,      176
transformer_forward:
        stp     x29, x30, [sp, #-F_SIZE]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        stp     x23, x24, [sp, #48]
        stp     x25, x26, [sp, #64]
        stp     x27, x28, [sp, #80]
        ADR_L   x19, g_ctx
        mov     w23, w0                 // token
        mov     x20, x1                 // pos
        str     x2, [sp, #F_LOGITS]
        ldr     x21, [x19, #CTX_DIM]
        ldr     x22, [x19, #CTX_HEAD_DIM]
        // per-token attention constants shared with the workers
        ADR_L   x9, tp_pos
        str     x20, [x9]
        ucvtf   s0, x22
        fsqrt   s0, s0
        fmov    s1, #1.0
        fdiv    s0, s1, s0              // 1 / sqrt(head_dim)
        ADR_L   x9, tp_scale
        str     s0, [x9]
        ldr     x0, [x19, #CTX_N_HEADS]
        ldr     x1, [x19, #CTX_N_KV_HEADS]
        udiv    x0, x0, x1
        ADR_L   x9, tp_gqa
        str     w0, [x9]
        // 1. embedding row (tied with the LM head) -> buf_x
        add     x2, x21, x21, lsr #4
        ldr     x0, [x19, #CTX_EMB_PTR]
        madd    x0, x23, x2, x0
        ldr     x1, [x19, #CTX_BUF_X]
        mov     x2, x21
        bl      dequant_row_q8_0
        // 2. RoPE cos/sin for this position (shared by all layers)
        ldr     x0, [x19, #CTX_BUF_COS]
        ldr     x1, [x19, #CTX_BUF_SIN]
        ldr     x2, [x19, #CTX_BUF_INV_FREQ]
        lsr     x3, x22, #1
        mov     x4, x20
        bl      compute_rope_cs
        mov     x24, #0                 // layer
.Llayer:
        ldr     x0, [x19, #CTX_N_LAYERS]
        cmp     x24, x0
        b.hs    .Llayers_done
        // tensor pointers of layer l (export_weights.py layout)
        ldr     x0, [x19, #CTX_LAYER_STRIDE]
        ldr     x1, [x19, #CTX_LAYERS_PTR]
        madd    x0, x24, x0, x1
        str     x0, [sp, #F_IN_NORM]
        add     x0, x0, x21, lsl #2
        str     x0, [sp, #F_Q_NORM]
        add     x0, x0, x22, lsl #2
        str     x0, [sp, #F_K_NORM]
        add     x0, x0, x22, lsl #2
        str     x0, [sp, #F_POST_NORM]
        add     x0, x0, x21, lsl #2
        ALIGN64 x0
        str     x0, [sp, #F_QKV]        // q, k, v rows are contiguous
        add     x1, x21, x21, lsr #4    // dim row stride
        ldr     x2, [x19, #CTX_Q_DIM]
        ldr     x3, [x19, #CTX_KV_DIM]
        add     x4, x2, x3, lsl #1
        madd    x0, x4, x1, x0
        ALIGN64 x0
        str     x0, [sp, #F_O]
        add     x5, x2, x2, lsr #4      // q_dim row stride
        madd    x0, x21, x5, x0
        ALIGN64 x0
        str     x0, [sp, #F_GATE_UP]    // gate, up rows are contiguous
        ldr     x6, [x19, #CTX_HIDDEN_DIM]
        lsl     x6, x6, #1
        madd    x0, x6, x1, x0
        ALIGN64 x0
        str     x0, [sp, #F_DOWN]
        // input RMSNorm
        ldr     x0, [x19, #CTX_BUF_XB]
        ldr     x1, [x19, #CTX_BUF_X]
        ldr     x2, [sp, #F_IN_NORM]
        mov     x3, x21
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        // fused Q + K + V
        ldr     x0, [sp, #F_QKV]
        ldr     x1, [x19, #CTX_BUF_XB]
        ldr     x2, [x19, #CTX_BUF_Q]
        ldr     x3, [x19, #CTX_Q_DIM]
        ldr     x4, [x19, #CTX_KV_DIM]
        add     x3, x3, x4, lsl #1
        mov     x4, x21
        bl      tp_matmul
        // QK-norm per head
        ldr     x25, [x19, #CTX_BUF_Q]
        ldr     x26, [x19, #CTX_N_HEADS]
1:      mov     x0, x25
        mov     x1, x25
        ldr     x2, [sp, #F_Q_NORM]
        mov     x3, x22
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        add     x25, x25, x22, lsl #2
        subs    x26, x26, #1
        b.ne    1b
        ldr     x25, [x19, #CTX_BUF_K]
        ldr     x26, [x19, #CTX_N_KV_HEADS]
2:      mov     x0, x25
        mov     x1, x25
        ldr     x2, [sp, #F_K_NORM]
        mov     x3, x22
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        add     x25, x25, x22, lsl #2
        subs    x26, x26, #1
        b.ne    2b
        // split-half RoPE on Q and K
        ldr     x0, [x19, #CTX_BUF_Q]
        ldr     x1, [x19, #CTX_BUF_COS]
        ldr     x2, [x19, #CTX_BUF_SIN]
        ldr     x3, [x19, #CTX_N_HEADS]
        mov     x4, x22
        bl      rope_apply
        ldr     x0, [x19, #CTX_BUF_K]
        ldr     x1, [x19, #CTX_BUF_COS]
        ldr     x2, [x19, #CTX_BUF_SIN]
        ldr     x3, [x19, #CTX_N_KV_HEADS]
        mov     x4, x22
        bl      rope_apply
        // K, V -> head-major cache[l, kv_h, pos, head_dim]
        ldr     x0, [x19, #CTX_MAX_SEQ]
        ldr     x1, [x19, #CTX_KV_DIM]
        mul     x2, x24, x0
        mul     x2, x2, x1
        lsl     x2, x2, #2              // layer_kv_base
        ADR_L   x9, tp_kv_base
        str     x2, [x9]
        lsl     x3, x22, #2             // head_bytes
        mul     x4, x0, x3              // kv_head_stride
        madd    x5, x20, x3, x2
        ldr     x6, [x19, #CTX_BUF_K_CACHE]
        add     x6, x6, x5
        ldr     x7, [x19, #CTX_BUF_V_CACHE]
        add     x7, x7, x5
        ldr     x10, [x19, #CTX_BUF_K]
        ldr     x11, [x19, #CTX_BUF_V]
        ldr     x12, [x19, #CTX_N_KV_HEADS]
3:      mov     x13, #0
4:      add     x14, x10, x13
        ldp     q0, q1, [x14]
        add     x14, x11, x13
        ldp     q2, q3, [x14]
        add     x14, x6, x13
        stp     q0, q1, [x14]
        add     x14, x7, x13
        stp     q2, q3, [x14]
        add     x13, x13, #32
        cmp     x13, x3
        b.lo    4b
        add     x10, x10, x3
        add     x11, x11, x3
        add     x6, x6, x4
        add     x7, x7, x4
        subs    x12, x12, #1
        b.ne    3b
        // parallel GQA attention over heads
        ldr     x0, [x19, #CTX_N_HEADS]
        bl      tp_attn
        // output projection + residual
        ldr     x0, [sp, #F_O]
        ldr     x1, [x19, #CTX_BUF_ATT_OUT]
        ldr     x2, [x19, #CTX_BUF_XB]
        mov     x3, x21
        ldr     x4, [x19, #CTX_Q_DIM]
        bl      tp_matmul
        ldr     x0, [x19, #CTX_BUF_X]
        ldr     x1, [x19, #CTX_BUF_XB]
        mov     x2, x21
        bl      vec_add
        // post-attention RMSNorm + SwiGLU MLP + residual
        ldr     x0, [x19, #CTX_BUF_XB]
        ldr     x1, [x19, #CTX_BUF_X]
        ldr     x2, [sp, #F_POST_NORM]
        mov     x3, x21
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        ldr     x0, [sp, #F_GATE_UP]
        ldr     x1, [x19, #CTX_BUF_XB]
        ldr     x2, [x19, #CTX_BUF_GATE]
        ldr     x3, [x19, #CTX_HIDDEN_DIM]
        lsl     x3, x3, #1
        mov     x4, x21
        bl      tp_matmul
        ldr     x0, [x19, #CTX_BUF_GATE]
        ldr     x1, [x19, #CTX_BUF_UP]
        ldr     x2, [x19, #CTX_HIDDEN_DIM]
        bl      silu_mul
        ldr     x0, [sp, #F_DOWN]
        ldr     x1, [x19, #CTX_BUF_GATE]
        ldr     x2, [x19, #CTX_BUF_XB]
        mov     x3, x21
        ldr     x4, [x19, #CTX_HIDDEN_DIM]
        bl      tp_matmul
        ldr     x0, [x19, #CTX_BUF_X]
        ldr     x1, [x19, #CTX_BUF_XB]
        mov     x2, x21
        bl      vec_add
        add     x24, x24, #1
        b       .Llayer
.Llayers_done:
        ldr     x0, [sp, #F_LOGITS]
        cbz     x0, 9f
        ldr     x0, [x19, #CTX_BUF_X]   // final RMSNorm + tied LM head
        mov     x1, x0
        ldr     x2, [x19, #CTX_FINAL_NORM_PTR]
        mov     x3, x21
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        ldr     x0, [x19, #CTX_EMB_PTR]
        ldr     x1, [x19, #CTX_BUF_X]
        ldr     x2, [x19, #CTX_BUF_LOGITS]
        ldr     x3, [x19, #CTX_VOCAB_SIZE]
        mov     x4, x21
        bl      tp_matmul
9:      ldp     x27, x28, [sp, #80]
        ldp     x25, x26, [sp, #64]
        ldp     x23, x24, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #F_SIZE
        ret

// void transformer_forward_batch(const uint32 *toks x0, uint64 B x1, uint64 pos0 x2,
//                                int compute_logits x3)
// Prompt prefill of B <= MAX_BATCH tokens at pos0 .. pos0 + B - 1: weights are
// streamed once per layer for the batch (tp_matmul_batch), all other steps run
// the single-token code per token, and token t attends to 0 .. pos0 + t after
// the whole batch is in the KV cache. Bit-identical to B transformer_forward calls.
        .equ FB_TOKS,     96
        .equ FB_POS0,     104
        .equ FB_LOGITS,   112
        .equ FB_IN_NORM,  120
        .equ FB_Q_NORM,   128
        .equ FB_K_NORM,   136
        .equ FB_POST_NORM, 144
        .equ FB_QKV,      152
        .equ FB_O,        160
        .equ FB_GU,       168
        .equ FB_DOWN,     176
        .equ FB_QKVM,     184
        .equ FB_SAVE_Q,   192
        .equ FB_SAVE_ATT, 200
        .equ FB_SZ,       208
transformer_forward_batch:
        stp     x29, x30, [sp, #-FB_SZ]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        stp     x23, x24, [sp, #48]
        stp     x25, x26, [sp, #64]
        stp     x27, x28, [sp, #80]
        ADR_L   x19, g_ctx
        str     x0, [sp, #FB_TOKS]
        mov     x20, x1                 // B
        str     x2, [sp, #FB_POS0]
        str     x3, [sp, #FB_LOGITS]
        ldr     x21, [x19, #CTX_DIM]
        ldr     x22, [x19, #CTX_HEAD_DIM]
        ucvtf   s0, x22
        fsqrt   s0, s0
        fmov    s1, #1.0
        fdiv    s0, s1, s0
        ADR_L   x9, tp_scale
        str     s0, [x9]
        ldr     x0, [x19, #CTX_N_HEADS]
        ldr     x1, [x19, #CTX_N_KV_HEADS]
        udiv    x0, x0, x1
        ADR_L   x9, tp_gqa
        str     w0, [x9]
        ldr     x0, [x19, #CTX_Q_DIM]
        ldr     x1, [x19, #CTX_KV_DIM]
        add     x0, x0, x1, lsl #1
        str     x0, [sp, #FB_QKVM]
        // embeddings and RoPE tables of every token
        mov     x23, #0
1:      cmp     x23, x20
        b.hs    2f
        ldr     x0, [sp, #FB_TOKS]
        ldr     w0, [x0, x23, lsl #2]
        add     x2, x21, x21, lsr #4
        ldr     x1, [x19, #CTX_EMB_PTR]
        madd    x0, x0, x2, x1
        mul     x1, x23, x21
        lsl     x1, x1, #2
        ldr     x2, [x19, #CTX_BUF_BX]
        add     x1, x1, x2
        mov     x2, x21
        bl      dequant_row_q8_0
        lsl     x5, x22, #1
        mul     x5, x5, x23             // t * (head_dim / 2) * 4
        ldr     x0, [x19, #CTX_BUF_BCOS]
        add     x0, x0, x5
        ldr     x1, [x19, #CTX_BUF_BSIN]
        add     x1, x1, x5
        ldr     x2, [x19, #CTX_BUF_INV_FREQ]
        lsr     x3, x22, #1
        ldr     x4, [sp, #FB_POS0]
        add     x4, x4, x23
        bl      compute_rope_cs
        add     x23, x23, #1
        b       1b
2:      mov     x24, #0                 // layer
.Lfb_layer:
        ldr     x0, [x19, #CTX_N_LAYERS]
        cmp     x24, x0
        b.hs    .Lfb_layers_done
        ldr     x0, [x19, #CTX_LAYER_STRIDE]
        ldr     x1, [x19, #CTX_LAYERS_PTR]
        madd    x0, x24, x0, x1
        str     x0, [sp, #FB_IN_NORM]
        add     x0, x0, x21, lsl #2
        str     x0, [sp, #FB_Q_NORM]
        add     x0, x0, x22, lsl #2
        str     x0, [sp, #FB_K_NORM]
        add     x0, x0, x22, lsl #2
        str     x0, [sp, #FB_POST_NORM]
        add     x0, x0, x21, lsl #2
        ALIGN64 x0
        str     x0, [sp, #FB_QKV]
        add     x1, x21, x21, lsr #4
        ldr     x4, [sp, #FB_QKVM]
        madd    x0, x4, x1, x0
        ALIGN64 x0
        str     x0, [sp, #FB_O]
        ldr     x2, [x19, #CTX_Q_DIM]
        add     x5, x2, x2, lsr #4
        madd    x0, x21, x5, x0
        ALIGN64 x0
        str     x0, [sp, #FB_GU]
        ldr     x6, [x19, #CTX_HIDDEN_DIM]
        lsl     x6, x6, #1
        madd    x0, x6, x1, x0
        ALIGN64 x0
        str     x0, [sp, #FB_DOWN]
        ldr     x0, [x19, #CTX_MAX_SEQ]
        ldr     x1, [x19, #CTX_KV_DIM]
        mul     x2, x24, x0
        mul     x2, x2, x1
        lsl     x2, x2, #2
        ADR_L   x9, tp_kv_base
        str     x2, [x9]
        // input RMSNorm of every token
        mov     x23, #0
3:      cmp     x23, x20
        b.hs    4f
        mul     x5, x23, x21
        lsl     x5, x5, #2
        ldr     x0, [x19, #CTX_BUF_BXB]
        add     x0, x0, x5
        ldr     x1, [x19, #CTX_BUF_BX]
        add     x1, x1, x5
        ldr     x2, [sp, #FB_IN_NORM]
        mov     x3, x21
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        add     x23, x23, #1
        b       3b
4:      lsl     x0, x21, #2             // fused Q + K + V for the batch
        ADR_L   x1, g_mm_xstride
        str     x0, [x1]
        ldr     x0, [sp, #FB_QKV]
        ldr     x1, [x19, #CTX_BUF_BXB]
        ldr     x2, [x19, #CTX_BUF_BQKV]
        ldr     x3, [sp, #FB_QKVM]
        mov     x4, x21
        mov     x5, x20
        bl      tp_matmul_batch
        // per token: QK-norm, RoPE, K/V into the cache at pos0 + t
        mov     x23, #0
5:      cmp     x23, x20
        b.hs    8f
        ldr     x25, [sp, #FB_QKVM]
        lsl     x25, x25, #2
        mul     x25, x25, x23
        ldr     x0, [x19, #CTX_BUF_BQKV]
        add     x25, x25, x0            // q of token t
        mov     x26, x25
        ldr     x27, [x19, #CTX_N_HEADS]
6:      mov     x0, x26
        mov     x1, x26
        ldr     x2, [sp, #FB_Q_NORM]
        mov     x3, x22
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        add     x26, x26, x22, lsl #2
        subs    x27, x27, #1
        b.ne    6b
        ldr     x27, [x19, #CTX_N_KV_HEADS]
7:      mov     x0, x26
        mov     x1, x26
        ldr     x2, [sp, #FB_K_NORM]
        mov     x3, x22
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        add     x26, x26, x22, lsl #2
        subs    x27, x27, #1
        b.ne    7b
        lsl     x28, x22, #1
        mul     x28, x28, x23           // RoPE table offset of token t
        mov     x0, x25
        ldr     x1, [x19, #CTX_BUF_BCOS]
        add     x1, x1, x28
        ldr     x2, [x19, #CTX_BUF_BSIN]
        add     x2, x2, x28
        ldr     x3, [x19, #CTX_N_HEADS]
        mov     x4, x22
        bl      rope_apply
        ldr     x0, [x19, #CTX_Q_DIM]
        add     x0, x25, x0, lsl #2
        ldr     x1, [x19, #CTX_BUF_BCOS]
        add     x1, x1, x28
        ldr     x2, [x19, #CTX_BUF_BSIN]
        add     x2, x2, x28
        ldr     x3, [x19, #CTX_N_KV_HEADS]
        mov     x4, x22
        bl      rope_apply
        ldr     x0, [x19, #CTX_MAX_SEQ]
        lsl     x3, x22, #2             // head_bytes
        mul     x4, x0, x3              // kv_head_stride
        ADR_L   x9, tp_kv_base
        ldr     x2, [x9]
        ldr     x5, [sp, #FB_POS0]
        add     x5, x5, x23
        madd    x5, x5, x3, x2
        ldr     x6, [x19, #CTX_BUF_K_CACHE]
        add     x6, x6, x5
        ldr     x7, [x19, #CTX_BUF_V_CACHE]
        add     x7, x7, x5
        ldr     x0, [x19, #CTX_Q_DIM]
        add     x10, x25, x0, lsl #2    // k_t
        ldr     x0, [x19, #CTX_KV_DIM]
        add     x11, x10, x0, lsl #2    // v_t
        ldr     x12, [x19, #CTX_N_KV_HEADS]
50:     mov     x13, #0
51:     add     x14, x10, x13
        ldp     q0, q1, [x14]
        add     x14, x11, x13
        ldp     q2, q3, [x14]
        add     x14, x6, x13
        stp     q0, q1, [x14]
        add     x14, x7, x13
        stp     q2, q3, [x14]
        add     x13, x13, #32
        cmp     x13, x3
        b.lo    51b
        add     x10, x10, x3
        add     x11, x11, x3
        add     x6, x6, x4
        add     x7, x7, x4
        subs    x12, x12, #1
        b.ne    50b
        add     x23, x23, #1
        b       5b
8:      // attention per token: point buf_q / buf_att_out at token t (no copies)
        ldr     x0, [x19, #CTX_BUF_Q]
        str     x0, [sp, #FB_SAVE_Q]
        ldr     x0, [x19, #CTX_BUF_ATT_OUT]
        str     x0, [sp, #FB_SAVE_ATT]
        mov     x23, #0
9:      cmp     x23, x20
        b.hs    10f
        ldr     x0, [sp, #FB_QKVM]
        lsl     x0, x0, #2
        mul     x0, x0, x23
        ldr     x1, [x19, #CTX_BUF_BQKV]
        add     x0, x0, x1
        str     x0, [x19, #CTX_BUF_Q]
        ldr     x0, [x19, #CTX_Q_DIM]
        lsl     x0, x0, #2
        mul     x0, x0, x23
        ldr     x1, [x19, #CTX_BUF_BATT]
        add     x0, x0, x1
        str     x0, [x19, #CTX_BUF_ATT_OUT]
        ldr     x0, [sp, #FB_POS0]
        add     x0, x0, x23
        ADR_L   x9, tp_pos
        str     x0, [x9]
        ldr     x0, [x19, #CTX_N_HEADS]
        bl      tp_attn
        add     x23, x23, #1
        b       9b
10:     ldr     x0, [sp, #FB_SAVE_Q]
        str     x0, [x19, #CTX_BUF_Q]
        ldr     x0, [sp, #FB_SAVE_ATT]
        str     x0, [x19, #CTX_BUF_ATT_OUT]
        ldr     x0, [x19, #CTX_Q_DIM]   // output projection for the batch
        lsl     x0, x0, #2
        ADR_L   x1, g_mm_xstride
        str     x0, [x1]
        ldr     x0, [sp, #FB_O]
        ldr     x1, [x19, #CTX_BUF_BATT]
        ldr     x2, [x19, #CTX_BUF_BXB]
        mov     x3, x21
        ldr     x4, [x19, #CTX_Q_DIM]
        mov     x5, x20
        bl      tp_matmul_batch
        mov     x23, #0
11:     cmp     x23, x20                // residual
        b.hs    12f
        mul     x5, x23, x21
        lsl     x5, x5, #2
        ldr     x0, [x19, #CTX_BUF_BX]
        add     x0, x0, x5
        ldr     x1, [x19, #CTX_BUF_BXB]
        add     x1, x1, x5
        mov     x2, x21
        bl      vec_add
        add     x23, x23, #1
        b       11b
12:     mov     x23, #0                 // post-attention RMSNorm
13:     cmp     x23, x20
        b.hs    14f
        mul     x5, x23, x21
        lsl     x5, x5, #2
        ldr     x0, [x19, #CTX_BUF_BXB]
        add     x0, x0, x5
        ldr     x1, [x19, #CTX_BUF_BX]
        add     x1, x1, x5
        ldr     x2, [sp, #FB_POST_NORM]
        mov     x3, x21
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        add     x23, x23, #1
        b       13b
14:     lsl     x0, x21, #2             // fused gate + up for the batch
        ADR_L   x1, g_mm_xstride
        str     x0, [x1]
        ldr     x0, [sp, #FB_GU]
        ldr     x1, [x19, #CTX_BUF_BXB]
        ldr     x2, [x19, #CTX_BUF_BGU]
        ldr     x3, [x19, #CTX_HIDDEN_DIM]
        lsl     x3, x3, #1
        mov     x4, x21
        mov     x5, x20
        bl      tp_matmul_batch
        mov     x23, #0
15:     cmp     x23, x20                // SwiGLU per token
        b.hs    16f
        ldr     x5, [x19, #CTX_HIDDEN_DIM]
        lsl     x6, x5, #3
        mul     x6, x6, x23
        ldr     x0, [x19, #CTX_BUF_BGU]
        add     x0, x0, x6
        add     x1, x0, x5, lsl #2
        mov     x2, x5
        bl      silu_mul
        add     x23, x23, #1
        b       15b
16:     ldr     x0, [x19, #CTX_HIDDEN_DIM] // down projection (input: gate half)
        lsl     x0, x0, #3
        ADR_L   x1, g_mm_xstride
        str     x0, [x1]
        ldr     x0, [sp, #FB_DOWN]
        ldr     x1, [x19, #CTX_BUF_BGU]
        ldr     x2, [x19, #CTX_BUF_BXB]
        mov     x3, x21
        ldr     x4, [x19, #CTX_HIDDEN_DIM]
        mov     x5, x20
        bl      tp_matmul_batch
        mov     x23, #0
17:     cmp     x23, x20                // residual
        b.hs    18f
        mul     x5, x23, x21
        lsl     x5, x5, #2
        ldr     x0, [x19, #CTX_BUF_BX]
        add     x0, x0, x5
        ldr     x1, [x19, #CTX_BUF_BXB]
        add     x1, x1, x5
        mov     x2, x21
        bl      vec_add
        add     x23, x23, #1
        b       17b
18:     add     x24, x24, #1
        b       .Lfb_layer
.Lfb_layers_done:
        ldr     x0, [sp, #FB_LOGITS]
        cbz     x0, 19f
        sub     x1, x20, #1             // final RMSNorm of the last token + LM head
        mul     x1, x1, x21
        lsl     x1, x1, #2
        ldr     x2, [x19, #CTX_BUF_BX]
        add     x1, x1, x2
        ldr     x0, [x19, #CTX_BUF_X]
        ldr     x2, [x19, #CTX_FINAL_NORM_PTR]
        mov     x3, x21
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        ldr     x0, [x19, #CTX_EMB_PTR]
        ldr     x1, [x19, #CTX_BUF_X]
        ldr     x2, [x19, #CTX_BUF_LOGITS]
        ldr     x3, [x19, #CTX_VOCAB_SIZE]
        mov     x4, x21
        bl      tp_matmul
19:     ldp     x27, x28, [sp, #80]
        ldp     x25, x26, [sp, #64]
        ldp     x23, x24, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #FB_SZ
        ret

// void gqa_attn_head(uint64 h x0) — one query head over cache[l, kv_h, 0..pos]
// Inputs come from tp_kv_base, tp_pos, tp_scale, tp_gqa (set per token/layer).
gqa_attn_head:
        stp     x29, x30, [sp, #-96]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        stp     x23, x24, [sp, #48]
        stp     x25, x26, [sp, #64]
        stp     x27, x28, [sp, #80]
        ADR_L   x19, g_ctx
        mov     x20, x0                 // h
        ldr     x21, [x19, #CTX_HEAD_DIM]
        lsl     x22, x21, #2            // head_bytes
        ADR_L   x9, tp_gqa
        ldr     w1, [x9]
        udiv    x2, x20, x1             // kv_h
        ldr     x3, [x19, #CTX_MAX_SEQ]
        mul     x4, x3, x22
        ADR_L   x9, tp_kv_base
        ldr     x5, [x9]
        madd    x23, x2, x4, x5         // byte offset of this kv head's rows
        ldr     x24, [x19, #CTX_BUF_Q]
        madd    x24, x20, x22, x24      // q_head
        lsl     x6, x3, #2
        ldr     x25, [x19, #CTX_BUF_ATT_SCR]
        madd    x25, x20, x6, x25       // private scores
        ADR_L   x9, tp_pos
        ldr     x26, [x9]
        ldr     x27, [x19, #CTX_BUF_K_CACHE]
        add     x27, x27, x23
        mov     x28, #0
1:      cmp     x28, x26                // scores[t] = q . k_t * scale
        b.hi    2f
        mov     x0, x24
        mov     x1, x27
        mov     x2, x21
        bl      dot_f32
        ADR_L   x9, tp_scale
        ldr     s1, [x9]
        fmul    s0, s0, s1
        str     s0, [x25, x28, lsl #2]
        add     x27, x27, x22
        add     x28, x28, #1
        b       1b
2:      mov     x0, x25
        add     x1, x26, #1
        bl      softmax
        ldr     x24, [x19, #CTX_BUF_ATT_OUT]
        madd    x24, x20, x22, x24      // att_head
        movi    v0.4s, #0
        mov     x9, #0
3:      str     q0, [x24, x9]
        add     x9, x9, #16
        cmp     x9, x22
        b.lo    3b
        ldr     x27, [x19, #CTX_BUF_V_CACHE]
        add     x27, x27, x23
        mov     x28, #0
4:      cmp     x28, x26                // att_head += p_t * v_t
        b.hi    5f
        ldr     s0, [x25, x28, lsl #2]
        mov     x0, x24
        mov     x1, x27
        mov     x2, x21
        bl      saxpy_f32
        add     x27, x27, x22
        add     x28, x28, #1
        b       4b
5:      ldp     x27, x28, [sp, #80]
        ldp     x25, x26, [sp, #64]
        ldp     x23, x24, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #96
        ret

        .bss
        .balign 64
g_ctx:  .skip   CTX_SIZE
