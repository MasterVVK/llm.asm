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
        ldr     x1, [x19, #CTX_N_HEADS]
        ldr     x3, [x19, #CTX_MAX_SEQ]
        mul     x1, x1, x3
        lsl     x1, x1, #2              // scores of every head of every token
        mul     x1, x1, x2
        ALLOC   CTX_BUF_BSCR, x1
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

// Per-token steps of the batched prefill. transformer_forward_batch runs them
// as pool jobs (tp_for) with one token per chunk (silu_item: a quarter of a
// token). A chunk makes the same calls on the same data as the single-thread
// loops they replace, so the results do not depend on the thread that takes
// it. Layer parameters: fb_norm, fb_res, fb_qnorm, fb_knorm, tp_pos, tp_kv_base.

// void norm_item(uint64 t x0)
// If fb_res: BX[t] += BXB[t] (residual of the projection that wrote BXB).
// Then BXB[t] = RMSNorm(BX[t]) * fb_norm, quantized into the q8b slot of t.
norm_item:
        stp     x29, x30, [sp, #-32]!
        stp     x19, x20, [sp, #16]
        ADR_L   x19, g_ctx
        mov     x20, x0                 // t
        ADR_L   x9, fb_res
        ldr     x9, [x9]
        cbz     x9, 1f
        ldr     x2, [x19, #CTX_DIM]
        mul     x5, x20, x2
        ldr     x0, [x19, #CTX_BUF_BX]
        add     x0, x0, x5, lsl #2
        ldr     x1, [x19, #CTX_BUF_BXB]
        add     x1, x1, x5, lsl #2
        bl      vec_add
1:      ldr     x3, [x19, #CTX_DIM]
        mul     x5, x20, x3
        ldr     x0, [x19, #CTX_BUF_BXB]
        add     x0, x0, x5, lsl #2
        ldr     x1, [x19, #CTX_BUF_BX]
        add     x1, x1, x5, lsl #2
        ADR_L   x9, fb_norm
        ldr     x2, [x9]
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        ldr     x1, [x19, #CTX_DIM]
        mul     x5, x20, x1
        ldr     x0, [x19, #CTX_BUF_BXB]
        add     x0, x0, x5, lsl #2
        mov     x2, x20
        mov     x3, #0
        bl      quantize_x_slot
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #32
        ret

// void qk_item(uint64 t x0)
// QK-norm and RoPE of token t, then its K and V into the cache at pos0 + t.
qk_item:
        stp     x29, x30, [sp, #-64]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        stp     x23, x24, [sp, #48]
        ADR_L   x19, g_ctx
        mov     x20, x0                 // t
        ldr     x22, [x19, #CTX_HEAD_DIM]
        ldr     x0, [x19, #CTX_Q_DIM]
        ldr     x1, [x19, #CTX_KV_DIM]
        add     x0, x0, x1, lsl #1      // floats per token in buf_bqkv
        mul     x0, x0, x20
        ldr     x1, [x19, #CTX_BUF_BQKV]
        add     x21, x1, x0, lsl #2     // q of token t (k, v follow)
        mov     x23, x21
        ldr     x24, [x19, #CTX_N_HEADS]
1:      mov     x0, x23
        mov     x1, x23
        ADR_L   x9, fb_qnorm
        ldr     x2, [x9]
        mov     x3, x22
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        add     x23, x23, x22, lsl #2
        subs    x24, x24, #1
        b.ne    1b
        ldr     x24, [x19, #CTX_N_KV_HEADS]
2:      mov     x0, x23
        mov     x1, x23
        ADR_L   x9, fb_knorm
        ldr     x2, [x9]
        mov     x3, x22
        ldr     s0, [x19, #CTX_RMS_EPS]
        bl      rmsnorm
        add     x23, x23, x22, lsl #2
        subs    x24, x24, #1
        b.ne    2b
        lsl     x23, x22, #1
        mul     x23, x23, x20           // RoPE table offset of token t
        mov     x0, x21
        ldr     x1, [x19, #CTX_BUF_BCOS]
        add     x1, x1, x23
        ldr     x2, [x19, #CTX_BUF_BSIN]
        add     x2, x2, x23
        ldr     x3, [x19, #CTX_N_HEADS]
        mov     x4, x22
        bl      rope_apply
        ldr     x0, [x19, #CTX_Q_DIM]
        add     x0, x21, x0, lsl #2
        ldr     x1, [x19, #CTX_BUF_BCOS]
        add     x1, x1, x23
        ldr     x2, [x19, #CTX_BUF_BSIN]
        add     x2, x2, x23
        ldr     x3, [x19, #CTX_N_KV_HEADS]
        mov     x4, x22
        bl      rope_apply
        ldr     x0, [x19, #CTX_MAX_SEQ]
        lsl     x3, x22, #2             // head_bytes
        mul     x4, x0, x3              // kv_head_stride
        ADR_L   x9, tp_kv_base
        ldr     x2, [x9]
        ADR_L   x9, tp_pos
        ldr     x5, [x9]
        add     x5, x5, x20
        madd    x5, x5, x3, x2
        ldr     x6, [x19, #CTX_BUF_K_CACHE]
        add     x6, x6, x5
        ldr     x7, [x19, #CTX_BUF_V_CACHE]
        add     x7, x7, x5
        ldr     x0, [x19, #CTX_Q_DIM]
        add     x10, x21, x0, lsl #2    // k_t
        ldr     x0, [x19, #CTX_KV_DIM]
        add     x11, x10, x0, lsl #2    // v_t
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
        ldp     x23, x24, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #64
        ret

// void silu_item(uint64 i x0)
// SwiGLU of quarter i % 4 of token i / 4 (gate = silu(gate) * up in place),
// then that quarter of the gate into the q8b slot of the token.
silu_item:
        stp     x29, x30, [sp, #-48]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        ADR_L   x19, g_ctx
        lsr     x20, x0, #2             // t
        and     x0, x0, #3              // quarter
        ldr     x5, [x19, #CTX_HIDDEN_DIM]
        lsr     x1, x5, #5              // blocks of 32
        mul     x2, x1, x0
        lsr     x2, x2, #2              // first block of the quarter
        madd    x3, x1, x0, x1          // (quarter + 1) * blocks
        lsr     x3, x3, #2
        sub     x3, x3, x2
        lsl     x22, x3, #5             // x22 = floats in the quarter
        lsl     x21, x2, #5             // x21 = k0
        lsl     x6, x5, #3
        mul     x6, x6, x20
        ldr     x0, [x19, #CTX_BUF_BGU]
        add     x0, x0, x6              // gate of token t
        add     x0, x0, x21, lsl #2     // + k0
        add     x1, x0, x5, lsl #2      // up + k0
        mov     x2, x22
        bl      silu_mul
        ldr     x5, [x19, #CTX_HIDDEN_DIM]
        lsl     x6, x5, #3
        mul     x6, x6, x20
        ldr     x0, [x19, #CTX_BUF_BGU]
        add     x0, x0, x6
        add     x0, x0, x21, lsl #2
        mov     x1, x22
        mov     x2, x20
        mov     x3, x21
        bl      quantize_x_slot
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #48
        ret

// void transformer_forward_batch(const uint32 *toks x0, uint64 B x1, uint64 pos0 x2,
//                                int compute_logits x3)
// Prompt prefill of B <= MAX_BATCH tokens at pos0 .. pos0 + B - 1: weights are
// streamed once per layer for the batch (tp_matmul_batch), the attention of all
// tokens is one pool job (attn_pair_item), and token t attends to 0 .. pos0 + t
// after the whole batch is in the KV cache. Between the matmuls, residual adds,
// RMSNorm, activation quantization, QK-norm, RoPE, the KV copy and SwiGLU run
// as pool jobs over the tokens (norm_item, qk_item, silu_item) with the
// single-token code. Bit-identical to B transformer_forward calls.
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
        .equ FB_PAIR,     208             // 1 = batched attention kernel applies
        .equ FB_SZ,       224
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
        and     x1, x0, #1              // attn_pair_item needs pairs of query
        cmp     x22, #128               // heads per kv head and head_dim 128
        ccmp    x1, #0, #0, eq
        cset    x1, eq
        str     x1, [sp, #FB_PAIR]
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
        // residual of the previous layer's FFN, input RMSNorm and quantization
        ldr     x0, [sp, #FB_IN_NORM]
        ADR_L   x9, fb_norm
        str     x0, [x9]
        cmp     x24, #0
        cset    x0, ne
        ADR_L   x9, fb_res
        str     x0, [x9]
        ADR_L   x0, norm_item
        mov     x1, x20
        bl      tp_for
        ldr     x0, [sp, #FB_QKV]       // fused Q + K + V for the batch
        mov     x1, #0                  // quantized by norm_item
        ldr     x2, [x19, #CTX_BUF_BQKV]
        ldr     x3, [sp, #FB_QKVM]
        mov     x4, x21
        mov     x5, x20
        bl      tp_matmul_batch
        // QK-norm, RoPE, K/V into the cache at pos0 + t
        ldr     x0, [sp, #FB_Q_NORM]
        ADR_L   x9, fb_qnorm
        str     x0, [x9]
        ldr     x0, [sp, #FB_K_NORM]
        ADR_L   x9, fb_knorm
        str     x0, [x9]
        ldr     x0, [sp, #FB_POS0]
        ADR_L   x9, tp_pos
        str     x0, [x9]
        ADR_L   x0, qk_item
        mov     x1, x20
        bl      tp_for
        // attention of the whole batch in one pool job: a chunk is a pair of
        // query heads of one token (its outputs come back quantized)
        ldr     x0, [sp, #FB_PAIR]
        cbz     x0, 80f
        ADR_L   x9, tp_ab_b
        str     x20, [x9]
        ADR_L   x0, attn_pair_item
        ldr     x1, [x19, #CTX_N_HEADS]
        lsr     x1, x1, #1              // head pairs
        mul     x1, x1, x20
        bl      tp_for
        mov     x1, #0                  // quantized by attn_pair_item
        b       81f
80:     // otherwise per token: point buf_q / buf_att_out at token t (no copies)
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
        ldr     x0, [x19, #CTX_Q_DIM]
        lsl     x0, x0, #2
        ADR_L   x1, g_mm_xstride
        str     x0, [x1]
        ldr     x1, [x19, #CTX_BUF_BATT] // tp_matmul_batch quantizes it
81:     ldr     x0, [sp, #FB_O]         // output projection for the batch
        ldr     x2, [x19, #CTX_BUF_BXB]
        mov     x3, x21
        ldr     x4, [x19, #CTX_Q_DIM]
        mov     x5, x20
        bl      tp_matmul_batch
        // residual, post-attention RMSNorm and quantization
        ldr     x0, [sp, #FB_POST_NORM]
        ADR_L   x9, fb_norm
        str     x0, [x9]
        mov     x0, #1
        ADR_L   x9, fb_res
        str     x0, [x9]
        ADR_L   x0, norm_item
        mov     x1, x20
        bl      tp_for
        ldr     x0, [sp, #FB_GU]        // fused gate + up for the batch
        mov     x1, #0                  // quantized by norm_item
        ldr     x2, [x19, #CTX_BUF_BGU]
        ldr     x3, [x19, #CTX_HIDDEN_DIM]
        lsl     x3, x3, #1
        mov     x4, x21
        mov     x5, x20
        bl      tp_matmul_batch
        ADR_L   x0, silu_item           // SwiGLU, a quarter of a token per chunk
        lsl     x1, x20, #2
        bl      tp_for
        ldr     x0, [sp, #FB_DOWN]      // down projection (input: gate half); the
        mov     x1, #0                  // next layer's norm_item adds the residual
        ldr     x2, [x19, #CTX_BUF_BXB]
        mov     x3, x21
        ldr     x4, [x19, #CTX_HIDDEN_DIM]
        mov     x5, x20
        bl      tp_matmul_batch
        add     x24, x24, #1
        b       .Lfb_layer
.Lfb_layers_done:
        mov     x23, #0
17:     cmp     x23, x20                // residual of the last FFN
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
18:     ldr     x0, [sp, #FB_LOGITS]
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

// void attn_pair_item(uint64 item x0) — one chunk of the batched prefill
// attention: query heads h and h + 1 of token b, h = 2 * (item / tp_ab_b),
// b = tp_ab_b - 1 - item % tp_ab_b. Consecutive chunks share a kv head, so its
// K/V rows stay in the core's L2 from token to token, and the longest rows of
// each pair go first. Both heads read kv head h / gqa (gqa is even); head_dim
// is 128. Scores use a 2 head x 2 position
// register tile (each q and K load serves two products); values keep 2 heads x
// 32 outputs in registers over all positions instead of a load and a store of
// the output per position. Every float operation matches gqa_attn_head
// (dot_f32, softmax, saxpy_f32) lane for lane: the output is bit-identical.
// The 256 outputs are then quantized into the q8b slot of token b (8 whole
// Q8_0 blocks) for the output projection.
attn_pair_item:
        stp     x29, x30, [sp, #-96]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        stp     x23, x24, [sp, #48]
        stp     x25, x26, [sp, #64]
        stp     x27, x28, [sp, #80]
        ADR_L   x19, g_ctx
        ADR_L   x9, tp_ab_b
        ldr     x4, [x9]
        udiv    x2, x0, x4              // head pair
        msub    x3, x2, x4, x0          // rank of the token
        sub     x4, x4, #1
        sub     x20, x4, x3             // token b
        lsl     x21, x2, #1             // h
        ADR_L   x9, tp_pos
        ldr     x22, [x9]
        add     x22, x22, x20
        add     x22, x22, #1            // n = pos0 + b + 1 positions
        ldr     x0, [x19, #CTX_Q_DIM]
        ldr     x1, [x19, #CTX_KV_DIM]
        add     x1, x0, x1, lsl #1      // floats per token in buf_bqkv
        mul     x1, x1, x20
        ldr     x2, [x19, #CTX_BUF_BQKV]
        add     x1, x2, x1, lsl #2
        add     x23, x1, x21, lsl #9    // q of head h (h + 1 at +512)
        mul     x1, x0, x20
        ldr     x2, [x19, #CTX_BUF_BATT]
        add     x1, x2, x1, lsl #2
        add     x28, x1, x21, lsl #9    // output of head h (h + 1 at +512)
        ldr     x3, [x19, #CTX_MAX_SEQ]
        lsl     x4, x3, #2              // score row bytes
        ldr     x5, [x19, #CTX_N_HEADS]
        madd    x5, x20, x5, x21        // b * n_heads + h
        ldr     x6, [x19, #CTX_BUF_BSCR]
        madd    x24, x5, x4, x6         // scores of head h
        add     x25, x24, x4            // scores of head h + 1
        ADR_L   x9, tp_gqa
        ldr     w6, [x9]
        udiv    x6, x21, x6             // kv head
        mul     x6, x6, x3
        lsl     x6, x6, #9              // kv_h * max_seq * 512
        ADR_L   x9, tp_kv_base
        ldr     x7, [x9]
        add     x6, x6, x7
        ldr     x7, [x19, #CTX_BUF_K_CACHE]
        add     x26, x7, x6             // K rows of the kv head
        ldr     x7, [x19, #CTX_BUF_V_CACHE]
        add     x27, x7, x6             // V rows

        // scores: positions t, t + 1 for both heads (odd tail: t twice)
        mov     x20, #0
1:      add     x10, x26, x20, lsl #9   // K row t
        add     x11, x10, #512          // K row t + 1
        add     x12, x20, #1
        cmp     x12, x22
        csel    x11, x11, x10, lo
        mov     x13, x23
        add     x14, x23, #512
        movi    v16.4s, #0
        movi    v17.4s, #0
        movi    v18.4s, #0
        movi    v19.4s, #0
        movi    v20.4s, #0
        movi    v21.4s, #0
        movi    v22.4s, #0
        movi    v23.4s, #0
        movi    v24.4s, #0
        movi    v25.4s, #0
        movi    v26.4s, #0
        movi    v27.4s, #0
        movi    v28.4s, #0
        movi    v29.4s, #0
        movi    v30.4s, #0
        movi    v31.4s, #0
        mov     x15, #8                 // 16 floats per step, accumulators as in dot_f32
2:      ldp     q0, q1, [x13], #32      // q h
        ldp     q8, q9, [x10], #32      // K t
        ldp     q12, q13, [x11], #32    // K t + 1
        ldp     q4, q5, [x14], #32      // q h + 1
        fmla    v16.4s, v0.4s, v8.4s    // (h, t): v16..v19
        ldp     q2, q3, [x13], #32
        fmla    v17.4s, v1.4s, v9.4s
        ldp     q10, q11, [x10], #32
        fmla    v20.4s, v0.4s, v12.4s   // (h, t + 1): v20..v23
        ldp     q14, q15, [x11], #32
        fmla    v21.4s, v1.4s, v13.4s
        ldp     q6, q7, [x14], #32
        fmla    v24.4s, v4.4s, v8.4s    // (h + 1, t): v24..v27
        fmla    v25.4s, v5.4s, v9.4s
        fmla    v28.4s, v4.4s, v12.4s   // (h + 1, t + 1): v28..v31
        fmla    v29.4s, v5.4s, v13.4s
        fmla    v18.4s, v2.4s, v10.4s
        fmla    v19.4s, v3.4s, v11.4s
        fmla    v22.4s, v2.4s, v14.4s
        fmla    v23.4s, v3.4s, v15.4s
        fmla    v26.4s, v6.4s, v10.4s
        fmla    v27.4s, v7.4s, v11.4s
        fmla    v30.4s, v6.4s, v14.4s
        fmla    v31.4s, v7.4s, v15.4s
        subs    x15, x15, #1
        b.ne    2b
        fadd    v16.4s, v16.4s, v17.4s  // reduction order of dot_f32
        fadd    v18.4s, v18.4s, v19.4s
        fadd    v16.4s, v16.4s, v18.4s
        fadd    v20.4s, v20.4s, v21.4s
        fadd    v22.4s, v22.4s, v23.4s
        fadd    v20.4s, v20.4s, v22.4s
        fadd    v24.4s, v24.4s, v25.4s
        fadd    v26.4s, v26.4s, v27.4s
        fadd    v24.4s, v24.4s, v26.4s
        fadd    v28.4s, v28.4s, v29.4s
        fadd    v30.4s, v30.4s, v31.4s
        fadd    v28.4s, v28.4s, v30.4s
        faddp   v16.4s, v16.4s, v20.4s
        faddp   v24.4s, v24.4s, v28.4s
        faddp   v16.4s, v16.4s, v24.4s  // (h, t) (h, t + 1) (h + 1, t) (h + 1, t + 1)
        ADR_L   x9, tp_scale
        ld1r    {v0.4s}, [x9]
        fmul    v16.4s, v16.4s, v0.4s
        add     x9, x24, x20, lsl #2
        add     x10, x25, x20, lsl #2
        cmp     x12, x22
        b.hs    3f
        str     d16, [x9]
        st1     {v16.d}[1], [x10]
        add     x20, x20, #2
        cmp     x20, x22
        b.lo    1b
        b       4f
3:      str     s16, [x9]               // last position of an odd count
        st1     {v16.s}[2], [x10]
4:      mov     x0, x24
        mov     x1, x22
        bl      softmax
        mov     x0, x25
        mov     x1, x22
        bl      softmax

        // outputs: 4 blocks of 32 floats per head, accumulated over all positions
        mov     x20, #0
5:      movi    v16.4s, #0
        movi    v17.4s, #0
        movi    v18.4s, #0
        movi    v19.4s, #0
        movi    v20.4s, #0
        movi    v21.4s, #0
        movi    v22.4s, #0
        movi    v23.4s, #0
        movi    v24.4s, #0
        movi    v25.4s, #0
        movi    v26.4s, #0
        movi    v27.4s, #0
        movi    v28.4s, #0
        movi    v29.4s, #0
        movi    v30.4s, #0
        movi    v31.4s, #0
        add     x10, x27, x20
        mov     x11, x24
        mov     x12, x25
        mov     x13, x22
6:      ldr     s8, [x11], #4           // weight of position t, head h
        ldr     s9, [x12], #4           // head h + 1
        ldp     q0, q1, [x10]
        ldp     q2, q3, [x10, #32]
        fmla    v16.4s, v0.4s, v8.s[0]
        ldp     q4, q5, [x10, #64]
        fmla    v24.4s, v0.4s, v9.s[0]
        ldp     q6, q7, [x10, #96]
        fmla    v17.4s, v1.4s, v8.s[0]
        add     x10, x10, #512
        fmla    v25.4s, v1.4s, v9.s[0]
        fmla    v18.4s, v2.4s, v8.s[0]
        fmla    v26.4s, v2.4s, v9.s[0]
        fmla    v19.4s, v3.4s, v8.s[0]
        fmla    v27.4s, v3.4s, v9.s[0]
        fmla    v20.4s, v4.4s, v8.s[0]
        fmla    v28.4s, v4.4s, v9.s[0]
        fmla    v21.4s, v5.4s, v8.s[0]
        fmla    v29.4s, v5.4s, v9.s[0]
        fmla    v22.4s, v6.4s, v8.s[0]
        fmla    v30.4s, v6.4s, v9.s[0]
        fmla    v23.4s, v7.4s, v8.s[0]
        fmla    v31.4s, v7.4s, v9.s[0]
        subs    x13, x13, #1
        b.ne    6b
        add     x9, x28, x20
        stp     q16, q17, [x9]
        stp     q18, q19, [x9, #32]
        stp     q20, q21, [x9, #64]
        stp     q22, q23, [x9, #96]
        add     x9, x9, #512
        stp     q24, q25, [x9]
        stp     q26, q27, [x9, #32]
        stp     q28, q29, [x9, #64]
        stp     q30, q31, [x9, #96]
        add     x20, x20, #128
        cmp     x20, #512
        b.lo    5b
        mov     x0, x28                 // heads h, h + 1 into the q8b slot of
        mov     x1, #256                // token b at element h * 128
        ADR_L   x9, tp_pos
        ldr     x9, [x9]
        sub     x2, x22, x9
        sub     x2, x2, #1              // b = n - 1 - pos0
        lsl     x3, x21, #7
        bl      quantize_x_slot
        ldp     x27, x28, [sp, #80]
        ldp     x25, x26, [sp, #64]
        ldp     x23, x24, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #96
        ret

        .bss
        .balign 64
g_ctx:  .skip   CTX_SIZE
        .balign 8
fb_norm:  .skip 8                       // RMSNorm weights of norm_item
fb_res:   .skip 8                       // 1 = norm_item adds the residual first
fb_qnorm: .skip 8                       // QK-norm weights of qk_item
fb_knorm: .skip 8
