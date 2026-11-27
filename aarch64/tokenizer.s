// ============================================================================
// tokenizer.s — BPE encoder over the pre-built merge table + Qwen3 chat
// template + token output (AArch64 port of shared/tokenizer.asm, same algorithm)
// ============================================================================
        .include "constants.s"

        .text
        .global tok_encode_bpe, tok_build_chat_prompt, tok_write_stdout

// uint64 tok_encode_bpe(const uint8 *text x0, uint64 len x1, uint32 *out x2, uint64 max x3)
// Byte -> base token, then repeatedly apply the lowest-rank merge.
tok_encode_bpe:
        cmp     x1, x3
        csel    x1, x1, x3, ls
        cbz     x1, 9f
        ADR_L   x9, g_ctx
        ldr     x4, [x9, #CTX_B2T_PTR]
        ldr     x5, [x9, #CTX_HT_PTR]
        ldr     x6, [x9, #CTX_HT_MASK]
        mov     x7, #0
1:      ldrb    w10, [x0, x7]           // map bytes to base tokens
        ldr     w10, [x4, x10, lsl #2]
        str     w10, [x2, x7, lsl #2]
        add     x7, x7, #1
        cmp     x7, x1
        b.lo    1b
        ldr     x15, =0x9E3779B97F4A7C15
2:      cmp     x1, #2                  // merge loop, n = x1
        b.lo    8f
        mov     w11, #-1                // best rank
        mov     x12, #-1                // best index
        mov     w13, #0                 // best merged id
        sub     x14, x1, #1
        mov     x7, #0
3:      cmp     x7, x14
        b.hs    6f
        add     x16, x2, x7, lsl #2
        ldp     w17, w10, [x16]         // toks[i], toks[i+1]
        orr     x10, x10, x17, lsl #32  // key
        mul     x16, x10, x15
        lsr     x16, x16, #HT_SHIFT     // slot
4:      add     x17, x5, x16, lsl #4
        ldr     w3, [x17, #HT_RANK]
        cmn     w3, #1
        b.eq    5f                      // empty: pair not mergeable
        ldr     x8, [x17, #HT_KEY]
        cmp     x8, x10
        b.eq    7f
        add     x16, x16, #1
        and     x16, x16, x6
        b       4b
7:      cmp     w3, w11
        b.hs    5f
        mov     w11, w3
        mov     x12, x7
        ldr     w13, [x17, #HT_MERGED]
5:      add     x7, x7, #1
        b       3b
6:      cmp     x12, #0
        b.lt    8f
        str     w13, [x2, x12, lsl #2]  // apply merge, shift tail left
        add     x7, x12, #1
        sub     x14, x1, #1
10:     cmp     x7, x14
        b.hs    11f
        add     x16, x7, #1
        ldr     w10, [x2, x16, lsl #2]
        str     w10, [x2, x7, lsl #2]
        add     x7, x7, #1
        b       10b
11:     sub     x1, x1, #1
        b       2b
8:      mov     x0, x1
        ret
9:      mov     x0, #0
        ret

// uint64 tok_build_chat_prompt(const char *prompt x0, uint32 *out x1, uint64 max x2,
//                              int raw x3, int think x4)
// <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n[<think>\n\n</think>\n\n]
tok_build_chat_prompt:
        stp     x29, x30, [sp, #-64]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        str     x23, [sp, #48]
        mov     x19, x0
        mov     x20, x1
        mov     x21, x2
        mov     x22, x3
        mov     x23, x4
        bl      str_len
        mov     x1, x0
        cbz     x22, 1f
        mov     x0, x19                 // raw: encode prompt as is
        mov     x2, x20
        mov     x3, x21
        bl      tok_encode_bpe
        b       9f
1:      ldr     w5, =TOK_IM_START
        str     w5, [x20, #0]
        mov     w5, #TOK_USER
        str     w5, [x20, #4]
        mov     w5, #TOK_NL
        str     w5, [x20, #8]
        mov     x0, x19
        add     x2, x20, #12
        sub     x3, x21, #16
        bl      tok_encode_bpe
        add     x6, x0, #3              // count so far
        add     x7, x20, x6, lsl #2
        ldr     w5, =TOK_IM_END
        str     w5, [x7], #4
        mov     w5, #TOK_NL
        str     w5, [x7], #4
        ldr     w5, =TOK_IM_START
        str     w5, [x7], #4
        ldr     w5, =TOK_ASSISTANT
        str     w5, [x7], #4
        mov     w5, #TOK_NL
        str     w5, [x7], #4
        add     x6, x6, #5
        cbnz    x23, 2f
        ldr     w5, =TOK_THINK_START    // empty think block (default no-think mode)
        str     w5, [x7], #4
        mov     w5, #TOK_NL2
        str     w5, [x7], #4
        ldr     w5, =TOK_THINK_END
        str     w5, [x7], #4
        mov     w5, #TOK_NL2
        str     w5, [x7], #4
        add     x6, x6, #4
2:      mov     x0, x6
9:      ldr     x23, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #64
        ret

// void tok_write_stdout(uint32 token w0) — raw UTF-8 bytes of the token to fd 1
tok_write_stdout:
        ADR_L   x9, g_ctx
        ldr     x10, [x9, #CTX_VOCAB_SIZE]
        mov     w0, w0
        cmp     x0, x10
        b.hs    1f
        ldr     x10, [x9, #CTX_TOK_OFFS_PTR]
        add     x10, x10, x0, lsl #2
        ldp     w11, w12, [x10]         // start, end
        subs    x2, x12, x11
        b.eq    1f
        ldr     x1, [x9, #CTX_TOK_BYTES_PTR]
        add     x1, x1, x11
        mov     x0, #1
        b       sys_write_all
1:      ret
