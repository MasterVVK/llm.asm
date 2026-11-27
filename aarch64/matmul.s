// ============================================================================
// matmul.s — Q8_0 x Q8_0 matmul with SDOT (ARMv8.2 dotprod), AArch64 port
//
// Weight layout (.asmbin v2, per-row split Q8_0, block = 32):
//   row = [K int8 weights][K/32 fp16 block scales], row_stride = K + K/16
// Activations are quantized once per matmul into q8_xq (int8) + q8_xs (f32),
// exactly like the x86 FAST_Q8=1 path. SDOT multiplies signed bytes directly,
// so the x86 abs/sign trick is not needed.
// ============================================================================
        .include "constants.s"

        .ifndef PF_DIST
        .equ    PF_DIST, 1024           // software prefetch distance in bytes
        .endif

        .text
        .global quantize_x_q8_0, matmul_q8_0_range, dequant_row_q8_0
        .global dot_f32, saxpy_f32, q8_xq, q8_xs
        .global repack_q8_0_4x4, matmul_q8_0_4x4_pf, matmul_q8_0_4x4_nopf
        .global quantize_x_batch, quantize_x_slot, matmul_q8_0_4x4b_pf, matmul_q8_0_4x4b_nopf
        .global g_mm_b, g_mm_m4

// void quantize_x_q8_0(const float *x x0, uint64 K x1)
// q8_xs[b] = max|x| / 127, q8_xq = round-half-even(x * 127 / max|x|) in [-127, 127]
quantize_x_q8_0:
        ADR_L   x2, q8_xq
        ADR_L   x3, q8_xs
// quantize_core: x0 = x, x1 = K, x2 = xq, x3 = xs. Clobbers x0-x4, SIMD.
quantize_core:
        ADR_L   x4, q8_consts
        ldp     s29, s30, [x4]          // 1/127, 127.0
        movi    v28.16b, #0x81          // -127 as int8
        movi    v31.4s, #0
        lsr     x1, x1, #5              // blocks
1:      cbz     x1, 2f
        ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
        ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
        fabs    v16.4s, v0.4s
        fabs    v17.4s, v1.4s
        fabs    v18.4s, v2.4s
        fabs    v19.4s, v3.4s
        fabs    v20.4s, v4.4s
        fabs    v21.4s, v5.4s
        fabs    v22.4s, v6.4s
        fabs    v23.4s, v7.4s
        fmax    v16.4s, v16.4s, v17.4s
        fmax    v18.4s, v18.4s, v19.4s
        fmax    v20.4s, v20.4s, v21.4s
        fmax    v22.4s, v22.4s, v23.4s
        fmax    v16.4s, v16.4s, v18.4s
        fmax    v20.4s, v20.4s, v22.4s
        fmax    v16.4s, v16.4s, v20.4s
        fmaxv   s16, v16.4s             // max |x|
        fmul    s17, s16, s29
        str     s17, [x3], #4           // scale
        fdiv    s18, s30, s16           // 127 / max (inf if max == 0)
        fcmp    s16, #0.0
        fcsel   s18, s18, s31, gt       // 0 if max == 0
        fmul    v0.4s, v0.4s, v18.s[0]
        fmul    v1.4s, v1.4s, v18.s[0]
        fmul    v2.4s, v2.4s, v18.s[0]
        fmul    v3.4s, v3.4s, v18.s[0]
        fmul    v4.4s, v4.4s, v18.s[0]
        fmul    v5.4s, v5.4s, v18.s[0]
        fmul    v6.4s, v6.4s, v18.s[0]
        fmul    v7.4s, v7.4s, v18.s[0]
        fcvtns  v0.4s, v0.4s
        fcvtns  v1.4s, v1.4s
        fcvtns  v2.4s, v2.4s
        fcvtns  v3.4s, v3.4s
        fcvtns  v4.4s, v4.4s
        fcvtns  v5.4s, v5.4s
        fcvtns  v6.4s, v6.4s
        fcvtns  v7.4s, v7.4s
        sqxtn   v16.4h, v0.4s
        sqxtn2  v16.8h, v1.4s
        sqxtn   v17.4h, v2.4s
        sqxtn2  v17.8h, v3.4s
        sqxtn   v18.4h, v4.4s
        sqxtn2  v18.8h, v5.4s
        sqxtn   v19.4h, v6.4s
        sqxtn2  v19.8h, v7.4s
        sqxtn   v20.8b, v16.8h
        sqxtn2  v20.16b, v17.8h
        sqxtn   v21.8b, v18.8h
        sqxtn2  v21.16b, v19.8h
        smax    v20.16b, v20.16b, v28.16b
        smax    v21.16b, v21.16b, v28.16b
        stp     q20, q21, [x2], #32
        sub     x1, x1, #1
        b       1b
2:      ret

// void matmul_q8_0_range(const void *W x0, float *out x1, uint64 K x2,
//                        uint64 r0 x3, uint64 r1 x4)
// out[r] = sum_b sdot(w[r,b], xq[b]) * scale_w[r,b] * xs[b] for r in [r0, r1).
// Each row is computed by one thread in a fixed order: results do not depend
// on the thread count or on which thread took the row.
matmul_q8_0_range:
        cmp     x4, x3
        b.ls    9f
        add     x5, x2, x2, lsr #4      // row_stride = K + K/16
        madd    x10, x3, x5, x0         // w row ptr
        add     x1, x1, x3, lsl #2      // out ptr
        sub     x4, x4, x3              // rows
        lsr     x6, x2, #7              // groups of 4 blocks (128 B)
        lsr     x7, x2, #5
        and     x7, x7, #3              // tail blocks
        ADR_L   x14, q8_xq
        ADR_L   x15, q8_xs
1:      mov     x11, x10                // weights
        add     x12, x10, x2            // fp16 scales of this row
        mov     x13, x14                // xq
        mov     x16, x15                // xs
        movi    v24.4s, #0
        movi    v25.4s, #0
        mov     x9, x6
        cbz     x9, 3f
2:      ldr     d26, [x12], #8          // 4 fp16 weight scales
        ldr     q27, [x16], #16         // 4 activation scales
        ldp     q0, q1, [x11], #32
        ldp     q16, q17, [x13], #32
        ldp     q2, q3, [x11], #32
        ldp     q18, q19, [x13], #32
        fcvtl   v26.4s, v26.4h
        movi    v28.4s, #0
        movi    v29.4s, #0
        ldp     q4, q5, [x11], #32
        ldp     q20, q21, [x13], #32
        sdot    v28.4s, v0.16b, v16.16b
        sdot    v29.4s, v2.16b, v18.16b
        ldp     q6, q7, [x11], #32
        ldp     q22, q23, [x13], #32
        sdot    v28.4s, v1.16b, v17.16b
        sdot    v29.4s, v3.16b, v19.16b
        movi    v30.4s, #0
        movi    v31.4s, #0
        fmul    v26.4s, v26.4s, v27.4s  // combined scales
        sdot    v30.4s, v4.16b, v20.16b
        sdot    v31.4s, v6.16b, v22.16b
        .ifndef NOPF
        prfm    pldl1keep, [x11, #512]
        .endif
        sdot    v30.4s, v5.16b, v21.16b
        sdot    v31.4s, v7.16b, v23.16b
        scvtf   v28.4s, v28.4s
        scvtf   v29.4s, v29.4s
        scvtf   v30.4s, v30.4s
        scvtf   v31.4s, v31.4s
        fmla    v24.4s, v28.4s, v26.s[0]
        fmla    v25.4s, v29.4s, v26.s[1]
        fmla    v24.4s, v30.4s, v26.s[2]
        fmla    v25.4s, v31.4s, v26.s[3]
        subs    x9, x9, #1
        b.ne    2b
3:      mov     x9, x7                  // tail: single 32-byte blocks
4:      cbz     x9, 5f
        ldr     h26, [x12], #2
        fcvt    s26, h26
        ldr     s27, [x16], #4
        fmul    s26, s26, s27
        ldp     q0, q1, [x11], #32
        ldp     q16, q17, [x13], #32
        movi    v28.4s, #0
        sdot    v28.4s, v0.16b, v16.16b
        sdot    v28.4s, v1.16b, v17.16b
        scvtf   v28.4s, v28.4s
        fmla    v24.4s, v28.4s, v26.s[0]
        sub     x9, x9, #1
        b       4b
5:      fadd    v24.4s, v24.4s, v25.4s
        faddp   v24.4s, v24.4s, v24.4s
        faddp   s24, v24.2s
        str     s24, [x1], #4
        add     x10, x10, x5
        subs    x4, x4, #1
        b.ne    1b
9:      ret

// void repack_q8_0_4x4(const void *src x0, void *dst x1, uint64 M x2, uint64 K x3)
// Interleaves every 4 rows the way SDOT-by-element wants them (llama.cpp's
// q8_0_4x4 idea). A 4-row group keeps its size and offset (4 * row_stride):
//   [K/32 blocks x 128 B: for k-group g (4 bytes) -> rows 0..3, 4 B each]
//   [K/32 blocks x 8 B: fp16 scales of rows 0..3]
repack_q8_0_4x4:
        add     x5, x3, x3, lsr #4      // row_stride
        lsr     x6, x3, #5              // blocks per row
        lsr     x2, x2, #2              // groups
1:      cbz     x2, 4f
        add     x7, x0, x5              // row 1
        add     x9, x7, x5              // row 2
        add     x10, x9, x5             // row 3
        mov     x11, x1                 // dst weights
        add     x12, x1, x3, lsl #2     // dst scales (after 4*K bytes)
        mov     x13, #0                 // block
2:      cmp     x13, x6
        b.hs    3f
        lsl     x14, x13, #5
        add     x15, x0, x14
        ldp     q0, q1, [x15]
        add     x15, x7, x14
        ldp     q2, q3, [x15]
        add     x15, x9, x14
        ldp     q4, q5, [x15]
        add     x15, x10, x14
        ldp     q6, q7, [x15]
        trn1    v16.4s, v0.4s, v2.4s    // first 16 bytes of rows 0..3
        trn2    v17.4s, v0.4s, v2.4s
        trn1    v18.4s, v4.4s, v6.4s
        trn2    v19.4s, v4.4s, v6.4s
        trn1    v20.2d, v16.2d, v18.2d
        trn1    v21.2d, v17.2d, v19.2d
        trn2    v22.2d, v16.2d, v18.2d
        trn2    v23.2d, v17.2d, v19.2d
        stp     q20, q21, [x11], #32
        stp     q22, q23, [x11], #32
        trn1    v16.4s, v1.4s, v3.4s    // last 16 bytes of rows 0..3
        trn2    v17.4s, v1.4s, v3.4s
        trn1    v18.4s, v5.4s, v7.4s
        trn2    v19.4s, v5.4s, v7.4s
        trn1    v20.2d, v16.2d, v18.2d
        trn1    v21.2d, v17.2d, v19.2d
        trn2    v22.2d, v16.2d, v18.2d
        trn2    v23.2d, v17.2d, v19.2d
        stp     q20, q21, [x11], #32
        stp     q22, q23, [x11], #32
        add     x14, x3, x13, lsl #1    // scale offset inside a row
        ldrh    w15, [x0, x14]
        strh    w15, [x12], #2
        ldrh    w15, [x7, x14]
        strh    w15, [x12], #2
        ldrh    w15, [x9, x14]
        strh    w15, [x12], #2
        ldrh    w15, [x10, x14]
        strh    w15, [x12], #2
        add     x13, x13, #1
        b       2b
3:      add     x0, x0, x5, lsl #2
        add     x1, x1, x5, lsl #2
        sub     x2, x2, #1
        b       1b
4:      ret

// void matmul_q8_0_4x4_{pf,nopf}(const void *Wr x0, float *out x1, uint64 K x2,
//                            uint64 r0 x3, uint64 r1 x4)   r0, r1 multiples of 4
// Repacked layout: one SDOT-by-element multiplies 4 bytes of x with 4 bytes of
// each of 4 rows, so x is loaded once per 4 rows and lanes are rows (no
// horizontal reduction per row). Two blocks per iteration, loads interleaved
// with independent SDOT chains for the in-order Cortex-A55.
// Two variants: Cortex-A76 gains from software prefetch, while on the in-order
// Cortex-A55 PRFM costs ~35% (measured), so A55 threads use the _nopf copy.
        .macro MM4X4 name, pf
\name:
        cmp     x4, x3
        b.ls    9f
        add     x5, x2, x2, lsr #4      // row_stride
        lsl     x6, x5, #2              // group stride
        lsr     x7, x3, #2
        madd    x10, x7, x6, x0         // first group
        add     x1, x1, x3, lsl #2
        sub     x4, x4, x3
        lsr     x4, x4, #2              // groups
        lsr     x9, x2, #6              // block pairs (K % 64 == 0)
        ADR_L   x14, q8_xq
        ADR_L   x15, q8_xs
1:      mov     x11, x10                // weights
        add     x12, x10, x2, lsl #2    // fp16 scales
        mov     x13, x14
        mov     x16, x15
        movi    v24.4s, #0
        movi    v25.4s, #0
        mov     x17, x9
2:      ld1     {v0.16b, v1.16b, v2.16b, v3.16b}, [x11], #64   // block b, k-groups 0..3
        ldp     q16, q17, [x13], #32    // x block b
        movi    v28.4s, #0
        movi    v29.4s, #0
        sdot    v28.4s, v0.16b, v16.4b[0]
        sdot    v29.4s, v1.16b, v16.4b[1]
        ld1     {v4.16b, v5.16b, v6.16b, v7.16b}, [x11], #64   // k-groups 4..7
        sdot    v28.4s, v2.16b, v16.4b[2]
        sdot    v29.4s, v3.16b, v16.4b[3]
        ldp     q18, q19, [x13], #32    // x block b+1
        sdot    v28.4s, v4.16b, v17.4b[0]
        sdot    v29.4s, v5.16b, v17.4b[1]
        ld1     {v0.16b, v1.16b, v2.16b, v3.16b}, [x11], #64   // block b+1
        movi    v30.4s, #0
        movi    v31.4s, #0
        sdot    v28.4s, v6.16b, v17.4b[2]
        sdot    v29.4s, v7.16b, v17.4b[3]
        sdot    v30.4s, v0.16b, v18.4b[0]
        sdot    v31.4s, v1.16b, v18.4b[1]
        ld1     {v4.16b, v5.16b, v6.16b, v7.16b}, [x11], #64
        ldr     q26, [x12], #16         // fp16 scales: 4 rows x 2 blocks
        sdot    v30.4s, v2.16b, v18.4b[2]
        sdot    v31.4s, v3.16b, v18.4b[3]
        ldr     d27, [x16], #8          // x scales of both blocks
        sdot    v30.4s, v4.16b, v19.4b[0]
        sdot    v31.4s, v5.16b, v19.4b[1]
        fcvtl   v20.4s, v26.4h
        fcvtl2  v21.4s, v26.8h
        sdot    v30.4s, v6.16b, v19.4b[2]
        sdot    v31.4s, v7.16b, v19.4b[3]
        add     v28.4s, v28.4s, v29.4s
        fmul    v20.4s, v20.4s, v27.s[0]
        fmul    v21.4s, v21.4s, v27.s[1]
        add     v30.4s, v30.4s, v31.4s
        scvtf   v28.4s, v28.4s
        scvtf   v30.4s, v30.4s
        .if \pf
        prfm    pldl1keep, [x11, #PF_DIST]       // one prefetch per 64-byte line
        prfm    pldl1keep, [x11, #(PF_DIST + 64)]
        prfm    pldl1keep, [x11, #(PF_DIST + 128)]
        prfm    pldl1keep, [x11, #(PF_DIST + 192)]
        .endif
        fmla    v24.4s, v28.4s, v20.4s
        fmla    v25.4s, v30.4s, v21.4s
        subs    x17, x17, #1
        b.ne    2b
        fadd    v24.4s, v24.4s, v25.4s
        str     q24, [x1], #16          // rows 4g .. 4g+3
        add     x10, x10, x6
        subs    x4, x4, #1
        b.ne    1b
9:      ret

        .endm

        MM4X4 matmul_q8_0_4x4_pf, 1
        MM4X4 matmul_q8_0_4x4_nopf, 0


// Batched prefill GEMM, 4 rows x 2 tokens register tile on the repacked
// weights. Each block's weight registers serve both tokens, so the group is
// read from L1 half as often per token. Per token the int32 SDOT sums are
// exact (the split into accumulators does not matter) and the float sequence
// is the GEMV one: scale = fp32(w scale) * x scale, acc_even/odd += sum *
// scale (FMLA), out = acc_even + acc_odd, so results are bit-identical to
// MM4X4. g_mm_b tokens, out[t] at out + t * g_mm_m4; an odd last token is
// computed as both halves of the tile and stored once.
//   v0-7 weights   v16-19 x (A: 16/17, B: 18/19)   v20-23 / v24-27 int sums
//   of blocks b / b+1   v28-31 acc (A even, A odd, B even, B odd)   v8-14 scales
        .macro MM4X4B name, pf
\name:
        stp     x29, x30, [sp, #-80]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        stp     x23, x24, [sp, #48]
        stp     x25, x26, [sp, #64]
        cmp     x4, x3
        b.ls    9f
        add     x5, x2, x2, lsr #4      // row_stride
        lsl     x6, x5, #2              // group stride
        lsr     x7, x3, #2
        madd    x10, x7, x6, x0         // first group
        add     x1, x1, x3, lsl #2      // out of token 0, first row
        sub     x4, x4, x3
        lsr     x4, x4, #2              // groups
        lsr     x9, x2, #6              // block pairs (K % 64 == 0)
        ADR_L   x19, g_mm_b
        ldr     x19, [x19]
        ADR_L   x20, g_mm_m4
        ldr     x20, [x20]
1:      ADR_L   x14, q8b_xq             // token A slots
        ADR_L   x15, q8b_xs
        mov     x21, x1                 // out of token A
        mov     x22, x19                // tokens left
2:      mov     x25, #XQ_STRIDE         // token B = A + 1
        mov     x26, #XS_STRIDE
        cmp     x22, #2
        b.hs    4f
        mov     x25, #0                 // odd last token: B = A, not stored
        mov     x26, #0
4:      mov     x11, x10                // weights
        add     x12, x10, x2, lsl #2    // fp16 scales
        mov     x13, x14
        add     x23, x14, x25
        mov     x16, x15
        add     x24, x15, x26
        movi    v28.4s, #0
        movi    v29.4s, #0
        movi    v30.4s, #0
        movi    v31.4s, #0
        mov     x17, x9
3:      ld1     {v0.16b, v1.16b, v2.16b, v3.16b}, [x11], #64   // block b, k-groups 0..3
        ldr     q16, [x13], #16
        ldr     q18, [x23], #16
        movi    v20.4s, #0
        movi    v21.4s, #0
        movi    v22.4s, #0
        movi    v23.4s, #0
        sdot    v20.4s, v0.16b, v16.4b[0]
        sdot    v22.4s, v0.16b, v18.4b[0]
        sdot    v21.4s, v1.16b, v16.4b[1]
        sdot    v23.4s, v1.16b, v18.4b[1]
        ld1     {v4.16b, v5.16b, v6.16b, v7.16b}, [x11], #64   // k-groups 4..7
        sdot    v20.4s, v2.16b, v16.4b[2]
        sdot    v22.4s, v2.16b, v18.4b[2]
        ldr     q17, [x13], #16
        sdot    v21.4s, v3.16b, v16.4b[3]
        sdot    v23.4s, v3.16b, v18.4b[3]
        ldr     q19, [x23], #16
        sdot    v20.4s, v4.16b, v17.4b[0]
        sdot    v22.4s, v4.16b, v19.4b[0]
        ld1     {v0.16b, v1.16b, v2.16b, v3.16b}, [x11], #64   // block b+1
        sdot    v21.4s, v5.16b, v17.4b[1]
        sdot    v23.4s, v5.16b, v19.4b[1]
        ldr     q16, [x13], #16
        sdot    v20.4s, v6.16b, v17.4b[2]
        sdot    v22.4s, v6.16b, v19.4b[2]
        ldr     q18, [x23], #16
        sdot    v21.4s, v7.16b, v17.4b[3]
        sdot    v23.4s, v7.16b, v19.4b[3]
        movi    v24.4s, #0
        movi    v25.4s, #0
        movi    v26.4s, #0
        movi    v27.4s, #0
        ldr     q8, [x12], #16          // fp16 scales: 4 rows x 2 blocks
        sdot    v24.4s, v0.16b, v16.4b[0]
        sdot    v26.4s, v0.16b, v18.4b[0]
        sdot    v25.4s, v1.16b, v16.4b[1]
        sdot    v27.4s, v1.16b, v18.4b[1]
        ld1     {v4.16b, v5.16b, v6.16b, v7.16b}, [x11], #64
        add     v20.4s, v20.4s, v21.4s
        add     v22.4s, v22.4s, v23.4s
        sdot    v24.4s, v2.16b, v16.4b[2]
        sdot    v26.4s, v2.16b, v18.4b[2]
        ldr     q17, [x13], #16
        fcvtl   v11.4s, v8.4h
        fcvtl2  v12.4s, v8.8h
        sdot    v25.4s, v3.16b, v16.4b[3]
        sdot    v27.4s, v3.16b, v18.4b[3]
        ldr     q19, [x23], #16
        ldr     d9, [x16], #8           // x scales of A (b, b+1)
        ldr     d10, [x24], #8          // x scales of B
        scvtf   v20.4s, v20.4s
        scvtf   v22.4s, v22.4s
        sdot    v24.4s, v4.16b, v17.4b[0]
        sdot    v26.4s, v4.16b, v19.4b[0]
        fmul    v13.4s, v11.4s, v9.s[0]
        fmul    v14.4s, v11.4s, v10.s[0]
        sdot    v25.4s, v5.16b, v17.4b[1]
        sdot    v27.4s, v5.16b, v19.4b[1]
        fmla    v28.4s, v20.4s, v13.4s
        fmla    v30.4s, v22.4s, v14.4s
        sdot    v24.4s, v6.16b, v17.4b[2]
        sdot    v26.4s, v6.16b, v19.4b[2]
        fmul    v13.4s, v12.4s, v9.s[1]
        fmul    v14.4s, v12.4s, v10.s[1]
        sdot    v25.4s, v7.16b, v17.4b[3]
        sdot    v27.4s, v7.16b, v19.4b[3]
        add     v24.4s, v24.4s, v25.4s
        add     v26.4s, v26.4s, v27.4s
        .if \pf
        prfm    pldl1keep, [x11, #PF_DIST]
        prfm    pldl1keep, [x11, #(PF_DIST + 64)]
        prfm    pldl1keep, [x11, #(PF_DIST + 128)]
        prfm    pldl1keep, [x11, #(PF_DIST + 192)]
        .endif
        scvtf   v24.4s, v24.4s
        scvtf   v26.4s, v26.4s
        fmla    v29.4s, v24.4s, v13.4s
        fmla    v31.4s, v26.4s, v14.4s
        subs    x17, x17, #1
        b.ne    3b
        fadd    v28.4s, v28.4s, v29.4s
        str     q28, [x21]              // rows 4g .. 4g+3 of token A
        cbz     x25, 5f
        fadd    v30.4s, v30.4s, v31.4s
        add     x21, x21, x20
        str     q30, [x21]              // token B
        add     x21, x21, x20
        add     x14, x14, x25, lsl #1
        add     x15, x15, x26, lsl #1
        subs    x22, x22, #2
        b.ne    2b
5:      add     x1, x1, #16
        add     x10, x10, x6
        subs    x4, x4, #1
        b.ne    1b
9:      ldp     x25, x26, [sp, #64]
        ldp     x23, x24, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #80
        ret
        .endm

        MM4X4B matmul_q8_0_4x4b_pf, 1
        MM4X4B matmul_q8_0_4x4b_nopf, 0

// void quantize_x_batch(const float *X x0, uint64 K x1, uint64 B x2, uint64 stride x3)
quantize_x_batch:
        stp     x29, x30, [sp, #-64]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        str     x23, [sp, #48]
        mov     x19, x0
        mov     x20, x1
        mov     x21, x2
        mov     x22, x3
        mov     x23, #0
1:      cmp     x23, x21
        b.hs    2f
        mov     x0, x19
        mov     x1, x20
        mov     x4, #XQ_STRIDE
        ADR_L   x2, q8b_xq
        madd    x2, x23, x4, x2
        mov     x4, #XS_STRIDE
        ADR_L   x3, q8b_xs
        madd    x3, x23, x4, x3
        bl      quantize_core
        add     x19, x19, x22
        add     x23, x23, #1
        b       1b
2:      ldr     x23, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #64
        ret

// void quantize_x_slot(const float *x x0, uint64 n x1, uint64 t x2, uint64 k0 x3)
// Quantizes x[0..n) into elements k0 .. k0 + n - 1 of the q8b slot of token t
// (k0 and n multiples of 32). Blocks are independent, so a token quantized in
// parts gets the same bytes as quantize_x_batch. Clobbers x0-x6, SIMD.
quantize_x_slot:
        mov     x4, #XQ_STRIDE
        ADR_L   x5, q8b_xq
        madd    x5, x2, x4, x5
        add     x5, x5, x3              // int8 of element k0
        mov     x4, #XS_STRIDE
        ADR_L   x6, q8b_xs
        madd    x6, x2, x4, x6
        add     x3, x6, x3, lsr #3      // scale of block k0 / 32
        mov     x2, x5
        b       quantize_core

// void dequant_row_q8_0(const void *row x0, float *out x1, uint64 K x2)
dequant_row_q8_0:
        add     x3, x0, x2              // fp16 scales
        lsr     x2, x2, #5
1:      cbz     x2, 2f
        ldr     h16, [x3], #2
        fcvt    s16, h16
        ldp     q0, q1, [x0], #32
        sxtl    v2.8h, v0.8b
        sxtl2   v3.8h, v0.16b
        sxtl    v4.8h, v1.8b
        sxtl2   v5.8h, v1.16b
        sxtl    v20.4s, v2.4h
        sxtl2   v21.4s, v2.8h
        sxtl    v22.4s, v3.4h
        sxtl2   v23.4s, v3.8h
        sxtl    v24.4s, v4.4h
        sxtl2   v25.4s, v4.8h
        sxtl    v26.4s, v5.4h
        sxtl2   v27.4s, v5.8h
        .irp r, 20, 21, 22, 23, 24, 25, 26, 27
        scvtf   v\r\().4s, v\r\().4s
        fmul    v\r\().4s, v\r\().4s, v16.s[0]
        .endr
        stp     q20, q21, [x1], #32
        stp     q22, q23, [x1], #32
        stp     q24, q25, [x1], #32
        stp     q26, q27, [x1], #32
        sub     x2, x2, #1
        b       1b
2:      ret

// float dot_f32(const float *a x0, const float *b x1, uint64 n x2) -> s0  (n % 16 == 0)
dot_f32:
        movi    v0.4s, #0
        movi    v1.4s, #0
        movi    v2.4s, #0
        movi    v3.4s, #0
1:      ldp     q4, q5, [x0], #32
        ldp     q6, q7, [x0], #32
        ldp     q16, q17, [x1], #32
        ldp     q18, q19, [x1], #32
        fmla    v0.4s, v4.4s, v16.4s
        fmla    v1.4s, v5.4s, v17.4s
        fmla    v2.4s, v6.4s, v18.4s
        fmla    v3.4s, v7.4s, v19.4s
        subs    x2, x2, #16
        b.gt    1b
        fadd    v0.4s, v0.4s, v1.4s
        fadd    v2.4s, v2.4s, v3.4s
        fadd    v0.4s, v0.4s, v2.4s
        faddp   v0.4s, v0.4s, v0.4s
        faddp   s0, v0.2s
        ret

// void saxpy_f32(float *dst x0, const float *src x1, uint64 n x2, float alpha s0)
saxpy_f32:
1:      ldp     q4, q5, [x0]
        ldp     q6, q7, [x0, #32]
        ldp     q16, q17, [x1], #32
        ldp     q18, q19, [x1], #32
        fmla    v4.4s, v16.4s, v0.s[0]
        fmla    v5.4s, v17.4s, v0.s[0]
        fmla    v6.4s, v18.4s, v0.s[0]
        fmla    v7.4s, v19.4s, v0.s[0]
        stp     q4, q5, [x0], #32
        stp     q6, q7, [x0], #32
        subs    x2, x2, #16
        b.gt    1b
        ret

        .section .rodata
        .balign 8
q8_consts:
        .float  0.007874015748031496, 127.0     // 1/127, 127

        .bss
        .balign 64
q8_xq:  .skip   Q8_MAX_K
        .balign 64
q8_xs:  .skip   (Q8_MAX_K / 32) * 4
        .balign 64
q8b_xq: .skip   MAX_BATCH * XQ_STRIDE
        .balign 64
q8b_xs: .skip   MAX_BATCH * XS_STRIDE
        .balign 8
g_mm_b: .skip   8
g_mm_m4: .skip  8
