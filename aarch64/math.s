// ============================================================================
// math.s — NEON math primitives for the AArch64 port (zero libm)
// Same algorithms as shared/math.asm (Cody-Waite + Taylor), with fused
// multiply-add like the upstream AVX2+FMA build.
//
// Register convention inside this port: every SIMD register is caller-saved.
// expf_neon clobbers only v0-v7, so callers keep their state in v16-v31.
// ============================================================================
        .include "constants.s"

        .text
        .global expf_neon, rmsnorm, rope_apply, softmax, silu_mul, vec_add
        .global argmax_f32, fast_sincos_f64, log_f64, exp_f64
        .global init_rope_freqs, compute_rope_cs

// v0.4s = expf(v0.4s); x < -87.33654 -> 0, x > 88.37626 clamped.
// Clobbers v0-v7, x9.
expf_neon:
        ADR_L   x9, expf_consts
        ld1r    {v4.4s}, [x9], #4       // exp_lo
        fcmge   v5.4s, v0.4s, v4.4s     // keep-mask
        fmax    v0.4s, v0.4s, v4.4s
        ld1r    {v4.4s}, [x9], #4       // exp_hi
        fmin    v0.4s, v0.4s, v4.4s
        ld1r    {v4.4s}, [x9], #4       // log2e
        fmul    v1.4s, v0.4s, v4.4s
        frintn  v1.4s, v1.4s            // k_f = round-half-even(x * log2e)
        fcvtzs  v2.4s, v1.4s            // k_i
        ld1r    {v4.4s}, [x9], #4       // ln2_hi
        fmls    v0.4s, v1.4s, v4.4s
        ld1r    {v4.4s}, [x9], #4       // ln2_lo
        fmls    v0.4s, v1.4s, v4.4s     // r
        ld1r    {v3.4s}, [x9], #4       // c6
        ld1r    {v4.4s}, [x9], #4       // c5
        fmla    v4.4s, v3.4s, v0.4s
        ld1r    {v3.4s}, [x9], #4       // c4
        fmla    v3.4s, v4.4s, v0.4s
        ld1r    {v4.4s}, [x9], #4       // c3
        fmla    v4.4s, v3.4s, v0.4s
        ld1r    {v3.4s}, [x9], #4       // c2
        fmla    v3.4s, v4.4s, v0.4s
        ld1r    {v6.4s}, [x9], #4       // 1.0
        mov     v4.16b, v6.16b
        fmla    v4.4s, v3.4s, v0.4s
        mov     v3.16b, v6.16b
        fmla    v3.4s, v4.4s, v0.4s     // poly
        movi    v7.4s, #127
        add     v2.4s, v2.4s, v7.4s
        shl     v2.4s, v2.4s, #23       // 2^k
        fmul    v0.4s, v3.4s, v2.4s
        and     v0.16b, v0.16b, v5.16b  // flush underflowed lanes to 0
        ret

// void rmsnorm(float *out x0, const float *x x1, const float *w x2, uint64 n x3, float eps s0)
// n multiple of 16; out may equal x.
rmsnorm:
        movi    v16.4s, #0
        movi    v17.4s, #0
        movi    v18.4s, #0
        movi    v19.4s, #0
        mov     x4, x1
        mov     x5, x3
1:      ldp     q20, q21, [x4], #32
        ldp     q22, q23, [x4], #32
        fmla    v16.4s, v20.4s, v20.4s
        fmla    v17.4s, v21.4s, v21.4s
        fmla    v18.4s, v22.4s, v22.4s
        fmla    v19.4s, v23.4s, v23.4s
        subs    x5, x5, #16
        b.gt    1b
        fadd    v16.4s, v16.4s, v17.4s
        fadd    v18.4s, v18.4s, v19.4s
        fadd    v16.4s, v16.4s, v18.4s
        faddp   v16.4s, v16.4s, v16.4s
        faddp   s16, v16.2s
        ucvtf   s17, x3
        fdiv    s16, s16, s17           // mean(x^2)
        fadd    s16, s16, s0
        fsqrt   s16, s16
        fmov    s17, #1.0
        fdiv    s16, s17, s16           // inv_rms
2:      ldp     q20, q21, [x1], #32
        ldp     q22, q23, [x2], #32
        fmul    v20.4s, v20.4s, v16.s[0]
        fmul    v21.4s, v21.4s, v16.s[0]
        fmul    v20.4s, v20.4s, v22.4s
        fmul    v21.4s, v21.4s, v23.4s
        stp     q20, q21, [x0], #32
        subs    x3, x3, #8
        b.gt    2b
        ret

// void rope_apply(float *vec x0, const float *cos x1, const float *sin x2,
//                 uint64 n_heads x3, uint64 head_dim x4)   split-half (NeoX)
rope_apply:
        lsr     x5, x4, #1              // half
        lsl     x6, x5, #2              // half_bytes
1:      cbz     x3, 3f
        add     x7, x0, x6              // x2 half
        mov     x9, #0
2:      ldr     q0, [x0, x9]            // x1
        ldr     q1, [x7, x9]            // x2
        ldr     q2, [x1, x9]            // cos
        ldr     q3, [x2, x9]            // sin
        fmul    v4.4s, v0.4s, v2.4s
        fmls    v4.4s, v1.4s, v3.4s     // x1*cos - x2*sin
        fmul    v5.4s, v1.4s, v2.4s
        fmla    v5.4s, v0.4s, v3.4s     // x2*cos + x1*sin
        str     q4, [x0, x9]
        str     q5, [x7, x9]
        add     x9, x9, #16
        cmp     x9, x6
        b.lo    2b
        add     x0, x0, x6, lsl #1
        sub     x3, x3, #1
        b       1b
3:      ret

// void softmax(float *x x0, uint64 n x1)  (n >= 1)
softmax:
        stp     x29, x30, [sp, #-48]!
        stp     x19, x20, [sp, #16]
        str     x21, [sp, #32]
        mov     x19, x0
        mov     x20, x1
        ldr     s16, [x19]
        mov     x21, #1
1:      cmp     x21, x20
        b.hs    2f
        ldr     s17, [x19, x21, lsl #2]
        fmax    s16, s16, s17
        add     x21, x21, #1
        b       1b
2:      dup     v16.4s, v16.s[0]        // max
        movi    v17.4s, #0              // sum
        mov     x21, #0
3:      add     x2, x21, #4
        cmp     x2, x20
        b.hi    4f
        add     x3, x19, x21, lsl #2
        ldr     q0, [x3]
        fsub    v0.4s, v0.4s, v16.4s
        bl      expf_neon               // clobbers v0-v7 and x9 only
        add     x3, x19, x21, lsl #2
        str     q0, [x3]
        fadd    v17.4s, v17.4s, v0.4s
        mov     x21, x2
        b       3b
4:      faddp   v17.4s, v17.4s, v17.4s
        faddp   s17, v17.2s
5:      cmp     x21, x20                // scalar tail
        b.hs    6f
        ldr     s0, [x19, x21, lsl #2]
        fsub    s0, s0, s16
        dup     v0.4s, v0.s[0]
        bl      expf_neon
        str     s0, [x19, x21, lsl #2]
        fadd    s17, s17, s0
        add     x21, x21, #1
        b       5b
6:      fmov    s18, #1.0
        fdiv    s17, s18, s17           // inv_sum
        mov     x21, #0
7:      add     x2, x21, #4
        cmp     x2, x20
        b.hi    8f
        add     x3, x19, x21, lsl #2
        ldr     q0, [x3]
        fmul    v0.4s, v0.4s, v17.s[0]
        str     q0, [x3]
        mov     x21, x2
        b       7b
8:      cmp     x21, x20
        b.hs    9f
        ldr     s0, [x19, x21, lsl #2]
        fmul    s0, s0, s17
        str     s0, [x19, x21, lsl #2]
        add     x21, x21, #1
        b       8b
9:      ldr     x21, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #48
        ret

// void silu_mul(float *gate x0, const float *up x1, uint64 n x2)  (n multiple of 4)
// e = exp(-|x|); num = x < 0 ? e : 1; gate = x * num / (1 + e) * up
silu_mul:
        stp     x29, x30, [sp, #-48]!
        stp     x19, x20, [sp, #16]
        str     x21, [sp, #32]
        mov     x19, x0
        mov     x20, x1
        mov     x21, x2
        fmov    v18.4s, #1.0
1:      cbz     x21, 2f
        ldr     q16, [x19]
        ldr     q17, [x20], #16
        fabs    v0.4s, v16.4s
        fneg    v0.4s, v0.4s
        bl      expf_neon               // v0 = e
        cmlt    v19.4s, v16.4s, #0      // sign mask of x
        bsl     v19.16b, v0.16b, v18.16b  // num
        fadd    v0.4s, v0.4s, v18.4s    // 1 + e
        fmul    v19.4s, v19.4s, v16.4s
        fdiv    v19.4s, v19.4s, v0.4s
        fmul    v19.4s, v19.4s, v17.4s
        str     q19, [x19], #16
        sub     x21, x21, #4
        b       1b
2:      ldr     x21, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #48
        ret

// void vec_add(float *dst x0, const float *src x1, uint64 n x2)  (n multiple of 8)
vec_add:
1:      ldp     q0, q1, [x0]
        ldp     q2, q3, [x1], #32
        fadd    v0.4s, v0.4s, v2.4s
        fadd    v1.4s, v1.4s, v3.4s
        stp     q0, q1, [x0], #32
        subs    x2, x2, #8
        b.gt    1b
        ret

// uint64 argmax_f32(const float *x x0, uint64 n x1) — first index on ties
argmax_f32:
        mov     x2, #0                  // best idx
        ldr     s0, [x0]                // best value
        mov     x5, #1                  // scalar loop start
        cmp     x1, #8
        b.lo    3f
        ldr     q1, [x0]                // per-lane best values
        ADR_L   x3, iota4
        ldr     q2, [x3]                // per-lane best indices
        movi    v4.4s, #4
        add     v3.4s, v2.4s, v4.4s     // current indices
        and     x4, x1, #~3
        mov     x5, #4
1:      cmp     x5, x4
        b.hs    2f
        add     x6, x0, x5, lsl #2
        ldr     q5, [x6]
        fcmgt   v6.4s, v5.4s, v1.4s     // strict > keeps the first index per lane
        bit     v1.16b, v5.16b, v6.16b
        bit     v2.16b, v3.16b, v6.16b
        add     v3.4s, v3.4s, v4.4s
        add     x5, x5, #4
        b       1b
2:      mov     s0, v1.s[0]
        umov    w2, v2.s[0]
        .irp lane, 1, 2, 3
        mov     s5, v1.s[\lane]
        umov    w6, v2.s[\lane]
        fcmp    s5, s0
        cset    w7, gt
        cset    w10, eq
        cmp     w6, w2
        cset    w11, lo
        and     w10, w10, w11
        orr     w7, w7, w10
        cbz     w7, 9f
        fmov    s0, s5
        mov     w2, w6
9:
        .endr
3:      cmp     x5, x1                  // scalar tail (or whole array if n < 8)
        b.hs    5f
        ldr     s5, [x0, x5, lsl #2]
        fcmp    s5, s0
        b.le    4f
        fmov    s0, s5
        mov     x2, x5
4:      add     x5, x5, #1
        b       3b
5:      mov     x0, x2
        ret

// (d0 = sin, d1 = cos) = fast_sincos_f64(d0 = angle)
// Cody-Waite reduction by pi/2 + degree-11/10 Taylor. Clobbers d0-d6, x9, x10.
fast_sincos_f64:
        ADR_L   x9, sincos_consts
        ldr     d1, [x9, #0]
        fmul    d1, d0, d1
        frintn  d1, d1                  // q_d = round(x * 2/pi)
        fcvtzs  x10, d1
        ldr     d2, [x9, #8]
        fmsub   d0, d1, d2, d0          // x - q*pio2_hi
        ldr     d2, [x9, #16]
        fmsub   d0, d1, d2, d0          // r
        fmul    d2, d0, d0              // r^2
        ldr     d3, [x9, #24]           // s5
        ldr     d4, [x9, #32]
        fmadd   d3, d3, d2, d4
        ldr     d4, [x9, #40]
        fmadd   d3, d3, d2, d4
        ldr     d4, [x9, #48]
        fmadd   d3, d3, d2, d4
        ldr     d4, [x9, #56]
        fmadd   d3, d3, d2, d4
        fmov    d4, #1.0
        fmadd   d3, d3, d2, d4
        fmul    d3, d3, d0              // sin(r)
        ldr     d5, [x9, #64]           // c5
        ldr     d6, [x9, #72]
        fmadd   d5, d5, d2, d6
        ldr     d6, [x9, #80]
        fmadd   d5, d5, d2, d6
        ldr     d6, [x9, #88]
        fmadd   d5, d5, d2, d6
        ldr     d6, [x9, #96]
        fmadd   d5, d5, d2, d6
        fmadd   d5, d5, d2, d4          // cos(r)
        and     x10, x10, #3
        cmp     x10, #1
        b.eq    1f
        cmp     x10, #2
        b.eq    2f
        cmp     x10, #3
        b.eq    3f
        fmov    d0, d3
        fmov    d1, d5
        ret
1:      fmov    d0, d5
        fneg    d1, d3
        ret
2:      fneg    d0, d3
        fneg    d1, d5
        ret
3:      fneg    d0, d5
        fmov    d1, d3
        ret

// d0 = log_f64(d0 > 0). Clobbers d0-d5, x9-x11.
log_f64:
        fmov    x9, d0
        lsr     x10, x9, #52
        and     x10, x10, #0x7FF
        sub     x10, x10, #1023         // e
        and     x9, x9, #0x000FFFFFFFFFFFFF
        orr     x9, x9, #0x3FF0000000000000
        fmov    d0, x9                  // m in [1, 2)
        ADR_L   x11, log_consts
        ldr     d1, [x11, #0]           // sqrt(2)
        fcmp    d0, d1
        b.lt    1f
        fmov    d1, #0.5
        fmul    d0, d0, d1
        add     x10, x10, #1
1:      fmov    d1, #1.0
        fsub    d2, d0, d1
        fadd    d3, d0, d1
        fdiv    d0, d2, d3              // u = (m - 1) / (m + 1)
        fmul    d2, d0, d0
        ldr     d3, [x11, #8]           // 1/13
        ldr     d4, [x11, #16]
        fmadd   d3, d3, d2, d4
        ldr     d4, [x11, #24]
        fmadd   d3, d3, d2, d4
        ldr     d4, [x11, #32]
        fmadd   d3, d3, d2, d4
        ldr     d4, [x11, #40]
        fmadd   d3, d3, d2, d4
        ldr     d4, [x11, #48]
        fmadd   d3, d3, d2, d4
        fmadd   d3, d3, d2, d1
        fmul    d3, d3, d0
        fadd    d3, d3, d3              // ln(m)
        scvtf   d4, x10
        ldr     d5, [x11, #56]          // ln(2)
        fmadd   d0, d4, d5, d3
        ret

// d0 = exp_f64(d0). Clobbers d0-d4, x10, x11.
exp_f64:
        ADR_L   x11, expd_consts
        ldr     d1, [x11, #0]
        fmul    d1, d0, d1
        frintn  d1, d1                  // k_d
        fcvtzs  x10, d1
        ldr     d2, [x11, #8]
        fmsub   d0, d1, d2, d0
        ldr     d2, [x11, #16]
        fmsub   d0, d1, d2, d0          // r
        ldr     d3, [x11, #24]          // e10
        .irp off, 32, 40, 48, 56, 64, 72, 80
        ldr     d4, [x11, #\off]
        fmadd   d3, d3, d0, d4
        .endr
        fmov    d4, #0.5
        fmadd   d3, d3, d0, d4
        fmov    d4, #1.0
        fmadd   d3, d3, d0, d4
        fmadd   d3, d3, d0, d4
        add     x10, x10, #1023
        lsl     x10, x10, #52
        fmov    d2, x10                 // 2^k
        fmul    d0, d3, d2
        ret

// void init_rope_freqs(double *out x0, uint64 head_dim x1, float theta s0)
// inv_freq[i] = exp(-ln(theta) * 2i / head_dim)
init_rope_freqs:
        stp     x29, x30, [sp, #-48]!
        stp     x19, x20, [sp, #16]
        str     x21, [sp, #40]
        mov     x19, x0
        mov     x20, x1
        fcvt    d0, s0
        bl      log_f64
        ucvtf   d1, x20
        fdiv    d0, d0, d1
        fneg    d0, d0
        str     d0, [sp, #32]           // step
        lsr     x20, x20, #1
        mov     x21, #0
1:      cmp     x21, x20
        b.hs    2f
        lsl     x2, x21, #1
        ucvtf   d0, x2
        ldr     d1, [sp, #32]
        fmul    d0, d0, d1
        bl      exp_f64
        str     d0, [x19, x21, lsl #3]
        add     x21, x21, #1
        b       1b
2:      ldr     x21, [sp, #40]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #48
        ret

// void compute_rope_cs(float *cos x0, float *sin x1, const double *inv_freq x2,
//                      uint64 half x3, uint64 pos x4)
compute_rope_cs:
        stp     x29, x30, [sp, #-64]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        str     x23, [sp, #48]
        mov     x19, x0
        mov     x20, x1
        mov     x21, x2
        mov     x22, x3
        ucvtf   d0, x4
        str     d0, [sp, #56]
        mov     x23, #0
1:      cmp     x23, x22
        b.hs    2f
        ldr     d0, [sp, #56]
        ldr     d1, [x21, x23, lsl #3]
        fmul    d0, d0, d1
        bl      fast_sincos_f64
        fcvt    s0, d0
        fcvt    s1, d1
        str     s0, [x20, x23, lsl #2]
        str     s1, [x19, x23, lsl #2]
        add     x23, x23, #1
        b       1b
2:      ldr     x23, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #64
        ret

        .section .rodata
        .balign 16
iota4:  .word   0, 1, 2, 3
expf_consts:                            // order matters: read sequentially by expf_neon
        .float  -87.3365447504000, 88.3762626647949, 1.4426950408889634
        .float  0.693359375, -0.000212194440
        .float  0.0013888889225199819, 0.008333333767950535, 0.0416666679084301
        .float  0.1666666716337204, 0.5, 1.0
        .balign 8
sincos_consts:
        .double 0.63661977236758134308
        .quad   0x3FF921FB54400000      // pi/2 high 33 bits
        .quad   0x3DD0B4611A626331      // pi/2 - high
        .double -0.00000002505210838544, 0.00000275573192239859, -0.00019841269841269841
        .double 0.00833333333333333333, -0.16666666666666666667
        .double -0.00000027557319223986, 0.00002480158730158730, -0.00138888888888888889
        .double 0.04166666666666666667, -0.5
log_consts:
        .double 1.4142135623730951
        .double 0.07692307692307692308, 0.09090909090909090909, 0.11111111111111111111
        .double 0.14285714285714285714, 0.2, 0.33333333333333333333
        .double 0.69314718055994530942
expd_consts:
        .double 1.44269504088896340736, 0.69314718055994530942, 0.0
        .double 0.00000027557319223986, 0.00000275573192239859, 0.00002480158730158730
        .double 0.00019841269841269841, 0.00138888888888888889, 0.00833333333333333333
        .double 0.04166666666666666667, 0.16666666666666666667
