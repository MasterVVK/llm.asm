// Kernel micro-benchmark (AArch64): batched GEMM vs GEMV on the repacked
// 4-row layout, plus optional candidate kernels matmul_q8_0_4x4t_{pf,nopf}
// (assemble with --defsym NEWK=1) checked for identical output.
// Run through aarch64/kbench.py.
//   stdin : [u64 K][u64 M][u64 B][u64 R][u64 pf]
//   stdout: [u64 cal_ns][u64 batch_ns][u64 gemv_ns][u64 new_ns][M*B f32][M*B f32]
        .include "constants.s"
        .equ MAX_K, 4096
        .equ MAX_M, 256
        .equ MAX_B, 16
        .text
        .global _start

now_ns:
        sub     sp, sp, #16
        mov     x0, #1
        mov     x1, sp
        mov     x8, #SYS_clock_gettime
        svc     #0
        ldp     x0, x1, [sp]
        ldr     x2, =1000000000
        madd    x0, x0, x2, x1
        add     sp, sp, #16
        ret

lcg:    ldr     x1, =6364136223846793005
        ldr     x2, =1442695040888963407
        madd    x28, x28, x1, x2
        ret

_start:
        ldr     x28, =0x9E3779B97F4A7C15
        mov     x0, #0
        ADR_L   x1, prm
        mov     x2, #40
        mov     x8, #SYS_read
        svc     #0
        ADR_L   x27, prm
        ldp     x19, x20, [x27]          // K, M
        ldp     x21, x22, [x27, #16]     // B, R
        ldr     x23, [x27, #32]          // pf
        // W: random int8 (no -128), then valid fp16 scales per 4-row group
        add     x24, x19, x19, lsr #4    // row_stride
        mul     x25, x24, x20            // bytes
        ADR_L   x26, W
        mov     x9, #0
1:      cmp     x9, x25
        b.hs    2f
        bl      lcg
        lsr     x0, x28, #33
        and     w0, w0, #0xFF
        cmp     w0, #0x80
        b.ne    11f
        mov     w0, #0x81
11:     strb    w0, [x26, x9]
        add     x9, x9, #1
        b       1b
2:      lsr     x10, x20, #2             // groups
        mov     x11, x26
3:      cbz     x10, 5f
        add     x12, x11, x19, lsl #2    // scales of the group
        lsr     x13, x19, #3             // K/32 * 4 scales
4:      bl      lcg
        lsr     x0, x28, #40
        and     w0, w0, #0x3FF
        orr     w0, w0, #0x2000
        strh    w0, [x12], #2
        subs    x13, x13, #1
        b.ne    4b
        add     x11, x11, x24, lsl #2
        sub     x10, x10, #1
        b       3b
5:      // X: B x K floats in [-1, 1)
        mul     x25, x19, x21
        ADR_L   x26, X
        mov     x9, #0
        ldr     s1, =0x30000000          // 2^-31
6:      cmp     x9, x25
        b.hs    7f
        bl      lcg
        asr     x0, x28, #32
        scvtf   s0, x0
        fmul    s0, s0, s1
        str     s0, [x26, x9, lsl #2]
        add     x9, x9, #1
        b       6b
7:      // frequency calibration: 2e7 x 10 dependent adds
        bl      now_ns
        mov     x25, x0
        ldr     x9, =20000000
        mov     x0, #0
8:      .rept 10
        add     x0, x0, #1
        .endr
        subs    x9, x9, #1
        b.ne    8b
        bl      now_ns
        sub     x0, x0, x25
        ADR_L   x27, res
        str     x0, [x27]

        // batched GEMM (current kernel)
        ADR_L   x0, X
        mov     x1, x19
        mov     x2, x21
        lsl     x3, x19, #2
        bl      quantize_x_batch
        ADR_L   x0, g_mm_b
        str     x21, [x0]
        lsl     x1, x20, #2
        ADR_L   x0, g_mm_m4
        str     x1, [x0]
        ADR_L   x26, matmul_q8_0_4x4b_nopf
        cbz     x23, 9f
        ADR_L   x26, matmul_q8_0_4x4b_pf
9:      ADR_L   x0, W
        ADR_L   x1, OUT
        mov     x2, x19
        mov     x3, #0
        mov     x4, x20
        blr     x26                      // warm-up
        bl      now_ns
        mov     x25, x0
        mov     x24, x22
10:     ADR_L   x0, W
        ADR_L   x1, OUT
        mov     x2, x19
        mov     x3, #0
        mov     x4, x20
        blr     x26
        subs    x24, x24, #1
        b.ne    10b
        bl      now_ns
        sub     x0, x0, x25
        ADR_L   x27, res
        str     x0, [x27, #8]
        .ifdef NEWK
        ADR_L   x26, matmul_q8_0_4x4t_nopf
        cbz     x23, 12f
        ADR_L   x26, matmul_q8_0_4x4t_pf
12:     ADR_L   x0, W
        ADR_L   x1, OUT3
        mov     x2, x19
        mov     x3, #0
        mov     x4, x20
        blr     x26
        bl      now_ns
        mov     x25, x0
        mov     x24, x22
13:     ADR_L   x0, W
        ADR_L   x1, OUT3
        mov     x2, x19
        mov     x3, #0
        mov     x4, x20
        blr     x26
        subs    x24, x24, #1
        b.ne    13b
        bl      now_ns
        sub     x0, x0, x25
        ADR_L   x27, res
        str     x0, [x27, #24]
        .endif
        // GEMV on token 0, R * B calls
        ADR_L   x0, X
        mov     x1, x19
        bl      quantize_x_q8_0
        ADR_L   x26, matmul_q8_0_4x4_nopf
        cbz     x23, 14f
        ADR_L   x26, matmul_q8_0_4x4_pf
14:     mul     x24, x22, x21
        bl      now_ns
        mov     x25, x0
15:     ADR_L   x0, W
        ADR_L   x1, OUT2
        mov     x2, x19
        mov     x3, #0
        mov     x4, x20
        blr     x26
        subs    x24, x24, #1
        b.ne    15b
        bl      now_ns
        sub     x0, x0, x25
        ADR_L   x27, res
        str     x0, [x27, #16]

        mov     x0, #1
        ADR_L   x1, res
        mov     x2, #32
        mov     x8, #SYS_write
        svc     #0
        mul     x2, x20, x21
        lsl     x2, x2, #2
        mov     x0, #1
        ADR_L   x1, OUT
        mov     x8, #SYS_write
        svc     #0
        mul     x2, x20, x21
        lsl     x2, x2, #2
        mov     x0, #1
        ADR_L   x1, OUT3
        mov     x8, #SYS_write
        svc     #0
        mov     x0, #0
        mov     x8, #SYS_exit_group
        svc     #0
        .ltorg

        .bss
        .balign 64
prm:    .skip   40
res:    .skip   32
        .balign 64
W:      .skip   MAX_M * (MAX_K + MAX_K / 16)
        .balign 64
X:      .skip   MAX_B * MAX_K * 4
        .balign 64
OUT:    .skip   MAX_B * MAX_M * 4
OUT2:   .skip   MAX_B * MAX_M * 4
OUT3:   .skip   MAX_B * MAX_M * 4
