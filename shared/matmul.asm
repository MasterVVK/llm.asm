; ============================================================================
; matmul.asm — High-throughput AVX2 + FMA + F16C Q8_0 matmul kernels (zero libc)
;
; Weight layout (per-row split Q8_0, block size = 32, int8 in [-127, 127]):
;   v2 (FP16 scales, HDR_FLAG_FP16_SCALES):
;     row_stride = K + (K / 32) * 2 = K + (K >> 4) bytes
;     [row + 0 .. row + K - 1]         : K int8 weights
;     [row + K .. row + row_stride - 1]: K/32 IEEE-754 fp16 block scales
;   v1 (FP32 scales, legacy):
;     row_stride = K + (K / 32) * 4 = K + (K >> 3) bytes
;     [row + 0 .. row + K - 1]         : K int8 weights
;     [row + K .. row + row_stride - 1]: K/32 fp32 block scales
;
; Ivy Bridge port (AVX + F16C, no AVX2 / FMA):
;   FAST_Q8=1 (only mode): Q8_0 x Q8_0 INT8xINT8 on 128-bit integer SIMD
;                        (vpsignb + vpmaddubsw + vpmaddwd + vcvtdq2ps, then
;                        separate vmulps + vaddps), |x| precomputed once per
;                        matmul, 1-row sequential 4-block (128B) unrolled with
;                        next-row software prefetch. Auto-detects v1 (fp32)
;                        and v2 (fp16) scales.
;   FAST_Q8=0 is not ported (assembly stops with %error).
;
; Functions:
;   matmul_set_fp16   : (uint64 is_fp16)
;   quantize_x_q8_0   : (const float *x, uint64 K)  [FAST_Q8=1]
;   dequant_row_q8_0  : (const void *row_ptr, float *out, uint64 K)
;   matmul_q8_0_range : (const void *W, const float *x, float *out,
;                        uint64 K, uint64 r0, uint64 r1)
;   dot_f32_avx2      : (const float *a, const float *b, uint64 n) -> xmm0
;   saxpy_f32_avx2    : (float *dst, const float *src, uint64 n, float alpha)
; ============================================================================

bits 64
default rel

%include "constants.inc"

%ifndef FAST_Q8
%define FAST_Q8 1
%endif

%ifndef Q8_PREFETCH
%define Q8_PREFETCH prefetcht0
%endif

%define Q8_MAX_K 16384
%define XQ_STRIDE (Q8_MAX_K + 64)            ; per-token slot, +64 avoids L1 set aliasing
%define XS_STRIDE ((Q8_MAX_K / 32) * 4 + 64)

section .text

global matmul_set_fp16
%if FAST_Q8
global quantize_x_q8_0
%ifndef BARE_METAL
global quantize_x_batch
global matmul_q8_0_batch_range
global g_mm_b
global g_mm_m4
global g_mm_xstride
%endif
%endif
global dequant_row_q8_0
global matmul_q8_0_range
global dot_f32_avx2
global saxpy_f32_avx2

; ----------------------------------------------------------------------------
; void matmul_set_fp16(uint64 is_fp16)
; rdi = 0 for v1 FP32 scales, non-zero for v2 FP16 scales
; ----------------------------------------------------------------------------
matmul_set_fp16:
    xor     eax, eax
    test    rdi, rdi
    setnz   al
    mov     [rel g_fp16_scales], eax
    ret

; ----------------------------------------------------------------------------
; void dequant_row_q8_0(const void *row_ptr, float *out, uint64 K)
; rdi = row_ptr, rsi = out, rdx = K (multiple of 32)
; Auto-detects v1 FP32 vs v2 FP16 scales via g_fp16_scales.
; ----------------------------------------------------------------------------
dequant_row_q8_0:
    lea     r8, [rdi + rdx]             ; scales_ptr = row_ptr + K
    mov     rcx, rdx
    shr     rcx, 5                      ; nb = K / 32
    mov     r11d, [rel g_fp16_scales]
    xor     eax, eax                    ; b = 0
.lp:
    cmp     rax, rcx
    jae     .done
    test    r11d, r11d
    jnz     .sc_f16
    vbroadcastss ymm4, [r8 + rax*4]     ; v1 fp32 scale[b]
    jmp     .sc_ok
.sc_f16:
    movzx   r10d, word [r8 + rax*2]     ; v2 fp16 scale[b]
    vmovd   xmm4, r10d
    vcvtph2ps xmm4, xmm4
    vshufps xmm4, xmm4, xmm4, 0
    vinsertf128 ymm4, ymm4, xmm4, 1
.sc_ok:
    mov     r9, rax
    shl     r9, 5                       ; b * 32 (q offset)
    mov     r10, rax
    shl     r10, 7                      ; b * 128 (out offset)

    ; 256-bit vpmovsxbd is AVX2: sign-extend 4 bytes at a time, join halves
    vpmovsxbd xmm0, dword [rdi + r9]
    vpmovsxbd xmm5, dword [rdi + r9 + 4]
    vinsertf128 ymm0, ymm0, xmm5, 1
    vpmovsxbd xmm1, dword [rdi + r9 + 8]
    vpmovsxbd xmm5, dword [rdi + r9 + 12]
    vinsertf128 ymm1, ymm1, xmm5, 1
    vpmovsxbd xmm2, dword [rdi + r9 + 16]
    vpmovsxbd xmm5, dword [rdi + r9 + 20]
    vinsertf128 ymm2, ymm2, xmm5, 1
    vpmovsxbd xmm3, dword [rdi + r9 + 24]
    vpmovsxbd xmm5, dword [rdi + r9 + 28]
    vinsertf128 ymm3, ymm3, xmm5, 1
    vcvtdq2ps ymm0, ymm0
    vcvtdq2ps ymm1, ymm1
    vcvtdq2ps ymm2, ymm2
    vcvtdq2ps ymm3, ymm3
    vmulps  ymm0, ymm0, ymm4
    vmulps  ymm1, ymm1, ymm4
    vmulps  ymm2, ymm2, ymm4
    vmulps  ymm3, ymm3, ymm4
    vmovups [rsi + r10], ymm0
    vmovups [rsi + r10 + 32], ymm1
    vmovups [rsi + r10 + 64], ymm2
    vmovups [rsi + r10 + 96], ymm3

    inc     rax
    jmp     .lp
.done:
    ret

%if FAST_Q8
; ----------------------------------------------------------------------------
; void quantize_x_q8_0(const float *x, uint64 K)
; rdi = x (f32), rsi = K (multiple of 32, <= Q8_MAX_K)
; Quantizes x into Q8_0 blocks of 32 ONCE before matmul dispatch:
;   q8_xq[0..K-1]  : int8 in [-127, 127] (fits in L1D)
;   q8_xa[0..K-1]  : |q8_xq| as uint8. It does not depend on the weight row,
;                    so it is computed here once instead of per row.
;   q8_xs[0..K/32] : float32 scale = max_abs / 127.0f
; Ivy Bridge: int32 -> int8 packing runs on 128-bit halves, which keeps the
; bytes in linear order (no vpermd fix-up needed).
; ----------------------------------------------------------------------------
quantize_x_q8_0:
    lea     r9, [rel q8_xq]
    lea     r10, [rel q8_xs]
    lea     r11, [rel q8_xa]
; quantize_core: rdi = x, rsi = K, r9 = xq, r10 = xs, r11 = xa.
; Clobbers rax, rcx, rdx, r8 and SIMD registers only.
quantize_core:
    mov     rcx, rsi
    shr     rcx, 5                      ; nb = K / 32

    vbroadcastss ymm12, [rel q8_abs_mask]
    vmovss  xmm13, [rel q8_inv_127]
    vmovss  xmm14, [rel q8_f127]
    vxorps  xmm15, xmm15, xmm15
    vmovdqa xmm10, [rel q8_min_i8x16]

    xor     eax, eax                    ; b = 0
    xor     r8d, r8d                    ; x byte offset (b * 128)
    xor     edx, edx                    ; xq byte offset (b * 32)
.qlp:
    cmp     rax, rcx
    jae     .qdone

    vmovups ymm0, [rdi + r8]
    vmovups ymm1, [rdi + r8 + 32]
    vmovups ymm2, [rdi + r8 + 64]
    vmovups ymm3, [rdi + r8 + 96]

    vandps  ymm4, ymm0, ymm12
    vandps  ymm5, ymm1, ymm12
    vandps  ymm6, ymm2, ymm12
    vandps  ymm7, ymm3, ymm12
    vmaxps  ymm4, ymm4, ymm5
    vmaxps  ymm6, ymm6, ymm7
    vmaxps  ymm4, ymm4, ymm6

    vextractf128 xmm5, ymm4, 1
    vmaxps  xmm4, xmm4, xmm5
    vshufps xmm5, xmm4, xmm4, 0x0E
    vmaxps  xmm4, xmm4, xmm5
    vshufps xmm5, xmm4, xmm4, 0x01
    vmaxss  xmm4, xmm4, xmm5            ; xmm4 = max_abs

    vmulss  xmm5, xmm4, xmm13           ; scale_x = max_abs / 127.0f
    vmovss  [r10 + rax*4], xmm5

    vxorps  xmm6, xmm6, xmm6            ; inv_scale = 0.0f if max_abs == 0
    vucomiss xmm4, xmm15
    jbe     .qinv_ok
    vdivss  xmm6, xmm14, xmm4           ; inv_scale = 127.0f / max_abs
.qinv_ok:
    vshufps xmm6, xmm6, xmm6, 0
    vinsertf128 ymm6, ymm6, xmm6, 1

    vmulps  ymm0, ymm0, ymm6
    vmulps  ymm1, ymm1, ymm6
    vmulps  ymm2, ymm2, ymm6
    vmulps  ymm3, ymm3, ymm6

    vcvtps2dq ymm0, ymm0
    vcvtps2dq ymm1, ymm1
    vcvtps2dq ymm2, ymm2
    vcvtps2dq ymm3, ymm3

    vextractf128 xmm4, ymm0, 1
    vextractf128 xmm5, ymm1, 1
    vextractf128 xmm6, ymm2, 1
    vextractf128 xmm7, ymm3, 1
    vpackssdw xmm0, xmm0, xmm4          ; x[0..7]   int16
    vpackssdw xmm1, xmm1, xmm5          ; x[8..15]
    vpackssdw xmm2, xmm2, xmm6          ; x[16..23]
    vpackssdw xmm3, xmm3, xmm7          ; x[24..31]
    vpacksswb xmm0, xmm0, xmm1          ; x[0..15]  int8
    vpacksswb xmm2, xmm2, xmm3          ; x[16..31] int8
    vpmaxsb xmm0, xmm0, xmm10           ; clamp >= -127 (prevents -128 overflow)
    vpmaxsb xmm2, xmm2, xmm10

    vmovdqa [r9 + rdx], xmm0            ; q8_xq[b*32 .. b*32+31]
    vmovdqa [r9 + rdx + 16], xmm2
    vpabsb  xmm0, xmm0
    vpabsb  xmm2, xmm2
    vmovdqa [r11 + rdx], xmm0           ; q8_xa[b*32 .. b*32+31]
    vmovdqa [r11 + rdx + 16], xmm2

    add     r8, 128
    add     rdx, 32
    inc     rax
    jmp     .qlp
.qdone:
    ret

; Macro: one 32-byte Q8_0 block of the current row, AVX1 (128-bit integer).
;   %1 = byte offset of the block inside the 128-byte group (0/32/64/96)
;   %2 = f32x4 accumulator, %3 = vpermilps immediate picking the block scale
; Preconditions:
;   rax = w (current row), rbx = byte offset into the row (multiple of 32)
;   r11 = q8_xq (signed x), r10 = q8_xa (|x|), xmm12 = 4 combined scales
;   xmm15 = 8x int16(1)
; The two 16-byte halves are summed in int32 before conversion; this is
; exact because |sum| <= 8 * 127 * 127.
%macro INT8_BLK_AVX1 3
    vmovdqu    xmm6, [rax + rbx + %1]
    vmovdqu    xmm7, [rax + rbx + %1 + 16]
    vpsignb    xmm6, xmm6, [r11 + rbx + %1]       ; w * sign(x)
    vpsignb    xmm7, xmm7, [r11 + rbx + %1 + 16]
    vmovdqa    xmm4, [r10 + rbx + %1]             ; |x|
    vmovdqa    xmm5, [r10 + rbx + %1 + 16]
    vpmaddubsw xmm6, xmm4, xmm6                   ; 8x int16 pair sums
    vpmaddubsw xmm7, xmm5, xmm7
    vpmaddwd   xmm6, xmm6, xmm15                  ; 4x int32
    vpmaddwd   xmm7, xmm7, xmm15
    vpaddd     xmm6, xmm6, xmm7
    vcvtdq2ps  xmm6, xmm6
    vpermilps  xmm8, xmm12, %3
    vmulps     xmm6, xmm6, xmm8
    vaddps     %2, %2, xmm6
%endmacro

; 1 row x 4 blocks (128 bytes = 2 cache lines). Prefetching [r8 + rbx]
; (= [w + K + rbx]) streams monotonically across the current row's scales
; and the next row, as upstream does.
%macro INT8_1ROW_4BLK 0
    Q8_PREFETCH [r8 + rbx]
    Q8_PREFETCH [r8 + rbx + 64]
    INT8_BLK_AVX1 0,  xmm0, 0x00
    INT8_BLK_AVX1 32, xmm1, 0x55
    INT8_BLK_AVX1 64, xmm2, 0xAA
    INT8_BLK_AVX1 96, xmm3, 0xFF
%endmacro

%macro ROW_REDUCE_STORE 0
    vaddps  xmm0, xmm0, xmm1
    vaddps  xmm2, xmm2, xmm3
    vaddps  xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss  [rdx], xmm0
    add     rdx, 4
%endmacro

; ----------------------------------------------------------------------------
; void matmul_q8_0_range(const void *W, const float *x, float *out,
;                        uint64 K, uint64 r0, uint64 r1)
; FAST_Q8=1, Ivy Bridge: 1-row sequential 4-block INT8xINT8 kernel on 128-bit
; integer SIMD (SSSE3 vpsignb/vpmaddubsw + vpmaddwd), F16C scales, separate
; multiply and add instead of FMA. Every row is computed by exactly one thread
; in a fixed order, so results do not depend on the thread count.
; ----------------------------------------------------------------------------
matmul_q8_0_range:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rsi, r9
    sub     rsi, r8                     ; rsi = rows_left = r1 - r0
    jle     .exit

    lea     rdx, [rdx + r8 * 4]         ; rdx = out_ptr = out + r0 * 4
    vmovdqa xmm15, [rel q8_one16x8]     ; 8x int16(1)
    lea     r11, [rel q8_xq]
    lea     r10, [rel q8_xa]
    lea     rbp, [rel q8_xs]
    mov     r15, rcx                    ; r15 = K
    mov     r13, rcx
    and     r13, -128                   ; r13 = K_128 = K & ~127

    cmp     dword [rel g_fp16_scales], 0
    jne     .fast_fp16

    ; ========================================================================
    ; v1 FP32 scales: row_stride = K + (K >> 3)
    ; ========================================================================
    mov     r14, rcx
    shr     r14, 3
    add     r14, rcx                    ; r14 = row_stride
    mov     rax, r8
    imul    rax, r14
    add     rax, rdi                    ; rax = w_cur = W + r0 * row_stride

align 16
.row_f32:
    lea     r12, [rax + r14]            ; r12 = w_next
    lea     r8,  [rax + r15]            ; r8  = s_cur (fp32)
    vxorps  xmm0, xmm0, xmm0
    vxorps  xmm1, xmm1, xmm1
    vxorps  xmm2, xmm2, xmm2
    vxorps  xmm3, xmm3, xmm3
    xor     ebx, ebx                    ; b_q: 0, 128, ...
    xor     r9d, r9d                    ; b_s: 0, 16, ... (fp32 scale & xs offset)
    test    r13, r13
    jz      .tail_f32

align 16
.blk4_f32:
    vmovups xmm12, [r8 + r9]            ; 4x fp32 scale_w
    vmulps  xmm12, xmm12, [rbp + r9]    ; * 4x fp32 scale_x
    INT8_1ROW_4BLK
    add     rbx, 128
    add     r9, 16
    cmp     rbx, r13
    jb      .blk4_f32

.tail_f32:
    cmp     rbx, r15
    jae     .red_f32
    vmovss  xmm12, [r8 + r9]
    vmulss  xmm12, xmm12, [rbp + r9]
    INT8_BLK_AVX1 0, xmm0, 0x00
    add     rbx, 32
    add     r9, 4
    jmp     .tail_f32

.red_f32:
    ROW_REDUCE_STORE
    mov     rax, r12
    dec     rsi
    jnz     .row_f32
    jmp     .exit

    ; ========================================================================
    ; v2 FP16 scales: row_stride = K + (K >> 4)
    ; ========================================================================
.fast_fp16:
    mov     r14, rcx
    shr     r14, 4
    add     r14, rcx                    ; r14 = row_stride
    mov     rax, r8
    imul    rax, r14
    add     rax, rdi                    ; rax = w_cur = W + r0 * row_stride

align 16
.row_f16:
    lea     r12, [rax + r14]            ; r12 = w_next
    lea     r8,  [rax + r15]            ; r8  = s_cur (fp16)
    vxorps  xmm0, xmm0, xmm0
    vxorps  xmm1, xmm1, xmm1
    vxorps  xmm2, xmm2, xmm2
    vxorps  xmm3, xmm3, xmm3
    xor     ebx, ebx                    ; b_q: 0, 128, ...
    xor     r9d, r9d                    ; b_s: 0, 8, ... (fp16 scale offset; xs = r9*2)
    test    r13, r13
    jz      .tail_f16

align 16
.blk4_f16:
    vmovq      xmm12, qword [r8 + r9]   ; 4x fp16 scale_w
    vcvtph2ps  xmm12, xmm12             ; -> 4x fp32
    vmulps     xmm12, xmm12, [rbp + r9*2] ; * 4x fp32 scale_x
    INT8_1ROW_4BLK
    add     rbx, 128
    add     r9, 8
    cmp     rbx, r13
    jb      .blk4_f16

.tail_f16:
    cmp     rbx, r15
    jae     .red_f16
    movzx   ecx, word [r8 + r9]
    vmovd   xmm12, ecx
    vcvtph2ps xmm12, xmm12
    vmulss  xmm12, xmm12, [rbp + r9*2]
    INT8_BLK_AVX1 0, xmm0, 0x00
    add     rbx, 32
    add     r9, 2
    jmp     .tail_f16

.red_f16:
    ROW_REDUCE_STORE
    mov     rax, r12
    dec     rsi
    jnz     .row_f16

.exit:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

%ifndef BARE_METAL
; ----------------------------------------------------------------------------
; void quantize_x_batch(const float *X, uint64 K, uint64 B, uint64 x_stride_bytes)
; Quantizes B activation vectors into per-token slots q8b_xq / q8b_xa / q8b_xs
; with exactly the same code as quantize_x_q8_0.
; ----------------------------------------------------------------------------
quantize_x_batch:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    mov     r12, rdi
    mov     r13, rsi
    mov     r14, rdx
    mov     r15, rcx
    xor     ebx, ebx
.qb_lp:
    cmp     rbx, r14
    jae     .qb_done
    mov     rdi, r12
    mov     rsi, r13
    imul    rax, rbx, XQ_STRIDE
    lea     r9, [rel q8b_xq]
    add     r9, rax
    lea     r11, [rel q8b_xa]
    add     r11, rax
    imul    rax, rbx, XS_STRIDE
    lea     r10, [rel q8b_xs]
    add     r10, rax
    call    quantize_core
    add     r12, r15
    inc     rbx
    jmp     .qb_lp
.qb_done:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void matmul_q8_0_batch_range(const void *W, (unused), float *out, uint64 K,
;                              uint64 r0, uint64 r1)
; Batched prefill GEMM over g_mm_b tokens (v2 fp16 scales). For every row the
; per-token dot product runs the very same instruction sequence as the GEMV
; kernel, so results are bit-identical to token-by-token prefill; the row is
; read from DRAM once and from L1 for the remaining tokens.
; out[t * M + r] with the token stride M * 4 taken from g_mm_m4.
; ----------------------------------------------------------------------------
matmul_q8_0_batch_range:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8                      ; [rsp] = out pointer of token 0 for this row

    mov     rsi, r9
    sub     rsi, r8                     ; rows
    jle     .b_exit
    lea     rdx, [rdx + r8 * 4]
    mov     [rsp], rdx
    vmovdqa xmm15, [rel q8_one16x8]
    mov     r15, rcx                    ; K
    mov     r13, rcx
    and     r13, -128                   ; K_128
    mov     r14, rcx
    shr     r14, 4
    add     r14, rcx                    ; row_stride (fp16 scales)
    mov     rax, r8
    imul    rax, r14
    add     rax, rdi                    ; first row
    mov     rcx, [rel g_mm_m4]          ; token stride in out (bytes)

align 16
.b_row:
    lea     r8, [rax + r15]             ; fp16 scales of this row
    mov     rdx, [rsp]
    lea     r11, [rel q8b_xq]
    lea     r10, [rel q8b_xa]
    lea     rbp, [rel q8b_xs]
    mov     rdi, [rel g_mm_b]
.b_tok:
    vxorps  xmm0, xmm0, xmm0
    vxorps  xmm1, xmm1, xmm1
    vxorps  xmm2, xmm2, xmm2
    vxorps  xmm3, xmm3, xmm3
    xor     ebx, ebx
    xor     r9d, r9d
    test    r13, r13
    jz      .b_tail
align 16
.b_blk4:
    vmovq      xmm12, qword [r8 + r9]
    vcvtph2ps  xmm12, xmm12
    vmulps     xmm12, xmm12, [rbp + r9*2]
    INT8_1ROW_4BLK
    add     rbx, 128
    add     r9, 8
    cmp     rbx, r13
    jb      .b_blk4
.b_tail:
    cmp     rbx, r15
    jae     .b_red
    movzx   r12d, word [r8 + r9]
    vmovd   xmm12, r12d
    vcvtph2ps xmm12, xmm12
    vmulss  xmm12, xmm12, [rbp + r9*2]
    INT8_BLK_AVX1 0, xmm0, 0x00
    add     rbx, 32
    add     r9, 2
    jmp     .b_tail
.b_red:
    vaddps  xmm0, xmm0, xmm1
    vaddps  xmm2, xmm2, xmm3
    vaddps  xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss  [rdx], xmm0
    add     rdx, rcx
    add     r11, XQ_STRIDE
    add     r10, XQ_STRIDE
    add     rbp, XS_STRIDE
    dec     rdi
    jnz     .b_tok
    add     qword [rsp], 4
    add     rax, r14
    dec     rsi
    jnz     .b_row
.b_exit:
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret
%endif ; !BARE_METAL

%else
%error "FAST_Q8=0 (FP32-activation path) is not ported to AVX1 / Ivy Bridge; build with FAST_Q8=1"
%endif

; ----------------------------------------------------------------------------
; float dot_f32_avx2(const float *a, const float *b, uint64 n) -> xmm0
; rdi = a, rsi = b, rdx = n (multiple of 32, e.g. head_dim = 128)
; ----------------------------------------------------------------------------
dot_f32_avx2:
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    shl     rdx, 2
    xor     eax, eax
.lp:
    cmp     rax, rdx
    jae     .red
    vmovups ymm4, [rdi + rax]
    vmovups ymm5, [rdi + rax + 32]
    vmovups ymm6, [rdi + rax + 64]
    vmovups ymm7, [rdi + rax + 96]
    vmulps  ymm4, ymm4, [rsi + rax]
    vmulps  ymm5, ymm5, [rsi + rax + 32]
    vmulps  ymm6, ymm6, [rsi + rax + 64]
    vmulps  ymm7, ymm7, [rsi + rax + 96]
    vaddps  ymm0, ymm0, ymm4
    vaddps  ymm1, ymm1, ymm5
    vaddps  ymm2, ymm2, ymm6
    vaddps  ymm3, ymm3, ymm7
    add     rax, 128
    jmp     .lp
.red:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    ret

; ----------------------------------------------------------------------------
; void saxpy_f32_avx2(float *dst, const float *src, uint64 n, float alpha)
; rdi = dst, rsi = src, rdx = n (multiple of 32, e.g. head_dim = 128), xmm0 = alpha
; dst[i] += alpha * src[i]
; ----------------------------------------------------------------------------
saxpy_f32_avx2:
    vshufps xmm0, xmm0, xmm0, 0
    vinsertf128 ymm0, ymm0, xmm0, 1
    shl     rdx, 2
    xor     eax, eax
.lp:
    cmp     rax, rdx
    jae     .done
    vmulps  ymm1, ymm0, [rsi + rax]
    vmulps  ymm2, ymm0, [rsi + rax + 32]
    vmulps  ymm3, ymm0, [rsi + rax + 64]
    vmulps  ymm4, ymm0, [rsi + rax + 96]
    vaddps  ymm1, ymm1, [rdi + rax]
    vaddps  ymm2, ymm2, [rdi + rax + 32]
    vaddps  ymm3, ymm3, [rdi + rax + 64]
    vaddps  ymm4, ymm4, [rdi + rax + 96]
    vmovups [rdi + rax], ymm1
    vmovups [rdi + rax + 32], ymm2
    vmovups [rdi + rax + 64], ymm3
    vmovups [rdi + rax + 96], ymm4
    add     rax, 128
    jmp     .lp
.done:
    ret

%if FAST_Q8
section .rodata
align 32
q8_one16x8:   times 8 dw 1
q8_min_i8x16: times 16 db -127
q8_abs_mask:  dd 0x7FFFFFFF
q8_f127:      dd 127.0
q8_inv_127:   dd 0.007874015748031496       ; 1.0f / 127.0f
%endif

section .bss
alignb 64
g_fp16_scales: resd 1
%if FAST_Q8
alignb 64
q8_xq:        resb Q8_MAX_K
alignb 64
q8_xa:        resb Q8_MAX_K
alignb 64
q8_xs:        resb (Q8_MAX_K / 32) * 4
%ifndef BARE_METAL
alignb 64
q8b_xq:       resb MAX_BATCH * XQ_STRIDE
alignb 64
q8b_xa:       resb MAX_BATCH * XQ_STRIDE
alignb 64
q8b_xs:       resb MAX_BATCH * XS_STRIDE
g_mm_b:       resq 1
g_mm_m4:      resq 1
g_mm_xstride: resq 1
%endif
%endif
