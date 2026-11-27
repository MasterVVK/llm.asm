; Kernel micro-benchmark: L1/L2-resident batched GEMM vs GEMV, plus an
; optional candidate kernel (-DNEWK=<symbol>) checked for bit-identical output.
;   stdin : [u64 K][u64 M][u64 B][u64 R]
;   stdout: [u64 cal_ns][u64 batch_ns][u64 gemv_ns][u64 new_ns][M*B f32 out][M*B f32 out_new]
bits 64
default rel
%include "constants.inc"

%define MAX_K 4096
%define MAX_M 256
%define MAX_B 16

extern quantize_x_batch
extern matmul_q8_0_batch_range
extern q8b_x
extern quantize_x_q8_0
extern matmul_q8_0_range
extern matmul_set_fp16
extern g_mm_b
extern g_mm_m4
%ifdef NEWK
extern NEWK
%endif

section .text
global _start

now_ns:
    sub     rsp, 24
    mov     edi, 1
    mov     rsi, rsp
    mov     eax, SYS_clock_gettime
    syscall
    mov     rax, [rsp]
    imul    rax, rax, 1000000000
    add     rax, [rsp + 8]
    add     rsp, 24
    ret

lcg:                                    ; rax = next 64-bit state
    mov     rax, [rel seed]
    mov     rcx, 6364136223846793005
    imul    rax, rcx
    mov     rcx, 1442695040888963407
    add     rax, rcx
    mov     [rel seed], rax
    ret

_start:
    and     rsp, -16
    xor     edi, edi
    lea     rsi, [rel prm]
    mov     edx, 32
    mov     eax, SYS_read
    syscall
    mov     edi, 1
    call    matmul_set_fp16

    ; W: M rows of K int8 (no -128) + K/32 fp16 scales
    mov     rax, [rel prm]
    shr     rax, 4
    add     rax, [rel prm]
    mov     [rel rstride], rax
    imul    rax, [rel prm + 8]
    mov     r12, rax                    ; total bytes
    xor     r13d, r13d
.fw:
    cmp     r13, r12
    jae     .fw_done
    call    lcg
    shr     rax, 33
    cmp     al, 0x80
    jne     .fw_ok
    mov     al, 0x81
.fw_ok:
    lea     rdx, [rel W]
    mov     [rdx + r13], al
    inc     r13
    jmp     .fw
.fw_done:
    xor     r13d, r13d                  ; row
.fs_row:
    cmp     r13, [rel prm + 8]
    jae     .fs_done
    mov     r14, r13
    imul    r14, [rel rstride]
    add     r14, [rel prm]
    lea     rdx, [rel W]
    add     r14, rdx                    ; scales of row
    mov     r15, [rel prm]
    shr     r15, 5
.fs:
    call    lcg
    shr     rax, 40
    and     eax, 0x3FF
    or      eax, 0x2000
    mov     [r14], ax
    add     r14, 2
    dec     r15
    jnz     .fs
    inc     r13
    jmp     .fs_row
.fs_done:
    ; X: B x K floats in [-1, 1)
    mov     r12, [rel prm]
    imul    r12, [rel prm + 16]
    xor     r13d, r13d
.fx:
    cmp     r13, r12
    jae     .fx_done
    call    lcg
    sar     rax, 32
    vcvtsi2ss xmm0, xmm0, rax
    vmulss  xmm0, xmm0, [rel inv31]
    lea     rdx, [rel X]
    vmovss  [rdx + r13*4], xmm0
    inc     r13
    jmp     .fx
.fx_done:

    ; frequency calibration: 3e7 x 10 dependent imul (3 cycles each)
    call    now_ns
    mov     r12, rax
    mov     rcx, 30000000
    mov     rax, 3
.cal:
%rep 10
    imul    rax, rax
%endrep
    dec     rcx
    jnz     .cal
    call    now_ns
    sub     rax, r12
    mov     [rel res], rax

    ; batched GEMM
    lea     rdi, [rel X]
    mov     rsi, [rel prm]
    mov     rdx, [rel prm + 16]
    mov     rcx, [rel prm]
    shl     rcx, 2
    call    quantize_x_batch
    mov     rax, [rel prm + 16]
    mov     [rel g_mm_b], rax
    mov     rax, [rel prm + 8]
    shl     rax, 2
    mov     [rel g_mm_m4], rax
    mov     r15, [rel prm + 24]
    inc     r15                         ; +1 warm-up
    xor     r12d, r12d
.bl:
    cmp     r15, [rel prm + 24]
    jne     .bl_go
    call    now_ns
    mov     r12, rax
.bl_go:
    lea     rdi, [rel W]
    lea     rsi, [rel q8b_x]
    lea     rdx, [rel OUT]
    mov     rcx, [rel prm]
    xor     r8d, r8d
    mov     r9, [rel prm + 8]
    call    matmul_q8_0_batch_range
    dec     r15
    jnz     .bl
    call    now_ns
    sub     rax, r12
    mov     [rel res + 8], rax

%ifdef NEWK
    mov     r15, [rel prm + 24]
    inc     r15
    xor     r12d, r12d
.nl:
    cmp     r15, [rel prm + 24]
    jne     .nl_go
    call    now_ns
    mov     r12, rax
.nl_go:
    lea     rdi, [rel W]
    lea     rsi, [rel q8b_x]
    lea     rdx, [rel OUT3]
    mov     rcx, [rel prm]
    xor     r8d, r8d
    mov     r9, [rel prm + 8]
    call    NEWK
    dec     r15
    jnz     .nl
    call    now_ns
    sub     rax, r12
    mov     [rel res + 24], rax
%endif

    ; GEMV on token 0, R * B calls
    lea     rdi, [rel X]
    mov     rsi, [rel prm]
    call    quantize_x_q8_0
    mov     r15, [rel prm + 24]
    imul    r15, [rel prm + 16]
    call    now_ns
    mov     r12, rax
.gl:
    lea     rdi, [rel W]
    lea     rsi, [rel X]
    lea     rdx, [rel OUT2]
    mov     rcx, [rel prm]
    xor     r8d, r8d
    mov     r9, [rel prm + 8]
    call    matmul_q8_0_range
    dec     r15
    jnz     .gl
    call    now_ns
    sub     rax, r12
    mov     [rel res + 16], rax

    mov     edi, 1
    lea     rsi, [rel res]
    mov     edx, 32
    mov     eax, SYS_write
    syscall
    mov     rdx, [rel prm + 8]
    imul    rdx, [rel prm + 16]
    shl     rdx, 2
    mov     edi, 1
    lea     rsi, [rel OUT]
    mov     eax, SYS_write
    syscall
    mov     rdx, [rel prm + 8]
    imul    rdx, [rel prm + 16]
    shl     rdx, 2
    mov     edi, 1
    lea     rsi, [rel OUT3]
    mov     eax, SYS_write
    syscall
    xor     edi, edi
    mov     eax, SYS_exit_group
    syscall

section .rodata
align 4
inv31: dd 0x30000000                    ; 2^-31

section .data
align 8
seed: dq 0x9E3779B97F4A7C15

section .bss
alignb 64
prm:     resq 4
res:     resq 4
rstride: resq 1
alignb 64
W:       resb MAX_M * (MAX_K + MAX_K / 16)
alignb 64
X:       resd MAX_B * MAX_K
alignb 64
OUT:     resd MAX_B * MAX_M
OUT2:    resd MAX_B * MAX_M
OUT3:    resd MAX_B * MAX_M
