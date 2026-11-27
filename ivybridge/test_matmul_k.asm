; ============================================================================
; test_matmul_k.asm — exact-reference check of quantize_x_q8_0 +
; matmul_q8_0_range at real model sizes. Upstream test_math uses K = 64, which
; only reaches the 32-byte tail; K = 1024 / 2048 / 3072 run the 4-block loop.
;
;   stdin : [u64 K][u64 M][u64 is_fp16][K f32 x][M Q8_0 rows]
;   stdout: [M f32 out]
; ============================================================================

bits 64
default rel

%include "constants.inc"

%define MAX_K   4096
%define MAX_M   512
%define MAX_W   (MAX_M * (MAX_K + MAX_K / 8))

extern sys_write_all
extern sys_exit_group
extern matmul_set_fp16
extern quantize_x_q8_0
extern matmul_q8_0_range

section .text
global _start

; read_all(buf=rdi, len=rsi) from stdin
read_all:
    push    rbx
    push    r12
    mov     rbx, rdi
    mov     r12, rsi
.lp:
    test    r12, r12
    jz      .done
    xor     edi, edi
    mov     rsi, rbx
    mov     rdx, r12
    mov     eax, SYS_read
    syscall
    cmp     rax, -4
    je      .lp
    test    rax, rax
    jle     .done
    add     rbx, rax
    sub     r12, rax
    jmp     .lp
.done:
    pop     r12
    pop     rbx
    ret

_start:
    and     rsp, -16

    lea     rdi, [rel hdr]
    mov     esi, 24
    call    read_all

    lea     rdi, [rel in_x]
    mov     rsi, [rel hdr]              ; K
    shl     rsi, 2
    call    read_all

    mov     rax, [rel hdr]              ; K
    mov     rcx, rax
    cmp     qword [rel hdr + 16], 0
    jne     .stride_f16
    shr     rcx, 3                      ; v1: K/32 fp32 scales
    jmp     .stride_ok
.stride_f16:
    shr     rcx, 4                      ; v2: K/32 fp16 scales
.stride_ok:
    add     rcx, rax                    ; row_stride
    imul    rcx, [rel hdr + 8]          ; * M
    lea     rdi, [rel in_w]
    mov     rsi, rcx
    call    read_all

    lea     rdi, [rel in_x]
    mov     rsi, [rel hdr]
    call    quantize_x_q8_0
    mov     rdi, [rel hdr + 16]
    call    matmul_set_fp16

    lea     rdi, [rel in_w]
    lea     rsi, [rel in_x]
    lea     rdx, [rel out]
    mov     rcx, [rel hdr]              ; K
    xor     r8d, r8d                    ; r0
    mov     r9, [rel hdr + 8]           ; r1 = M
    call    matmul_q8_0_range

    mov     edi, 1
    lea     rsi, [rel out]
    mov     rdx, [rel hdr + 8]
    shl     rdx, 2
    call    sys_write_all

    xor     edi, edi
    call    sys_exit_group

section .bss
alignb 64
hdr:    resq 3
alignb 64
in_x:   resb MAX_K * 4
alignb 64
in_w:   resb MAX_W
alignb 64
out:    resb MAX_M * 4
