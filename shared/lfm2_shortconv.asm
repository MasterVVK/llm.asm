; LFM2-700M bias-free, depthwise causal short convolution, one token.
; System V AMD64: out, BCx (three planes of n floats), transposed kernel
; [oldest..newest][channel], previous B*x (two planes, initially zero), n.
; out[c] = C[c] * (w0*prev2 + w1*prev1 + w2*(B[c]*x[c])).
; No allocation, syscalls, AVX2 or FMA. Safe for n=0 and nonvector multiples.
bits 64
default rel
section .text
global lfm2_shortconv_step
lfm2_shortconv_step:
    lea r9, [r8*4]
    lea r10, [rsi+r9]           ; C
    lea r11, [r10+r9]           ; x
    xor eax, eax
.loop:
    cmp r8, 8
    jb .tail
    vmovups ymm0, [rsi+rax]
    vmulps  ymm0, ymm0, [r11+rax]
    vmovups ymm1, [rcx+rax]     ; prev2
    vmulps  ymm1, ymm1, [rdx+rax]
    add rcx, r9
    add rdx, r9
    vmovups ymm2, [rcx+rax]     ; prev1
    vmulps  ymm3, ymm2, [rdx+rax]
    vaddps  ymm1, ymm1, ymm3
    add rdx, r9
    vmulps  ymm3, ymm0, [rdx+rax]
    vaddps  ymm1, ymm1, ymm3
    vmulps  ymm1, ymm1, [r10+rax]
    vmovups [rdi+rax], ymm1
    vmovups [rcx+rax], ymm0
    sub rcx, r9
    vmovups [rcx+rax], ymm2
    sub rdx, r9
    sub rdx, r9
    add rax, 32
    sub r8, 8
    jmp .loop
.tail:
    test r8, r8
    jz .done
    vmovss xmm0, [rsi+rax]
    vmulss xmm0, xmm0, [r11+rax]
    vmovss xmm1, [rcx+rax]
    vmulss xmm1, xmm1, [rdx+rax]
    add rcx, r9
    add rdx, r9
    vmovss xmm2, [rcx+rax]
    vmulss xmm3, xmm2, [rdx+rax]
    vaddss xmm1, xmm1, xmm3
    add rdx, r9
    vmulss xmm3, xmm0, [rdx+rax]
    vaddss xmm1, xmm1, xmm3
    vmulss xmm1, xmm1, [r10+rax]
    vmovss [rdi+rax], xmm1
    vmovss [rcx+rax], xmm0
    sub rcx, r9
    vmovss [rcx+rax], xmm2
    sub rdx, r9
    sub rdx, r9
    add rax, 4
    dec r8
    jmp .tail
.done:
    vzeroupper
    ret
section .note.GNU-stack noalloc noexec nowrite progbits
