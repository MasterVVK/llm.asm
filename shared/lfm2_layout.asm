; Validate the fixed LFM2-700M ALF2 v1 layout before resolving pointers.
; rdi = mapping base, rsi = actual size. Returns eax = 0 or -1.
bits 64
default rel
%include "constants.inc"
section .text
global lfm2_validate
lfm2_validate:
    cmp     rsi, 1024*1024*1024
    ja      .bad
    cmp     rsi, 128+64
    jb      .bad
    cmp     [rdi+HDR_FILE_SIZE], rsi
    jne     .bad
%macro EXACT 2
    cmp     dword [rdi+%1], %2
    jne     .bad
%endmacro
    EXACT HDR_MAGIC, ALF2_MAGIC
    EXACT HDR_VERSION, 1
    EXACT HDR_DIM, 1536
    EXACT HDR_HIDDEN_DIM, 6912
    EXACT HDR_N_LAYERS, 16
    EXACT HDR_N_HEADS, 24
    EXACT HDR_N_KV_HEADS, 8
    EXACT HDR_HEAD_DIM, 64
    EXACT HDR_VOCAB_SIZE, 65536
    EXACT HDR_FLAGS, 7
    EXACT HDR_HT_MASK, 524287
    EXACT HDR_EOS_1, 7
    EXACT HDR_EOS_2, 2
    EXACT HDR_RMS_EPS, __float32__(0.00001)
    EXACT HDR_ROPE_THETA, __float32__(1000000.0)
    mov     eax, [rdi+HDR_MAX_SEQ_LEN]
    dec     eax
    cmp     eax, 4095
    ja      .bad
    cmp     qword [rdi+HDR_LAYER_STRIDE], 128
    jne     .bad
    cmp     qword [rdi+HDR_OFF_TOKENIZER], 128
    jne     .bad
    lea     r10, [rdi+128]
    cmp     qword [r10], 64
    jne     .bad
    cmp     qword [r10+8], 1088
    jne     .bad
    cmp     qword [r10+16], 263296
    jne     .bad
    mov     rax, [r10+24]               ; relative hash table offset
    cmp     rax, 263296
    jb      .bad
    cmp     rax, rsi
    jae     .bad
    test    eax, 63
    jnz     .bad
    add     rax, 128 + 524288*16
    cmp     rax, [rdi+HDR_OFF_EMBED]
    jne     .bad
    add     rax, 65536*(1536+96)
    cmp     rax, [rdi+HDR_OFF_LAYERS]
    jne     .bad
    lea     r8, [rdi+rax]
    add     rax, 16*128
    cmp     rax, rsi
    ja      .bad
    xor     ecx, ecx
; Each descriptor must point to the exact start of each tensor. In particular,
; Q/K/V and gate/up are contiguous as required by the fused matrix multiplies.
%macro FIELD 2
    cmp     [r8+8*%1], rax
    jne     .bad
    add     rax, %2
%endmacro
%macro ABSENT 1
    cmp     qword [r8+8*%1], 0
    jne     .bad
%endmacro
.loop:
    mov     edx, (1<<2)|(1<<5)|(1<<8)|(1<<10)|(1<<12)|(1<<14)
    bt      edx, ecx
    setc    dl
    movzx   edx, dl
    cmp     [r8], rdx
    jne     .bad
    FIELD   1,1536*4
    FIELD   2,1536*4
    test    edx, edx
    jz      .conv
    FIELD   3,64*4
    FIELD   4,64*4
    FIELD   5,1536*1632
    FIELD   6,512*1632
    FIELD   7,512*1632
    FIELD   8,1536*1632
    ABSENT  9
    jmp     .ffn
.conv:
    ABSENT  3
    ABSENT  4
    FIELD   5,3*1536*1632
    ABSENT  6
    ABSENT  7
    FIELD   8,1536*1632
    FIELD   9,3*1536*4
.ffn:
    FIELD   10,6912*1632
    FIELD   11,6912*1632
    FIELD   12,1536*(6912+432)
    ABSENT  13
    ABSENT  14
    ABSENT  15
    add     r8, 128
    inc     ecx
    cmp     ecx, 16
    jb      .loop
    cmp     rax, [rdi+HDR_OFF_FINAL_NORM]
    jne     .bad
    add     rax, 1536*4
    cmp     rax, rsi
    jne     .bad
    ; Check token byte slices, byte token IDs, merge output IDs and a vacant slot.
    mov     r8, [r10+24]
    sub     r8, 263296
    xor     ecx, ecx
    xor     edx, edx
.tok:
    mov     eax, [r10+1088+rcx*4]
    cmp     eax, edx
    jb      .bad
    cmp     rax, r8
    ja      .bad
    mov     edx, eax
    inc     ecx
    cmp     ecx, 65537
    jb      .tok
    xor     ecx, ecx
.bytes:
    cmp     dword [r10+64+rcx*4], 65536
    jae     .bad
    inc     ecx
    cmp     ecx, 256
    jb      .bytes
    mov     r8, [r10+24]
    add     r8, r10
    mov     ecx, 524288
    xor     edx, edx
.merges:
    cmp     dword [r8+12], -1
    jne     .occupied
    mov     edx, 1
    jmp     .advance
.occupied:
    cmp     dword [r8+8], 65536
    jae     .bad
.advance:
    add     r8, 16
    dec     ecx
    jnz     .merges
    test    edx, edx
    jz      .bad
    xor     eax, eax
    ret
.bad:
    mov     eax, -1
    ret
section .note.GNU-stack noalloc noexec nowrite progbits
