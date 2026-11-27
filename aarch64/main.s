// ============================================================================
// main.s — _start, CLI and generation loop of the AArch64 port (zero libc)
// Output and stats lines match the x86-64 asm-llm, so the same scripts work.
// ============================================================================
        .include "constants.s"

        .equ O_MODEL,   0
        .equ O_PROMPT,  8
        .equ O_MAX_TOK, 16
        .equ O_THREADS, 24
        .equ O_CTX,     32
        .equ O_THINK,   40
        .equ O_RAW,     48
        .equ O_PRINT,   56
        .equ O_DUMP,    64
        .equ O_HP,      72
        .equ O_T0,      80
        .equ O_T1,      88
        .equ O_T2,      96
        .equ O_T3,      104
        .equ O_NOREPACK, 112


        .text
        .global _start

        .macro FLAG sym, target
        mov     x0, x22
        ADR_L   x1, \sym
        bl      str_eq
        cbnz    x0, \target
        .endm

        .macro NEXT_ARG dst
        add     x21, x21, #1
        cmp     x21, x19
        b.hs    .Lusage
        ldr     \dst, [x20, x21, lsl #3]
        .endm

        .macro ERR msg
        mov     x0, #2
        ADR_L   x1, \msg
        bl      write_cstr
        mov     x0, #1
        b       sys_exit_group
        .endm

_start:
        ldr     x19, [sp]               // argc
        add     x20, sp, #8             // argv
        mov     x29, #0
        ADR_L   x24, opts
        ADR_L   x0, def_prompt
        str     x0, [x24, #O_PROMPT]
        mov     x0, #128
        str     x0, [x24, #O_MAX_TOK]
        str     xzr, [x24, #O_THREADS]  // 0 = auto (big cores of the taskset mask)
        mov     x0, #2048
        str     x0, [x24, #O_CTX]
        mov     x0, #1
        str     x0, [x24, #O_HP]        // 2 MiB pages for the repacked weights
        mov     x21, #1
.Larg:  cmp     x21, x19
        b.hs    .Largs_done
        ldr     x22, [x20, x21, lsl #3]
        FLAG    flg_m, .Lopt_m
        FLAG    flg_p, .Lopt_p
        FLAG    flg_n, .Lopt_n
        FLAG    flg_t, .Lopt_t
        FLAG    flg_c, .Lopt_c
        FLAG    flg_think, .Lopt_think
        FLAG    flg_raw, .Lopt_raw
        FLAG    flg_print, .Lopt_print
        FLAG    flg_dump, .Lopt_dump
        FLAG    flg_hp, .Lopt_hp
        FLAG    flg_nohp, .Lopt_nohp
        FLAG    flg_norepack, .Lopt_norepack
        FLAG    flg_chunk, .Lopt_chunk
        b       .Lusage
.Lopt_m: NEXT_ARG x0
        str     x0, [x24, #O_MODEL]
        b       .Lnext
.Lopt_p: NEXT_ARG x0
        str     x0, [x24, #O_PROMPT]
        b       .Lnext
.Lopt_n: NEXT_ARG x0
        bl      parse_u64
        str     x0, [x24, #O_MAX_TOK]
        b       .Lnext
.Lopt_t: NEXT_ARG x0
        bl      parse_u64
        str     x0, [x24, #O_THREADS]
        b       .Lnext
.Lopt_c: NEXT_ARG x0
        bl      parse_u64
        str     x0, [x24, #O_CTX]
        b       .Lnext
.Lopt_think:
        mov     x0, #1
        str     x0, [x24, #O_THINK]
        b       .Lnext
.Lopt_raw:
        mov     x0, #1
        str     x0, [x24, #O_RAW]
        b       .Lnext
.Lopt_print:
        mov     x0, #1
        str     x0, [x24, #O_PRINT]
        b       .Lnext
.Lopt_hp:
        mov     x0, #1
        str     x0, [x24, #O_HP]
        b       .Lnext
.Lopt_chunk: NEXT_ARG x0
        bl      parse_u64
        ADR_L   x1, tp_chunk_shift
        str     x0, [x1]
        b       .Lnext
.Lopt_nohp:
        str     xzr, [x24, #O_HP]
        b       .Lnext
.Lopt_norepack:
        mov     x0, #1
        str     x0, [x24, #O_NOREPACK]
        b       .Lnext
.Lopt_dump: NEXT_ARG x0
        str     x0, [x24, #O_DUMP]
.Lnext: add     x21, x21, #1
        b       .Larg
.Lusage:
        mov     x0, #2
        ADR_L   x1, msg_usage
        bl      write_cstr
        mov     x0, #2
        b       sys_exit_group

.Largs_done:
        ldr     x0, [x24, #O_MODEL]
        cbz     x0, .Lusage
        bl      sys_clock_ns
        str     x0, [x24, #O_T0]
        ldr     x0, [x24, #O_MODEL]
        ldr     x1, [x24, #O_CTX]
        ldr     x2, [x24, #O_HP]
        ldr     x3, [x24, #O_NOREPACK]
        eor     x3, x3, #1
        bl      model_init
        cmp     x0, #0
        b.ge    1f
        ERR     msg_err_model
1:      bl      sys_clock_ns
        ldr     x1, [x24, #O_T0]
        sub     x0, x0, x1
        ucvtf   d0, x0
        ADR_L   x9, d_1e6
        ldr     d1, [x9]
        fdiv    d0, d0, d1
        mov     x0, #2
        ADR_L   x1, msg_load
        bl      write_cstr
        mov     x0, #2
        bl      write_f64_2dp
        mov     x0, #2
        ADR_L   x1, msg_ms_nl
        bl      write_cstr
        ldr     x0, [x24, #O_THREADS]
        bl      tp_init
        ADR_L   x25, g_ctx
        ldr     x0, [x24, #O_PROMPT]
        ldr     x1, [x25, #CTX_BUF_PROMPT]
        mov     x2, #MAX_PROMPT_TOKENS
        ldr     x3, [x24, #O_RAW]
        ldr     x4, [x24, #O_THINK]
        bl      tok_build_chat_prompt
        mov     x23, x0                 // n_prompt
        cbnz    x23, 2f
        ERR     msg_err_prompt
2:      ldr     x0, [x25, #CTX_MAX_SEQ]
        cmp     x23, x0
        csel    x23, x23, x0, ls
        ldr     x0, [x24, #O_PRINT]
        cbz     x0, 3f
        ADR_L   x0, msg_ptoks
        ldr     x1, [x25, #CTX_BUF_PROMPT]
        mov     x2, x23
        bl      write_tok_csv
3:      // ---------------- prefill ----------------
        bl      sys_clock_ns
        str     x0, [x24, #O_T0]
        mov     x26, #0
4:      cmp     x26, x23
        b.hs    5f
        ldr     x0, [x25, #CTX_BUF_PROMPT]
        ldr     w0, [x0, x26, lsl #2]
        mov     x1, x26
        add     x2, x26, #1
        cmp     x2, x23
        cset    x2, eq                  // logits only for the last prompt token
        bl      transformer_forward
        add     x26, x26, #1
        b       4b
5:      bl      sys_clock_ns
        str     x0, [x24, #O_T1]
        ldr     x0, [x24, #O_DUMP]
        cbz     x0, 6f
        bl      sys_open_wr
        cmp     x0, #0
        b.lt    6f
        mov     x27, x0
        ldr     x1, [x25, #CTX_BUF_LOGITS]
        ldr     x2, [x25, #CTX_VOCAB_SIZE]
        lsl     x2, x2, #2
        bl      sys_write_all
        mov     x0, x27
        bl      sys_close
6:      ldr     x0, [x25, #CTX_BUF_LOGITS]
        ldr     x1, [x25, #CTX_VOCAB_SIZE]
        bl      argmax_f32
        mov     x27, x0                 // next token
        // ---------------- decode ----------------
        bl      sys_clock_ns
        str     x0, [x24, #O_T2]
        mov     x28, #0                 // n_gen
        mov     x26, x23                // pos
7:      ldr     x0, [x24, #O_MAX_TOK]
        cmp     x28, x0
        b.hs    8f
        ldr     x0, [x25, #CTX_EOS_1]
        cmp     x27, x0
        b.eq    8f
        ldr     x0, [x25, #CTX_EOS_2]
        cmp     x27, x0
        b.eq    8f
        ldr     x0, [x25, #CTX_BUF_GEN]
        str     w27, [x0, x28, lsl #2]
        add     x28, x28, #1
        mov     w0, w27
        bl      tok_write_stdout
        ldr     x0, [x24, #O_MAX_TOK]
        cmp     x28, x0
        b.hs    8f
        ldr     x0, [x25, #CTX_MAX_SEQ]
        cmp     x26, x0
        b.hs    8f
        mov     w0, w27
        mov     x1, x26
        mov     x2, #1
        bl      transformer_forward
        add     x26, x26, #1
        ldr     x0, [x25, #CTX_BUF_LOGITS]
        ldr     x1, [x25, #CTX_VOCAB_SIZE]
        bl      argmax_f32
        mov     x27, x0
        b       7b
8:      bl      sys_clock_ns
        str     x0, [x24, #O_T3]
        mov     x0, #1
        ADR_L   x1, msg_nl
        mov     x2, #1
        bl      sys_write_all
        ldr     x0, [x24, #O_PRINT]
        cbz     x0, 9f
        ADR_L   x0, msg_gtoks
        ldr     x1, [x25, #CTX_BUF_GEN]
        mov     x2, x28
        bl      write_tok_csv
9:      // ---------------- stats (stderr) ----------------
        ADR_L   x1, msg_prefill
        mov     x2, x23
        ldr     x3, [x24, #O_T0]
        ldr     x4, [x24, #O_T1]
        mov     x5, x23
        bl      write_stat
        ADR_L   x1, msg_tps_end
        mov     x0, #2
        bl      write_cstr
        ADR_L   x1, msg_decode
        mov     x2, x28
        ldr     x3, [x24, #O_T2]
        ldr     x4, [x24, #O_T3]
        subs    x5, x28, #1             // exact number of decode forward passes
        csel    x5, x5, x28, gt
        bl      write_stat
        mov     x0, #2
        ADR_L   x1, msg_tps_thr
        bl      write_cstr
        mov     x0, #2
        ADR_L   x1, tp_n_threads
        ldr     x1, [x1]
        bl      write_u64
        mov     x0, #2
        ADR_L   x1, msg_little
        bl      write_cstr
        mov     x0, #2
        ADR_L   x1, tp_n_little
        ldr     x1, [x1]
        bl      write_u64
        mov     x0, #2
        ADR_L   x1, msg_close_nl
        bl      write_cstr
        mov     x0, #0
        b       sys_exit_group

// write_stat(label x1, count x2, t_start x3, t_end x4, passes x5):
// "<label><count> tok in <ms> ms (<passes*1e9/dt>" to stderr
write_stat:
        stp     x29, x30, [sp, #-48]!
        stp     x19, x20, [sp, #16]
        sub     x19, x4, x3
        cmp     x19, #0
        csinc   x19, x19, xzr, gt       // dt >= 1 ns
        mov     x20, x5
        str     x2, [sp, #32]
        mov     x0, #2
        bl      write_cstr
        mov     x0, #2
        ldr     x1, [sp, #32]
        bl      write_u64
        mov     x0, #2
        ADR_L   x1, msg_in
        bl      write_cstr
        ucvtf   d0, x19
        ADR_L   x9, d_1e6
        ldr     d1, [x9]
        fdiv    d0, d0, d1
        mov     x0, #2
        bl      write_f64_2dp
        mov     x0, #2
        ADR_L   x1, msg_ms_open
        bl      write_cstr
        ucvtf   d0, x20
        ADR_L   x9, d_1e9
        ldr     d1, [x9]
        fmul    d0, d0, d1
        ucvtf   d1, x19
        fdiv    d0, d0, d1
        mov     x0, #2
        bl      write_f64_2dp
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #48
        ret

// write_tok_csv(label x0, const uint32 *toks x1, uint64 n x2) -> stderr
write_tok_csv:
        stp     x29, x30, [sp, #-48]!
        stp     x19, x20, [sp, #16]
        str     x21, [sp, #32]
        mov     x19, x1
        mov     x20, x2
        mov     x1, x0
        mov     x0, #2
        bl      write_cstr
        mov     x21, #0
1:      cmp     x21, x20
        b.hs    3f
        cbz     x21, 2f
        mov     x0, #2
        ADR_L   x1, msg_comma
        mov     x2, #1
        bl      sys_write_all
2:      mov     x0, #2
        ldr     w1, [x19, x21, lsl #2]
        bl      write_u64
        add     x21, x21, #1
        b       1b
3:      mov     x0, #2
        ADR_L   x1, msg_nl
        mov     x2, #1
        bl      sys_write_all
        ldr     x21, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #48
        ret

        .section .rodata
        .balign 8
d_1e6:  .double 1.0e6
d_1e9:  .double 1.0e9
flg_m:  .asciz "-m"
flg_p:  .asciz "-p"
flg_n:  .asciz "-n"
flg_t:  .asciz "-t"
flg_c:  .asciz "-c"
flg_think: .asciz "--think"
flg_raw: .asciz "--raw"
flg_print: .asciz "--print-tokens"
flg_dump: .asciz "--dump-logits"
flg_hp: .asciz "--hugepages"
flg_norepack: .asciz "--norepack"
flg_nohp: .asciz "--nohugepages"
flg_chunk: .asciz "--chunk-shift"
msg_little: .asciz ", a55="
msg_load: .asciz "[asm-llm] load   : "
msg_ms_nl: .asciz " ms\n"
def_prompt: .asciz "Hello!"
msg_nl: .ascii "\n"
msg_comma: .ascii ","
msg_ptoks: .asciz "PROMPT_TOKENS:"
msg_gtoks: .asciz "GEN_TOKENS:"
msg_prefill: .asciz "[asm-llm] prefill: "
msg_decode: .asciz "[asm-llm] decode : "
msg_in: .asciz " tok in "
msg_ms_open: .asciz " ms ("
msg_tps_end: .asciz " tok/s)\n"
msg_tps_thr: .asciz " tok/s, threads="
msg_close_nl: .asciz ")\n"
msg_err_model: .asciz "error: failed to load .asmbin v2 model\n"
msg_err_prompt: .asciz "error: empty prompt\n"
msg_usage:
        .ascii "asm-llm (pure AArch64 NEON+SDOT Qwen3-0.6B Q8_0, zero libc)\n"
        .ascii "Usage: asm-llm -m <model.asmbin> [options]\n"
        .ascii "  -p <prompt>          Prompt text (default: Hello!)\n"
        .ascii "  -n <max_tokens>      Max generated tokens (default: 128)\n"
        .ascii "  -t <threads>         Threads (default: one per big core of the taskset mask)\n"
        .ascii "  -c <ctx_len>         KV cache length (default: 2048)\n"
        .ascii "  --think | --raw      Qwen3 reasoning mode | no chat template\n"
        .ascii "  --print-tokens       Print PROMPT_TOKENS/GEN_TOKENS to stderr\n"
        .ascii "  --dump-logits <path> Dump float32 logits after prefill\n"
        .ascii "  --nohugepages        Do not request 2 MiB pages (default: on)\n"
        .ascii "  --hugepages          (default) 2 MiB pages; with --norepack copies the file\n"
        .ascii "  --norepack           Row-major kernel instead of 4-row interleaved\n"
        .asciz "  --chunk-shift <n>    Rows per work chunk = 2^n (default 4; 8 on A55-only sets)\n"

        .bss
        .balign 8
opts:   .skip   128
