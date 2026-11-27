// ============================================================================
// constants.s — AArch64 Linux port: syscalls, .asmbin layout, runtime context
// GNU as syntax, zero libc. Mirrors shared/constants.inc of the x86-64 build.
// ============================================================================

        .arch   armv8.2-a+dotprod+fp16

// --- Linux AArch64 (asm-generic) syscall numbers ---
        .equ SYS_openat,        56
        .equ SYS_close,         57
        .equ SYS_read,          63
        .equ SYS_write,         64
        .equ SYS_fstat,         80
        .equ SYS_exit,          93
        .equ SYS_exit_group,    94
        .equ SYS_clock_gettime, 113
        .equ SYS_munmap,        215
        .equ SYS_clone,         220
        .equ SYS_mmap,          222
        .equ SYS_madvise,       233
        .equ SYS_sched_setaffinity, 122
        .equ SYS_sched_getaffinity, 123
        .equ MIDR_PART_A55,     0xD05

        .equ AT_FDCWD,          -100
        .equ O_RDONLY,          0
        .equ O_WRONLY_CREAT_TR, 0x241
        .equ PROT_READ,         1
        .equ PROT_RW,           3
        .equ MAP_PRIVATE,       2
        .equ MAP_PRIV_ANON,     0x22
        .equ MADV_HUGEPAGE,     14
        .equ CLOCK_MONOTONIC,   1
        // CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD
        .equ CLONE_THREAD_FLAGS, 0x00010F00
        // struct stat (asm-generic): st_size at 48, sizeof = 128
        .equ STAT_SIZE,         128
        .equ STAT_ST_SIZE,      48

// --- .asmbin header (128 bytes) ---
        .equ ASMBIN_MAGIC,      0x4C4D5341      // "ASML"
        .equ HDR_FLAG_FP16_SCALES, 4
        .equ HDR_MAGIC,         0
        .equ HDR_VERSION,       4
        .equ HDR_DIM,           8
        .equ HDR_HIDDEN_DIM,    12
        .equ HDR_N_LAYERS,      16
        .equ HDR_N_HEADS,       20
        .equ HDR_N_KV_HEADS,    24
        .equ HDR_HEAD_DIM,      28
        .equ HDR_VOCAB_SIZE,    32
        .equ HDR_MAX_SEQ_LEN,   36
        .equ HDR_RMS_EPS,       40
        .equ HDR_ROPE_THETA,    44
        .equ HDR_EOS_1,         48
        .equ HDR_EOS_2,         52
        .equ HDR_HT_MASK,       56
        .equ HDR_FLAGS,         60
        .equ HDR_OFF_TOKENIZER, 64
        .equ HDR_OFF_EMBED,     72
        .equ HDR_OFF_LAYERS,    80
        .equ HDR_LAYER_STRIDE,  88
        .equ HDR_OFF_FINAL_NORM, 96

// --- Tokenizer section ---
        .equ TOKH_OFF_B2T,      0
        .equ TOKH_OFF_OFFSETS,  8
        .equ TOKH_OFF_BYTES,    16
        .equ TOKH_OFF_HT,       24
        .equ HT_KEY,            0
        .equ HT_MERGED,         8
        .equ HT_RANK,           12
        .equ HT_SHIFT,          45

// --- Qwen3 chat template token IDs ---
        .equ TOK_IM_START,      151644
        .equ TOK_IM_END,        151645
        .equ TOK_THINK_START,   151667
        .equ TOK_THINK_END,     151668
        .equ TOK_USER,          872
        .equ TOK_ASSISTANT,     77091
        .equ TOK_NL,            198
        .equ TOK_NL2,           271

// --- Limits ---
        .equ MAX_THREADS,       64
        .equ WORKER_STACK_SIZE, 65536
        .equ MAX_PROMPT_TOKENS, 8192
        .equ MAX_GEN_TOKENS,    8192
        .equ Q8_MAX_K,          16384
        .equ MAX_BATCH,         16              // prompt tokens per weight pass
        .ifndef DEFAULT_BATCH
        .equ DEFAULT_BATCH,     16              // best on A76 (64 KB L1D) and A55 alike
        .endif
        .equ XQ_STRIDE,         (Q8_MAX_K + 64) // per-token int8 slot (+64: no L1 aliasing)
        .equ XS_STRIDE,         ((Q8_MAX_K / 32) * 4 + 64)
        .ifndef CH_SHIFT
        .equ CH_SHIFT,          4               // 16 rows per dynamically claimed chunk
        .endif
        .ifndef CH_SHIFT_LITTLE
        .equ CH_SHIFT_LITTLE,   8               // 256 rows when only A55 cores work
        .endif

// --- Runtime context (g_ctx) ---
        .equ CTX_MMAP_BASE,     0
        .equ CTX_MMAP_SIZE,     8
        .equ CTX_ARENA_BASE,    16
        .equ CTX_ARENA_SIZE,    24
        .equ CTX_B2T_PTR,       32
        .equ CTX_TOK_OFFS_PTR,  40
        .equ CTX_TOK_BYTES_PTR, 48
        .equ CTX_HT_PTR,        56
        .equ CTX_EMB_PTR,       64
        .equ CTX_LAYERS_PTR,    72
        .equ CTX_FINAL_NORM_PTR, 80
        .equ CTX_DIM,           88
        .equ CTX_HIDDEN_DIM,    96
        .equ CTX_N_LAYERS,      104
        .equ CTX_N_HEADS,       112
        .equ CTX_N_KV_HEADS,    120
        .equ CTX_HEAD_DIM,      128
        .equ CTX_Q_DIM,         136
        .equ CTX_KV_DIM,        144
        .equ CTX_VOCAB_SIZE,    152
        .equ CTX_MAX_SEQ,       160
        .equ CTX_LAYER_STRIDE,  168
        .equ CTX_HT_MASK,       176
        .equ CTX_EOS_1,         184
        .equ CTX_EOS_2,         192
        .equ CTX_RMS_EPS,       200             // f32
        .equ CTX_ROPE_THETA,    204             // f32
        .equ CTX_BUF_X,         208             // first of 16 arena pointers
        .equ CTX_BUF_XB,        216
        .equ CTX_BUF_Q,         224
        .equ CTX_BUF_K,         232
        .equ CTX_BUF_V,         240
        .equ CTX_BUF_ATT_OUT,   248
        .equ CTX_BUF_ATT_SCR,   256
        .equ CTX_BUF_GATE,      264
        .equ CTX_BUF_UP,        272
        .equ CTX_BUF_LOGITS,    280
        .equ CTX_BUF_INV_FREQ,  288
        .equ CTX_BUF_COS,       296
        .equ CTX_BUF_SIN,       304
        .equ CTX_BUF_K_CACHE,   312
        .equ CTX_BUF_V_CACHE,   320
        .equ CTX_BUF_PROMPT,    328
        .equ CTX_BUF_GEN,       336
        .equ CTX_BUF_BX,        344             // batched prefill: [MAX_BATCH][dim]
        .equ CTX_BUF_BXB,       352             // [MAX_BATCH][dim]
        .equ CTX_BUF_BQKV,      360             // [MAX_BATCH][q_dim + 2 kv_dim]
        .equ CTX_BUF_BATT,      368             // [MAX_BATCH][q_dim]
        .equ CTX_BUF_BGU,       376             // [MAX_BATCH][2 hidden_dim]
        .equ CTX_BUF_BCOS,      384             // [MAX_BATCH][head_dim / 2]
        .equ CTX_BUF_BSIN,      392
        .equ CTX_BUF_BSCR,      400             // [MAX_BATCH][n_heads][max_seq] attention scores
        .equ CTX_N_BUFS,        25
        .equ CTX_SIZE,          448

// adrp + add for a symbol in a static executable
        .macro ADR_L reg, sym
        adrp    \reg, \sym
        add     \reg, \reg, :lo12:\sym
        .endm

// raw syscall: number in x8
        .macro SYSCALL nr
        mov     x8, #\nr
        svc     #0
        .endm
