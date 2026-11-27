// ============================================================================
// threads.s — zero-libc clone() thread pool for the AArch64 port
//
// - Dynamic work distribution: every thread (main included) claims the next
//   chunk with an atomic LDADD, so fast A76 cores take more rows than slow
//   A55 cores instead of everyone waiting for the slowest static slice.
// - Results stay deterministic: a row (or attention head) is always computed
//   by exactly one thread with the same code, whatever thread takes it.
// - Waiting uses LDAXR + WFE: the core sleeps until another core writes the
//   watched word (or the kernel event stream ticks) instead of spinning hot.
// ============================================================================
        .include "constants.s"

        .equ JOB_KIND,    0
        .equ JOB_NCHUNKS, 4
        .equ JOB_W,       8
        .equ JOB_OUT,     16
        .equ JOB_K,       24
        .equ JOB_M,       32


        .text
        .global tp_init, tp_matmul, tp_attn, tp_n_threads, g_mm_delta, tp_n_little, tp_chunk_shift
        .global tp_matmul_batch, tp_main_variant
        .global tp_kv_base, tp_pos, tp_scale, tp_gqa
        .global tp_attn_batch, tp_ab_b

// void tp_init(uint64 n x0) — spawns n-1 workers (n clamped to [1, MAX_THREADS])
// n = 0 picks the thread count automatically: one per big core of the
// taskset mask, or one per allowed CPU when the mask has only A55 cores.
tp_init:
        stp     x29, x30, [sp, #-32]!
        stp     x19, x20, [sp, #16]
        mov     x20, x0
        bl      tp_build_cpu_list
        ADR_L   x1, tp_big_count
        ldr     x2, [x1]
        ADR_L   x1, tp_cpu_count
        ldr     x3, [x1]
        cbnz    x20, 4f
        cmp     x2, #0
        csel    x20, x2, x3, ne         // auto: big cores, else all allowed
4:      cmp     x20, #1
        mov     x1, #1
        csel    x20, x20, x1, ge
        cmp     x20, #MAX_THREADS
        mov     x1, #MAX_THREADS
        csel    x20, x20, x1, le
        ADR_L   x1, tp_n_threads
        str     x20, [x1]
        // chunk size: 16 rows if any thread sits on a big core (best for A76
        // and for mixed sets), 256 rows when every thread is on an A55, which
        // streams memory better in long runs (+23..27% measured on 4x A55)
        ADR_L   x1, tp_chunk_shift
        ldr     x4, [x1]
        cbnz    x4, 5f                  // set by --chunk-shift
        mov     x4, #CH_SHIFT
        mov     x5, #CH_SHIFT_LITTLE
        cmp     x2, #0
        csel    x4, x4, x5, ne
        str     x4, [x1]
5:      mov     x0, #0                  // main thread is tid 0
        bl      tp_pin_and_probe
        ADR_L   x1, tp_main_variant
        str     x0, [x1]
        mov     x19, #1                 // tid of the next worker
1:      cmp     x19, x20
        b.hs    3f
        mov     x0, #WORKER_STACK_SIZE
        bl      sys_mmap_anon
        cmn     x0, #4096
        b.hi    2f                      // mmap failed
        add     x1, x0, #WORKER_STACK_SIZE
        ldr     x0, =CLONE_THREAD_FLAGS
        mov     x2, #0
        mov     x3, #0
        mov     x4, #0
        mov     x8, #SYS_clone
        svc     #0
        cbz     x0, worker_main         // child: x19 = tid, sp = new stack
        cmp     x0, #0
        b.lt    2f
        add     x19, x19, #1
        b       1b
2:      ADR_L   x1, tp_n_threads        // run with the workers we got
        str     x19, [x1]
3:      ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #32
        ret

// Collect the CPUs this process may run on (taskset mask) into tp_cpu_list,
// big cores first: the main thread (tid 0) does all sequential work and must
// not land on a slow Cortex-A55. Each CPU is visited once to read its MIDR.
tp_build_cpu_list:
        stp     x29, x30, [sp, #-176]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        mov     x0, #0
        mov     x1, #128
        add     x2, sp, #48
        mov     x8, #SYS_sched_getaffinity
        svc     #0
        mov     x21, #0                 // big count
        mov     x22, #0                 // little count
        cmp     x0, #0
        b.le    4f
        ldr     x19, [sp, #48]          // first 64 CPUs
        mov     x20, #0
1:      cmp     x20, #64
        b.hs    3f
        lsr     x7, x19, x20
        tbz     x7, #0, 2f
        mov     x7, #1
        lsl     x7, x7, x20
        str     x7, [sp, #176 - 8]
        mov     x0, #0
        mov     x1, #8
        add     x2, sp, #176 - 8
        mov     x8, #SYS_sched_setaffinity
        svc     #0
        mrs     x0, midr_el1
        ubfx    x0, x0, #4, #12
        mov     x1, #MIDR_PART_A55
        cmp     x0, x1
        b.eq    5f
        ADR_L   x3, tp_cpu_list         // big core: append to the front list
        strb    w20, [x3, x21]
        add     x21, x21, #1
        b       2f
5:      ADR_L   x3, tp_little_tmp
        strb    w20, [x3, x22]
        add     x22, x22, #1
2:      add     x20, x20, #1
        b       1b
3:      ADR_L   x3, tp_cpu_list         // then the little cores
        ADR_L   x4, tp_little_tmp
        mov     x5, #0
6:      cmp     x5, x22
        b.hs    4f
        ldrb    w6, [x4, x5]
        add     x7, x21, x5
        strb    w6, [x3, x7]
        add     x5, x5, #1
        b       6b
4:      add     x0, x21, x22
        ADR_L   x3, tp_cpu_count
        str     x0, [x3]
        ADR_L   x3, tp_big_count
        str     x21, [x3]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #176
        ret

// uint64 tp_pin_and_probe(uint64 tid x0) -> 1 = prefetching kernel, 0 = A55 kernel
// Pins the calling thread to tp_cpu_list[tid % count] and reads MIDR_EL1
// (Linux emulates the read for EL0 and returns the current core's value).
tp_pin_and_probe:
        sub     sp, sp, #16
        ADR_L   x3, tp_cpu_count
        ldr     x4, [x3]
        cbz     x4, 1f
        udiv    x5, x0, x4
        msub    x5, x5, x4, x0
        ADR_L   x3, tp_cpu_list
        ldrb    w6, [x3, x5]
        mov     x7, #1
        lsl     x7, x7, x6
        str     x7, [sp]
        mov     x0, #0                  // this thread
        mov     x1, #8
        mov     x2, sp
        mov     x8, #SYS_sched_setaffinity
        svc     #0
1:      mrs     x0, midr_el1
        ubfx    x0, x0, #4, #12         // part number
        mov     x1, #MIDR_PART_A55
        cmp     x0, x1
        cset    x0, ne
        cbnz    x0, 2f
        ADR_L   x3, tp_n_little         // count A55 threads for the stats line
        mov     x4, #1
        ldadd   x4, x5, [x3]
2:      add     sp, sp, #16
        ret

// Worker: wait for a new generation, run chunks, report done, repeat.
worker_main:
        mov     x0, x19
        bl      tp_pin_and_probe
        mov     x23, x0                 // kernel variant of this core
        mov     w20, #0                 // last seen generation
        ADR_L   x21, tp_gen
        ADR_L   x22, tp_done
1:      ldaxr   w0, [x21]
        cmp     w0, w20
        b.ne    2f
        .ifdef NOWFE
        yield
        .else
        wfe
        .endif
        b       1b
2:      mov     w20, w0
        mov     x0, x23
        bl      tp_run_chunks
        mov     w1, #1
        ldaddal w1, w2, [x22]           // done += 1 (release our results)
        sev
        b       1b

// Claim and run chunks of the current job until none are left.
tp_run_chunks:
        stp     x29, x30, [sp, #-48]!
        stp     x19, x20, [sp, #16]
        str     x21, [sp, #32]
        mov     x21, x0                 // kernel variant
        ADR_L   x19, tp_job
        ADR_L   x20, tp_next
1:      mov     w1, #1
        ldadd   w1, w2, [x20]           // chunk index
        ldr     w3, [x19, #JOB_NCHUNKS]
        cmp     w2, w3
        b.hs    9f
        ldr     w7, [x19, #JOB_KIND]
        cmp     w7, #1
        b.eq    2f
        cmp     w7, #4
        b.eq    7f
        // matmul rows [c*CH, min((c+1)*CH, M))
        ADR_L   x5, tp_chunk_shift
        ldr     x5, [x5]
        lsl     x3, x2, x5              // r0 = chunk << shift
        mov     x4, #1
        lsl     x4, x4, x5
        add     x4, x3, x4
        ldr     x6, [x19, #JOB_M]
        cmp     x4, x6
        csel    x4, x4, x6, lo
        ldr     x0, [x19, #JOB_W]
        ldr     x1, [x19, #JOB_OUT]
        ldr     x2, [x19, #JOB_K]
        cmp     w7, #2
        b.eq    3f
        cmp     w7, #3
        b.eq    5f
        bl      matmul_q8_0_range
        b       1b
3:      cbz     x21, 4f
        bl      matmul_q8_0_4x4_pf
        b       1b
4:      bl      matmul_q8_0_4x4_nopf
        b       1b
5:      cbz     x21, 6f
        bl      matmul_q8_0_4x4b_pf
        b       1b
6:      bl      matmul_q8_0_4x4b_nopf
        b       1b
2:      mov     x0, x2                  // attention head
        bl      gqa_attn_head
        b       1b
7:      mov     x0, x2                  // two heads of one token of the batch
        bl      attn_pair_item
        b       1b
9:      ldr     x21, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #48
        ret

// Publish the job in tp_job, run chunks on this thread, wait for workers.
tp_dispatch:
        stp     x29, x30, [sp, #-32]!
        str     x19, [sp, #16]
        ADR_L   x0, tp_next
        str     wzr, [x0]
        ADR_L   x0, tp_done
        str     wzr, [x0]
        ADR_L   x0, tp_gen
        ldr     w1, [x0]
        add     w1, w1, #1
        stlr    w1, [x0]                // release job + counters, wakes waiters
        sev
        ADR_L   x0, tp_main_variant
        ldr     x0, [x0]
        bl      tp_run_chunks
        ADR_L   x1, tp_n_threads
        ldr     x19, [x1]
        sub     w19, w19, #1            // workers to wait for
        ADR_L   x1, tp_done
1:      ldaxr   w0, [x1]
        cmp     w0, w19
        b.eq    2f
        .ifdef NOWFE
        yield
        .else
        wfe
        .endif
        b       1b
2:      ldr     x19, [sp, #16]
        ldp     x29, x30, [sp], #32
        ret

// void tp_matmul(const void *W x0, const float *x x1, float *out x2, uint64 M x3, uint64 K x4)
tp_matmul:
        stp     x29, x30, [sp, #-48]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        mov     x19, x0
        mov     x20, x2
        mov     x21, x3
        mov     x22, x4
        mov     x0, x1
        mov     x1, x4
        bl      quantize_x_q8_0         // once, shared by all threads
        ADR_L   x0, tp_n_threads
        ldr     x0, [x0]
        cmp     x0, #1
        b.gt    1f
        mov     x0, x19                 // single thread: whole matrix
        mov     x1, x20
        mov     x2, x22
        mov     x3, #0
        mov     x4, x21
        ADR_L   x5, g_mm_delta
        ldr     x5, [x5]
        cbz     x5, 3f
        add     x0, x0, x5              // repacked copy of the same matrix
        ADR_L   x6, tp_main_variant
        ldr     x6, [x6]
        cbz     x6, 5f
        bl      matmul_q8_0_4x4_pf
        b       2f
5:      bl      matmul_q8_0_4x4_nopf
        b       2f
3:      bl      matmul_q8_0_range
        b       2f
1:      ADR_L   x0, tp_job
        ADR_L   x5, g_mm_delta
        ldr     x5, [x5]
        mov     w1, #0
        cbz     x5, 4f
        add     x19, x19, x5
        mov     w1, #2                  // 4x4 kernel on the repacked weights
4:      str     w1, [x0, #JOB_KIND]
        str     x19, [x0, #JOB_W]
        str     x20, [x0, #JOB_OUT]
        str     x22, [x0, #JOB_K]
        str     x21, [x0, #JOB_M]
        ADR_L   x5, tp_chunk_shift
        ldr     x5, [x5]
        mov     x1, #1
        lsl     x1, x1, x5
        sub     x1, x1, #1
        add     x1, x21, x1
        lsr     x1, x1, x5              // chunks = ceil(M / rows)
        str     w1, [x0, #JOB_NCHUNKS]
        bl      tp_dispatch
2:      ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #48
        ret

// void tp_matmul_batch(const void *W x0, const float *X x1, float *OUT x2,
//                      uint64 M x3, uint64 K x4, uint64 B x5)
// Batched prefill GEMM on the repacked weights; X token stride in g_mm_xstride.
tp_matmul_batch:
        stp     x29, x30, [sp, #-64]!
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        str     x23, [sp, #48]
        mov     x19, x0
        mov     x20, x2
        mov     x21, x3
        mov     x22, x4
        mov     x23, x5
        mov     x0, x1
        mov     x1, x4
        mov     x2, x5
        ADR_L   x3, g_mm_xstride
        ldr     x3, [x3]
        bl      quantize_x_batch
        ADR_L   x0, g_mm_b
        str     x23, [x0]
        lsl     x1, x21, #2
        ADR_L   x0, g_mm_m4
        str     x1, [x0]
        ADR_L   x5, g_mm_delta
        ldr     x5, [x5]
        add     x19, x19, x5            // repacked copy of the matrix
        ADR_L   x0, tp_n_threads
        ldr     x0, [x0]
        cmp     x0, #1
        b.gt    1f
        mov     x0, x19
        mov     x1, x20
        mov     x2, x22
        mov     x3, #0
        mov     x4, x21
        ADR_L   x6, tp_main_variant
        ldr     x6, [x6]
        cbz     x6, 3f
        bl      matmul_q8_0_4x4b_pf
        b       2f
3:      bl      matmul_q8_0_4x4b_nopf
        b       2f
1:      ADR_L   x0, tp_job
        mov     w1, #3
        str     w1, [x0, #JOB_KIND]
        str     x19, [x0, #JOB_W]
        str     x20, [x0, #JOB_OUT]
        str     x22, [x0, #JOB_K]
        str     x21, [x0, #JOB_M]
        ADR_L   x5, tp_chunk_shift
        ldr     x5, [x5]
        mov     x1, #1
        lsl     x1, x1, x5
        sub     x1, x1, #1
        add     x1, x21, x1
        lsr     x1, x1, x5
        str     w1, [x0, #JOB_NCHUNKS]
        bl      tp_dispatch
2:      ldr     x23, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #64
        ret

// void tp_attn(uint64 n_heads x0) — attention params are in tp_kv_base/tp_pos/tp_scale/tp_gqa
tp_attn:
        stp     x29, x30, [sp, #-32]!
        str     x19, [sp, #16]
        mov     x19, x0
        ADR_L   x0, tp_n_threads
        ldr     x0, [x0]
        cmp     x0, #1
        b.gt    2f
        str     x20, [sp, #24]
        mov     x20, #0
1:      cmp     x20, x19
        b.hs    3f
        mov     x0, x20
        bl      gqa_attn_head
        add     x20, x20, #1
        b       1b
3:      ldr     x20, [sp, #24]
        b       4f
2:      ADR_L   x0, tp_job
        mov     w1, #1
        str     w1, [x0, #JOB_KIND]
        str     w19, [x0, #JOB_NCHUNKS]
        bl      tp_dispatch
4:      ldr     x19, [sp, #16]
        ldp     x29, x30, [sp], #32
        ret

// void tp_attn_batch(uint64 n_items x0) — batched prefill attention:
// attn_pair_item(i) for every i < n_items, claimed dynamically by the pool.
tp_attn_batch:
        stp     x29, x30, [sp, #-32]!
        stp     x19, x20, [sp, #16]
        mov     x19, x0
        ADR_L   x0, tp_n_threads
        ldr     x0, [x0]
        cmp     x0, #1
        b.gt    2f
        mov     x20, #0
1:      cmp     x20, x19
        b.hs    3f
        mov     x0, x20
        bl      attn_pair_item
        add     x20, x20, #1
        b       1b
2:      ADR_L   x0, tp_job
        mov     w1, #4
        str     w1, [x0, #JOB_KIND]
        str     w19, [x0, #JOB_NCHUNKS]
        bl      tp_dispatch
3:      ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #32
        ret

        .bss
        .balign 64
tp_n_threads: .skip 8
        .balign 64
tp_gen:     .skip 4
        .balign 64
tp_done:    .skip 4
        .balign 64
tp_next:    .skip 4
        .balign 64
tp_job:     .skip 64
tp_kv_base: .skip 8
tp_pos:     .skip 8
tp_scale:   .skip 4
tp_gqa:     .skip 4
tp_ab_b:    .skip 8                     // tokens in the batch (batched attention)
        .balign 8
g_mm_delta: .skip 8                     // repacked_base - mmap_base (0 = row kernel)
tp_main_variant: .skip 8
tp_cpu_count: .skip 8
tp_big_count: .skip 8
tp_chunk_shift: .skip 8                 // log2(rows per chunk); 0 = choose in tp_init
tp_n_little: .skip 8
tp_cpu_list: .skip 64
tp_little_tmp: .skip 64
