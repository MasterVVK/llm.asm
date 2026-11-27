// ============================================================================
// sys.s — raw Linux syscalls and tiny formatting helpers (AArch64, zero libc)
// ============================================================================
        .include "constants.s"

        .text
        .global sys_write_all, sys_clock_ns, sys_exit_group, sys_open_ro
        .global sys_open_wr, sys_close, sys_fstat_size, sys_mmap_file
        .global sys_mmap_anon, sys_munmap, sys_madvise, sys_read_all
        .global str_len, str_eq, parse_u64, write_cstr, write_u64, write_f64_2dp

// void sys_write_all(int fd x0, const void *buf x1, uint64 len x2)
sys_write_all:
        mov     x9, x0
        mov     x10, x1
        mov     x11, x2
1:      cbz     x11, 3f
        mov     x0, x9
        mov     x1, x10
        mov     x2, x11
        SYSCALL SYS_write
        cmn     x0, #4                  // -EINTR -> retry
        b.eq    1b
        cmp     x0, #0
        b.le    3f
        add     x10, x10, x0
        sub     x11, x11, x0
        b       1b
3:      ret

// int64 sys_read_all(int fd x0, void *buf x1, uint64 len x2) -> bytes read
sys_read_all:
        mov     x9, x0
        mov     x10, x1
        mov     x11, x2
        mov     x12, #0
1:      cbz     x11, 3f
        mov     x0, x9
        mov     x1, x10
        mov     x2, x11
        SYSCALL SYS_read
        cmn     x0, #4
        b.eq    1b
        cmp     x0, #0
        b.le    3f
        add     x10, x10, x0
        sub     x11, x11, x0
        add     x12, x12, x0
        b       1b
3:      mov     x0, x12
        ret

// uint64 sys_clock_ns(void) — CLOCK_MONOTONIC in nanoseconds
sys_clock_ns:
        sub     sp, sp, #16
        mov     x0, #CLOCK_MONOTONIC
        mov     x1, sp
        SYSCALL SYS_clock_gettime
        ldp     x1, x2, [sp]
        ldr     x3, =1000000000
        madd    x0, x1, x3, x2
        add     sp, sp, #16
        ret

// noreturn sys_exit_group(int code x0)
sys_exit_group:
        SYSCALL SYS_exit_group
        b       sys_exit_group

// int64 sys_open_ro(const char *path x0) -> fd or -errno
sys_open_ro:
        mov     x1, x0
        mov     x0, #AT_FDCWD
        mov     x2, #O_RDONLY
        mov     x3, #0
        SYSCALL SYS_openat
        ret

// int64 sys_open_wr(const char *path x0) -> fd (O_WRONLY|O_CREAT|O_TRUNC, 0644)
sys_open_wr:
        mov     x1, x0
        mov     x0, #AT_FDCWD
        mov     x2, #O_WRONLY_CREAT_TR
        mov     x3, #420                // 0644
        SYSCALL SYS_openat
        ret

// void sys_close(int fd x0)
sys_close:
        SYSCALL SYS_close
        ret

// int64 sys_fstat_size(int fd x0) -> st_size or -errno
sys_fstat_size:
        sub     sp, sp, #STAT_SIZE
        mov     x1, sp
        SYSCALL SYS_fstat
        cmp     x0, #0
        b.lt    1f
        ldr     x0, [sp, #STAT_ST_SIZE]
1:      add     sp, sp, #STAT_SIZE
        ret

// void *sys_mmap_file(uint64 size x0, int fd x1) — PROT_READ, MAP_PRIVATE
sys_mmap_file:
        mov     x4, x1
        mov     x1, x0
        mov     x0, #0
        mov     x2, #PROT_READ
        mov     x3, #MAP_PRIVATE
        mov     x5, #0
        SYSCALL SYS_mmap
        ret

// void *sys_mmap_anon(uint64 size x0) — PROT_READ|WRITE, MAP_PRIVATE|ANON
sys_mmap_anon:
        mov     x1, x0
        mov     x0, #0
        mov     x2, #PROT_RW
        mov     x3, #MAP_PRIV_ANON
        mov     x4, #-1
        mov     x5, #0
        SYSCALL SYS_mmap
        ret

// void sys_munmap(void *p x0, uint64 len x1)
sys_munmap:
        SYSCALL SYS_munmap
        ret

// int64 sys_madvise(void *p x0, uint64 len x1, int advice x2)
sys_madvise:
        SYSCALL SYS_madvise
        ret

// uint64 str_len(const char *s x0)
str_len:
        mov     x1, x0
1:      ldrb    w2, [x1], #1
        cbnz    w2, 1b
        sub     x0, x1, x0
        sub     x0, x0, #1
        ret

// int str_eq(const char *a x0, const char *b x1) -> 1 if equal
str_eq:
1:      ldrb    w2, [x0], #1
        ldrb    w3, [x1], #1
        cmp     w2, w3
        b.ne    2f
        cbnz    w2, 1b
        mov     x0, #1
        ret
2:      mov     x0, #0
        ret

// uint64 parse_u64(const char *s x0)
parse_u64:
        mov     x1, #0
        mov     x3, #10
1:      ldrb    w2, [x0], #1
        sub     w2, w2, #'0'
        cmp     w2, #9
        b.hi    2f
        madd    x1, x1, x3, x2
        b       1b
2:      mov     x0, x1
        ret

// void write_cstr(int fd x0, const char *s x1)
write_cstr:
        stp     x29, x30, [sp, #-32]!
        stp     x0, x1, [sp, #16]
        mov     x0, x1
        bl      str_len
        mov     x2, x0
        ldp     x0, x1, [sp, #16]
        bl      sys_write_all
        ldp     x29, x30, [sp], #32
        ret

// void write_u64(int fd x0, uint64 v x1)
write_u64:
        stp     x29, x30, [sp, #-48]!
        add     x4, sp, #48             // end of 32-byte digit buffer at [sp+16, sp+48)
        mov     x5, x4
        mov     x6, #10
1:      udiv    x7, x1, x6
        msub    x3, x7, x6, x1
        add     w3, w3, #'0'
        strb    w3, [x5, #-1]!
        mov     x1, x7
        cbnz    x1, 1b
        mov     x1, x5
        sub     x2, x4, x5
        bl      sys_write_all
        ldp     x29, x30, [sp], #48
        ret

// void write_f64_2dp(int fd x0, double v d0) — v >= 0, printed as %.2f
write_f64_2dp:
        stp     x29, x30, [sp, #-48]!
        str     x19, [sp, #16]
        mov     x19, x0
        ADR_L   x1, d_hundred
        ldr     d1, [x1]
        fmul    d0, d0, d1
        fcvtns  x2, d0                  // round(v * 100)
        mov     x3, #100
        udiv    x4, x2, x3
        msub    x5, x4, x3, x2          // cents
        str     x5, [sp, #24]
        mov     x0, x19
        mov     x1, x4
        bl      write_u64
        ldr     x5, [sp, #24]
        mov     w6, #'.'
        strb    w6, [sp, #32]
        mov     x3, #10
        udiv    x4, x5, x3
        msub    x7, x4, x3, x5
        add     w4, w4, #'0'
        add     w7, w7, #'0'
        strb    w4, [sp, #33]
        strb    w7, [sp, #34]
        mov     x0, x19
        add     x1, sp, #32
        mov     x2, #3
        bl      sys_write_all
        ldr     x19, [sp, #16]
        ldp     x29, x30, [sp], #48
        ret

        .section .rodata
        .balign 8
d_hundred: .double 100.0
