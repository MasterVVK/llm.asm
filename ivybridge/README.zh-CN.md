# llm.asm 的 Ivy Bridge 移植版（AVX + F16C，无需 AVX2/FMA）

[English](README.md) · [Русский](README.ru.md) · **简体中文**

本项目将 [alesha-pro/llm.asm](https://github.com/alesha-pro/llm.asm)（上游提交 `48feccb`）移植到不支持 AVX2 和 FMA 的 x86-64 处理器上。它只需要 AVX 和 F16C，因此可以在 Intel Ivy Bridge（2012 年）及更新的处理器上运行，包括至今仍常见于家庭实验室和 Proxmox 集群的至强 E5 v2 服务器。上游版本要求 AVX2 + FMA（Haswell 或更新），在这些机器上会以 `Illegal instruction`（Linux）或 `[FATAL] CPU lacks AVX2+FMA`（裸机操作系统）退出。

上游的两个程序都已移植：Linux 推理引擎（`build/asm-llm`，21,128 字节）和可引导的裸机操作系统（`build/asm-llm-os.elf`，25,144 字节）。

## 改动内容

- `shared/matmul.asm`：INT8×INT8 内核（`FAST_Q8=1`）改用 128 位整数 SIMD 指令（`vpsignb`、`vpmaddubsw`、`vpmaddwd`）。每个 32 字节的块拆成两个 16 字节的半块处理，两半的 int32 和先精确相加，再一次性转换为浮点数。激活向量的 |x| 在每次矩阵乘法中只计算一次，而不是每一行权重都计算一次。FMA 改为 `vmulps` + `vaddps`。int32→int8 打包和嵌入反量化都在 128 位半块上完成。
- `shared/math.asm`：FMA 改为乘法加加法（宏 `MADD213SD` / `MADD213PS`）；`expf` 中的 2^k 在 128 位半块上构造；`argmax` 以浮点数保存索引；以寄存器为源操作数的 `vbroadcastss` 由宏 `BCAST_SS` 代替。
- `02-baremetal-os/boot.asm`、`kernel.asm`：CPUID 检查改为要求 XSAVE + AVX + F16C，并更新了启动横幅和错误信息。
- `FAST_Q8=0`（FP32 激活）未移植，构建时会以 `%error` 停止。
- 已用反汇编器检查：两个二进制文件中没有任何 AVX2 或 FMA 指令，也没有传统（非 VEX 编码）SSE 指令，因此不存在 AVX–SSE 切换惩罚。

## 正确性

- 在上游基准提示词上，用 `qemu-x86_64 -cpu max` 模拟运行的上游 AVX2 版本与本移植版**生成的 64 个贪心解码 token 完全一致（64/64）**，输出文本逐字节相同。预填充后 logits 的余弦相似度为 0.99970：不使用 FMA 时舍入方式不同，少数激活的量化结果会发生变化。上游自身相对其 FP32 参考实现为 0.99965。见 `results/parity-*`。
- 上游单元测试（`python3 scripts/test_math.py build/test_math`）全部通过，各项误差均不高于上游 README 中的数值。
- `ivybridge/test_matmul_k.py` 在模型实际尺寸下测试每次处理 4 个块的主循环；上游测试只用 K = 64，只覆盖尾部分支。在 K = 1024 / 2048 / 3072、FP32 与 FP16 两种缩放因子下，相对精确 Q8_0×Q8_0 参考结果的最大相对误差为 4e-8…1.4e-7。激活量化本身带来约 1e-2 的误差。
- 1 线程与 4 线程的输出逐位一致。

## 测试结果

硬件：一台 Proxmox VE 8.4 主机，2 颗至强 E5-2696 v2（Ivy Bridge-EP，每颗 12 核），内存为 DDR3-1600。Linux 数据来自该主机上一台 32 vCPU 的 Ubuntu 22.04 KVM 虚拟机。模型：Qwen3-0.6B Q8_0（asm-llm 使用 `.asmbin` v2，llama.cpp 使用官方 `Qwen3-0.6B-Q8_0.gguf`）。与上游表格相同：提示词 34 个 token，生成 64 个 token。llama.cpp：官方发布版 b12810（`ca123ba76`，CPU 后端 `ivybridge`），`llama-bench -ngl 0`，flash attention 为 `auto`（即开启，比关闭时更快）。数值范围来自两轮测试，每轮 3 次（`results/linux-vm-run*.csv`），单位为 token/秒：

| 线程数 | 生成：本移植版 | 生成：llama.cpp | 提升 | 预填充：本移植版 | 预填充：llama.cpp |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.9 | 7.6–8.1 | +36…44% | 14.5 | 20.9–21.9 |
| 4 | 33.7–33.9 | 25.7–26.0 | +31% | 44.3–44.6 | 72–73 |
| 8 | 47.7–49.9 | 39.1–40.0 | +22…25% | 61.5–62.1 | 143–146 |
| 16 | 51.4–54.4 | 39.9–40.9 | +29…33% | 65–69 | 200–223 |
| 24 | 49.5 | 32.2 | +54% | 63.1 | 249 |

此表中的预填充为逐个 token 处理，即移植版最初的测量结果；当前数据见[“批量预填充”](#批量预填充)一节。

同一主机上无磁盘 Proxmox 虚拟机中的裸机操作系统（`results/proxmox-os-vm250.txt`）：内核启动耗时 170–180 毫秒，虚拟机启动 8 秒后出现提示符。在同一提示词下，其生成速度比 Linux 版本快：1 核快 24%，4 核快 12%，8 核快 4%，12 核快 5%；12 核时达到 55.4 token/秒。

## 批量预填充

提示词按每批最多 16 个 token 处理（`--batch n`，默认 16；`--batch 1` 为逐个 token）。每批使用“1 行 × 2 个 token”的寄存器分块内核：两个 token 共用每个 32 字节块的 |w|，每个 token 计算 x·sign(w)，因此 `vpmaddubsw(|w|, x·sign(w))` 得到的 int16 和与单 token 内核完全相同。浮点部分在 256 位寄存器中计算，每个 128 位半部分对应一个 token，所用的 IEEE 运算与单 token 内核相同。无论批大小和线程数如何，KV 缓存和 logits 都与逐 token 预填充逐位一致。线程通过原子计数器领取每块 16 行的任务，而不是使用预先划分的固定区间；当 vCPU 共用物理核心时，这样更快。

单独测试内核，数据位于 L1（`ivybridge/kbench.asm`）：每行、每个 token、每个 32 值块需要 6.0–6.4 个周期，之前为 8.4–8.9。

每批的注意力在每层只作为一个线程池任务派发，而不是每个 token 派发一次：每个任务块是同一 token 的两个 query 头，它们读取同一个 kv 头，因此每次加载 K 和 V 都同时服务两个头。分数计算保留单 token 点积的四个累加器和归约顺序；两个头的每个 32 值输出块在所有位置上都保留在寄存器中，而不是每个位置都从内存读写一次输出。同一对头的任务块连续排列，因此其 K 和 V 行在 token 之间一直留在核心的 L2 中。logits 仍然逐位一致。

在矩阵乘法批次之间，残差加法、RMSNorm 和激活量化作为线程池任务执行，每个任务块处理一个 token，注意力任务则直接量化自己的输出。QK 归一化、RoPE、写入 KV 缓存和 SwiGLU 直接在矩阵乘法批次内部执行，处理的是核心刚刚算出的行：Q/K/V 的任务块是半个头，完成某个头第二半的任务块负责执行该头的这些步骤；gate/up 的任务块是 32 行 gate 加上对应的 32 行 up，随后执行 SwiGLU，并把这 32 个值的 Q8_0 块写入第二个激活缓冲区，因为其他任务块仍在读取第一个缓冲区。每个步骤执行的都是单 token 代码，因此 logits 仍然逐位一致。在投影的任务块内部做残差加法的方案更慢，未被采用（`results/prefill-fused.txt`）。

预填充速度，Qwen3-0.6B，token/秒（各 5 次；llama.cpp b12810，`llama-bench -p 34,227`，3 次重复的均值 ± 标准差；`results/prefill-tiles.txt`、`results/prefill-attention.txt`、`results/prefill-steps.txt`、`results/prefill-fused.txt`）：

| 线程数 | 34 个 token：本移植版 | 34 个 token：llama.cpp | 227 个 token：本移植版 | 227 个 token：llama.cpp |
|---:|---:|---:|---:|---:|
| 1 | 26.5–27.5 | 21.6 | 28.2–29.7 | 21.0 |
| 4 | 97–101 | 75 | 99–104 | 75.5 |
| 8 | 177–185 | 124 ± 19 | 188–195 | 145 |
| 16 | 230–252 | 248 ± 19 | 302–318 | 240 |

与上一版交替运行对比（5 次的中位数），把这些步骤放进矩阵乘法批次内部后，34 个 token 的提示词在 4–16 线程提升 5–22%（16 线程从 205 提高到 249 token/秒），227 个 token 的提示词提升 0.5–5%（16 线程从 301 提高到 316）；单线程以及 Qwen3-1.7B / 4B 的差异在 ±1.3% 以内。在此之前，把逐 token 步骤交给线程池在 8–16 线程提升了 5–16%，批量注意力在 227 个 token 的提示词上提升了 3–8%。第一版批量预填充（每次处理一行和一个 token，线程使用固定区间，每批 8 个 token）在 227 个 token 的提示词上，1 / 4 / 8 / 16 线程分别为 20 / 70 / 125–130 / 141–183 token/秒。

短提示词在 16 线程下，本移植版现在与 llama.cpp 持平（249 对 248 ± 19 token/秒）。

## 更大的 Qwen3 模型

同一个引擎可以运行 Qwen3-1.7B 和 Qwen3-4B；`scripts/export_weights.py` 现在支持分片存储的模型。同一台主机上的生成速度（token/秒），3 次测量取中位数，llama.cpp b12810 使用官方 Q8_0 GGUF：

| 模型 | 线程数 | 本移植版 | llama.cpp | 提升 |
|---|---:|---:|---:|---:|
| 1.7B | 1 | 4.7 | 2.8 | +69% |
| 1.7B | 4 | 13.6 | 10.0 | +36% |
| 1.7B | 8 | 19.0 | 15.7 | +21% |
| 1.7B | 16 | 19.0 | 17.0 | +12% |
| 4B | 4 | 5.7 | 4.6 | +24% |
| 4B | 8 | 8.4 | 7.3 | +16% |
| 4B | 16 | 8.8 | 7.8 | +13% |

227 个 token 提示词的批量预填充速度，token/秒（本移植版取 3 次的中位数；llama.cpp 为 `llama-bench -p 227` 3 次的均值；`results/prefill-tiles.txt`、`results/prefill-attention.txt`、`results/prefill-steps.txt`、`results/prefill-fused.txt`）：

| 模型 | 线程数 | 本移植版 | llama.cpp | 提升 |
|---|---:|---:|---:|---:|
| 1.7B | 4 | 36.3 | 26.0 | +40% |
| 1.7B | 8 | 69.8 | 48.8 | +43% |
| 1.7B | 16 | 124.1 | 71.5 | +74% |
| 4B | 4 | 14.1 | 10.2 | +38% |
| 4B | 8 | 28.0 | 19.7 | +42% |
| 4B | 16 | 50.9 | 33.1 | +54% |

`--hugepages` 会把权重复制到 2 MiB 透明大页中：1.7B 和 4B 在 4 线程时提升 8%，8–16 线程时没有提升，因此默认关闭。

## 局限

- 预填充（见上文）在 227 个 token 的提示词上，1–16 线程均快于 llama.cpp（16 线程时快 32%）；在 34 个 token 的提示词上，16 线程时与 llama.cpp 持平。
- 仅支持 Qwen3 稠密模型（已测试 0.6B、1.7B 和 4B）：模型结构写死在代码中，上下文最多 2048 个 token，不支持 GPU，也没有网络 API。
- **裸机操作系统即使空闲也会让所有 vCPU 保持 100% 占用**：工作核心在 `pause` 循环中等待，没有使用 `hlt`。在共享的虚拟化主机上，只在使用时运行它。
- 在 Linux 下，`-t` 不要超过空闲核心数。默认是 32 个线程，在繁忙的机器上吞吐量会下降一个数量级以上（上游 issue #9 和 #10）。
- FP16 块缩放因子需要 F16C，因此不支持 Sandy Bridge（至强 E5 v1）。

## 复现

```bash
make inference os test-math          # 需要 nasm 和 binutils
python3 scripts/test_math.py build/test_math
nasm -f elf64 -DFAST_Q8=1 -Ishared/ ivybridge/test_matmul_k.asm -o build/test_matmul_k.o
ld --gc-sections -s -z noexecstack build/test_matmul_k.o build/syscalls.o build/matmul.o -o build/test_matmul_k
python3 ivybridge/test_matmul_k.py build/test_matmul_k
nasm -f elf64 -DFAST_Q8=1 -Ishared/ ivybridge/kbench.asm -o build/kbench.o
ld --gc-sections -z noexecstack build/kbench.o build/matmul.o -o build/kbench
python3 ivybridge/kbench.py build/kbench   # 内核每行、每个 token、每个块的周期数
# 模型导出：见 01-qwen3-inference/README.md
ivybridge/bench.sh build/asm-llm models/qwen3-0.6b-q8_0.asmbin /path/to/llama-bench /path/to/Qwen3-0.6B-Q8_0.gguf "1 4 8 16" 3 out.csv
```

可引导 ISO：`make os-iso MODEL=models/qwen3-0.6b-q8_0.asmbin`（需要 `grub-pc-bin grub-efi-amd64-bin mtools xorriso`）。无磁盘、无网络的 Proxmox VE 虚拟机：

```bash
qm create 250 --name asm-llm-os --machine pc --bios seabios --cpu host --cores 8 \
  --memory 2048 --balloon 0 --vga std --serial0 socket \
  --sata0 local:iso/asm-llm-os.iso,media=cdrom --boot order=sata0 --ostype other --onboot 0
qm start 250 && python3 ivybridge/serial_bench.py 250 3; qm stop 250
```

请将 ISO 挂载为 `sata0`。按上游说明使用 `ide2` 时，GRUB 读取 614 MiB 的模型约需 64 秒；使用 `sata0` 约需 8 秒。`serial_bench.py` 在 Proxmox 主机上运行：它在虚拟机串口套接字上等待 `asm-llm>` 提示符，然后执行 `/bench`。

## 许可证

MIT，与上游相同。原始作品 © 2026 Alexey Fateev（[alesha-pro/llm.asm](https://github.com/alesha-pro/llm.asm)）。
