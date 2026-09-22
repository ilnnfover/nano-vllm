# 本地 LLM 推理引擎学习环境

目标：参考 vLLM，从零手写一个推理引擎，覆盖 KV Cache / 分页显存 / 前缀缓存 / 连续批处理 / CUDA Graph / Triton 算子。

## 一、本机现状（已实测）

| 项目 | 实测值 | 是否满足 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER | ✅ |
| 显存 | 16376 MiB（16 GB） | ✅ 够跑 1B–8B |
| 计算能力 | **8.9（Ada / sm_89）** | ✅ vLLM 要求 ≥ 7.5 |
| 驱动 | 610.74，CUDA UMD 13.3 | ✅ 非常新，cu126/129/130/132 通吃 |
| 内存 | 31.8 GB | ✅ 建议给 WSL 分配 16 GB |
| WSL2 | 已装 **Ubuntu-24.04** | ✅ 直接用 |
| Docker | 未安装 | 不需要 |
| 磁盘 | C:115 / D:91 / E:256 GB 可用 | ✅ |
| VS2022 | 未安装 | 不需要（走 WSL） |

## 二、为什么必须用 WSL2 而不是原生 Windows

| 组件 | Windows 原生 | WSL2 |
|---|---|---|
| vLLM | ❌ 官方不支持，只有社区 fork | ✅ 官方支持 |
| Triton | ⚠️ 需 `triton-windows` 第三方 fork（原仓库已 archived） | ✅ 官方 wheel，随 PyTorch 自动装 |
| flash-attn | ⚠️ 需手工编译，坑多 | ✅ `pip install` 可编译 |
| FlashInfer | ❌ 无 Windows 构建 | ✅ 有 |
| CUDA Graph | ⚠️ WDDM 下受限 | ✅ 正常 |

结论：**主力环境放 WSL2**，Windows 只负责驱动和编辑器。
vLLM 官方文档也明确写了：Windows 需使用 WSL。

## 三、目录与环境布局

```
Windows 宿主
└─ NVIDIA 驱动 610.74  ── 驱动透传 ──>  WSL2 Ubuntu-24.04
                                        ├─ ~/venvs/dev   自研引擎 (torch + triton)
                                        ├─ ~/venvs/vllm  vLLM 源码 (editable，只读参考)
                                        └─ ~/llm-engine-lab/
                                            ├─ vllm-src/   (git clone)
                                            └─ myengine/    (你自己的代码)
```

**两个环境必须分开。** vLLM 对 PyTorch 版本强绑定，而 `flashinfer-python` 安装时会把 torch 升上去、直接把 vLLM 的编译产物搞崩（vLLM 官方 issue 反复出现）。隔离是唯一省事的做法。

## 四、执行步骤

```bash
# 在 Windows PowerShell 里确认 WSL 正常
wsl -d Ubuntu-24.04

# 在 WSL 内
cp -r /mnt/c/Users/25184/WorkBuddy/2026-09-20-10-47-54/llm-engine-env ~/llm-engine-env
cd ~/llm-engine-env && bash setup_wsl.sh
```

脚本分 6 阶段，任何阶段失败都不会污染前面的成果，可单独重跑。

装完自检：

```bash
source ~/venvs/dev/bin/activate
python ~/llm-engine-env/scripts/verify_env.py
```

## 五、几个必踩的坑（提前避）

1. **WSL 里千万不要 `apt install cuda`**。那会装 Linux 版驱动，覆盖掉 WSL 的驱动透传机制。只装 `cuda-toolkit-12-9`，而且纯 Triton 开发根本不需要它——PyTorch/Triton 的 wheel 自带 ptxas。只有在你写 `.cu` 文件时才需要 nvcc。

2. **显存占用一定要手动压低**。WSL 里 `nvidia-smi` 看到的是整张 16 GB 卡，但 Windows 桌面侧已经吃掉约 2.6 GB。vLLM 默认 `gpu-memory-utilization=0.9` 会直接 OOM。实测建议：
   ```bash
   vllm serve Qwen/Qwen2.5-1.5B-Instruct \
     --gpu-memory-utilization 0.65 --max-model-len 4096
   ```

3. **代码放 WSL 原生 ext4**（`~/llm-engine-lab`），**不要放 `/mnt/c/...`**。跨 9p 文件系统做 pip install 和 git 操作会慢一个数量级。

4. **给 WSL 加内存限制**，否则默认策略下加载 7B 模型容易 OOM。在 Windows 侧 `C:\Users\25184\.wslconfig`：
   ```ini
   [wsl2]
   memory=16GB
   processors=8
   swap=8GB
   ```
   然后 `wsl --shutdown` 重启生效。

5. **conda 装的 PyTorch 静态链接 NCCL**，vLLM 用 NCCL 时会出问题（官方 issue #8420）。你机器上 `D:\Anaconda` 存在，别用它建环境，用 `uv`。

6. 国内下载模型加 `HF_ENDPOINT=https://hf-mirror.com`（脚本已写入 `~/.bashrc`）。

## 六、学习内容 → 环境组件对照

> **Python 版本冲突提醒**：如果你要用 `Wenyueh/MinivLLM` 自带的 `uv` 环境，它的 `pyproject.toml` 写死了 `requires-python = ">=3.11,<3.12"`，必须 Python **3.11**。本脚本默认建 **3.12** 的环境（vLLM 官方推荐 3.12）。
> 二选一：
> - 走 MinivLLM 路线 → 直接用它仓库的 `uv sync`，别用本脚本的 venv；
> - 走本脚本路线 → 改 `PYVER=3.11` 也能兼容 MinivLLM。
> 不要试图让一个环境同时满足两者。

| 学习主题 | 依赖的环境能力 | 建议动手顺序 |
|---|---|---|
| KV Cache | `transformers` 读权重 + 手写 attention 循环 | 先跑通 HF `model.generate()`，再自己写一遍 `for` 循环版，对比 KV 复用带来的加速 |
| 分页显存 (PagedAttention) | 大块显存预分配 + block table + **Triton kernel** | 把 KV cache 改成 `[num_blocks, block_size, n_kv_heads, head_dim]` 的池子，写 Triton kernel 按 block table 索引 |
| 前缀缓存 | block 内容哈希（脚本已装 `xxhash`）+ 引用计数 | 在 block manager 里加 `ref_cnt`，实现 copy-on-write |
| 连续批处理 | 纯 Python 调度器 + `asyncio` | 先做最简单的 FCFS + 每步重排 batch，再加 prefill/decode 分离 |
| CUDA Graph | `torch.cuda.CUDAGraph` + 固定 shape + 独立 stream 预热 | 只对 decode 阶段的固定 batch 做 capture；注意 WSL 下同样要先预热再 capture |
| 动态图 / torch.compile | `torch.compile(mode="reduce-overhead")` | 和手写 CUDA Graph 对比，理解 cudagraph trees 做了什么 |
| 算子级优化 | Triton + `torch.profiler` + Nsight | 用 profiler 找瓶颈 kernel，再写 Triton 版本打表对比 |

## 七、推荐的小模型（16 GB 显存）

- `Qwen/Qwen2.5-0.5B-Instruct` — 起步首选，秒级加载
- `Qwen/Qwen2.5-1.5B-Instruct` — 主力对照
- `meta-llama/Llama-3.2-3B-Instruct` — 需要 GQA，正好验证你的 KV head 复用逻辑写对没
- `Qwen/Qwen2.5-7B-Instruct-AWQ` — 压测极限，验证分页显存的价值

**一定要选一个带 GQA（num_key_value_heads < num_attention_heads）的模型**——现代模型基本都是 GQA，你的 KV cache 布局必须处理 head 数不匹配，这是最容易写错的地方。

## 八、学习路径建议

1. **先读 vLLM 的两个文件**：
   - `vllm-src/vllm/v1/core/kv_cache_manager.py`（块分配 + 前缀缓存）
   - `vllm-src/vllm/v1/core/sched/scheduler.py`（连续批处理调度）
   V1 引擎核心是 Python 写的，可读性远好于 V0。
2. **先写正确，再写快**：先用纯 PyTorch 实现一个正确但慢的版本，建立 baseline 和单元测试。
3. **再逐层替换**：KV 池化 → 分页 kernel → prefix 复用 → 连续批处理 → CUDA Graph。每一步都用 `scripts/verify_env.py` 里的 profiler 量一遍。
4. **对照 vLLM 的行为**：同样请求跑你自己的引擎和 vLLM，比 TTFT / TPOT / 显存占用，差异就是你要学的东西。
