#!/usr/bin/env bash
# ============================================================================
#  LLM 推理引擎学习环境 —— 单文件入口 (WSL2 / Ubuntu 24.04 内执行)
#  目标机器: RTX 4070 Ti SUPER 16GB (sm_89) + 32GB RAM + WSL2
#
#  用法:
#     bash setup_wsl.sh              完整搭建(幂等: 已完成的阶段自动跳过)
#     bash setup_wsl.sh --check      只体检, 不装任何东西(失败时打印 GPU 透传诊断)
#     bash setup_wsl.sh --fix        只修 CUDA 环境变量 + 清理降级开关(秒级)
#     bash setup_wsl.sh --verify     跑 verify_env.py 自检(加 RUN_SMOKE=1 再跑 vLLM 冒烟)
#     bash setup_wsl.sh --help
#
#  完整搭建的 7 个阶段:
#     0 体检(GPU 透传)  1 apt  2 uv  3 dev 环境  4 CUDA Toolkit
#     5 vLLM            6 算子库(flashinfer / flash-attn)  7 自检
#
#  设计原则: 建【两个互相隔离的虚拟环境】
#    ~/venvs/dev   <- 你自己写引擎的地方, PyTorch + Triton + 算子库
#    ~/venvs/vllm  <- 只读参考用, vLLM 官方 wheel, 用来读代码/跑对照实验
#    绝不能混: vLLM 对 torch 版本强绑定, flashinfer 安装时会偷偷升级 torch 把 vLLM 搞崩。
#
#  环境变量可调:
#     PYVER=3.12  VLLM_VERSION=0.29.0  SKIP_FLASH_ATTN=1  RUN_SMOKE=1
# ============================================================================
set -euo pipefail

# ------------------------------------------------------------------ 配置
LAB="$HOME/llm-engine-lab"     # 项目代码目录(放 WSL 原生 ext4, 不要放 /mnt/c)
VENVS="$HOME/venvs"
PYVER="${PYVER:-3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINT="$LAB/constraints.txt"

# 国内 HuggingFace 加速(网络通畅可改成 https://huggingface.co)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ------------------------------------------------------------------ 输出
log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
dim()  { printf '\033[2m    %s\033[0m\n' "$*"; }

# 用 heredoc 而不是 sed -n 抽注释: 后者依赖行号, 改一次文件就错位
usage() {
  cat <<'USAGE'
LLM 推理引擎学习环境 —— 单文件入口 (WSL2 / Ubuntu 24.04 内执行)

用法:
  bash setup_wsl.sh              完整搭建(幂等: 已完成的阶段自动跳过)
  bash setup_wsl.sh --check      只体检, 不装任何东西(失败时打印 GPU 透传诊断)
  bash setup_wsl.sh --fix        只修 CUDA 环境变量 + 清理降级开关(秒级)
  bash setup_wsl.sh --verify     跑 verify_env.py 自检(加 RUN_SMOKE=1 再跑 vLLM 冒烟)
  bash setup_wsl.sh --help

7 个阶段:
  0 体检(GPU 透传)  1 apt  2 uv  3 dev 环境  4 CUDA Toolkit
  5 vLLM            6 算子库(flashinfer / flash-attn)  7 自检

两个隔离环境:
  ~/venvs/dev   自己写引擎的地方 (PyTorch + Triton + 算子库)
  ~/venvs/vllm  vLLM 官方 wheel, 用来读代码/跑对照实验

可调环境变量:
  PYVER=3.12  VLLM_VERSION=0.29.0  SKIP_FLASH_ATTN=1  RUN_SMOKE=1  MAX_JOBS=4
USAGE
}

# ------------------------------------------------------------------ 模式
MODE="all"
case "${1:-}" in
  ""|all)        MODE="all" ;;
  --check|check) MODE="check" ;;
  --fix|fix)     MODE="fix" ;;
  --verify|verify) MODE="verify" ;;
  -h|--help)     usage; exit 0 ;;
  *)
    printf '未知参数: %s\n\n' "$1"
    usage
    exit 1
    ;;
esac

# ================================================================== 工具函数

# 找 nvcc。要覆盖 PATH / CUDA_HOME / CUDA_PATH / 符号链接 / 版本目录五种情况,
# 只查 `command -v` 会在"装了但没配 PATH"时误判, 导致重复 apt(慢)。
find_nvcc() {
  NVCC_BIN=""
  local c
  for c in "$(command -v nvcc 2>/dev/null || true)" \
           "${CUDA_HOME:-}/bin/nvcc" "${CUDA_PATH:-}/bin/nvcc" /usr/local/cuda/bin/nvcc; do
    if [[ -n "$c" && -x "$c" ]]; then NVCC_BIN="$c"; return 0; fi
  done
  local d
  for d in /usr/local/cuda-*/bin/nvcc; do
    [[ -x "$d" ]] && { NVCC_BIN="$d"; return 0; }
  done
  return 1
}

cuda_root_of() { # $1 = nvcc 完整路径 -> /usr/local/cuda-13.3
  local d1 d2
  d1="$(dirname "$1")"; d2="$(dirname "$d1")"
  printf '%s\n' "$d2"
}

# ---------------------------------------------------------------------------
# 同步 ~/venvs/vllm/bin/activate 里由本脚本管理的那段配置
# ---------------------------------------------------------------------------
# 为什么要有这个函数:
#   脚本从"降级版"演进到"完整版"时, 上一版写进去的
#       export VLLM_USE_V2_MODEL_RUNNER=0
#       export VLLM_USE_FLASHINFER_SAMPLER=0
#   不会因为重跑而消失(旧代码只做"不存在才追加"的检查)。
#   而 --fix / --check 这些子命令根本不走阶段 5, 于是完整配置所需的
#       export VLLM_WSL2_ENABLE_PIN_MEMORY=1
#   又永远不会被写入。两边一叠加: 开关删了、能力没开 -> V2 runner 照样崩在 UVA。
#
# 做法: 用一对唯一标记界定"本脚本管理的区域", 每次全量重写这段区域,
#       再兜底删掉散落在文件里的旧版裸 export。这样无论跑哪条路径,
#       activate 的最终状态都是收敛的、可预测的。
ACT_MARK_BEGIN="# >>> setup_wsl.sh managed block"
ACT_MARK_END="# <<< setup_wsl.sh managed block"

sync_vllm_activate() {
  local act="$VENVS/vllm/bin/activate"
  [[ -f "$act" ]] || return 0

  # 1) 删掉上一次写入的整个管理块(含块内注释)。
  #    必须两个标记都在才用范围删除 —— 否则(写入被中断、只剩 BEGIN)范围删除
  #    会一路删到文件末尾, 把用户自己的内容全带走。退化成逐行删标记更安全。
  if grep -qxF "$ACT_MARK_BEGIN" "$act" && grep -qxF "$ACT_MARK_END" "$act"; then
    sed -i -E "/^${ACT_MARK_BEGIN}$/,/^${ACT_MARK_END}$/d" "$act"
  else
    sed -i -E "/^${ACT_MARK_BEGIN}$/d; /^${ACT_MARK_END}$/d" "$act"
  fi
  # 2) 兜底: 删掉更早版本散落的裸 export 与注释行
  sed -i -E '/^export VLLM_(USE_V2_MODEL_RUNNER|USE_FLASHINFER_SAMPLER|WSL2_ENABLE_PIN_MEMORY)=/d; /^# --- setup_wsl\.sh/d' "$act"

  # 3) 按当前实际能力重写。
  #    注意: 空行必须放在标记【内部】。放外面的话它不在删除范围内,
  #    每跑一次就多一个空行 —— 幂等性会被悄悄破坏(实测踩到)。
  {
    echo "$ACT_MARK_BEGIN"
    echo ''
    echo '# WSL2 完整配置: V2 model runner(默认)依赖 UVA, UVA 依赖 pinned host memory。'
    echo '# vLLM 在 WSL2 上默认关闭 pin memory -> 显式打开, 保住 V2 runner + FlashInfer。'
    echo '# 这行是【启用能力】不是降级。若确实不稳定, 再手动: export VLLM_USE_V2_MODEL_RUNNER=0'
    echo 'export VLLM_WSL2_ENABLE_PIN_MEMORY=1'
    if ! find_nvcc; then
      echo '# 未检测到 nvcc: FlashInfer 无法 JIT, 临时退回 PyTorch 原生采样。'
      echo '# 装好 CUDA Toolkit 后跑 bash setup_wsl.sh --fix 会自动删掉本行。'
      echo 'export VLLM_USE_FLASHINFER_SAMPLER=0'
    fi
    echo "$ACT_MARK_END"
  } >> "$act"

  # 4) 当前 shell 里也同步, 免得本次会话还残留旧值
  unset VLLM_USE_V2_MODEL_RUNNER VLLM_USE_FLASHINFER_SAMPLER || true
  export VLLM_WSL2_ENABLE_PIN_MEMORY=1
  if ! find_nvcc; then export VLLM_USE_FLASHINFER_SAMPLER=0; fi
}

# 真编译并跑一个 kernel —— 只查 nvcc --version 是不够的, 得确认整条链通
cuda_smoke_test() {
  local nv="$1" td
  td="$(mktemp -d)"
  cat > "$td/hello.cu" <<'CU'
#include <cstdio>
__global__ void hello() { printf("kernel running on GPU\n"); }
int main() {
  hello<<<1, 1>>>();
  cudaError_t e = cudaDeviceSynchronize();
  printf("cudaDeviceSynchronize -> %s\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}
CU
  # -cudart static: 静态链入 CUDA 运行时, 运行产物不需要 LD_LIBRARY_PATH。
  # (我们刻意不把系统 CUDA 的 lib64 加进 LD_LIBRARY_PATH —— 会盖掉 torch 自带的运行时。)
  if "$nv" -arch=sm_89 -cudart static -o "$td/hello" "$td/hello.cu" >/dev/null && "$td/hello"; then
    ok "nvcc 编译 + GPU 运行通过 (sm_89)"
  else
    warn "nvcc 编译或运行失败 —— 看上面的输出"
  fi
  rm -rf "$td"
}

# GPU 透传诊断 —— 阶段 0 失败时打全, 便于一次定位断在哪一层
dump_gpu_diag() {
  echo
  echo "─── 1. 内核"
  uname -r
  grep -qi microsoft /proc/version && echo "WSL ✓" || echo "不是 WSL"
  echo
  echo "─── 2. 宿主驱动注入的 CUDA 文件 (/usr/lib/wsl/lib)"
  ls -la /usr/lib/wsl/lib 2>/dev/null | head -20 || echo "  ✗ 目录不存在 —— 宿主机驱动没注入"
  echo
  echo "─── 3. PATH 是否含 /usr/lib/wsl/lib"
  case ":$PATH:" in *":/usr/lib/wsl/lib:"*) echo "  ✓ 在 PATH 中" ;; *) echo "  ✗ 不在 PATH 中" ;; esac
  echo
  echo "─── 4. 绝对路径调用 nvidia-smi"
  /usr/lib/wsl/lib/nvidia-smi 2>&1 | head -12 || echo "  调用失败"
  echo
  echo "─── 5. libcuda 动态库"
  ldconfig -p 2>/dev/null | grep -i 'libcuda\|libnvidia-ml' || echo "  ldconfig 里没有"
  echo
  echo "─── 6. GPU 设备节点"
  ls -la /dev/dxg 2>/dev/null || echo "  ✗ 没有 /dev/dxg"
  # WSL2 走 dxgkrnl 通道, 没有 /dev/nvidia* 是正常的, 不要当成故障
  ls -la /dev/nvidia* 2>/dev/null || echo "  (没有 /dev/nvidia* —— WSL2 下正常, 不影响 CUDA)"
  echo
}

# ================================================================== 阶段 0
stage0_check() {
  log "阶段 0/7  检查运行环境"

  local kern; kern="$(uname -r)"
  if ! grep -qi microsoft /proc/version 2>/dev/null; then
    warn "没检测到 WSL 环境。本脚本必须在 WSL2 内运行。按 Ctrl-C 退出或回车继续。"
    read -r _
  fi

  # WSL1 内核是 4.4.0-Microsoft, 且没有 GPU 访问能力
  case "$kern" in
    *WSL2*|5.*|6.*) ok "内核 $kern (WSL2)" ;;
    *)
      warn "内核 $kern — 疑似 WSL1, WSL1 没有 GPU 访问能力。"
      echo "  Windows PowerShell 里执行:  wsl -l -v"
      echo "  VERSION 列为 1 则转换:      wsl --set-version Ubuntu-24.04 2"
      exit 1
      ;;
  esac

  # GPU 没打通必须 fail fast: uv --torch-backend=auto 探测不到 GPU 会把 torch
  # 装成 CPU 版, 几个 GB 下完才发现等于全白装。
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv 2>/dev/null
  elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    ok "nvidia-smi 不在 PATH, 但在 /usr/lib/wsl/lib — 自动补上"
    export PATH="/usr/lib/wsl/lib:$PATH"
    grep -q '/usr/lib/wsl/lib' "$HOME/.bashrc" 2>/dev/null || \
      echo 'export PATH="/usr/lib/wsl/lib:$PATH"' >> "$HOME/.bashrc"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
  else
    warn "找不到 nvidia-smi, /usr/lib/wsl/lib 也不存在 —— 宿主驱动没把 CUDA 注入 WSL。"
    cat <<'FIX'
  在 GPU 可用之前不能继续装。按顺序排查(前三条在 Windows PowerShell 里执行):

    1) 确认发行版是 WSL2 (VERSION 列必须是 2):
         wsl -l -v
       若为 1:  wsl --set-version Ubuntu-24.04 2

    2) 升级 WSL 内核并冷重启:
         wsl --update
         wsl --shutdown
       然后重新打开 Ubuntu-24.04 再跑本脚本

    3) 确认 Windows 宿主侧 GPU 正常: PowerShell 里跑 nvidia-smi 应有输出

    4) 仍不行, 跑 bash setup_wsl.sh --check 看完整诊断
FIX
    exit 1
  fi
  ok "GPU 可用"
}

# ================================================================== 阶段 1
stage1_apt() {
  log "阶段 1/7  安装系统依赖"
  local pkgs=(build-essential ca-certificates curl wget git git-lfs python3-dev \
              pkg-config ccache cmake ninja-build htop tmux nvtop bc)
  local missing=()
  local p
  for p in "${pkgs[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    ok "系统依赖已齐, 跳过"
  else
    dim "待装: ${missing[*]}"
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends "${missing[@]}"
    ok "系统依赖装完"
  fi
  git lfs install --skip-repo >/dev/null 2>&1 || true
}

# ================================================================== 阶段 2
stage2_uv() {
  log "阶段 2/7  安装 uv (vLLM 官方推荐的包管理器)"
  if command -v uv >/dev/null 2>&1; then
    ok "uv 已装: $(uv --version)"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ok "uv 装完"
  fi
  export PATH="$HOME/.local/bin:$PATH"
  grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null || \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  grep -q 'HF_ENDPOINT' "$HOME/.bashrc" 2>/dev/null || \
    echo "export HF_ENDPOINT=$HF_ENDPOINT" >> "$HOME/.bashrc"
  mkdir -p "$LAB" "$VENVS"
}

# ================================================================== 阶段 3
stage3_dev() {
  log "阶段 3/7  自研引擎开发环境 ~/venvs/dev"
  if [[ -x "$VENVS/dev/bin/python" ]] &&
     "$VENVS/dev/bin/python" -c "import torch, triton, transformers" >/dev/null 2>&1; then
    ok "dev 环境已就绪, 跳过"
  else
    uv venv "$VENVS/dev" --python "$PYVER" --seed
    # shellcheck disable=SC1091
    source "$VENVS/dev/bin/activate"

    # --torch-backend=auto: 让 uv 探测本机驱动挑 CUDA 版本, 避免硬编码成过时值
    uv pip install --torch-backend=auto torch torchvision torchaudio

    # 把刚装好的 torch 版本写成约束文件。后续所有安装都带 -c,
    # 既保住依赖树完整, 又不会被 flashinfer 之流顺手把 torch 升到不兼容版本。
    python - <<'PY'
import torch, torchvision, pathlib
p = pathlib.Path.home() / "llm-engine-lab" / "constraints.txt"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(f"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n")
print("已写入约束:", p)
PY
    uv pip install -c "$CONSTRAINT" \
      "triton" \
      "transformers>=4.56" "datasets" "accelerate" "tokenizers" "safetensors" \
      "numpy" "einops" "sentencepiece" "tiktoken" \
      "pytest" "ipython" "jupyter" "matplotlib" "pandas" "rich" \
      "nvidia-ml-py" "py-cpuinfo" "psutil" \
      "xxhash" "zstandard"
    ok "dev 环境装完"
  fi
  # 跳过安装时约束文件可能缺失(比如上次只装了 torch), 这里兜底补生成
  if [[ ! -f "$CONSTRAINT" ]]; then
    # shellcheck disable=SC1091
    source "$VENVS/dev/bin/activate"
    uv pip freeze 2>/dev/null | grep -E '^torch==|^torchvision==' > "$CONSTRAINT" || true
  fi
  "$VENVS/dev/bin/python" - <<'PY'
import torch, triton
print("  torch", torch.__version__, "| cuda", torch.version.cuda)
print("  triton", triton.__version__, "| device", torch.cuda.get_device_name(0),
      "| sm", torch.cuda.get_device_capability(0))
PY
}

# ================================================================== 阶段 4
stage4_cuda() {
  log "阶段 4/7  CUDA Toolkit"

  # 为什么必须装(不是可选项):
  #   1. FlashInfer 是 JIT 编译型库, vLLM 默认拿它做 top-k/top-p 采样,
  #      首次调用现场生成 CUDA C++ 再调 nvcc 编译。没 nvcc 就崩在 profile_run。
  #   2. flash-attn / 自定义 .cu / torch.utils.cpp_extension 都需要 nvcc。
  #   3. 只有"纯 Triton 开发"不需要(wheel 自带 ptxas)。
  #
  # ⚠ WSL 里【绝对不要】apt install cuda 或 cuda-drivers —— 那是 Linux 内核驱动,
  #   会覆盖掉 Windows 宿主的驱动透传, GPU 直接消失。只能装 cuda-toolkit-<版本>。
  if find_nvcc; then
    ok "nvcc 已存在: $NVCC_BIN"
  else
    if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
      wget -q -O /tmp/cuda-keyring_1.1-1_all.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
      sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
      sudo apt-get update -y
    fi

    local torch_cuda want pkg
    torch_cuda="$("$VENVS/dev/bin/python" -c 'import torch; print(torch.version.cuda)')"
    want="$(printf '%s' "$torch_cuda" | cut -d. -f1,2 | tr . -)"   # 13.0 -> 13-0
    if apt-cache pkgnames 2>/dev/null | grep -qx "cuda-toolkit-$want"; then
      pkg="cuda-toolkit-$want"
    else
      pkg="cuda-toolkit-$(apt-cache pkgnames 2>/dev/null \
            | grep -E '^cuda-toolkit-[0-9]+-[0-9]+$' \
            | sed 's/^cuda-toolkit-//' | sort -V | tail -1)"
      warn "源里没有 cuda-toolkit-$want (torch CUDA 运行时=$torch_cuda), 改用 $pkg"
    fi

    dim "安装 $pkg (约 3-4GB, 几分钟)"
    if ! sudo apt-get install -y "$pkg"; then
      warn "$pkg 安装失败。手动排查: apt-cache policy $pkg"
    fi
  fi

  # 重新找一次。
  # ⚠ 坑(实测踩过): apt 包名用【短横线】cuda-toolkit-13-3, 安装目录却用【点】cuda-13.3。
  #   靠包名拼路径会得到不存在的 /usr/local/cuda-13-3 -> nvcc 不进 PATH ->
  #   连带 .cu 编译测试失败、flash-attn 编译失败、vLLM 兜底关掉 FlashInfer。
  #   所以不猜路径, 一律 glob 实测。
  if ! find_nvcc; then
    warn "装完仍未在 /usr/local/cuda* 下找到 nvcc —— 环境变量不写入。诊断:"
    ls -d /usr/local/cuda* 2>/dev/null || echo "  /usr/local 下没有 cuda 目录"
    dpkg -l 2>/dev/null | grep -i 'cuda-toolkit\|cuda-nvcc' || echo "  dpkg 里没有 cuda 相关包"
    return 0
  fi

  local root; root="$(cuda_root_of "$NVCC_BIN")"
  ok "CUDA Toolkit: $root"
  "$NVCC_BIN" --version | tail -1

  # 版本不一致提醒: 一般是源里没有精确对应版本, 退到了最新的。CUDA 13.x minor 兼容,
  # 且驱动 610.x 支持到 13.3, 通常可用; 真撞上不兼容会报 undefined symbol。
  local torch_cuda2 want_root
  torch_cuda2="$("$VENVS/dev/bin/python" -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '')"
  if [[ -n "$torch_cuda2" ]]; then
    want_root="/usr/local/cuda-$(printf '%s' "$torch_cuda2" | cut -d. -f1,2)"
    [[ "$root" != "$want_root" ]] && \
      warn "工具链($root)与 torch CUDA 运行时($torch_cuda2)不同 —— 一般可用(13.x minor 兼容)。
  若 FlashInfer JIT 报 undefined symbol, 再装同版本 toolkit 对齐。"
  fi

  # 只加 CUDA_HOME + PATH。刻意【不】加 LD_LIBRARY_PATH: torch 的 pip wheel 自带
  # CUDA 运行时, 把系统 CUDA 的 lib64 顶到前面会盖掉它, 反而把能跑的环境搞坏。
  # 系统 CUDA 只负责【编译】, 运行时交给 torch 自己管。
  if ! grep -q "CUDA_HOME=$root" "$HOME/.bashrc" 2>/dev/null; then
    echo "export CUDA_HOME=$root"                >> "$HOME/.bashrc"
    echo 'export PATH=$CUDA_HOME/bin:$PATH'      >> "$HOME/.bashrc"
    echo "# 注: 刻意不加 LD_LIBRARY_PATH, 见 setup_wsl.sh 阶段 4 注释" >> "$HOME/.bashrc"
  fi
  export CUDA_HOME="$root"
  export PATH="$CUDA_HOME/bin:$PATH"

  cuda_smoke_test "$NVCC_BIN"
}

# ================================================================== 阶段 5
stage5_vllm() {
  log "阶段 5/7  vLLM $VLLM_VERSION"

  # 源码按【tag】拉, 不要用 main HEAD: vLLM 预编译 wheel 只发布到"最近一次成功构建
  # 的 commit", main HEAD 通常比它新, 拿 HEAD 装 editable 必然 404。
  if [[ -d "$LAB/vllm-src" ]]; then
    ok "源码已存在: $LAB/vllm-src"
  else
    git clone --branch "v$VLLM_VERSION" --depth 1 \
      https://github.com/vllm-project/vllm.git "$LAB/vllm-src"
  fi

  uv venv "$VENVS/vllm" --python "$PYVER" --seed
  # shellcheck disable=SC1091
  source "$VENVS/vllm/bin/activate"

  if "$VENVS/vllm/bin/python" -c "import vllm" >/dev/null 2>&1; then
    ok "vLLM 已装: $("$VENVS/vllm/bin/python" -c 'import vllm; print(vllm.__version__)')"
  else
    # 二进制直接用官方 wheel, 不本地编译。"读代码"和"能跑"是两件事:
    # 源码靠 clone、运行靠 wheel, 两者锁同一 tag 即严格对齐。
    if ! uv pip install "vllm==$VLLM_VERSION" --torch-backend=auto \
          --extra-index-url "https://wheels.vllm.ai/$VLLM_VERSION/cu130" \
          --index-strategy unsafe-best-match 2>/dev/null; then
      warn "cu130 变体拉取失败, 回退 PyPI 默认 wheel (仍是 GPU 版)"
      uv pip install "vllm==$VLLM_VERSION" --torch-backend=auto
    fi
    ok "vLLM 装完"
  fi

  # WSL2 完整配置: 打开 pinned host memory 保住 V2 runner(默认路径),
  # 同时清掉上一版遗留的降级开关。逻辑统一收敛在 sync_vllm_activate() 里。
  #
  # 背景: V2 runner 依赖 UVA 缓冲区, 而 is_uva_available() 恒等于
  #       is_pin_memory_available(); vLLM 在 WSL2 上默认把 pin memory 关了,
  #       于是 V2 runner 一 init_device() 就 raise "UVA is not available"。
  #       上游 PR #54655 / #47579 至今未合入。
  # 这是【启用能力】不是【降级】: V2 runner + FlashInfer + UVA 全套保持默认。
  # 若确实不稳定, 再退回 V1: export VLLM_USE_V2_MODEL_RUNNER=0
  sync_vllm_activate

  "$VENVS/vllm/bin/python" -c "
import vllm, torch
print('  vLLM', vllm.__version__, '| torch', torch.__version__,
      '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"

  cat <<'NOTE'
  源码在 ~/llm-engine-lab/vllm-src (tag), 用来【读】; 装进环境的是官方 wheel, 用来【跑】。
  两者版本严格对齐 —— 你读到的就是跑着的。
  只有想改 vLLM 的 Python 代码并立即生效时才需要 editable 安装:
    cd ~/llm-engine-lab/vllm-src && VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
  这一步依赖 wheels.vllm.ai 上有该 commit 的预编译产物, 失败是正常的, 不必强求。
NOTE
}

# ================================================================== 阶段 6
stage6_ops() {
  log "阶段 6/7  attention 算子库 (装在 dev 环境)"
  # shellcheck disable=SC1091
  source "$VENVS/dev/bin/activate"

  if "$VENVS/dev/bin/python" -c "import flashinfer" >/dev/null 2>&1; then
    ok "flashinfer-python 已装, 跳过"
  else
    # 【不能用 --no-deps】flashinfer-python 虽是纯 Python wheel, 但依赖
    # apache-tvm-ffi / cuda-python / ninja / einops 等十余个包, 全跳过会导致
    # import 直接失败。正确做法是 -c 锁住 torch, 其余依赖照常装全。
    if uv pip install -c "$CONSTRAINT" flashinfer-python 2>/dev/null; then
      ok "flashinfer-python 装完"
    else
      warn "flashinfer-python 安装失败 —— 可跳过, 自己写 Triton kernel 不依赖它"
    fi
  fi

  if "$VENVS/dev/bin/python" -c "import flash_attn" >/dev/null 2>&1; then
    ok "flash-attn 已装, 跳过"
  elif [[ "${SKIP_FLASH_ATTN:-0}" == "1" ]]; then
    ok "SKIP_FLASH_ATTN=1, 跳过 flash-attn"
  else
    dim "flash-attn 需要从源码编译, 约 10-20 分钟"
    if MAX_JOBS="${MAX_JOBS:-4}" uv pip install -c "$CONSTRAINT" flash-attn --no-build-isolation 2>/dev/null; then
      ok "flash-attn 装完"
    else
      warn "flash-attn 编译失败 —— 可重试: MAX_JOBS=2 bash setup_wsl.sh (并行过高会 OOM)。
  不影响主线: 自己写 Triton kernel 不需要它。"
    fi
  fi
}

# ================================================================== 阶段 7
stage7_verify() {
  log "阶段 7/7  自检"
  if [[ -f "$SCRIPT_DIR/verify_env.py" ]]; then
    # shellcheck disable=SC1091
    source "$VENVS/dev/bin/activate"
    python "$SCRIPT_DIR/verify_env.py" || warn "自检有失败项, 看上面的输出"
  else
    warn "找不到 verify_env.py (应在 $SCRIPT_DIR)"
  fi

  if [[ "${RUN_SMOKE:-0}" == "1" ]]; then
    # shellcheck disable=SC1091
    source "$VENVS/vllm/bin/activate"
    [[ -f "$SCRIPT_DIR/test_vllm.py" ]] && python "$SCRIPT_DIR/test_vllm.py" \
      || warn "找不到 test_vllm.py"
  else
    dim "加 RUN_SMOKE=1 可再跑 vLLM 冒烟测试(会下载模型)"
  fi
}

# ================================================================== 主流程
case "$MODE" in
  check)
    stage0_check
    log "组件状态"
    find_nvcc && ok "nvcc: $NVCC_BIN" || warn "nvcc: 未找到"
    [[ -x "$VENVS/dev/bin/python" ]] && ok "dev 环境存在" || warn "dev 环境不存在"
    [[ -x "$VENVS/vllm/bin/python" ]] && ok "vllm 环境存在" || warn "vllm 环境不存在"
    "$VENVS/vllm/bin/python" -c "import vllm; print('  vLLM', vllm.__version__)" 2>/dev/null || true
    echo
    dump_gpu_diag
    ;;

  fix)
    # 只做 CUDA 环境变量修复 + 清理降级开关, 秒级完成, 不用重跑全量安装
    # 注意: 这里是脚本顶层, 不能用 local(会报 "local: can only be used in a function")
    if find_nvcc; then
      root="$(cuda_root_of "$NVCC_BIN")"
      ok "找到 CUDA Toolkit: $root"
      if ! grep -q "CUDA_HOME=$root" "$HOME/.bashrc" 2>/dev/null; then
        echo "export CUDA_HOME=$root"           >> "$HOME/.bashrc"
        echo 'export PATH=$CUDA_HOME/bin:$PATH' >> "$HOME/.bashrc"
      fi
      export CUDA_HOME="$root"; export PATH="$CUDA_HOME/bin:$PATH"
      cuda_smoke_test "$NVCC_BIN"
    else
      warn "仍没找到 nvcc —— 请先跑 bash setup_wsl.sh (阶段 4) 安装 CUDA Toolkit"
      exit 1
    fi

    # 关键: 光删降级开关不够。还必须补上完整配置所需的 pin memory,
    # 否则 V2 runner(默认)仍会因 UVA 不可用而崩 —— 这正是上一轮的坑。
    sync_vllm_activate
    ok "vllm 环境激活脚本已同步: pin memory 开启, 降级开关清除"

    if ! find_nvcc; then
      warn "仍无 nvcc —— FlashInfer 采样器已临时关闭(兜底)"
    else
      ok "nvcc 可用, FlashInfer 保持默认(JIT 编译)"
    fi

    cat <<'NEXT'
  接下来:
    source ~/venvs/dev/bin/activate
    MAX_JOBS=4 uv pip install -c ~/llm-engine-lab/constraints.txt flash-attn --no-build-isolation
    source ~/venvs/vllm/bin/activate && python ~/llm-engine-env/test_vllm.py
NEXT
    ;;

  verify)
    stage7_verify
    ;;

  all)
    stage0_check
    stage1_apt
    stage2_uv
    stage3_dev
    stage4_cuda
    stage5_vllm
    stage6_ops
    stage7_verify
    cat <<'DONE'

============================================================================
 完成。日常使用:

   source ~/venvs/dev/bin/activate     # 自己写引擎
   source ~/venvs/vllm/bin/activate    # 跑 vLLM 对照实验

 常用命令:
   bash setup_wsl.sh --check     体检(不装东西)
   bash setup_wsl.sh --fix       修 CUDA 环境变量 + 清降级开关(秒级)
   bash setup_wsl.sh --verify    重跑自检
   bash setup_wsl.sh             重跑全量(幂等, 已完成的会跳过)

 ⚠ 16GB 卡跑 vLLM 务必显式限制显存: 在 WSL 里 nvidia-smi 看到的是整张卡,
   但 Windows 桌面侧已占掉 ~2.6GB, vLLM 默认 0.9 会 OOM。建议 0.65:
     vllm serve Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.65 --max-model-len 4096
============================================================================
DONE
    ;;
esac
