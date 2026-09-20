#!/usr/bin/env bash
# ============================================================================
#  WSL2 GPU 透传诊断脚本 —— 在 WSL 内执行:  bash diag_wsl_gpu.sh
#  用途: 当 setup_wsl.sh 报 "WSL 内找不到 nvidia-smi" 时, 定位到底断在哪一层。
# ============================================================================

hr() { printf '\n\033[1;36m─── %s\033[0m\n' "$*"; }

hr "1. 发行版与内核 (必须是 WSL2)"
cat /proc/version 2>/dev/null
echo "uname -r : $(uname -r)"
case "$(uname -r)" in
  *WSL2*|5.*|6.*) echo "判定    : WSL2  ✓" ;;
  *)             echo "判定    : 疑似 WSL1  ✗  WSL1 没有 GPU 访问能力" ;;
esac
echo "分发版本: $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"

hr "2. 宿主机驱动注入的 CUDA 文件 (/usr/lib/wsl/lib)"
if [[ -d /usr/lib/wsl/lib ]]; then
  ls -la /usr/lib/wsl/lib/ 2>&1
else
  echo "✗ /usr/lib/wsl/lib 不存在 —— 驱动未注入 CUDA 到 WSL (最常见的原因)"
fi

hr "3. 驱动目录 (/usr/lib/wsl/drivers)"
ls /usr/lib/wsl/drivers/ 2>&1 | head -20 || true

hr "4. PATH 是否包含 /usr/lib/wsl/lib"
if echo "$PATH" | tr ':' '\n' | grep -q '/usr/lib/wsl/lib'; then
  echo "✓ 已在 PATH 中"
else
  echo "✗ 不在 PATH 中 (但 nvidia-smi 存在的话可以用绝对路径调用)"
fi

hr "5. 直接以绝对路径调用 nvidia-smi"
if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  /usr/lib/wsl/lib/nvidia-smi 2>&1 | head -15
else
  echo "✗ /usr/lib/wsl/lib/nvidia-smi 不存在或不可执行"
fi

hr "6. libcuda 动态库"
ldconfig -p 2>/dev/null | grep -i 'libcuda\|nvidia-ml' || echo "✗ ldconfig 缓存里没有 libcuda / nvidia-ml"
echo "--- find 兜底 ---"
find /usr/lib/wsl -maxdepth 3 -name 'libcuda*' -o -maxdepth 3 -name 'libnvidia-ml*' 2>/dev/null | head -10 || true

hr "7. GPU 设备节点"
ls -la /dev/dxg 2>&1 || echo "✗ 没有 /dev/dxg"
ls -la /dev/nvidia* 2>&1 || echo "✗ 没有 /dev/nvidia*"

hr "8. Python 侧的最终结论"
if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY' 2>&1 || true
try:
    import torch
    print("torch", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
except ImportError:
    print("尚未安装 torch (属正常, 环境还没搭)")
PY
else
  echo "python3 不存在"
fi

hr "诊断结束"
cat <<'NEXT'
把上面 1–7 的输出整体发出来即可定位。最常见的三种结果:

  A. /usr/lib/wsl/lib 不存在
     → 在 Windows PowerShell 执行:
         wsl --update
         wsl --shutdown
       然后重开 Ubuntu-24.04

  B. uname -r 显示 4.4.0-Microsoft (WSL1)
     → wsl -l -v 看 VERSION 列, 若是 1:
         wsl --set-version Ubuntu-24.04 2

  C. /usr/lib/wsl/lib/nvidia-smi 存在但 PATH 里没有
     → echo 'export PATH="/usr/lib/wsl/lib:$PATH"' >> ~/.bashrc && source ~/.bashrc
NEXT
