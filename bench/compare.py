#!/usr/bin/env python3
"""P0 · benchmark 结果对比：汇总多份 results JSON 为表格，可选画图。

用法:
  python bench/compare.py                     # 汇总 bench/results/*.json
  python bench/compare.py --plot out.png     # 追加 TPOT/吞吐对比图
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="*", help="results JSON；缺省读 bench/results/*.json")
    p.add_argument("--plot", default=None, help="输出对比图 png 路径")
    p.add_argument("--sort", default="tag", choices=["tag", "tpot"])
    args = p.parse_args()

    paths = [Path(x) for x in args.inputs] or sorted(Path("bench/results").glob("*.json"))
    if not paths:
        print("没有结果文件（bench/results/*.json），先跑 bench/run_local.py")
        return 1

    rows = []
    for fp in paths:
        d = json.loads(fp.read_text())
        s = d["summary"]
        rows.append(
            {
                "tag": fp.stem,
                "backend": d["backend"],
                "device": f"{d['device']}/{d['dtype']}",
                "n": s["n_requests"],
                "ttft_p50": s["ttft_p50_ms"],
                "tpot_p50": s["tpot_p50_ms"],
                "tpot_p99": s["tpot_p99_ms"],
                "tok_s": s["output_tok_per_s"],
            }
        )
    rows.sort(key=lambda r: r[args.sort])

    header = (
        f"{'tag':<40}{'backend':<8}{'dev':<10}{'n':>3}"
        f"{'TTFT p50':>12}{'TPOT p50':>12}{'TPOT p99':>12}{'tok/s':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['tag']:<40}{r['backend']:<8}{r['device']:<10}{r['n']:>3}"
            f"{r['ttft_p50']:>12.2f}{r['tpot_p50']:>12.2f}{r['tpot_p99']:>12.2f}{r['tok_s']:>10.1f}"
        )
    print("-" * len(header))
    print("单位: ms / output tok/s")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        names = [r["tag"] for r in rows]
        ax1.barh(names, [r["tpot_p50"] for r in rows])
        ax1.set_xlabel("TPOT p50 (ms)")
        ax1.invert_yaxis()
        ax2.barh(names, [r["tok_s"] for r in rows])
        ax2.set_xlabel("output tok/s")
        ax2.invert_yaxis()
        fig.suptitle("P0 benchmark compare (HF baseline)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"[compare] 图 -> {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())