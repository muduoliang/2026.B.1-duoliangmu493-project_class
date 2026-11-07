# -*- coding: utf-8 -*-
"""
tools/plot_offline.py —— 离线重绘全部图表（不重新训练、不读取 ./data）

用途：
  1) 训练已完成，想换格式（png/pdf/svg）、换分辨率、换平滑窗口后重新出图；
  2) 训练中断后，用已经写出的 train_log.csv / eval_log.csv / series/*.npz 出图；
  3) 论文排版需要单图版本（--individual）；
  4) 训练脚本内部也调用本脚本（--training-only 用于中途刷新训练曲线）——
     绘图放在子进程里，matplotlib 若原生崩溃（缺 DLL 等）也不会影响训练进程。

数据来源（只读，全部为真实运行产物；文件缺失会明确报错，不生成占位图）：
    <output-root>/logs/train_log.csv            训练逐 episode 指标
    <output-root>/logs/eval_log.csv             贪心评测逐次指标
    <output-root>/logs/series/*.npz             基线与评测 episode 的逐秒序列
    <output-root>/reports/run_report.json       基线汇总与最优评测

用法：
    python tools/plot_offline.py                                  # 默认 output/
    python tools/plot_offline.py --output-root output/_smoke       # 冒烟产物
    python tools/plot_offline.py --fmt pdf --dpi 300 --lang en     # 论文排版/英文标签
    python tools/plot_offline.py --no-cdf --no-individual          # 只要主图
    python tools/plot_offline.py --training-only                   # 只刷新训练过程 4 曲线

退出码：
    0 成功；4 绘图依赖不可用（matplotlib 缺失/损坏）；5 缺少训练产物（train_log.csv）；
    6 绘图过程异常；7 未生成任何图；其他（Windows 上常见为 3221225472 级别的大负数）
    表示 matplotlib 原生扩展崩溃（缺 DLL / 版本不匹配），此时请重装 matplotlib：
        pip install --force-reinstall --no-cache-dir "matplotlib>=3.5"
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Sequence

# 让 tools/ 下的脚本能 import 上级目录的工程模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="离线重绘训练曲线与测试评估图（只读 output/ 运行产物，不读 ./data）")
    p.add_argument("--output-root", default=None,
                   help="产物根目录（默认用 config.py 的 output/；冒烟产物为 output/_smoke）")
    p.add_argument("--fmt", default="png", choices=["png", "pdf", "svg", "jpg"], help="图片格式")
    p.add_argument("--dpi", type=int, default=150, help="图片分辨率（默认 150）")
    p.add_argument("--smooth", type=int, default=5, help="训练曲线滑动平均窗口（episode）")
    p.add_argument("--ts-smooth", type=int, default=15, help="评估时序曲线滑动平均窗口（仿真秒）")
    p.add_argument("--lang", default="auto", choices=["auto", "cjk", "en"], help="标签语言")
    p.add_argument("--no-cdf", action="store_true", help="不生成 CDF 累积分布图")
    p.add_argument("--no-individual", action="store_true", help="训练曲线只出 2×2 拼图")
    p.add_argument("--training-only", action="store_true",
                   help="只刷新训练过程 4 曲线（训练中途刷新用），不动测试评估图")
    return p.parse_args(argv)


def resolve_dirs(output_root: Optional[str]) -> Dict[str, str]:
    """确定各产物目录；未指定 --output-root 时沿用 config.ensure_output_dirs()。"""
    if output_root is None:
        import config
        return config.ensure_output_dirs()
    root = os.path.abspath(output_root)
    return {
        "output": root,
        "model": os.path.join(root, "rl_model"),
        "log": os.path.join(root, "logs"),
        "generated": os.path.join(root, "generated"),
        "report": os.path.join(root, "reports"),
    }


def matplotlib_preflight() -> Optional[str]:
    """
    显式验证绘图后端可用：matplotlib 存在但原生扩展损坏时，导入 pyplot 会让进程
    直接以原生异常码退出（本机 Windows conda 实测 -1066598273 / 0xC06D007F），
    因此这里先做一次最小渲染，失败就返回原因，让上层给出可读提示。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")                     # 服务器/WSL 无显示环境，必须用 Agg
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(1, 1))
        plt.plot([0.0, 1.0], [0.0, 1.0])
        plt.close(fig)
        return None
    except BaseException as e:                    # noqa: BLE001 - 依赖缺失/损坏都要给出提示
        return "{}: {}".format(type(e).__name__, e)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dirs = resolve_dirs(args.output_root)

    print("=" * 78)
    print("离线重绘图表（论文 4.2 节 Our RL）{}".format("［仅训练曲线］" if args.training_only else ""))
    print("=" * 78)
    print("[产物目录] {}".format(dirs["output"]))
    print("[数据来源] {} / {} / {}".format(
        os.path.join(dirs["log"], "train_log.csv"),
        os.path.join(dirs["log"], "eval_log.csv"),
        os.path.join(dirs["log"], "series")))

    try:
        import plot_metrics
    except Exception as e:                     # matplotlib 缺失等
        print("[错误] 无法导入 plot_metrics：{}".format(e))
        print("       请先安装绘图依赖：pip install 'matplotlib>=3.5'")
        return 4

    bad = matplotlib_preflight()
    if bad is not None:
        print("[错误] 绘图后端不可用（matplotlib 缺失或原生扩展损坏）：{}".format(bad))
        print("       修复：pip install --force-reinstall --no-cache-dir 'matplotlib>=3.5'")
        print("       （若为 Windows 宿主环境，建议在 WSL 的 conda 环境里绘图）")
        return 4

    try:
        if args.training_only:
            fig_dir = os.path.join(dirs["output"], "figures")
            files = plot_metrics.plot_training_curves(
                os.path.join(dirs["log"], "train_log.csv"),
                os.path.join(fig_dir, "training_curves.{}".format(args.fmt)),
                eval_log_path=os.path.join(dirs["log"], "eval_log.csv"),
                smooth=args.smooth, dpi=args.dpi, lang=args.lang,
                also_individual=not args.no_individual)
            if not files:
                print("[提示] 未生成任何图。")
                return 7
            print("[绘图完成] 训练曲线 → {}".format(files[0]))
            for f in files[1:]:
                print("   {}".format(f))
            return 0

        info = plot_metrics.generate_all(
            dirs, fmt=args.fmt, dpi=args.dpi, smooth=args.smooth, ts_smooth=args.ts_smooth,
            lang=args.lang, also_individual=not args.no_individual,
            with_cdf=not args.no_cdf, verbose=True)
    except FileNotFoundError as e:
        print("[数据缺失，无法绘图] {}".format(e))
        return 5
    except Exception as e:
        print("[绘图失败] {}: {}".format(type(e).__name__, e))
        return 6

    if not info.get("figures"):
        print("[提示] 未生成任何图。")
        return 7
    if info.get("missing_sources"):
        print("\n[提示] 以下产物缺失（对应图未生成或只有单侧数据）：")
        for m in info["missing_sources"]:
            print("   - {}".format(m))
    print("\n图表索引: {}".format(info.get("index_file")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
