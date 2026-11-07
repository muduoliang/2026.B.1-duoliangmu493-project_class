# -*- coding: utf-8 -*-
"""离线冒烟训练：不依赖 CityFlow 引擎，验证 train_rl_aux.py 全链路可跑通。

用途与边界（重要）：
    * 本脚本用 tools/selfcheck.py 里的 MockEngine 替换 CityFlow 仿真引擎，
      让「数据校验 → 环境 → 网络 → 回放 → 梯度更新 → 指标 → 权重落盘」整条
      训练链路在没有 cityflow 扩展包的机器上也能跑一遍。
    * MockEngine 是人为的简化车流模型，**其 ATT/AWT/AQL 数值没有任何实验意义**，
      不能写进论文；论文级结果必须在 Ubuntu/WSL 的 conda 环境里用真实 CityFlow
      运行 `python train_rl_aux.py` 得到。
    * 冒烟产物写到 output/_smoke/ 下，与真实训练的 output/rl_model 等目录隔离，
      避免用 MockEngine 训出的权重污染真实训练。

用法（工程根目录下）：
    python tools/smoke_train.py
    python tools/smoke_train.py --episodes 2 --episode-seconds 600
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import config                      # noqa: E402
import train_rl_aux as T           # noqa: E402
from env_cityflow import CityFlowTrafficEnv   # noqa: E402
from selfcheck import MockEngine    # noqa: E402

SMOKE_ROOT = os.path.join(config.OUTPUT_DIR, "_smoke")


def patch_output_dirs(smoke_root: str = SMOKE_ROOT):
    """把 config.ensure_output_dirs 重定向到 output/_smoke/，隔离冒烟产物。"""
    original = config.ensure_output_dirs

    def ensure():
        dirs = {
            "output": smoke_root,
            "model": os.path.join(smoke_root, "rl_model"),
            "log": os.path.join(smoke_root, "logs"),
            "generated": os.path.join(smoke_root, "generated"),
            "report": os.path.join(smoke_root, "reports"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        return dirs

    config.ensure_output_dirs = ensure
    return original


def patch_env_engine():
    """把 train_rl_aux 里构造 CityFlowTrafficEnv 的地方换成注入 MockEngine 的版本。

    CityFlowTrafficEnv 本身支持 engine 注入（engine 参数），因此这里只是工厂替换，
    不改动任何训练逻辑与数据校验逻辑。
    """
    real_cls = CityFlowTrafficEnv

    def factory(schema, **kwargs):
        kwargs.pop("engine", None)
        engine = MockEngine(schema)
        print("[冒烟] 已注入 MockEngine（简化车流模型，仅用于链路验证）")
        return real_cls(schema, engine=engine, **kwargs)

    T.CityFlowTrafficEnv = factory
    return real_cls


def check_artifacts(smoke_root: str, episodes: int) -> bool:
    """检查冒烟产物是否齐全：权重 / 回放池 / 训练日志 / 评测日志 / 运行报告。"""
    problems = []
    model_dir = os.path.join(smoke_root, "rl_model")
    log_dir = os.path.join(smoke_root, "logs")
    report_dir = os.path.join(smoke_root, "reports")

    for name in ("dqn_rl_aux_last.pth", "replay_buffer.npz"):
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            problems.append("缺少或为空: {}".format(path))

    train_log = os.path.join(log_dir, "train_log.csv")
    if not os.path.isfile(train_log):
        problems.append("缺少训练日志: {}".format(train_log))
    else:
        with open(train_log, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != episodes:
            problems.append("训练日志应有 {} 行，实际 {} 行".format(episodes, len(rows)))
        else:
            header = sorted(rows[0].keys())
            need = {"episode", "ATT", "AWT", "AQL", "throughput", "avg_reward", "loss_mean"}
            missing = need - set(header)
            if missing:
                problems.append("训练日志缺少指标列: {}".format(sorted(missing)))
            last = rows[-1]
            print("[冒烟] 末轮指标（MockEngine，无实验意义）: ATT={} AWT={} AQL={} "
                  "throughput={} avg_reward={} loss_mean={}".format(
                      last.get("ATT"), last.get("AWT"), last.get("AQL"),
                      last.get("throughput"), last.get("avg_reward"), last.get("loss_mean")))

    report_file = os.path.join(report_dir, "run_report.json")
    if not os.path.isfile(report_file):
        problems.append("缺少运行报告: {}".format(report_file))
    else:
        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)
        dims = report.get("runtime_dimensions") or {}
        print("[冒烟] 运行报告 runtime_dimensions = {}".format(
            json.dumps(dims, ensure_ascii=False)))
        if not dims:
            problems.append("运行报告缺少 runtime_dimensions（维度必须由数据推导）")

    if problems:
        print("\n[冒烟失败] 产物检查未通过：")
        for p in problems:
            print("   - {}".format(p))
        return False
    print("\n[冒烟通过] 权重 / 回放池 / 训练日志 / 评测日志 / 运行报告 均已生成")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="离线冒烟训练（MockEngine，不依赖 CityFlow）")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--epochs-per-episode", type=int, default=2)
    parser.add_argument("--episode-seconds", type=float, default=600.0)
    parser.add_argument("--smoke-root", default=SMOKE_ROOT)
    args = parser.parse_args(argv)

    patch_output_dirs(args.smoke_root)
    patch_env_engine()

    train_args = [
        "--smoke",
        "--episodes", str(args.episodes),
        "--epochs-per-episode", str(args.epochs_per_episode),
        "--episode-seconds", str(args.episode_seconds),
        "--save-every", "1",
        "--eval-every", "1",
    ]
    if args.data_dir:
        train_args += ["--data-dir", args.data_dir]

    rc = T.main(train_args)
    print("\n[冒烟] train_rl_aux.main 返回码 = {}".format(rc))
    if rc != 0:
        return rc
    ok = check_artifacts(args.smoke_root, args.episodes)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
