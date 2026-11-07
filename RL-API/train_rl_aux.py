# -*- coding: utf-8 -*-
"""
train_rl_aux.py —— 论文4.2节「融合邻居信息的动态交通信号控制强化学习算法」主训练脚本

对应论文：王之颢《城市交通信号控制大语言模型方法研究与系统实现》
         第四章 4.2 节 RL 辅助模型（Our RL），本脚本实现该节的 DQN 五模块框架：
    ① 环境      → env_cityflow.CityFlowTrafficEnv（CityFlow 仿真 + 状态/动作/奖励）
    ② 经验回放区 → replay_buffer.ReplayBuffer（容量 config.REPLAY_CAPACITY=12000）
    ③ 特征提取  → model_net.ObservationEncoder + NeighborRelationFusion
    ④ 相位网络  → model_net 相位选择头（相位动作 + 动作掩码）
    ⑤ 持续时间网络 → model_net 持续时间选择头（7 档真实持续时间集合）
  经验元组 (S, a_p, a_d, R, S')，损失 = MSE(R + γ·max_{a'}Q(S',a') − Q(S,a))，γ=0.8。

【数据约束（需求方硬性要求，本脚本严格执行）】
  - 启动即调用 data_schema.build_schema('./data')：解析并校验真实数据字段，
    打印「检测到的全部可用字段 + 数据维度」；
  - 论文4.2节状态空间所需的字段若在 ./data 或 CityFlow 引擎返回中缺失，
    直接抛 ValueError 并打印缺失字段名，程序立刻终止；
    **绝不自动填充、绝不虚构数据集不存在的特征/观测/维度**；
  - 所有维度（状态维度、车道数、相位动作数、邻居数、相对位置类别数）运行时推导，
    本文件不出现任何写死的特征维度；
  - ./data 全程只读，一切生成物（CityFlow 配置、日志、权重、报告）写入 ./output。

【运行】
  python train_rl_aux.py                        # 按 config.py 的论文参数训练
  python train_rl_aux.py --smoke                # 小规模冒烟（验证链路是否跑通）
  python train_rl_aux.py --episodes 10 --eval-every 2
"""

# 兼容 Python 3.8
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import config
from data_schema import DataSchema, DataValidationError, SignalPlan, build_schema
from eval_metric import (format_metrics, relative_improvement, save_series,
                         save_series_csv, series_from_metrics)
from env_cityflow import (CITYFLOW_INSTALL_HINT, CityFlowTrafficEnv,
                          build_cityflow_config)

# torch 单独 try：缺失时给出明确的安装提示（需求：异常处理）
try:
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_
    _TORCH_IMPORT_ERROR: Optional[BaseException] = None
except BaseException as _e:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = _e

from model_net import DQNNetwork, count_parameters, epsilon_greedy_select
from replay_buffer import ReplayBuffer


# ===========================================================================
# 0. 命令行参数（默认值全部来自 config.py，与论文参数一一对应）
# ===========================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="论文4.2节 RL 辅助模型（融合邻居信息的动态交通信号控制）训练脚本")
    p.add_argument("--data-dir", default=None,
                   help="原始数据集目录（默认 ./data，只读；不得修改其中任何文件）")
    p.add_argument("--episodes", type=int, default=None,
                   help="总训练 episode 数（论文：{}）".format(config.TOTAL_EPISODES))
    p.add_argument("--epochs-per-episode", type=int, default=None,
                   help="每个 episode 的梯度更新轮数（论文：{}）".format(config.EPOCHS_PER_EPISODE))
    p.add_argument("--episode-seconds", type=float, default=None,
                   help="单 episode 仿真时长（默认由 ./data 需求数据的 startTime/endTime 推导）")
    p.add_argument("--warmup-seconds", type=float, default=None, help="预热时长（不计入指标）")
    p.add_argument("--batch-size", type=int, default=None,
                   help="回放池采样批量（论文：{}）".format(config.BATCH_SIZE))
    p.add_argument("--capacity", type=int, default=None,
                   help="经验回放缓冲区容量（论文：{}）".format(config.REPLAY_CAPACITY))
    p.add_argument("--gamma", type=float, default=None,
                   help="折扣因子（论文：{}）".format(config.GAMMA))
    p.add_argument("--lr", type=float, default=None,
                   help="Adam 初始学习率（论文：{}）".format(config.LEARNING_RATE))
    p.add_argument("--eta", type=float, default=None,
                   help="邻居奖励平衡系数（论文：{}）".format(config.ETA))
    p.add_argument("--hidden-dim", type=int, default=None,
                   help="隐藏维度 h（论文：{}）".format(config.HIDDEN_DIM))
    p.add_argument("--n-heads", type=int, default=None,
                   help="注意力头数（论文：{}）".format(config.N_HEADS))
    p.add_argument("--n-self-attn-layers", type=int, default=None,
                   help="自注意力堆叠层数 N（论文：{}）".format(config.N_SELF_ATTN_LAYERS))
    p.add_argument("--eps-start", type=float, default=None, help="ε 初值（论文未给出，程序可调）")
    p.add_argument("--eps-end", type=float, default=None, help="ε 终值（论文未给出，程序可调）")
    p.add_argument("--eps-decay-steps", type=int, default=None,
                   help="ε 线性衰减所用的智能体决策次数（论文未给出，程序可调）")
    p.add_argument("--target-update-every", type=int, default=None,
                   help="目标网络同步间隔（梯度更新次数，论文未给出，程序可调）")
    p.add_argument("--eval-every", type=int, default=5,
                   help="每隔多少 episode 做一次贪心评测（默认 5）")
    p.add_argument("--save-every", type=int, default=None,
                   help="每隔多少 episode 存一次权重（默认 config.SAVE_EVERY_EPISODE）")
    p.add_argument("--thread-num", type=int, default=1, help="CityFlow 仿真线程数")
    p.add_argument("--device", default=None, help="计算设备（默认自动：cuda 可用则 cuda）")
    p.add_argument("--seed", type=int, default=None, help="随机种子")
    p.add_argument("--no-baseline", action="store_true",
                   help="跳过定时配时（FixedTime）基线评测")
    p.add_argument("--resume", default=None, help="从指定权重文件继续训练")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟模式：极小规模参数，仅验证数据校验/张量维度/训练链路是否跑通")
    # ---- 绘图相关（训练曲线 + 测试评估图，数据全部来自本次真实运行产物）----
    p.add_argument("--no-plots", action="store_true",
                   help="不生成任何图（只训练，不写 output/figures/）")
    p.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg", "jpg"],
                   help="图片格式（默认 png）")
    p.add_argument("--plot-dpi", type=int, default=150, help="图片分辨率（默认 150）")
    p.add_argument("--plot-smooth", type=int, default=5,
                   help="训练曲线滑动平均窗口（episode 粒度，默认 5）")
    p.add_argument("--plot-ts-smooth", type=int, default=15,
                   help="评估时序图滑动平均窗口（仿真秒，默认 15）")
    p.add_argument("--plot-lang", default="auto", choices=["auto", "cjk", "en"],
                   help="图标签语言（auto=有中文字体则中文，否则英文）")
    p.add_argument("--no-cdf", action="store_true", help="不生成 CDF 累积分布图（可选深度分析图）")
    p.add_argument("--no-individual-plots", action="store_true",
                   help="训练曲线只出 2×2 拼图，不额外输出单图")
    p.add_argument("--plot-every", type=int, default=0,
                   help="训练中途每隔 N 个 episode 刷新一次训练曲线图（0=只在训练结束时出图）")
    return p.parse_args(argv)


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """把命令行参数与 config.py 合并成一份运行时设置（config 里的论文参数为默认值）。"""
    s: Dict[str, Any] = {
        "data_dir": args.data_dir or config.DATA_DIR,
        "episodes": config.TOTAL_EPISODES if args.episodes is None else int(args.episodes),
        "epochs_per_episode": (config.EPOCHS_PER_EPISODE if args.epochs_per_episode is None
                               else int(args.epochs_per_episode)),
        "episode_seconds": args.episode_seconds,
        "warmup_seconds": (config.WARMUP_SECONDS if args.warmup_seconds is None
                           else float(args.warmup_seconds)),
        "batch_size": config.BATCH_SIZE if args.batch_size is None else int(args.batch_size),
        "capacity": config.REPLAY_CAPACITY if args.capacity is None else int(args.capacity),
        "gamma": config.GAMMA if args.gamma is None else float(args.gamma),
        "lr": config.LEARNING_RATE if args.lr is None else float(args.lr),
        "eta": config.ETA if args.eta is None else float(args.eta),
        "hidden_dim": config.HIDDEN_DIM if args.hidden_dim is None else int(args.hidden_dim),
        "n_heads": config.N_HEADS if args.n_heads is None else int(args.n_heads),
        "n_self_attn_layers": (config.N_SELF_ATTN_LAYERS if args.n_self_attn_layers is None
                               else int(args.n_self_attn_layers)),
        "eps_start": config.EPS_START if args.eps_start is None else float(args.eps_start),
        "eps_end": config.EPS_END if args.eps_end is None else float(args.eps_end),
        "eps_decay_steps": (config.EPS_DECAY_STEPS if args.eps_decay_steps is None
                            else int(args.eps_decay_steps)),
        "target_update_every": (config.TARGET_UPDATE_EVERY if args.target_update_every is None
                                else int(args.target_update_every)),
        "eval_every": int(args.eval_every),
        "save_every": (config.SAVE_EVERY_EPISODE if args.save_every is None
                       else int(args.save_every)),
        "thread_num": int(args.thread_num),
        "seed": config.SEED if args.seed is None else int(args.seed),
        "run_baseline": not args.no_baseline,
        "resume": args.resume,
        "smoke": bool(args.smoke),
        # 绘图设置（不属于论文超参，仅影响出图，不影响训练数值）
        "plots": not args.no_plots,
        "plot_format": args.plot_format,
        "plot_dpi": int(args.plot_dpi),
        "plot_smooth": int(args.plot_smooth),
        "plot_ts_smooth": int(args.plot_ts_smooth),
        "plot_lang": args.plot_lang,
        "plot_cdf": not args.no_cdf,
        "plot_individual": not args.no_individual_plots,
        "plot_every": int(args.plot_every),
    }
    if args.smoke:      # 冒烟参数：只验证链路，不作为论文复现结果
        s.update({
            "episodes": min(2, s["episodes"]),
            "epochs_per_episode": min(2, s["epochs_per_episode"]),
            "episode_seconds": s["episode_seconds"] or 600.0,
            "batch_size": min(32, s["batch_size"]),
            "capacity": min(2000, s["capacity"]),
            "hidden_dim": min(32, s["hidden_dim"]),
            "n_heads": max(1, min(4, s["n_heads"])),
            "n_self_attn_layers": min(1, s["n_self_attn_layers"]),
            "eval_every": 1,
            "save_every": 1,
        })
    return s


def set_global_seed(seed: int) -> None:
    np.random.seed(seed % (2 ** 32))
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def pick_device(name: Optional[str]) -> str:
    if name:
        return name
    if config.DEVICE:
        return config.DEVICE
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ===========================================================================
# 1. 批数据 → torch 张量（键名与 replay_buffer 中的经验元组一致）
# ===========================================================================

_LONG_KEYS = ("rel", "group", "phase_avail", "next_rel", "next_group",
              "next_phase_avail", "phase_action", "duration_action")


def batch_to_torch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    """把回放池采样出的 numpy 批次转成 torch 张量（维度即运行时推导维度）。"""
    t: Dict[str, Any] = {}
    for k, v in batch.items():
        if k == "index":
            continue
        arr = np.asarray(v)
        if k in _LONG_KEYS:
            t[k] = torch.as_tensor(arr.astype(np.int64), dtype=torch.long, device=device)
        else:
            t[k] = torch.as_tensor(arr.astype(np.float32), dtype=torch.float32, device=device)
    return t


def decision_to_torch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    """把 env.decision_batch() 的结果转成 torch 张量（仅含当前状态相关键）。"""
    t: Dict[str, Any] = {}
    for k in ("own", "own_mask", "nbr", "nbr_mask", "rel", "group", "phase_avail"):
        arr = np.asarray(batch[k])
        if k in _LONG_KEYS:
            t[k] = torch.as_tensor(arr.astype(np.int64), dtype=torch.long, device=device)
        else:
            t[k] = torch.as_tensor(arr.astype(np.float32), dtype=torch.float32, device=device)
    return t


def valid_phase_mask(phase_avail: "torch.Tensor", n_phase_max: int) -> "torch.Tensor":
    """
    动作掩码：异构交叉口的合法相位动作数不同，用真实可选相位个数生成布尔掩码。
    掩码外的槽位在 argmax/softmax 前被填 −inf，从而不会选出该交叉口不存在的相位。
    """
    ar = torch.arange(n_phase_max, device=phase_avail.device).unsqueeze(0)
    return ar < phase_avail.unsqueeze(1)


def model_forward(model: DQNNetwork, t: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """统一的前向封装（prefix='' 为 S，prefix='next_' 为 S'）。"""
    return model(own=t[prefix + "own"], own_mask=t[prefix + "own_mask"],
                 nbr=t[prefix + "nbr"], nbr_mask=t[prefix + "nbr_mask"],
                 rel=t[prefix + "rel"], group=t[prefix + "group"])


def epsilon_by_decisions(decisions: int, s: Dict[str, Any]) -> float:
    """
    ε 线性衰减（论文未给出 ε 设置，属程序级补充超参，见 config.py 注释）。
    计数口径 = 智能体决策次数（每次交叉口决策计 1）。
    """
    frac = min(1.0, float(decisions) / max(1.0, float(s["eps_decay_steps"])))
    return float(s["eps_start"] + (s["eps_end"] - s["eps_start"]) * frac)


# ===========================================================================
# 2. 定时配时基线（FixedTime）：只回放 ./data 中真实存在的配时方案
# ===========================================================================

def make_plan_schedule(plan: SignalPlan):
    """
    由 ./data 中 signal_inter_*.txt 的真实「时间→相位」事件生成相位查询函数。

    说明：多个交叉口的配时文件可能相同也可能不同，本函数不做任何假设；
    若 episode 时长超过配时文件覆盖范围且文件给出了周期，则按该周期循环回放，
    否则保持最后一个相位（均为真实数据，不构造新相位）。
    """
    if not plan.events:
        raise DataValidationError(
            "[数据校验失败] 交叉口 {} 的配时文件 {} 中没有事件行".format(plan.inter_id, plan.path))
    times = [float(t) for t, _ in plan.events]
    phases = [int(p) for _, p in plan.events]
    t0, t_last = times[0], times[-1]
    cycle = float(plan.cycle_seconds) if plan.cycle_seconds else 0.0

    def schedule(t: float) -> int:
        if t <= t_last:
            idx = bisect.bisect_right(times, t) - 1
        elif cycle > 0:
            t_eff = t0 + math.fmod(t - t0, cycle)
            idx = bisect.bisect_right(times, t_eff) - 1
        else:
            idx = len(times) - 1
        return phases[max(0, idx)]

    return schedule


# ===========================================================================
# 2b. 运行产物落盘：逐秒序列（绘图的唯一数据来源）
# ===========================================================================

def dump_episode_series(env: CityFlowTrafficEnv, dirs: Dict[str, str], tag: str,
                        also_csv: bool = True, verbose: bool = False) -> Optional[str]:
    """
    把本 episode 的逐秒真实序列存到 output/logs/series/<tag>_series.npz（另存一份 CSV）。

    序列内容（全部由 env_cityflow 在仿真中逐秒采集，无插值、无补造）：
        排队长度 Q(t)、等待车辆数 W(t)、在途车辆数、出现车辆总数 N_seen(t)、
        引擎 ATT、累计等待车辆秒数、每车平均延误、各交叉口决策时刻排队采样。
    绘图模块（plot_metrics.py）只读取这些真值。
    """
    if env.metrics is None or len(env.metrics.queue_samples) == 0:
        return None
    series = series_from_metrics(env.metrics)
    sdir = os.path.join(dirs["log"], "series")
    npz = save_series(series, os.path.join(sdir, "{}_series.npz".format(tag)))
    if also_csv:
        try:
            save_series_csv(series, os.path.join(sdir, "{}_series.csv".format(tag)))
        except Exception as e:                     # CSV 只是附带产物，失败不影响训练
            print("[提示] 序列 CSV 写出失败（npz 已保存）：{}".format(e))
    if verbose:
        print("[序列已保存] {}".format(npz))
    return npz


def generate_figures(dirs: Dict[str, str], s: Dict[str, Any], training_only: bool = False,
                     verbose: bool = True) -> Optional[Dict[str, Any]]:
    """
    生成/刷新图表（训练曲线 + 测试评估图）。

    training_only=True 用于训练中途刷新训练曲线（此时还没有完整评测序列）。

    【为什么绘图放在子进程里跑】
    matplotlib 的原生扩展一旦损坏（缺 DLL / 版本不匹配），`import matplotlib.pyplot`
    会让**整个进程**以原生异常码直接死掉（本机 Windows conda 环境实测
    exit=-1066598273 / 0xC06D007F），Python 的 try/except 根本拦不住。
    训练可能已经跑了几小时，绝不能因为出图而丢掉权重：
    因此这里用 `sys.executable tools/plot_offline.py` 子进程出图，
    子进程怎么崩都只打印告警，训练结果、权重、日志不受影响。
    """
    if not s.get("plots", True):
        return None
    root = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(root, "tools", "plot_offline.py")
    if not os.path.exists(script):
        print("[绘图跳过] 未找到绘图脚本：{}".format(script))
        return None

    fig_dir = os.path.join(dirs["output"], "figures")
    cmd = [sys.executable, "-u", script,
           "--output-root", dirs["output"],
           "--fmt", s["plot_format"], "--dpi", str(s["plot_dpi"]),
           "--smooth", str(s["plot_smooth"]), "--ts-smooth", str(s["plot_ts_smooth"]),
           "--lang", s["plot_lang"]]
    if training_only:
        cmd += ["--training-only", "--no-individual"]
    else:
        if not s["plot_cdf"]:
            cmd.append("--no-cdf")
        if not s["plot_individual"]:
            cmd.append("--no-individual")

    log_name = "plot_refresh.log" if training_only else "plot_run.log"
    log_path = os.path.join(dirs["log"], log_name)
    env_vars = dict(os.environ)
    env_vars["PYTHONIOENCODING"] = "utf-8"          # 子进程中文输出不乱码
    try:
        os.makedirs(dirs["log"], exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("[绘图子进程] {}\n".format(" ".join(cmd)))
            fh.flush()
            proc = subprocess.run(cmd, cwd=root, stdout=fh, stderr=subprocess.STDOUT,
                                  env=env_vars)
    except Exception as e:                          # 子进程都起不来（极少见）
        print("[绘图失败，训练结果与权重不受影响] {}: {}".format(type(e).__name__, e))
        return None

    if proc.returncode != 0:
        hints = {
            4: "绘图依赖不可用（matplotlib 缺失或原生扩展损坏）：pip install --force-reinstall "
               "--no-cache-dir 'matplotlib>=3.5'",
            5: "缺少 train_log.csv（尚未开始训练或 --output-root 指错目录）",
            6: "绘图过程出现异常",
            7: "没有生成任何图",
        }
        hint = hints.get(proc.returncode,
                         "绘图子进程异常退出（Windows 上多为 matplotlib 原生扩展崩溃/缺 DLL）")
        print("[绘图失败，训练结果与权重不受影响] 退出码 {}：{}".format(proc.returncode, hint))
        print("   详情见日志：{}".format(log_path))
        return None

    idx_path = os.path.join(fig_dir, "figures_index.json")
    info: Dict[str, Any] = {"log_file": log_path}
    if os.path.exists(idx_path):
        try:
            with open(idx_path, "r", encoding="utf-8") as fh:
                info.update(json.load(fh))
        except Exception as e:
            print("[提示] 图表索引读取失败（图已生成）：{}".format(e))
    if verbose:
        figs = info.get("figures") or {}
        print("[绘图完成] {} 张图 → {}".format(len(figs), fig_dir))
        for k, v in figs.items():
            print("   {:<22} {}".format(k, v))
    return info


def run_fixed_time_baseline(env: CityFlowTrafficEnv, schema: DataSchema,
                            verbose: bool = True) -> Dict[str, float]:
    """
    FixedTime 基线：完全按 ./data 的 signal_inter_*.txt 定时方案控制（不做任何学习）。
    该基线的 ATT/AQL 是论文第四章对照实验的参照（论文杭州 HZ-1 FixedTime ATT=497.87s）。
    """
    schedules = {}
    for iid in env.inter_ids:
        plan = schema.signal_plans.get(iid)
        if plan is None:
            raise DataValidationError(
                "[数据校验失败] 交叉口 {} 在 ./data 中缺少配时文件（signal_inter_{}.txt），"
                "无法进行 FixedTime 基线评测；缺失即终止，不构造虚构配时。".format(iid, iid))
        schedules[iid] = make_plan_schedule(plan)

    env.reset()
    step_s = float(config.SIM_INTERVAL)
    while True:
        env.apply_light_phases({iid: schedules[iid](env.sim_time) for iid in env.inter_ids})
        env.advance(step_s)
        if env.is_done():
            break
    m = env.episode_metrics()
    if verbose:
        print(format_metrics("FixedTime 基线(./data 真实配时)", m))
    return m


# ===========================================================================
# 3. 评测（贪心策略，ε=0）
# ===========================================================================

@torch.no_grad()
def evaluate_policy(env: CityFlowTrafficEnv, model: DQNNetwork, device: str,
                    n_phase_max: int, epsilon: float = 0.0,
                    generator: Optional["torch.Generator"] = None) -> Dict[str, float]:
    """跑一个完整 episode 的贪心评测，返回 ATT/AWT/AQL 等指标。"""
    model.eval()
    env.reset()
    n_decisions = 0
    reward_sum, reward_n = 0.0, 0
    while True:
        batch = env.decision_batch()
        t = decision_to_torch(batch, device)
        out = model_forward(model, t)
        valid = valid_phase_mask(t["phase_avail"], n_phase_max)
        a_p, a_d = epsilon_greedy_select(out["phase_q"], out["duration_q"], valid,
                                         epsilon, generator)
        phase_actions = {iid: int(a_p[i]) for i, iid in enumerate(batch["inter_ids"])}
        duration_actions = {iid: int(a_d[i]) for i, iid in enumerate(batch["inter_ids"])}
        _, _, rewards, done, _ = env.step(phase_actions, duration_actions)
        n_decisions += len(batch["inter_ids"])
        if rewards:
            reward_sum += float(np.mean(list(rewards.values())))
            reward_n += 1
        if done:
            break
    m = env.episode_metrics()
    m["avg_decision_reward"] = reward_sum / max(1, reward_n)
    m["n_decisions"] = float(n_decisions)
    model.train()
    return m


# ===========================================================================
# 4. 训练主循环
# ===========================================================================

def checkpoint_payload(episode: int, model: DQNNetwork, target: DQNNetwork,
                       optimizer: "torch.optim.Optimizer", env: CityFlowTrafficEnv,
                       s: Dict[str, Any], updates: int, decisions: int,
                       best_att: float, metrics: Optional[Dict[str, float]] = None,
                       schema: Optional[DataSchema] = None) -> Dict[str, Any]:
    """
    权重文件内容：网络参数 + 运行时推导的维度 + 数据字段报告。

    保存维度是为了让评测/消融脚本在没有 ./data 变化时能自校验维度一致性
    （若 ./data 改变导致维度变化，加载时会明确报错而不是静默错配）。
    """
    spec = env.spec
    return {
        "episode": int(episode),
        "model_state": model.state_dict(),
        "target_state": target.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "updates": int(updates),
        "decisions": int(decisions),
        "best_eval_att": float(best_att),
        "last_metrics": metrics or {},
        "dims": {
            "state_dim_max": int(spec.state_dim_max),
            "state_dim_per_inter": list(spec.state_dim_per_inter),
            "n_rel_pos_classes": int(spec.n_rel_pos_classes),
            "n_rel_class_groups": int(spec.n_rel_class_groups),
            "n_selectable_phases_max": int(spec.n_selectable_phases_max),
            "n_duration_actions": int(spec.n_duration_actions),
            "neighbor_degree_max": int(spec.neighbor_degree_max),
            "hidden_dim": int(s["hidden_dim"]),
            "n_heads": int(s["n_heads"]),
            "n_self_attn_layers": int(s["n_self_attn_layers"]),
        },
        "inter_ids": list(env.inter_ids),
        "settings": {k: v for k, v in s.items() if k != "resume"},
        "data_files": {k: os.path.basename(v) if isinstance(v, str) else v
                       for k, v in env.schema.data_files.items()},
        "data_fields_report": (schema.fields_report if schema is not None else {}),
        "paper_section": "第四章 4.2 节 融合邻居信息的动态交通信号控制强化学习算法（Our RL）",
    }


def train(args: argparse.Namespace) -> int:
    s = resolve_settings(args)
    dirs = config.ensure_output_dirs()
    set_global_seed(s["seed"])
    device = pick_device(args.device)

    print("=" * 78)
    print("论文4.2节 RL 辅助模型复现 —— 融合邻居信息的动态交通信号控制（Our RL）")
    print("=" * 78)
    print("[配置] {} → 输出目录 {}".format(config.DATA_DIR, dirs["output"]))
    for name, value, source in config.describe():
        print("   {:<26} = {:<22} ({})".format(name, str(value), source))
    print("[运行参数] episodes={episodes} epochs/episode={epochs} batch={batch} "
          "capacity={cap} episode_seconds={eps_s} device={dev}".format(
              episodes=s["episodes"], epochs=s["epochs_per_episode"], batch=s["batch_size"],
              cap=s["capacity"], eps_s=s["episode_seconds"] or "自动(由数据推导)", dev=device))

    # ---------------- ① 数据读取 + 合法性校验（缺失字段直接终止） ----------------
    print("\n[1/5] 读取并校验 ./data（只读，不修改任何原始文件）...")
    schema = build_schema(s["data_dir"], verbose=config.PRINT_FIELDS_REPORT)
    report_path = os.path.join(dirs["report"], "data_fields_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(schema.fields_report, f, ensure_ascii=False, indent=2)
    print("[数据校验通过] 字段与维度报告已写入: {}".format(report_path))
    print("[数据检查] 受控交叉口 {} 个；论文4.2节状态五元组 {} ；"
          "每交叉口驶入车道数 {}；最大邻居数 {}；相位动作数 {}（上界 {}）；"
          "持续时间动作数 {}".format(
              schema.dims.n_intersections, "/".join(schema.dims.lane_feature_names),
              sorted(set(schema.dims.n_incoming_lanes_per_inter.values())),
              schema.dims.neighbor_degree_max,
              sorted(set(schema.dims.n_selectable_phases_per_inter.values())),
              schema.dims.n_selectable_phases_max, schema.dims.n_duration_actions))

    # ---------------- ② 环境 ----------------
    print("\n[2/5] 构建 CityFlow 环境（配置写入 output/generated，./data 只读）...")
    env = CityFlowTrafficEnv(schema, episode_seconds=s["episode_seconds"],
                             warmup_seconds=s["warmup_seconds"], seed=s["seed"],
                             thread_num=s["thread_num"], verbose=True)
    spec = env.spec
    print("[维度校验] {}".format(spec.summary()))

    # ---------------- ③ 网络 / 回放池 / 优化器 ----------------
    print("\n[3/5] 构建 DQN 网络（输入维度来自上面的真实数据推导结果，非写死）...")
    if torch is None:
        raise DataValidationError(
            "[依赖缺失] 无法导入 torch：{!r}\n  conda activate cityflow 后执行: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121".format(
                _TORCH_IMPORT_ERROR))
    model = DQNNetwork(
        state_dim=spec.state_dim_max,                      # ← 运行时推导维度
        n_rel_pos_classes=spec.n_rel_pos_classes,          # ← 运行时推导维度
        n_selectable_phases_max=spec.n_selectable_phases_max,
        n_duration_actions=spec.n_duration_actions,
        n_neighbor_groups=spec.n_rel_class_groups,
        hidden_dim=s["hidden_dim"], n_heads=s["n_heads"],
        n_self_attn_layers=s["n_self_attn_layers"],
    ).to(device)
    target = DQNNetwork(**{k: v for k, v in dict(
        state_dim=spec.state_dim_max, n_rel_pos_classes=spec.n_rel_pos_classes,
        n_selectable_phases_max=spec.n_selectable_phases_max,
        n_duration_actions=spec.n_duration_actions,
        n_neighbor_groups=spec.n_rel_class_groups, hidden_dim=s["hidden_dim"],
        n_heads=s["n_heads"], n_self_attn_layers=s["n_self_attn_layers"]).items()}).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=s["lr"])
    buffer = ReplayBuffer(capacity=s["capacity"], state_dim_max=spec.state_dim_max,
                          neighbor_degree_max=spec.neighbor_degree_max,
                          n_rel_pos_classes=spec.n_rel_pos_classes,
                          n_duration_actions=spec.n_duration_actions, seed=s["seed"])
    print("[网络] 参数量 = {}（隐藏维度 h={}，注意力头={}，自注意力层数 N={}）".format(
        count_parameters(model), s["hidden_dim"], s["n_heads"], s["n_self_attn_layers"]))
    print("[回放池] 容量 = {}，采样批量 = {}".format(s["capacity"], s["batch_size"]))

    # ---------------- ④ 定时配时基线 ----------------
    baseline: Dict[str, float] = {}
    if s["run_baseline"]:
        print("\n[4/5] FixedTime 基线评测（回放 ./data 中的真实定时方案）...")
        baseline = run_fixed_time_baseline(env, schema)
        # 保存基线的逐秒真实序列：测试评估图（时序/箱线/柱状/CDF）的对照组数据
        dump_episode_series(env, dirs, "baseline_fixedtime")
        print("[基线序列] 已保存逐秒序列，供测试评估图对比: {}".format(
            os.path.join(dirs["log"], "series", "baseline_fixedtime_series.npz")))
    else:
        print("\n[4/5] 已跳过 FixedTime 基线（--no-baseline）")

    # ---------------- ⑤ 训练 ----------------
    print("\n[5/5] 开始训练：{} episode × {} epoch/ episode".format(
        s["episodes"], s["epochs_per_episode"]))
    log_path = os.path.join(dirs["log"], "train_log.csv")
    eval_path = os.path.join(dirs["log"], "eval_log.csv")
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    log_w.writerow(["episode", "sim_seconds", "decisions", "epsilon", "avg_reward",
                    "ATT", "AWT", "AQL", "throughput", "loss_mean", "grad_updates",
                    "replay_size", "wall_seconds"])
    eval_f = open(eval_path, "w", newline="", encoding="utf-8")
    eval_w = csv.writer(eval_f)
    eval_w.writerow(["episode", "ATT", "AWT", "AQL", "throughput", "avg_decision_reward"])

    start_episode, updates, decisions, best_att = 0, 0, 0, float("inf")
    if s["resume"]:
        ckpt = torch.load(s["resume"], map_location=device)
        d = ckpt.get("dims", {})
        mismatch = [k for k, v in (("state_dim_max", spec.state_dim_max),
                                   ("n_rel_pos_classes", spec.n_rel_pos_classes),
                                   ("n_selectable_phases_max", spec.n_selectable_phases_max),
                                   ("n_duration_actions", spec.n_duration_actions),
                                   ("neighbor_degree_max", spec.neighbor_degree_max))
                    if k in d and int(d[k]) != int(v)]
        if mismatch:
            raise DataValidationError(
                "[权重维度不匹配] {} 中的维度 {} 与当前 ./data 推导维度不一致，"
                "拒绝加载（避免静默错配）".format(s["resume"], mismatch))
        model.load_state_dict(ckpt["model_state"])
        target.load_state_dict(ckpt.get("target_state", ckpt["model_state"]))
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_episode = int(ckpt.get("episode", 0)) + 1
        updates = int(ckpt.get("updates", 0))
        decisions = int(ckpt.get("decisions", 0))
        best_att = float(ckpt.get("best_eval_att", float("inf")))
        print("[续训] 从 {} 恢复：episode={} updates={} decisions={}".format(
            s["resume"], start_episode, updates, decisions))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(s["seed"])
    rng = np.random.default_rng(s["seed"])
    model.train()
    run_t0 = time.time()

    for episode in range(start_episode, s["episodes"]):
        ep_t0 = time.time()
        eps = epsilon_by_decisions(decisions, s)
        env.reset()
        n_inter_decisions = 0
        reward_sum, reward_n = 0.0, 0
        ep_steps = 0

        while True:
            batch = env.decision_batch()
            t = decision_to_torch(batch, device)
            with torch.no_grad():
                out = model_forward(model, t)
            valid = valid_phase_mask(t["phase_avail"], spec.n_selectable_phases_max)
            a_p, a_d = epsilon_greedy_select(out["phase_q"], out["duration_q"], valid,
                                             eps, generator)
            phase_actions = {iid: int(a_p[i]) for i, iid in enumerate(batch["inter_ids"])}
            duration_actions = {iid: int(a_d[i]) for i, iid in enumerate(batch["inter_ids"])}
            _, transitions, rewards, done, _ = env.step(phase_actions, duration_actions)

            # 经验元组 (S, a_p, a_d, R, S') 入回放池
            for tr in transitions:
                buffer.push(tr)
            n_inter_decisions += len(batch["inter_ids"])
            decisions += len(batch["inter_ids"])
            ep_steps += 1
            if rewards:
                reward_sum += float(np.mean(list(rewards.values())))
                reward_n += 1
            if done:
                break

        metrics = env.episode_metrics()

        # ---- 每个 episode 做 epochs_per_episode 轮梯度更新（论文：100 轮）----
        ep_losses: List[float] = []
        for _ in range(s["epochs_per_episode"]):
            if buffer.is_ready(s["batch_size"]):
                batch = buffer.sample(s["batch_size"], rng=rng)
            elif config.STRICT_BATCH_FROM_FULL_BUFFER:
                break       # 缓冲区未满，按论文 batch_size 不降级采样（可改 config 调整）
            else:
                batch = buffer.sample(max(1, len(buffer)), rng=rng)
            tb = batch_to_torch(batch, device)

            out = model_forward(model, tb)
            q_phase = out["phase_q"].gather(1, tb["phase_action"].unsqueeze(1)).squeeze(1)
            q_dur = out["duration_q"].gather(1, tb["duration_action"].unsqueeze(1)).squeeze(1)
            # 双输出动作的动作价值：Q(S,a_p,a_d) = Q_phase(S,a_p) + Q_dur(S,a_d)
            q_pred = q_phase + q_dur

            with torch.no_grad():
                t_out = model_forward(target, tb, prefix="next_")
                t_valid = valid_phase_mask(tb["next_phase_avail"], spec.n_selectable_phases_max)
                next_phase_q = t_out["phase_q"].masked_fill(~t_valid, -1e9).max(dim=1).values
                next_dur_q = t_out["duration_q"].max(dim=1).values
                # 论文4.2节 DQN 目标：y = R + γ·max_{a'} Q(S',a')，γ=0.8
                y = tb["reward"] + s["gamma"] * (1.0 - tb["done"]) * (next_phase_q + next_dur_q)

            loss = F.mse_loss(q_pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            updates += 1
            ep_losses.append(float(loss.item()))
            if updates % s["target_update_every"] == 0:
                target.load_state_dict(model.state_dict())

        wall = time.time() - ep_t0
        att = metrics.get("ATT", float("nan"))
        awt = metrics.get("AWT", float("nan"))
        aql = metrics.get("AQL", float("nan"))
        loss_mean = float(np.mean(ep_losses)) if ep_losses else float("nan")
        log_w.writerow([episode, round(metrics.get("episode_seconds", 0.0), 1),
                        n_inter_decisions, round(eps, 4), round(reward_sum / max(1, reward_n), 4),
                        round(att, 3), round(awt, 3), round(aql, 3),
                        metrics.get("throughput", 0), round(loss_mean, 5), updates,
                        len(buffer), round(wall, 2)])
        log_f.flush()

        if episode % max(1, config.LOG_EVERY_EPISODE) == 0 or episode == s["episodes"] - 1:
            print("[ep {:>3}/{}] eps={:.3f} R={:>8.2f} ATT={:>8.2f}s AWT={:>7.2f}s "
                  "AQL={:>6.2f} 通过={:>5.0f} loss={:>8.4f} updates={:<6d} 用时={:.1f}s".format(
                      episode, s["episodes"], eps, reward_sum / max(1, reward_n),
                      att, awt, aql, metrics.get("throughput", 0.0), loss_mean, updates, wall))

        # ---- 贪心评测（论文未给评测频率，属程序级设置）----
        if s["eval_every"] > 0 and ((episode + 1) % s["eval_every"] == 0
                                    or episode == s["episodes"] - 1):
            ev = evaluate_policy(env, model, device, spec.n_selectable_phases_max, 0.0)
            eval_w.writerow([episode, round(ev.get("ATT", float("nan")), 3),
                             round(ev.get("AWT", float("nan")), 3),
                             round(ev.get("AQL", float("nan")), 3),
                             ev.get("throughput", 0),
                             round(ev.get("avg_decision_reward", float("nan")), 4)])
            eval_f.flush()
            print(format_metrics("贪心评测 ep{}".format(episode), ev))
            # 保存本次贪心评测的逐秒真实序列（测试评估图使用其中 ATT 最优的一次）
            dump_episode_series(env, dirs, "eval_ep{:03d}".format(episode))
            if ev.get("ATT", float("inf")) < best_att:
                best_att = float(ev["ATT"])
                torch.save(checkpoint_payload(episode, model, target, optimizer, env, s,
                                              updates, decisions, best_att, ev, schema),
                           os.path.join(dirs["model"], "dqn_rl_aux_best.pth"))
                print("   ↑ 目前最优 ATT={:.2f}s，已保存 dqn_rl_aux_best.pth".format(best_att))

        # ---- 周期性保存权重 ----
        if s["save_every"] > 0 and ((episode + 1) % s["save_every"] == 0):
            torch.save(checkpoint_payload(episode, model, target, optimizer, env, s,
                                          updates, decisions, best_att, metrics, schema),
                       os.path.join(dirs["model"], "dqn_rl_aux_ep{}.pth".format(episode)))

        # ---- 训练中途刷新训练曲线（可选 --plot-every N；长训练可随时看图）----
        if (s["plot_every"] > 0 and (episode + 1) % s["plot_every"] == 0
                and episode != s["episodes"] - 1):
            log_f.flush()
            eval_f.flush()
            generate_figures(dirs, s, training_only=True)

    # ---------------- 收尾：保存最终权重与运行报告 ----------------
    torch.save(checkpoint_payload(s["episodes"] - 1, model, target, optimizer, env, s,
                                  updates, decisions, best_att, None, schema),
               os.path.join(dirs["model"], "dqn_rl_aux_last.pth"))
    buffer.save(os.path.join(dirs["model"], "replay_buffer.npz"))

    run_report = {
        "paper_section": "第四章 4.2 节（Our RL，融合邻居信息的动态交通信号控制强化学习算法）",
        "settings": {k: v for k, v in s.items() if k != "resume"},
        "runtime_dimensions": {
            "state_dim_per_inter": schema.dims.state_dim_per_inter,
            "state_dim_max": schema.dims.state_dim_max,
            "n_incoming_lanes_per_inter": schema.dims.n_incoming_lanes_per_inter,
            "n_phases_per_inter": schema.dims.n_phases_per_inter,
            "n_selectable_phases_per_inter": schema.dims.n_selectable_phases_per_inter,
            "neighbor_degree_max": schema.dims.neighbor_degree_max,
            "n_rel_pos_classes": schema.dims.n_rel_pos_classes,
            "n_rel_class_groups": schema.dims.n_rel_class_groups,
            "n_duration_actions": schema.dims.n_duration_actions,
            "lane_feature_names": list(schema.dims.lane_feature_names),
        },
        "fixed_time_baseline": baseline,
        "best_eval_ATT": best_att if best_att != float("inf") else None,
        "relative_improvement_vs_fixed_time_pct": (
            relative_improvement(baseline.get("ATT"), best_att)
            if baseline and best_att != float("inf") else None),
        "replay": buffer.stats(),
        "total_grad_updates": updates,
        "total_decisions": decisions,
        "wall_seconds": time.time() - run_t0,
        "outputs": dirs,
        "data_read_only_notice": schema.read_only_notice(),
    }
    # ---------------- 收尾：生成图表（数据全部来自本次真实运行产物） ----------------
    log_f.flush()
    eval_f.flush()
    fig_info = generate_figures(dirs, s)
    if fig_info:
        run_report["figures"] = fig_info.get("figures", {})
        run_report["figures_index_file"] = fig_info.get("index_file")
        run_report["figure_data_sources"] = fig_info.get("data_sources", {})

    report_file = os.path.join(dirs["report"], "run_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(run_report, f, ensure_ascii=False, indent=2)
    log_f.close()
    eval_f.close()

    print("\n" + "=" * 78)
    print("训练完成。权重目录: {}".format(dirs["model"]))
    print("  - dqn_rl_aux_last.pth（最后一轮） / dqn_rl_aux_best.pth（最优 ATT）")
    print("  - dqn_rl_aux_ep*.pth（每 {} 个 episode 一次）".format(s["save_every"]))
    print("日志: {} ；评测: {}".format(log_path, eval_path))
    print("运行报告: {}".format(report_file))
    if fig_info:
        print("图表目录: {}（训练曲线 4 条 + 测试评估 4 类图）".format(
            os.path.join(dirs["output"], "figures")))
    if baseline:
        print("FixedTime 基线 ATT={:.2f}s / AQL={:.2f}；本模型最优贪心评测 ATT={}".format(
            baseline.get("ATT", float("nan")), baseline.get("AQL", float("nan")),
            "N/A" if best_att == float("inf") else "{:.2f}s".format(best_att)))
    print("=" * 78)
    return 0


# ===========================================================================
# 5. 入口（含路径/字段/依赖异常的集中处理）
# ===========================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return train(args)
    except DataValidationError as e:
        # 数据/引擎字段校验失败：打印缺失字段与原因后立即退出，绝不虚构数据继续训练
        print("\n" + "!" * 78)
        print("[数据校验失败，训练终止]")
        print(str(e))
        print("!" * 78)
        return 2
    except FileNotFoundError as e:
        print("\n[路径错误] {}：请确认 ./data 目录存在，且包含 roadnet/flow 数据文件".format(e))
        return 3
    except KeyboardInterrupt:
        print("\n[中断] 训练被手动终止。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
