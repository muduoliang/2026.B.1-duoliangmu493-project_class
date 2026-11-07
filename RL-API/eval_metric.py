# -*- coding: utf-8 -*-
"""
eval_metric.py —— 评价指标：ATT（平均旅行时间）/ AWT（平均等待时间）/ AQL（平均排队长度）

对应论文：王之颢《城市交通信号控制大语言模型方法研究与系统实现》
         第四章 4.2 节 RL 辅助模型的实验评价指标（论文对比表中 HZ-1 / HZ-2 两组场景的
         FixedTime 与 Our RL 就是以 ATT 为对照量）。

【三个指标的口径（本实现的明确定义，避免歧义）】
设仿真区间为 [0, T]，采样粒度 Δt（= config.SIM_INTERVAL，本数据为 1 s），
第 t 个采样时刻路网中「速度 < 0.1 m/s」的车辆数为 W(t)（口径与 CityFlow 引擎一致：
engine.cpp:641 `get_lane_waiting_vehicle_count`），全部车道排队车辆数之和为 Q(t)，
则该 episode 内：

  AQL (平均排队长度, 辆)  = (1/n) Σ_t Q(t)                      —— 逐秒采样求平均
  AWT (平均等待时间, s)   = Σ_t [W(t)·Δt] / N_total              —— 车辆平均累计等待时间
  ATT (平均旅行时间, s)   = (Σ 已完成车辆行程时间 + Σ 在途车辆已用时间) / 车辆数
                           直接采用引擎 get_average_travel_time()（engine.cpp:670-691）

其中 N_total = 本 episode 中出现过的唯一车辆 id 数（已到达 + 在途），由 env 逐秒
用 get_vehicles(include_waiting=True) 统计得到——CityFlow 不对外暴露「累计进入车辆数」，
故按此口径统计。

【只读】本文件只做计算，不读写 ./data。
"""

# 兼容 Python 3.8
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ===========================================================================
# 单 episode 指标累积器
# ===========================================================================

@dataclass
class EpisodeMetrics:
    """
    逐秒累积 ATT / AWT / AQL。

    用法（由 env_cityflow.CityFlowTrafficEnv 在每个仿真秒调用 observe）：
        m = EpisodeMetrics(interval=1.0)
        m.observe(t, queue_sum=..., n_active=..., n_seen_total=..., att_now=...)
        ...
        result = m.finalize(att_final=engine.get_average_travel_time())
    """
    interval: float = 1.0
    queue_samples: List[float] = field(default_factory=list)
    waiting_samples: List[float] = field(default_factory=list)
    att_samples: List[float] = field(default_factory=list)
    active_samples: List[int] = field(default_factory=list)
    # 逐秒「本 episode 至今出现过的车辆总数」N_seen(t)
    # 用途：把累计等待车辆秒数 Σ W(τ)Δτ 换算成「每车平均延误」曲线（绘图/分析用），
    #       口径与 finalize() 中 AWT 的 N_total 一致（同一来源变量）。
    seen_samples: List[int] = field(default_factory=list)
    # 决策时刻采样的「各交叉口排队车辆数之和」，用于给出 per-intersection AQL
    per_intersection_queue: Dict[str, List[float]] = field(default_factory=dict)
    # 决策时刻路网剩余车辆（用于判断 episode 是否可提前结束）
    last_active: int = 0
    last_time: float = 0.0

    # --- 逐秒采样 ---
    def observe(self, t: float, queue_sum: float, waiting_sum: float,
                n_active: int, n_seen_total: int, att_now: float) -> None:
        self.queue_samples.append(float(queue_sum))
        self.waiting_samples.append(float(waiting_sum))
        self.att_samples.append(float(att_now))
        self.active_samples.append(int(n_active))
        self.seen_samples.append(int(n_seen_total))
        self.last_active = int(n_active)
        self.last_time = float(t)

    # --- 决策时刻的 per-intersection 排队 ---
    def observe_decision(self, per_inter_queue: Dict[str, float]) -> None:
        for k, v in per_inter_queue.items():
            self.per_intersection_queue.setdefault(k, []).append(float(v))

    # --- 汇总 ---
    def finalize(self, att_final: Optional[float] = None,
                 n_total_override: Optional[int] = None) -> Dict[str, float]:
        """
        计算 episode 级指标。

        n_total_override: 由 env 传入的「出现过车辆总数」；缺省则用 active 峰值近似。
        """
        n = len(self.queue_samples)
        if n == 0:
            return {"ATT": float("nan"), "AWT": float("nan"), "AQL": float("nan"),
                    "throughput": 0.0, "n_samples": 0, "episode_seconds": 0.0}

        aql = float(np.mean(self.queue_samples))
        waited_vehicle_seconds = float(np.sum(self.waiting_samples) * self.interval)

        if n_total_override is not None and n_total_override > 0:
            n_total = int(n_total_override)
        else:
            # 兜底：用整个 episode 观测到的在途车辆峰值近似总车辆数
            n_total = int(max(self.active_samples)) if self.active_samples else 0

        awt = waited_vehicle_seconds / n_total if n_total > 0 else float("nan")

        if att_final is not None and np.isfinite(att_final):
            att = float(att_final)
        elif self.att_samples:
            att = float(self.att_samples[-1])
        else:
            att = float("nan")

        finished = max(0, n_total - int(self.last_active))
        per_inter_aql = {k: float(np.mean(v)) for k, v in self.per_intersection_queue.items()
                         if len(v) > 0}

        return {
            "ATT": att,                                        # 平均旅行时间 (s)
            "AWT": awt,                                        # 平均等待时间 (s)
            "AQL": aql,                                        # 平均排队长度 (辆)
            "throughput": float(finished),                     # 完成行程车辆数（辅助指标）
            "n_total_vehicles": float(n_total),
            "waited_vehicle_seconds": waited_vehicle_seconds,
            "n_samples": float(n),
            "episode_seconds": float(self.last_time),
            "att_series_mean": float(np.mean(self.att_samples)) if self.att_samples else float("nan"),
            "per_intersection_AQL": per_inter_aql,
        }


# ===========================================================================
# 便捷函数：固定配时基线 / 论文对照
# ===========================================================================

def compute_att_from_series(att_series: Sequence[float]) -> float:
    """给定逐秒 ATT 序列，返回其时间平均（用于绘制收敛曲线时取每 episode 的平均 ATT）。"""
    arr = np.asarray(list(att_series), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def relative_improvement(baseline: float, ours: float) -> float:
    """
    相对改善率（论文中「Our RL 相比 FixedTime 的 ATT 下降百分比」的口径）：
        improvement = (baseline - ours) / baseline * 100 %
    """
    if baseline is None or not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return float("nan")
    return float((baseline - ours) / baseline * 100.0)


def format_metrics(tag: str, metrics: Dict[str, float]) -> str:
    """把指标 dict 格式化成一行日志文本。"""
    return ("{tag:<18} ATT={att:8.2f}s  AWT={awt:8.2f}s  AQL={aql:6.2f}辆  "
            "完成车辆={thr:6.0f}  采样数={ns:5.0f}").format(
        tag=tag,
        att=metrics.get("ATT", float("nan")),
        awt=metrics.get("AWT", float("nan")),
        aql=metrics.get("AQL", float("nan")),
        thr=metrics.get("throughput", float("nan")),
        ns=metrics.get("n_samples", float("nan")),
    )


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """滑动平均（用于把逐 episode 指标平滑后观察收敛趋势）。"""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0 or window <= 1:
        return [float(v) for v in arr]
    out: List[float] = []
    for i in range(arr.size):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(arr[lo:i + 1])))
    return out


# ===========================================================================
# 逐秒序列导出（供 plot_metrics.py 绘图；只搬运真实采样值，不做插值或补造）
# ===========================================================================

def series_from_metrics(m: EpisodeMetrics) -> Dict[str, np.ndarray]:
    """
    把逐秒累积器转成绘图用的 numpy 序列字典。

    键（全部为真实采样/真实累计，无任何虚构或插值）：
        t                     采样时刻 (s)，t = Δt, 2Δt, ...
        queue                 逐秒「全部车道排队车辆数之和」Q(t)
        waiting               逐秒「等待车辆数」W(t)（速度 < 0.1 m/s）
        active                逐秒在途车辆数
        seen                  逐秒「本 episode 至今出现过车辆总数」N_seen(t)
        att                   逐秒引擎 get_average_travel_time()
        waited_vehicle_seconds 累计等待车辆秒数 Σ W(τ)Δτ
        delay                 每车平均延误 = 累计等待车辆秒数 / N_seen(t)（s/辆）
        per_inter__<iid>      各交叉口在决策时刻的排队车辆数采样序列
    """
    n = len(m.queue_samples)
    interval = float(m.interval)
    t = np.arange(1, n + 1, dtype=np.float64) * interval
    queue = np.asarray(m.queue_samples, dtype=np.float64)
    waiting = np.asarray(m.waiting_samples, dtype=np.float64)
    active = np.asarray(m.active_samples, dtype=np.int64)
    att = np.asarray(m.att_samples, dtype=np.float64)
    seen = (np.asarray(m.seen_samples, dtype=np.int64)
            if len(m.seen_samples) == n else np.zeros(n, dtype=np.int64))
    waited = np.cumsum(waiting) * interval
    delay = waited / np.maximum(1, seen)
    out: Dict[str, np.ndarray] = {
        "t": t, "queue": queue, "waiting": waiting, "active": active, "seen": seen,
        "att": att, "waited_vehicle_seconds": waited, "delay": delay,
    }
    for k, v in m.per_intersection_queue.items():
        out["per_inter__" + str(k)] = np.asarray(v, dtype=np.float64)
    return out


def per_intersection_ids(series: Dict[str, np.ndarray]) -> List[str]:
    """从序列字典中取出 per-intersection 序列对应的交叉口 id 列表（保持写入顺序）。"""
    return [k[len("per_inter__"):] for k in series.keys() if k.startswith("per_inter__")]


def save_series(series: Dict[str, np.ndarray], path: str) -> str:
    """把序列字典存成压缩 npz（供离线绘图 tools/plot_offline.py 复用）。"""
    import os
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in series.items()})
    return path


def load_series(path: str) -> Dict[str, np.ndarray]:
    """读取 save_series 写出的 npz。"""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_series_csv(series: Dict[str, np.ndarray], path: str) -> str:
    """把逐秒序列另存为 CSV（便于用 Excel/其他工具二次分析；行数较多但每行都是真值）。"""
    import csv
    import os
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    cols = [k for k in series.keys() if not k.startswith("per_inter__")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        n = len(series[cols[0]]) if cols else 0
        for i in range(n):
            w.writerow([float(series[c][i]) for c in cols])
    return path
