# -*- coding: utf-8 -*-
"""
plot_metrics.py —— 训练过程曲线 + 测试评估图表（离线绘图，只读运行产物）

对应论文：王之颢《城市交通信号控制大语言模型方法研究与系统实现》
         第四章 4.2 节 RL 辅助模型（Our RL）的实验分析图。

【本模块图表的唯一数据来源（不生成任何虚构数据）】
  1) output/logs/train_log.csv       —— train_rl_aux.py 每个 episode 写出的真实指标
  2) output/logs/eval_log.csv        —— 每次贪心评测的真实指标（ATT/AWT/AQL/吞吐量）
  3) output/logs/series/*.npz        —— 基线与评测 episode 的逐秒真实序列（env 采集）
  4) output/reports/run_report.json  —— FixedTime 基线汇总与最优评测
上述文件缺失时直接报错并列出缺失项，绝不使用占位/示例数据补图。

【产出图（默认 output/figures/）】
  训练过程（用户需求 1）：
    training_curves.png    2×2 拼图：Episode Reward / 平均延误 / 平均排队长度 AQL / 吞吐量
    training_curve_*.png   同一组曲线的单图版本（便于论文排版）
  测试评估（用户需求 2）：
    eval_timeseries.png    时间序列折线图：延误、排队长度随仿真步数变化（FixedTime vs Our RL）
    eval_boxplot.png       箱线图：逐秒排队长度分布 + 各交叉口平均排队长度分布
    eval_bar.png           柱状图：ATT / AWT / AQL / 吞吐量对比（含相对改善率标注）
    eval_cdf.png           累积分布图 CDF（深度分析，可选）

【中文字体】自动在系统/WSL(/mnt/c/Windows/Fonts)/Linux 字体目录中寻找 CJK 字体；
  找不到时自动改用英文标签（避免出现方框乱码），并打印安装建议。

【只读】本模块不读写 ./data，只读 output/ 下的运行产物。
"""

# 兼容 Python 3.8
from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from eval_metric import moving_average


# ===========================================================================
# 0. 绘图环境（Agg 后端 + 中文字体自动选择）
# ===========================================================================

_CJK_FONT_CANDIDATES: Tuple[str, ...] = (
    "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Noto Sans CJK JP",
    "Source Han Sans SC", "Source Han Sans CN", "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei", "AR PL UMing CN", "AR PL UKai CN", "Droid Sans Fallback",
)

_CJK_FONT_FILES: Tuple[str, ...] = (
    # Windows 字体（WSL 下 /mnt/c 挂载可直接注册使用，无需额外安装）
    "/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/msyh.ttf",
    "/mnt/c/Windows/Fonts/msyhbd.ttc", "/mnt/c/Windows/Fonts/simhei.ttf",
    "/mnt/c/Windows/Fonts/simsun.ttc",
    # Linux 常见中文字体
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)


def select_cjk_font() -> Optional[str]:
    """返回可用的中文字体名；找不到返回 None（调用方改用英文标签）。"""
    from matplotlib import font_manager
    for path in _CJK_FONT_FILES:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass
    for name in _CJK_FONT_CANDIDATES:
        try:
            font_manager.findfont(font_manager.FontProperties(family=name),
                                  fallback_to_default=False)
            return name
        except Exception:
            continue
    return None


def setup_matplotlib(lang: str = "auto", dpi: int = 150) -> Tuple[Any, bool, str]:
    """
    初始化 matplotlib（无界面 Agg 后端）。

    lang: auto（默认，有中文字体则中文）/ cjk（强制中文）/ en（强制英文）
    返回 (pyplot, 是否使用中文标签, 实际使用的字体名)
    """
    import matplotlib
    matplotlib.use("Agg")                 # 服务器/WSL 无显示环境，必须用 Agg
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    font = select_cjk_font() if lang in ("auto", "cjk") else None
    cjk = font is not None
    if lang == "cjk" and font is None:
        print("[绘图警告] 未找到中文字体，改用英文标签；"
              "如需中文可执行: sudo apt install -y fonts-noto-cjk fonts-wqy-zenhei")
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = [font, "DejaVu Sans"] if cjk else ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = float(dpi)
    rcParams["savefig.dpi"] = float(dpi)
    rcParams["axes.grid"] = True
    rcParams["grid.alpha"] = 0.3
    rcParams["legend.fontsize"] = 9
    rcParams["axes.titlesize"] = 11
    rcParams["axes.labelsize"] = 10
    return plt, cjk, (font or "DejaVu Sans")


# ---------------------------------------------------------------------------
# 标签：中文 / 英文两套（按字体可用性自动切换）
# ---------------------------------------------------------------------------

_LABELS_CJK: Dict[str, str] = {
    "training_title": "训练过程曲线 —— 融合邻居信息的动态交通信号控制 RL 辅助模型（论文 4.2 节）",
    "reward": "Episode Reward 回合奖励",
    "delay": "Episode 平均延误（AWT 平均等待时间, s）",
    "queue": "平均排队长度 AQL（辆）",
    "throughput": "通行量 / 吞吐量（完成行程车辆数）",
    "episode": "训练 episode",
    "x_time": "仿真步数 / 仿真时间（s）",
    "y_reward": "平均回合奖励",
    "y_delay": "平均延误（s）",
    "y_queue": "平均排队长度（辆）",
    "y_throughput": "完成行程车辆数（辆）",
    "raw": "逐 episode 原始值",
    "smooth": "滑动平均（窗口 {}）",
    "eval_pts": "贪心评测点",
    "att_short": "ATT 平均旅行时间（s）",
    "awt_short": "AWT 平均等待时间（s）",
    "ts_title": "测试评估：延误 / 排队长度随仿真步数变化",
    "ts_delay": "延误：累计等待车辆秒数 / 出现车辆总数（s/辆）",
    "ts_queue": "排队长度：全部车道排队车辆数之和（辆）",
    "box_title": "测试评估：箱线图 Boxplot —— 排队长度分布",
    "box_a": "逐秒排队长度分布（每 tag 为整段 episode 的逐秒真值）",
    "box_b": "各交叉口平均排队长度分布（每 tag 为 16 个受控交叉口）",
    "bar_title": "测试评估：柱状图 Bar Chart —— 关键指标对比",
    "cdf_title": "测试评估：累积分布图 CDF（深度分析，可选）",
    "cdf_a": "逐秒排队长度 CDF",
    "cdf_b": "逐秒每车平均延误 CDF",
    "x_queue": "排队长度（辆）",
    "x_delay": "每车平均延误（s）",
    "cdf_y": "累积概率 P(X ≤ x)",
    "tag_baseline": "FixedTime 基线（./data 真实配时）",
    "tag_rl": "Our RL（贪心评测 ep{}）",
    "median": "中位数",
    "mean": "均值",
    "impr_vs": "相对基线改善 {:+.2f}%",
    "source_note": "数据来源：{}（真实运行产物，未做任何虚构或插值）",
    "ep": "episode",
    "value": "数值",
}

_LABELS_EN: Dict[str, str] = {
    "training_title": "Training Curves - Neighbor-Fused Dynamic Traffic Signal Control RL (Paper Sec. 4.2)",
    "reward": "Episode Reward",
    "delay": "Episode Average Delay (AWT, s)",
    "queue": "Average Queue Length AQL (veh)",
    "throughput": "Throughput (finished trips)",
    "episode": "training episode",
    "x_time": "simulation step / time (s)",
    "y_reward": "mean episode reward",
    "y_delay": "average delay (s)",
    "y_queue": "average queue length (veh)",
    "y_throughput": "finished trips (veh)",
    "raw": "per-episode raw value",
    "smooth": "moving average (window {})",
    "eval_pts": "greedy evaluation",
    "att_short": "ATT (s)",
    "awt_short": "AWT (s)",
    "ts_title": "Evaluation: delay / queue length vs simulation step",
    "ts_delay": "delay: cumulated waiting veh-seconds / total vehicles (s/veh)",
    "ts_queue": "queue length: sum over all lanes (veh)",
    "box_title": "Evaluation: Boxplot - queue length distribution",
    "box_a": "per-second queue length distribution (whole episode)",
    "box_b": "per-intersection mean queue length (16 controlled intersections)",
    "bar_title": "Evaluation: Bar Chart - key metrics comparison",
    "cdf_title": "Evaluation: CDF (deep-dive, optional)",
    "cdf_a": "CDF of per-second queue length",
    "cdf_b": "CDF of per-second average delay",
    "x_queue": "queue length (veh)",
    "x_delay": "average delay per vehicle (s)",
    "cdf_y": "cumulative probability P(X <= x)",
    "tag_baseline": "FixedTime baseline (./data real plan)",
    "tag_rl": "Our RL (greedy eval ep{})",
    "median": "median",
    "mean": "mean",
    "impr_vs": "{:+.2f}% vs baseline",
    "source_note": "Source: {} (real run artifacts only, no synthetic data)",
    "ep": "episode",
    "value": "value",
}


def labels_for(cjk: bool) -> Dict[str, str]:
    return _LABELS_CJK if cjk else _LABELS_EN


# ===========================================================================
# 1. 读取运行产物（只读）
# ===========================================================================

def _read_csv_dicts(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError("[绘图数据缺失] 未找到 {}；请先完成训练再绘图。".format(path))
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_metric_csv(path: str) -> Dict[str, np.ndarray]:
    """把 train_log.csv / eval_log.csv 读成 {列名: float 数组}（非数值单元格记为 NaN）。"""
    rows = _read_csv_dicts(path)
    if not rows:
        raise ValueError("[绘图数据为空] {} 没有任何数据行。".format(path))
    cols: Dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        vals: List[float] = []
        for r in rows:
            v = (r.get(key) or "").strip()
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        cols[key] = np.asarray(vals, dtype=np.float64)
    return cols


def load_run_report(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _fmt_ext(fmt: str) -> str:
    fmt = (fmt or "png").lower().lstrip(".")
    if fmt not in ("png", "pdf", "svg", "jpg", "jpeg"):
        raise ValueError("[绘图参数错误] 不支持的图片格式 {}（可选 png/pdf/svg/jpg）".format(fmt))
    return fmt


def _save(fig: Any, out_path: str, dpi: int) -> str:
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


def _finite(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


# ===========================================================================
# 2. 训练过程曲线（用户需求 1）
# ===========================================================================

def plot_training_curves(train_log_path: str, out_path: str, eval_log_path: Optional[str] = None,
                         smooth: int = 5, dpi: int = 150, lang: str = "auto",
                         figsize: Tuple[float, float] = (13.0, 8.5),
                         also_individual: bool = False) -> List[str]:
    """
    训练过程 4 条曲线（2×2 拼图 + 可选单图）：
        Episode Reward / Episode 平均延误 / 平均排队长度 AQL / 吞吐量

    数据列全部来自 train_log.csv（train_rl_aux.py 每个 episode 写出的真实指标）；
    若存在 eval_log.csv，则把贪心评测点叠加到对应子图上（真实的独立评测，非拟合）。
    """
    plt, cjk, font = setup_matplotlib(lang, dpi)
    L = labels_for(cjk)
    tr = load_metric_csv(train_log_path)
    ev = load_metric_csv(eval_log_path) if (eval_log_path and os.path.exists(eval_log_path)) else None

    ep = tr.get("episode", np.arange(len(next(iter(tr.values()))), dtype=np.float64))
    panels = [
        ("reward", L["reward"], L["y_reward"], tr.get("avg_reward"), "tab:blue", None),
        ("delay", L["delay"], L["y_delay"], tr.get("AWT"), "tab:red", tr.get("ATT")),
        ("queue", L["queue"], L["y_queue"], tr.get("AQL"), "tab:orange", None),
        ("throughput", L["throughput"], L["y_throughput"], tr.get("throughput"), "tab:green", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    files: List[str] = []
    for ax, (key, title, ylab, series, color, extra) in zip(axes.ravel(), panels):
        if series is None:
            ax.set_visible(False)
            continue
        m = _finite(series)
        ax.plot(ep[m], series[m], color=color, alpha=0.35, lw=1.0, label=L["raw"])
        sm = np.asarray(moving_average(series[m], int(smooth)), dtype=np.float64)
        ax.plot(ep[m], sm, color=color, lw=2.0, label=L["smooth"].format(int(smooth)))
        if extra is not None and key == "delay":
            me = _finite(extra)
            ax.plot(ep[me], extra[me], color="tab:purple", lw=1.4, ls="--", label=L["att_short"])
        if ev is not None and len(ev.get("episode", [])) > 0:
            col = {"reward": "avg_decision_reward", "delay": "AWT", "queue": "AQL",
                   "throughput": "throughput"}[key]
            if col in ev:
                e_ep, e_v = ev["episode"], ev[col]
                mm = _finite(e_v)
                ax.scatter(e_ep[mm], e_v[mm], s=22, marker="o", facecolors="none",
                           edgecolors="black", linewidths=0.8, zorder=5,
                           label=L["eval_pts"])
        ax.set_xlabel(L["episode"])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(loc="best", framealpha=0.85)
        if also_individual:
            f2, a2 = plt.subplots(figsize=(7.2, 4.2))
            m2 = _finite(series)
            a2.plot(ep[m2], series[m2], color=color, alpha=0.35, lw=1.0, label=L["raw"])
            a2.plot(ep[m2], moving_average(series[m2], int(smooth)), color=color, lw=2.0,
                    label=L["smooth"].format(int(smooth)))
            a2.set_xlabel(L["episode"])
            a2.set_ylabel(ylab)
            a2.set_title(title)
            a2.legend(loc="best", framealpha=0.85)
            stem, ext = os.path.splitext(out_path)
            # 单图命名固定为 training_curve_<key>.<ext>，便于脚本/论文稳定引用
            files.append(_save(f2, os.path.join(os.path.dirname(stem),
                                                "training_curve_{}{}".format(key, ext)), dpi))
            plt.close(f2)

    fig.suptitle(L["training_title"], fontsize=13)
    fig.text(0.01, 0.005, L["source_note"].format(os.path.basename(train_log_path)),
             fontsize=7.5, alpha=0.7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    files.insert(0, _save(fig, out_path, dpi))
    plt.close(fig)
    return files


# ===========================================================================
# 3. 测试评估图（用户需求 2）
# ===========================================================================

def _normalize_series_by_tag(series_by_tag: "Dict[str, Dict[str, np.ndarray]]"
                             ) -> "List[Tuple[str, Dict[str, np.ndarray]]]":
    out: List[Tuple[str, Dict[str, np.ndarray]]] = []
    for tag, s in series_by_tag.items():
        if s is None or "queue" not in s:
            continue
        out.append((tag, s))
    if not out:
        raise ValueError("[绘图数据缺失] 没有任何可用的逐秒序列（series npz）。")
    return out


def plot_eval_timeseries(series_by_tag: "Dict[str, Dict[str, np.ndarray]]", out_path: str,
                         smooth: int = 15, dpi: int = 150, lang: str = "auto",
                         figsize: Tuple[float, float] = (12.5, 8.0),
                         source_note: Optional[str] = None) -> str:
    """时间序列折线图：延误 / 排队长度随仿真步数变化（各 tag 叠加，细线为真值 + 粗线为滑动平均）。"""
    plt, cjk, _ = setup_matplotlib(lang, dpi)
    L = labels_for(cjk)
    items = _normalize_series_by_tag(series_by_tag)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    specs = [(axes[0], "delay", L["ts_delay"], "tab:red"),
             (axes[1], "queue", L["ts_queue"], "tab:blue")]
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for ax, key, title, _c in specs:
        for i, (tag, s) in enumerate(items):
            if key not in s:
                continue
            y = np.asarray(s[key], dtype=np.float64)
            x = np.asarray(s.get("t", np.arange(1, len(y) + 1, dtype=np.float64)))
            col = colors[i % len(colors)]
            ax.plot(x, y, color=col, alpha=0.25, lw=0.9)
            if int(smooth) > 1:
                ax.plot(x, moving_average(y, int(smooth)), color=col, lw=2.0,
                        label="{}（{} {}）".format(tag, L["smooth"].format(int(smooth)), "")
                        if cjk else "{} ({})".format(tag, L["smooth"].format(int(smooth))))
            else:
                ax.plot(x, y, color=col, lw=1.6, label=tag)
        ax.set_ylabel(title)
        ax.legend(loc="best", framealpha=0.85)
    axes[1].set_xlabel(L["x_time"])
    fig.suptitle(L["ts_title"], fontsize=13)
    if source_note:
        fig.text(0.01, 0.005, L["source_note"].format(source_note), fontsize=7.5, alpha=0.7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _save(fig, out_path, dpi)
    plt.close(fig)
    return out_path


def plot_eval_boxplot(series_by_tag: "Dict[str, Dict[str, np.ndarray]]", out_path: str,
                      dpi: int = 150, lang: str = "auto",
                      figsize: Tuple[float, float] = (12.5, 5.6),
                      source_note: Optional[str] = None) -> str:
    """
    箱线图：左 = 逐秒排队长度分布；右 = 各交叉口平均排队长度分布（per-intersection AQL）。

    箱体为真实分位数（matplotlib 默认 Q1/中位数/Q3 + 1.5IQR 须线），均值另以三角标记。
    """
    plt, cjk, _ = setup_matplotlib(lang, dpi)
    L = labels_for(cjk)
    items = _normalize_series_by_tag(series_by_tag)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左：逐秒排队长度
    data_a = [np.asarray(s["queue"], dtype=np.float64) for _, s in items]
    names = [t for t, _ in items]
    pos = np.arange(1, len(data_a) + 1)
    bp = axes[0].boxplot(data_a, positions=pos, widths=0.5, patch_artist=True,
                         showmeans=True)
    for patch, c in zip(bp["boxes"], ["tab:red", "tab:blue", "tab:green", "tab:orange"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(names, fontsize=8, rotation=8)
    axes[0].set_ylabel(L["y_queue"])
    axes[0].set_title(L["box_a"])

    # 右：各交叉口平均排队长度（从 per_inter__<iid> 序列取时间平均）
    data_b, names_b = [], []
    for tag, s in items:
        per = [k for k in s.keys() if k.startswith("per_inter__")]
        if not per:
            continue
        vals = [float(np.mean(np.asarray(s[k], dtype=np.float64))) for k in per]
        data_b.append(vals)
        names_b.append(tag)
    if data_b:
        pos_b = np.arange(1, len(data_b) + 1)
        bp2 = axes[1].boxplot(data_b, positions=pos_b, widths=0.5, patch_artist=True,
                              showmeans=True)
        for patch, c in zip(bp2["boxes"], ["tab:red", "tab:blue", "tab:green", "tab:orange"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        axes[1].set_xticks(pos_b)
        axes[1].set_xticklabels(names_b, fontsize=8, rotation=8)
        axes[1].set_title(L["box_b"])
    else:
        axes[1].set_visible(False)
    axes[1].set_ylabel(L["y_queue"])

    fig.suptitle(L["box_title"], fontsize=13)
    if source_note:
        fig.text(0.01, 0.005, L["source_note"].format(source_note), fontsize=7.5, alpha=0.7)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    _save(fig, out_path, dpi)
    plt.close(fig)
    return out_path


def plot_eval_bar(metrics_by_tag: "Dict[str, Dict[str, float]]", out_path: str,
                  dpi: int = 150, lang: str = "auto",
                  figsize: Tuple[float, float] = (12.5, 7.0),
                  source_note: Optional[str] = None) -> str:
    """
    柱状图：ATT / AWT / AQL / 吞吐量 四个面板对比（含相对第一组基线的改善率标注）。
    metrics_by_tag 中的数值必须来自真实评测（run_report.json / eval_log.csv）。
    """
    plt, cjk, _ = setup_matplotlib(lang, dpi)
    L = labels_for(cjk)
    tags = list(metrics_by_tag.keys())
    specs = [("ATT", "ATT 平均旅行时间（s）" if cjk else "ATT (s)", "%.1f"),
             ("AWT", "AWT 平均等待时间（s）" if cjk else "AWT (s)", "%.1f"),
             ("AQL", "AQL 平均排队长度（辆）" if cjk else "AQL (veh)", "%.2f"),
             ("throughput", "吞吐量（完成行程车辆数）" if cjk else "Throughput (trips)", "%.0f")]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for ax, (key, title, fmt) in zip(axes.ravel(), specs):
        vals = [float(metrics_by_tag[t].get(key, float("nan"))) for t in tags]
        x = np.arange(len(tags))
        bars = ax.bar(x, vals, width=0.55,
                      color=[colors[i % len(colors)] for i in range(len(tags))],
                      alpha=0.75)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt % v, ha="center",
                        va="bottom", fontsize=9)
        base = vals[0] if vals else float("nan")
        for i, v in enumerate(vals[1:], start=1):
            if np.isfinite(v) and np.isfinite(base) and abs(base) > 1e-9:
                ax.text(i, v * 0.5, L["impr_vs"].format((base - v) / base * 100.0),
                        ha="center", va="center", fontsize=8, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0))
        ax.set_xticks(x)
        ax.set_xticklabels(tags, fontsize=8, rotation=8)
        ax.set_title(title)
    fig.suptitle(L["bar_title"], fontsize=13)
    if source_note:
        fig.text(0.01, 0.005, L["source_note"].format(source_note), fontsize=7.5, alpha=0.7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _save(fig, out_path, dpi)
    plt.close(fig)
    return out_path


def plot_eval_cdf(series_by_tag: "Dict[str, Dict[str, np.ndarray]]", out_path: str,
                  dpi: int = 150, lang: str = "auto",
                  figsize: Tuple[float, float] = (12.5, 5.6),
                  source_note: Optional[str] = None) -> str:
    """
    累积分布图 CDF（深度分析，可选）：左 = 逐秒排队长度 CDF，右 = 逐秒每车平均延误 CDF。
    CDF 由真实逐秒序列排序得到：F(x_k) = k / n。
    """
    plt, cjk, _ = setup_matplotlib(lang, dpi)
    L = labels_for(cjk)
    items = _normalize_series_by_tag(series_by_tag)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for ax, key, title, xlab in ((axes[0], "queue", L["cdf_a"], L["x_queue"]),
                                 (axes[1], "delay", L["cdf_b"], L["x_delay"])):
        for i, (tag, s) in enumerate(items):
            if key not in s:
                continue
            y = np.sort(np.asarray(s[key], dtype=np.float64))
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            p = np.arange(1, y.size + 1, dtype=np.float64) / float(y.size)
            col = colors[i % len(colors)]
            ax.plot(y, p, color=col, lw=1.8, label=tag)
            for q, ls in ((0.5, ":"), (0.9, "--")):
                xq = float(np.quantile(y, q))
                ax.axvline(xq, color=col, ls=ls, lw=0.9, alpha=0.6)
                ax.annotate("P{:.0f}={:.1f}".format(q * 100, xq), xy=(xq, q),
                            xytext=(xq, min(0.99, q + 0.06)), fontsize=7.5, color=col)
        ax.set_xlabel(xlab)
        ax.set_ylabel(L["cdf_y"])
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", framealpha=0.85)
    fig.suptitle(L["cdf_title"], fontsize=13)
    if source_note:
        fig.text(0.01, 0.005, L["source_note"].format(source_note), fontsize=7.5, alpha=0.7)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    _save(fig, out_path, dpi)
    plt.close(fig)
    return out_path


def plot_evaluation_report(series_by_tag: "Dict[str, Dict[str, np.ndarray]]",
                           metrics_by_tag: "Dict[str, Dict[str, float]]", out_dir: str,
                           prefix: str = "eval", fmt: str = "png", dpi: int = 150,
                           lang: str = "auto", smooth: int = 15,
                           source_note: Optional[str] = None,
                           with_cdf: bool = True) -> Dict[str, str]:
    """一次性生成测试评估的 4 类图（时序折线 / 箱线图 / 柱状图 / CDF）。"""
    ext = _fmt_ext(fmt)
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, str] = {}
    out["timeseries"] = plot_eval_timeseries(
        series_by_tag, os.path.join(out_dir, "{}_timeseries.{}".format(prefix, ext)),
        smooth=smooth, dpi=dpi, lang=lang, source_note=source_note)
    out["boxplot"] = plot_eval_boxplot(
        series_by_tag, os.path.join(out_dir, "{}_boxplot.{}".format(prefix, ext)),
        dpi=dpi, lang=lang, source_note=source_note)
    out["bar"] = plot_eval_bar(
        metrics_by_tag, os.path.join(out_dir, "{}_bar.{}".format(prefix, ext)),
        dpi=dpi, lang=lang, source_note=source_note)
    if with_cdf:
        out["cdf"] = plot_eval_cdf(
            series_by_tag, os.path.join(out_dir, "{}_cdf.{}".format(prefix, ext)),
            dpi=dpi, lang=lang, source_note=source_note)
    return out


# ===========================================================================
# 4. 一键生成全部图（训练收尾 / 离线重绘 共用）
# ===========================================================================

SERIES_BASELINE = "baseline_fixedtime_series.npz"


def series_dir_of(log_dir: str) -> str:
    return os.path.join(log_dir, "series")


def list_eval_series(log_dir: str) -> "Dict[int, str]":
    """扫描 output/logs/series/ 下的 eval_ep*_series.npz，返回 {episode: 路径}。"""
    d = series_dir_of(log_dir)
    out: Dict[int, str] = {}
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if name.startswith("eval_ep") and name.endswith("_series.npz"):
            try:
                ep = int(name[len("eval_ep"):-len("_series.npz")])
            except ValueError:
                continue
            out[ep] = os.path.join(d, name)
    return out


def build_eval_inputs(log_dir: str, report_dir: str, verbose: bool = True
                      ) -> "Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]], Optional[str], List[str]]":
    """
    汇集测试评估图所需输入：
      逐秒序列（基线 + 最优贪心评测）+ 对应指标 + 数据来源说明 + 缺失项列表。
    全部来自真实产物；缺哪一项就明确记录在 missing 中，不补造。
    """
    from eval_metric import load_series, per_intersection_ids  # noqa: F401  (接口自检)

    missing: List[str] = []
    sdir = series_dir_of(log_dir)
    series_by_tag: Dict[str, Dict[str, np.ndarray]] = {}
    metrics_by_tag: Dict[str, Dict[str, float]] = {}
    srcs: List[str] = []

    base_path = os.path.join(sdir, SERIES_BASELINE)
    if os.path.exists(base_path):
        series_by_tag[LBL_BASELINE_KEY] = load_series(base_path)
        srcs.append(os.path.join("logs", "series", SERIES_BASELINE))
    else:
        missing.append(base_path)

    evals = list_eval_series(log_dir)
    eval_csv = os.path.join(log_dir, "eval_log.csv")
    best_ep: Optional[int] = None
    ev_cols: Optional[Dict[str, np.ndarray]] = None
    if os.path.exists(eval_csv):
        try:
            ev_cols = load_metric_csv(eval_csv)
            att = ev_cols.get("ATT")
            if att is not None and np.isfinite(att).any():
                idx = int(np.nanargmin(np.where(np.isfinite(att), att, np.inf)))
                best_ep = int(ev_cols["episode"][idx])
                srcs.append(os.path.join("logs", "eval_log.csv"))
        except Exception as e:
            if verbose:
                print("[绘图警告] 读取 eval_log.csv 失败：{}".format(e))
    if evals:
        use_ep = best_ep if (best_ep is not None and best_ep in evals) else max(evals.keys())
        if best_ep is not None and use_ep != best_ep and verbose:
            print("[绘图提示] 最优评测 ep{} 的逐秒序列缺失，改用 ep{} 的真实序列。".format(best_ep, use_ep))
        series_by_tag[LBL_RL_KEY.format(use_ep)] = load_series(evals[use_ep])
        srcs.append(os.path.join("logs", "series", "eval_ep{}_series.npz".format(use_ep)))
        # 指标：优先取 eval_log.csv 中该 episode 的真实行
        if ev_cols is not None and "episode" in ev_cols:
            hit = np.where(ev_cols["episode"] == float(use_ep))[0]
            if hit.size > 0:
                row = {k: float(v[hit[0]]) for k, v in ev_cols.items()}
                metrics_by_tag[LBL_RL_KEY.format(use_ep)] = row
    else:
        missing.append(os.path.join(sdir, "eval_ep*_series.npz"))

    rep = load_run_report(os.path.join(report_dir, "run_report.json"))
    base_metrics = rep.get("fixed_time_baseline") or {}
    if base_metrics and LBL_BASELINE_KEY in series_by_tag:
        metrics_by_tag[LBL_BASELINE_KEY] = {k: float(v) for k, v in base_metrics.items()
                                            if isinstance(v, (int, float))}
        srcs.append(os.path.join("reports", "run_report.json"))
    elif not base_metrics:
        missing.append(os.path.join(report_dir, "run_report.json", "fixed_time_baseline"))

    # 基线指标缺失但序列存在时：用序列真值补 AQL（ATT/AWT/吞吐量只能来自 run_report.json）
    if LBL_BASELINE_KEY in series_by_tag and LBL_BASELINE_KEY not in metrics_by_tag:
        s = series_by_tag[LBL_BASELINE_KEY]
        metrics_by_tag[LBL_BASELINE_KEY] = {
            "AQL": float(np.mean(s["queue"])) if "queue" in s else float("nan"),
            "ATT": float(s["att"][-1]) if "att" in s and s["att"].size else float("nan"),
            "AWT": float("nan"), "throughput": float("nan"),
        }
    note = "、".join(sorted(set(srcs))) if srcs else None
    return series_by_tag, metrics_by_tag, note, missing


LBL_BASELINE_KEY = "__BASELINE__"      # 内部键：绘图时替换成中/英文标签
LBL_RL_KEY = "__RL_EP{}__"             # 内部键：Our RL（贪心评测 ep{N}）


def _localize_tag(tag: str, cjk: bool) -> str:
    L = labels_for(cjk)
    if tag == LBL_BASELINE_KEY:
        return L["tag_baseline"]
    if tag.startswith("__RL_EP") and tag.endswith("__"):
        return L["tag_rl"].format(tag[len("__RL_EP"):-2])
    return tag


def _localize_tags(d: Dict[str, Any], cjk: bool) -> Dict[str, Any]:
    return {_localize_tag(k, cjk): v for k, v in d.items()}


def generate_all(dirs: Optional[Dict[str, str]] = None, fmt: str = "png", dpi: int = 150,
                 smooth: int = 5, ts_smooth: int = 15, lang: str = "auto",
                 also_individual: bool = True, with_cdf: bool = True,
                 verbose: bool = True) -> Dict[str, Any]:
    """
    一键生成本项目的全部图（训练 4 曲线 + 评估 4 类图）并写出 figures_index.json。

    返回 {"figures": {...}, "sources": [...], "missing": [...], "font": str}
    """
    import config
    if dirs is None:
        dirs = config.ensure_output_dirs()
    log_dir, report_dir = dirs["log"], dirs["report"]
    fig_dir = os.path.join(dirs["output"], "figures")
    train_log = os.path.join(log_dir, "train_log.csv")
    if not os.path.exists(train_log):
        raise FileNotFoundError(
            "[绘图数据缺失] 未找到 {}：训练过程曲线必须来自真实训练日志，"
            "不生成占位图。请先运行 python train_rl_aux.py。".format(train_log))

    _, cjk, font = setup_matplotlib(lang, dpi)     # 先探测字体，供标签本地化
    figures: Dict[str, str] = {}
    files = plot_training_curves(
        train_log, os.path.join(fig_dir, "training_curves.{}".format(_fmt_ext(fmt))),
        eval_log_path=os.path.join(log_dir, "eval_log.csv"), smooth=smooth, dpi=dpi,
        lang=lang, also_individual=also_individual)
    figures["training_curves"] = files[0]
    for f in files[1:]:
        # 单图键名 = 文件名（training_curve_reward / _delay / _queue / _throughput）
        figures[os.path.splitext(os.path.basename(f))[0]] = f

    series_by_tag, metrics_by_tag, note, missing = build_eval_inputs(log_dir, report_dir, verbose)
    if series_by_tag:
        series_by_tag = _localize_tags(series_by_tag, cjk)
        metrics_by_tag = _localize_tags(metrics_by_tag, cjk)
        if len(series_by_tag) == 1 and missing and verbose:
            print("[绘图提示] 仅有一个数据源可对比，评估图仍按真实序列绘制；缺失：{}".format(missing))
        ev_figs = plot_evaluation_report(
            series_by_tag, metrics_by_tag, fig_dir, prefix="eval", fmt=fmt, dpi=dpi,
            lang=lang, smooth=ts_smooth, source_note=note, with_cdf=with_cdf)
        figures.update({("eval_" + k): v for k, v in ev_figs.items()})
    elif verbose:
        print("[绘图跳过] 缺少逐秒序列（{}），无法生成测试评估图；"
              "训练脚本会在基线与评测时自动保存 output/logs/series/*.npz。".format(missing))

    index = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paper_section": "第四章 4.2 节 融合邻居信息的动态交通信号控制强化学习算法（Our RL）",
        "font_used": font,
        "figure_format": _fmt_ext(fmt),
        "figures": figures,
        "data_sources": {"train_log": train_log,
                         "eval_log": os.path.join(log_dir, "eval_log.csv"),
                         "series_dir": series_dir_of(log_dir),
                         "run_report": os.path.join(report_dir, "run_report.json")},
        "note": "所有曲线均来自上述真实运行产物；本模块不生成任何虚构/占位数据。",
        "missing_sources": missing,
    }
    idx_path = os.path.join(fig_dir, "figures_index.json")
    os.makedirs(fig_dir, exist_ok=True)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    if verbose:
        print("[绘图完成] 共 {} 张图 → {}".format(len(figures), fig_dir))
        for k, v in figures.items():
            print("   {:<22} {}".format(k, v))
        if missing:
            print("[绘图缺失项] {}".format(missing))
    index["index_file"] = idx_path
    return index
