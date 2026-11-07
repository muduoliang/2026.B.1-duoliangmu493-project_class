# -*- coding: utf-8 -*-
"""
webster.py —— 韦伯斯特（Webster）信号配时计算（本次新增的应用层算法模块）

【重要说明】
    ★ 本文件不修改任何原有 RL 代码（RL-API/ 目录只读引用）。
      这里实现的是"配时方案怎么算出来"的经典交通工程公式，用于替代原先由 LLM 自由生成的方案；
      RL 模型仍然只负责读转换后的数据 + 施加方案 + 打分评价（rl_scorer.py）。

【采用的方法（韦伯斯特经典公式）】
    1. 各相位流量比      y_j = q_j / S_j
         q_j：该相位关键进口道的最大流率（veh/h，本模块由上传车流按小时折算得到）
         S_j：该进口道的饱和流量 = 单车道饱和流量 × 车道数
    2. 各相位流量比之和  Y = Σ y_j
    3. 总损失时间        L = Σ (黄灯时长 + 启动损失时间) = 相位数 × (3 + 3) 秒
    4. 最优周期          C0 = (1.5L + 5) / (1 - Y)          ← 韦伯斯特经典式
       周期取值范围限制在 [60, 120] 秒（题目要求）；Y ≥ 0.95 视为过饱和，直接取上限 120s
    5. 有效绿灯总时长    G_total = C - L
    6. 各相位绿灯        g_j = G_total × y_j / Y           （按流量比分配）
       附加最短绿灯保护：不足 10s 的相位抬到 10s，其余相位按比例让出（min_green 可关）
    7. 其余输出量        黄灯 = 3s（常量）、红灯 = C - g_j - 黄灯、绿信比 λ_j = g_j / C

【数据来源】
    车流：第③步转换后 data 目录里的 flow（已由 data_convert 校验过是原 RL 能读的格式）。
          逐条路线 route=[road0, road1, ...]，相邻两条路 (road_i → road_{i+1}) 就是一次
          "通过某个交叉口的转向动作"；用 roadnet 的 roadLink(phase) 表把它映射到具体相位槽。
    车道数：roadnet 里 roads[x]["lanes"] 的长度（决定饱和流量）。
"""
from __future__ import print_function

import data_convert
import rl_core

# ============================== 韦伯斯特法参数 ==============================
SATURATION_FLOW_PER_LANE = 1800.0   # 单车道饱和流量（veh/h/lane），韦伯斯特法常用取值
YELLOW_SECONDS = 3.0                # 每相位黄灯时长（s）
START_LOST_SECONDS = 3.0            # 每相位启动损失时间（s），计入总损失 L
CYCLE_MIN_SECONDS = 60.0            # 周期下界（s）
CYCLE_MAX_SECONDS = 120.0           # 周期上界（s）
MIN_GREEN_SECONDS = 5.0             # 最短绿灯保护（s）：工程下界，纯公式模式可设 0
SATURATED_Y = 0.95                  # Y ≥ 该值视为过饱和：C0 无意义，取周期上限


# ===========================================================================
# 1. 流量：flow 条目 → (交叉口, 相位槽, 进口道路) 的流率
# ===========================================================================

def _slot_tables(schema):
    """构建「(交叉口, 进口道路, 出口道路) → 相位槽」映射，以及每条道路的车道数。"""
    _points, roads, links = data_convert.build_tables(schema)
    lane_count = {}
    for rid, road in roads.items():
        try:
            lane_count[rid] = max(1, len(road.get("lanes") or []))
        except TypeError:
            lane_count[rid] = 1

    slot_of = {}
    for inter in schema.intersections:
        iid = inter.inter_id
        for slot, phase_index in enumerate(inter.selectable_phase_indices):
            phase = inter.phases[phase_index]
            for link_idx in phase.available_road_links:
                link = links.get(iid, {}).get(link_idx)
                if not link:
                    continue
                # 同一 (进口, 出口) 在多个相位里重复出现时，保留第一个（相位本身不应重叠）
                slot_of.setdefault((iid, link["start"], link["end"]), slot)
    return roads, lane_count, slot_of


def _collect_volumes(flow_json, schema, flow_stats=None):
    """
    把 flow 条目按「交叉口 + 相位槽 + 进口道路」累加车辆数，并折算成小时流率。

    返回 (volumes, info)
        volumes[iid][slot][road] = 小时流率 q（veh/h）
        info  : 车辆数/时间跨度/未映射动作数等审计信息
    """
    entries = data_convert.expand_entries(flow_json)
    roads, lane_count, slot_of = _slot_tables(schema)

    counts = {}          # (iid, slot, road) -> 车辆数
    stat = {"n_vehicles": 0, "movements": 0, "mapped_movements": 0,
            "t_min": None, "t_max": None}
    for entry in entries:
        route = entry.get("route") or []
        if not isinstance(route, list) or len(route) < 2:
            continue
        try:
            count = int(entry.get("_count", 1) or 1)
        except (TypeError, ValueError):
            count = 1
        start = _to_float(entry.get("startTime"), 0.0)
        end = _to_float(entry.get("endTime"), start)
        stat["n_vehicles"] += count
        stat["t_min"] = start if stat["t_min"] is None else min(stat["t_min"], start)
        stat["t_max"] = end if stat["t_max"] is None else max(stat["t_max"], end)

        for i in range(len(route) - 1):
            start_road, end_road = route[i], route[i + 1]
            road = roads.get(start_road)
            if not road:
                continue
            iid = road.get("endIntersection")
            slot = slot_of.get((iid, start_road, end_road))
            stat["movements"] += 1
            if slot is None:
                continue
            stat["mapped_movements"] += 1
            key = (iid, slot, start_road)
            counts[key] = counts.get(key, 0) + count

    # 时间跨度：优先用转换后的 flow 统计（整份上传数据），否则用本条目的时间范围
    span = None
    if flow_stats:
        t0, t1 = flow_stats.get("t_min"), flow_stats.get("t_max")
        if t0 is not None and t1 is not None and float(t1) > float(t0):
            span = float(t1) - float(t0)
    if not span and stat["t_min"] is not None and (stat["t_max"] - stat["t_min"]) > 0:
        span = stat["t_max"] - stat["t_min"]
    if not span or span <= 0:
        span = 3600.0
    stat["span_seconds"] = round(span, 1)
    stat["rate_factor"] = 3600.0 / span       # 换算成 veh/h 的系数

    volumes = {}
    for (iid, slot, road), count in counts.items():
        q = count * stat["rate_factor"]
        volumes.setdefault(iid, {}).setdefault(slot, {})[road] = q
    stat["mapped_ratio"] = (round(float(stat["mapped_movements"]) / stat["movements"], 3)
                            if stat["movements"] else 0.0)
    return volumes, stat


def _to_float(value, default=None):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


# ===========================================================================
# 2. 韦伯斯特核心计算
# ===========================================================================

def _allocate_green(green_total, ratios, min_green):
    """把有效绿灯总时长按流量比分配，并做最短绿灯保护（返回秒数列表）。"""
    n = len(ratios)
    if n == 0:
        return []
    total = max(0.0, green_total)
    if total <= 0:
        return [0.0] * n
    sum_y = sum(ratios)
    if sum_y <= 1e-9:                         # 没有任何流量信息 → 只能等分
        return [total / n] * n
    # 注：调用方已把 min_green 夹到 total/n 以内，因此 n*min_green <= total 恒成立，
    #     下面先按流量比等比分配，再把不足最短绿的相位抬上去（不会被等分兜底覆盖）。

    greens = [total * y / sum_y for y in ratios]
    for _ in range(n * 3):
        short = [i for i, g in enumerate(greens) if g < min_green - 1e-9]
        if not short:
            break
        need = sum(min_green - greens[i] for i in short)
        for i in short:
            greens[i] = min_green
        spare_idx = [i for i, g in enumerate(greens) if g > min_green + 1e-9]
        spare = sum(greens[i] - min_green for i in spare_idx)
        if spare <= 1e-9:
            break
        take = min(need, spare)
        for i in spare_idx:
            share = (greens[i] - min_green) / spare
            greens[i] -= take * share
    return greens


def _short_flows(flows, limit=4):
    """把相位放行的中文流向压短：直行 → 左转 → 右转，最多 limit 条。"""
    if not flows:
        return []

    def rank(text):
        if "右转" in text:
            return 2
        if "左转" in text:
            return 1
        return 0

    return sorted(flows, key=rank)[:limit]


def compute_intersection(row, volumes_for_inter, lane_count, labels, params=None):
    """
    单个交叉口的韦伯斯特配时。

    row             : data_convert.build_plan_context() 里的一个路口（含相位槽与中文流向）
    volumes_for_inter: {slot: {road: q}}
    lane_count      : {road: 车道数}
    labels          : {slot: [中文流向, ...]}
    params          : 计算参数（饱和流量/黄灯/损失/周期上下界），默认取本模块常量
    返回该路口的完整配时明细（相位方案 / 周期 / 绿黄红 / 绿信比）。
    """
    params = params or {}
    sat_flow = float(params.get("saturation_flow", SATURATION_FLOW_PER_LANE))
    yellow = float(params.get("yellow_seconds", YELLOW_SECONDS))
    start_lost = float(params.get("start_lost_seconds", START_LOST_SECONDS))
    c_min = float(params.get("cycle_min", CYCLE_MIN_SECONDS))
    c_max = float(params.get("cycle_max", CYCLE_MAX_SECONDS))
    saturated_y = float(params.get("saturated_y", SATURATED_Y))
    min_green_cfg = float(params.get("min_green", MIN_GREEN_SECONDS))

    slots = row.get("slots") or []
    detail = []
    for item in slots:
        slot = item["slot"]
        approaches = volumes_for_inter.get(slot) or {}
        critical = None
        for road, q in approaches.items():
            lanes = lane_count.get(road, 1)
            saturation = sat_flow * lanes
            ratio = q / saturation if saturation > 0 else 0.0
            cand = {"road": road, "q": q, "lanes": lanes,
                    "saturation": saturation, "y": ratio}
            if critical is None or cand["y"] > critical["y"]:
                critical = cand
        detail.append({
            "slot": slot,
            "light_phase": item.get("light_phase"),
            "flows": _short_flows(labels.get(slot) or item.get("flows") or []),
            "critical_road": (critical or {}).get("road"),
            "volume": round((critical or {}).get("q", 0.0), 1),
            "lanes": (critical or {}).get("lanes", 0),
            "saturation": round((critical or {}).get("saturation", 0.0), 1),
            "y": round((critical or {}).get("y", 0.0), 4),
        })

    # 相位方案本身是交叉口的固定配置：即使某相位当前没有需求也保留在方案里
    # （流量比为 0 → 分不到绿灯，只拿最短绿灯保护），这样输出不会被"删相位"破坏。
    no_demand = not any(p["y"] > 0 for p in detail)
    n_phase = len(detail)
    big_y = sum(p["y"] for p in detail)
    lost_per_phase = yellow + start_lost
    total_lost = round(n_phase * lost_per_phase, 1)

    saturated = big_y >= saturated_y
    if saturated or big_y <= 1e-9:
        c0_raw = None if saturated else 0.0
        cycle = c_max if saturated else c_min
    else:
        c0_raw = (1.5 * total_lost + 5.0) / (1.0 - big_y)
        cycle = c0_raw
    clamped = None
    if cycle < c_min:
        cycle, clamped = c_min, "min"
    elif cycle > c_max:
        cycle, clamped = c_max, "max"
    if saturated:
        clamped = "saturated"

    green_total = cycle - total_lost
    min_green = min(min_green_cfg, green_total / n_phase) if n_phase else 0.0
    greens = _allocate_green(green_total, [p["y"] for p in detail], min_green)

    phases = []
    for item, green in zip(detail, greens):
        green = round(green, 1)
        red = round(cycle - green - yellow, 1)
        phases.append({
            "slot": item["slot"],
            "light_phase": item["light_phase"],
            "flows": item["flows"],
            "critical_road": item["critical_road"],
            "volume": item["volume"],
            "lanes": item["lanes"],
            "saturation": item["saturation"],
            "y": item["y"],
            "green": green,
            "yellow": yellow,
            "red": red,
            "green_ratio": round(green / cycle, 3) if cycle > 0 else 0.0,
        })

    return {
        "intersection_id": row.get("intersection_id"),
        "cycle_seconds": round(cycle, 1),
        "c0_seconds": round(c0_raw, 1) if c0_raw is not None else None,
        "clamped": clamped,
        "Y": round(big_y, 4),
        "L_seconds": total_lost,
        "n_phases": n_phase,
        "green_total": round(green_total, 1),
        "min_green_used": round(min_green, 1),
        "no_demand": no_demand,
        "phases": phases,
    }


def compute(schema, flow_json, flow_stats=None, options=None):
    """
    全路网韦伯斯特配时计算（对外主入口）。

    返回 dict：
        method / params / demand / intersections[] / summary / plan / warnings
        plan 是可直接交给 rl_scorer 执行的方案（{"intersections": {iid: [{phase_slot,duration}]}}）
    """
    options = options or {}
    params = {
        "saturation_flow": float(options.get("saturation_flow") or SATURATION_FLOW_PER_LANE),
        "yellow_seconds": float(options.get("yellow_seconds") or YELLOW_SECONDS),
        "start_lost_seconds": float(options.get("start_lost_seconds") or START_LOST_SECONDS),
        "cycle_min": float(options.get("cycle_min") or CYCLE_MIN_SECONDS),
        "cycle_max": float(options.get("cycle_max") or CYCLE_MAX_SECONDS),
        "min_green": float(options.get("min_green") if options.get("min_green") is not None
                           else MIN_GREEN_SECONDS),
        "saturated_y": SATURATED_Y,
    }
    sat = params["saturation_flow"]
    yellow = params["yellow_seconds"]
    start_lost = params["start_lost_seconds"]
    c_min = params["cycle_min"]
    c_max = params["cycle_max"]

    volumes, demand = _collect_volumes(flow_json, schema, flow_stats)
    context = data_convert.build_plan_context(schema, max_intersections=1000)
    labels = {}
    for row in context["intersections"]:
        labels[row["intersection_id"]] = dict(
            (s["slot"], s["flows"]) for s in row["slots"])

    _points, roads, _links = data_convert.build_tables(schema)
    lane_count = dict((rid, max(1, len(road.get("lanes") or []))) for rid, road in roads.items())

    rows, warnings = [], []
    for row in context["intersections"]:
        iid = row["intersection_id"]
        rows.append(compute_intersection(row, volumes.get(iid) or {}, lane_count,
                                        labels.get(iid) or {}, params))
    if demand["mapped_movements"] == 0:
        warnings.append("车流里没有可映射到相位的转向动作，各路口相位按等分绿灯处理")

    cycles = [r["cycle_seconds"] for r in rows] or [0.0]
    clamped_min = [r for r in rows if r["clamped"] == "min"]
    clamped_max = [r for r in rows if r["clamped"] == "max"]
    saturated = [r for r in rows if r["clamped"] == "saturated"]

    plan = {"intersections": {}, "notes": "韦伯斯特法：周期 {}~{}s，绿灯按流量比分配".format(
        int(c_min), int(c_max))}
    for row in rows:
        plan["intersections"][row["intersection_id"]] = [
            {"phase_slot": p["slot"], "duration": max(1, int(round(p["green"])))}
            for p in row["phases"]]

    return {
        "method": "webster",
        "params": {
            "saturation_flow_per_lane": sat,
            "yellow_seconds": yellow,
            "start_lost_seconds": start_lost,
            "lost_per_phase": round(yellow + start_lost, 1),
            "cycle_min": c_min,
            "cycle_max": c_max,
            "min_green": MIN_GREEN_SECONDS,
            "formula": "C0 = (1.5L + 5) / (1 - Y)",
        },
        "demand": demand,
        "intersections": rows,
        "summary": {
            "n_intersections": len(rows),
            "cycle_min": round(min(cycles), 1),
            "cycle_max": round(max(cycles), 1),
            "cycle_mean": round(sum(cycles) / len(cycles), 1),
            "n_clamped_min": len(clamped_min),
            "n_clamped_max": len(clamped_max),
            "n_saturated": len(saturated),
            "Y_min": round(min([r["Y"] for r in rows] or [0.0]), 4),
            "Y_max": round(max([r["Y"] for r in rows] or [0.0]), 4),
        },
        "plan": plan,
        "warnings": warnings,
    }


# ===========================================================================
# 3. 纯文本摘要（给日志/复制用，保持简短）
# ===========================================================================

def brief(row):
    """单路口一行的简短描述：周期 + 各相位绿/黄/红。"""
    parts = []
    for p in row["phases"]:
        parts.append("相位{} G{:.1f}/Y{:.1f}/R{:.1f}".format(
            p["light_phase"], p["green"], p["yellow"], p["red"]))
    return "{}｜周期 {:.1f}s｜Y={:.3f}｜{}".format(
        row["intersection_id"], row["cycle_seconds"], row["Y"], " ".join(parts))
