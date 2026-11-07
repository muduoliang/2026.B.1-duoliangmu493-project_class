# -*- coding: utf-8 -*-
"""
rl_scorer.py —— 第④步：把「转换后的数据 + LLM 配时方案」喂进原有 RL 环境，做打分与效果评价

本文件只做「应用层编排」，不含也不修改任何 RL 算法：
    * RL 基线臂：直接调用原有 rl_aux_infer.FrozenRLAuxPolicy（掩码 argmax 双输出），一行未改；
    * 方案臂   ：把 LLM 方案解读成 (phase_slot, duration_index) 动作序列，喂给原有
                 env_cityflow.CityFlowTrafficEnv.step()；
    * 打分     ：只对两个臂的 episode 指标与冻结网络的 Q 值做归一化对照（口径见 SCORING_NOTE）。
"""
from __future__ import print_function

import math
import time

import rl_core
from offline_engine import make_engine

# 打分口径（应用层定义，不影响 RL 核心计算）：
#   score_sim : 仿真指标分。ATT / AWT / AQL 三个指标「越小越好」，对每个指标取
#               100 * m_rl / (m_plan + m_rl)：与 RL 基线持平时 = 50，方案更优则 > 50。
#   score_q   : 动作质量分。用冻结网络对方案实际执行的动作算归一化 regret
#               regret = (Q_best - Q_plan) / (Q_best - Q_worst)，score_q = 100 * (1 - mean(regret))。
#               Q 值区间很小（实测 gap≈0.02），所以必须按本状态下的 valid 区间归一化。
#   score_total = 0.6 * score_sim + 0.4 * score_q
SCORING_NOTE = (
    "score_sim=仿真指标(ATT/AWT/AQL)相对 RL 基线的归一化对照；"
    "score_q=冻结网络对方案动作的归一化 regret；total=0.6*sim+0.4*q。"
    "注①：原有 env_cityflow._sample_metrics 把同一 queue_sum 同时计入排队与等待，"
    "故 AWT 与 AQL 的份额通常相同（属原有口径，未改动）。"
    "注②：仿真时长短到没有车辆完成行程时，ATT 不计入（避免 0/0 拉偏分数）。"
)
SIM_METRICS = (("ATT", "平均旅行时间", "s"), ("AWT", "平均等待时间", "s"), ("AQL", "平均排队长度", "辆"))


# ===========================================================================
# 1. 方案规范化：把 LLM 输出变成可执行动作（不改数据结构，只做校验/夹取）
# ===========================================================================

def normalize_plan(plan, schema):
    """
    支持两种 LLM 输出形态：
        {"intersections": {"intersection_1_1": [{"phase_slot":1,"duration":30}, ...]}, "default_cycle": [...]}
        {"default_cycle": [{"phase_slot":2,"duration":25}, ...]}        （对所有交叉口统一配时）
    返回 (norm, warnings)：norm[iid] = [{"phase_slot","duration","duration_index","duration_seconds"}, ...]
    """
    cfg = rl_core.core("config")
    durations = [int(x) for x in cfg.DURATION_OPTIONS]
    warnings = []

    by_inter = {}
    if isinstance(plan, dict):
        raw = plan.get("intersections")
        if isinstance(raw, dict):
            by_inter = raw
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and item.get("intersection_id"):
                    by_inter[item["intersection_id"]] = item
        default_cycle = plan.get("default_cycle") or plan.get("cycle")
    else:
        default_cycle = None

    norm = {}
    for inter in schema.intersections:
        iid = inter.inter_id
        n_slots = len(inter.selectable_phase_indices)
        entry = by_inter.get(iid)
        cycle_raw = None
        if isinstance(entry, dict):
            cycle_raw = entry.get("cycle") or entry.get("phases") or entry.get("steps")
            if cycle_raw is None and ("phase_slot" in entry or "phase" in entry):
                cycle_raw = [entry]
        elif isinstance(entry, list):
            cycle_raw = entry
        if not cycle_raw:
            cycle_raw = default_cycle
        if not cycle_raw:
            continue

        cycle = []
        for step in cycle_raw:
            if not isinstance(step, dict):
                continue
            slot = step.get("phase_slot", step.get("phase", step.get("slot", 0)))
            try:
                slot = int(round(float(slot)))
            except (TypeError, ValueError):
                slot = 0
            if slot < 0 or slot >= n_slots:
                warnings.append("{} 的相位槽 {} 超出可选范围 0~{}，已夹取".format(iid, slot, n_slots - 1))
                slot = max(0, min(n_slots - 1, slot))
            dur_sec = step.get("duration", step.get("green", step.get("seconds")))
            dur_idx = step.get("duration_index")
            if dur_idx is not None:
                try:
                    dur_idx = int(round(float(dur_idx)))
                except (TypeError, ValueError):
                    dur_idx = None
            if dur_idx is None or dur_idx < 0 or dur_idx >= len(durations):
                try:
                    dur_sec = float(dur_sec)
                except (TypeError, ValueError):
                    dur_sec = durations[len(durations) // 2]
                dur_idx = min(range(len(durations)), key=lambda k: abs(durations[k] - dur_sec))
            cycle.append({"phase_slot": slot, "duration": durations[dur_idx],
                          "duration_index": dur_idx, "duration_seconds": durations[dur_idx]})
        if cycle:
            norm[iid] = cycle

    if not norm:
        warnings.append("LLM 未给出可执行配时，方案臂使用内置相位文件的顺序周期（仅用于对照）")
    return norm, warnings


def fallback_plan(schema):
    """兜底方案：用原有 signal_inter_*.txt 的前两个相位各 30s 组成周期（仅作对照，不改原文件）。"""
    out = {}
    for inter in schema.intersections:
        choices = list(inter.selectable_phase_indices)
        if not choices:
            continue
        out[inter.inter_id] = [
            {"phase_slot": 0, "duration": 30, "duration_index": _nearest_index(30), "duration_seconds": 30},
            {"phase_slot": min(1, len(choices) - 1), "duration": 30,
             "duration_index": _nearest_index(30), "duration_seconds": 30},
        ]
    return out


def _nearest_index(seconds):
    durations = [int(x) for x in rl_core.core("config").DURATION_OPTIONS]
    return min(range(len(durations)), key=lambda k: abs(durations[k] - seconds))


# ===========================================================================
# 2. 两个评估臂
# ===========================================================================

def run_rl_arm(schema, data_dir, episode_seconds, seed, engine_factory):
    """RL 基线臂：直接复用原有 FrozenRLAuxPolicy 的逐决策推理循环（未改一行）。"""
    aux = rl_core.core("rl_aux_infer")
    t0 = time.time()
    policy = aux.FrozenRLAuxPolicy(frozen_path=rl_core.frozen_path(), data_dir=data_dir,
                                   seed=seed, verbose=False)
    try:
        policy.create_env(engine=engine_factory(), episode_seconds=episode_seconds, verbose=False)
        policy.reset(seed)
        max_steps = int(episode_seconds / 10.0) + 200
        for _ in range(max_steps):
            res = policy.advance(1)
            if res.get("done"):
                break
        metrics = policy.metrics()
        status = policy.status()
    finally:
        policy.close()
    return {
        "metrics": _pick_metrics(metrics),
        "raw": metrics,
        "n_decisions": int(status.get("n_decisions") or 0),
        "sim_time": float(status.get("sim_time") or 0.0),
        "wall_seconds": round(time.time() - t0, 2),
    }


def run_plan_arm(schema, flow_json, engine_factory_kind, plan_norm, episode_seconds, seed,
                 model=None, progress=None):
    """
    方案臂：按 LLM 方案顺序循环施加 (phase_slot, duration_index)。
    同时在每个决策点用冻结网络给「方案实际选的动作」算 Q，用于 score_q。
    """
    cfg = rl_core.core("config")
    durations = [int(x) for x in cfg.DURATION_OPTIONS]
    engine, kind = engine_factory_kind()
    env = rl_core.make_env(schema, engine=engine, episode_seconds=episode_seconds, seed=seed,
                           verbose=False)
    t0 = time.time()
    pointer = dict((iid, 0) for iid in plan_norm)
    regrets = []
    n_decisions = 0
    max_steps = int(episode_seconds / min(durations)) + 200
    try:
        env.reset()
        for _ in range(max_steps):
            batch = env.decision_batch()
            pending = list(batch["inter_ids"])
            if not pending:
                break
            phase_actions, dur_actions, slots_used = {}, {}, {}
            for iid in pending:
                cycle = plan_norm.get(iid)
                if not cycle:
                    cycle = fallback_plan(schema).get(iid)
                step = cycle[pointer.get(iid, 0) % len(cycle)]
                pointer[iid] = pointer.get(iid, 0) + 1
                phase_actions[iid] = int(step["phase_slot"])
                dur_actions[iid] = int(step["duration_index"])
                slots_used[iid] = int(step["phase_slot"])
            if model is not None:
                regrets.extend(_q_regrets(model, batch, pending, slots_used))
            _, _, _, done, info = env.step(phase_actions, dur_actions)
            n_decisions += len(pending)
            if progress is not None and n_decisions % 40 == 0:
                progress(round(float(env.sim_time), 1), episode_seconds)
            if done:
                break
        metrics = env.episode_metrics()
        sim_time = float(env.sim_time)
    finally:
        env.close()

    mean_regret = sum(regrets) / float(len(regrets)) if regrets else None
    return {
        "metrics": _pick_metrics(metrics),
        "raw": metrics,
        "n_decisions": n_decisions,
        "sim_time": sim_time,
        "wall_seconds": round(time.time() - t0, 2),
        "engine_kind": kind,
        "q": {"n_eval": len(regrets), "mean_regret": mean_regret},
    }


def _q_regrets(model, batch, pending, slots_used):
    """用冻结网络（原有 DQNNetwork，未改动）评估方案动作的相对质量。"""
    out = model.forward(batch)
    phase_q = out["phase_q"]                       # [n, n_slots]
    avail = list(batch["phase_avail"])
    row_of = dict((iid, i) for i, iid in enumerate(batch["inter_ids"]))
    result = []
    for iid in pending:
        row = row_of.get(iid)
        if row is None:
            continue
        n_valid = int(avail[row])
        if n_valid <= 0:
            continue
        values = [float(phase_q[row][k]) for k in range(n_valid)]
        q_best, q_worst = max(values), min(values)
        slot = slots_used.get(iid, 0)
        q_plan = values[slot] if 0 <= slot < n_valid else q_worst
        denom = q_best - q_worst
        if denom <= 1e-9:
            result.append(0.0 if abs(q_best - q_plan) <= 1e-9 else 1.0)
        else:
            result.append(max(0.0, min(1.0, (q_best - q_plan) / denom)))
    return result


# ===========================================================================
# 3. 打分与结论
# ===========================================================================

def _pick_metrics(raw):
    out = {}
    for key in ("ATT", "AWT", "AQL", "throughput", "n_total_vehicles", "episode_seconds",
                "n_samples", "waited_vehicle_seconds"):
        value = raw.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        if value is not None and (math.isnan(value) or math.isinf(value)):
            value = None
        out[key] = round(value, 3) if value is not None else None
    per_inter = raw.get("per_intersection_AQL") or {}
    out["per_intersection_AQL"] = dict((k, round(float(v), 3)) for k, v in per_inter.items())
    # ATT 只在有车辆真正完成行程时才有意义（短演示窗口里常常为 0）
    throughput = out.get("throughput") or 0.0
    out["att_valid"] = bool(throughput > 0 and (out.get("ATT") or 0) > 0)
    return out


def score_arms(plan_metrics, rl_metrics, q_mean_regret):
    """按 SCORING_NOTE 的口径算三个分数（全部有界 0~100）。

    detail 只保留「指标名 / 方案值 / 基线值 / 相对差异」四项，不再输出份额与说明字段。
    不计入的指标把方案值与基线值置为 None（前端显示「—」）。
    """
    parts, detail = [], {}
    for key, label, unit in SIM_METRICS:
        p, r = plan_metrics.get(key), rl_metrics.get(key)
        skip = None
        if key == "ATT" and not (plan_metrics.get("att_valid") and rl_metrics.get("att_valid")):
            skip = True
        if skip is None and (p is None or r is None or (p + r) <= 1e-9):
            skip = True
        if skip is not None:
            detail[key] = {"plan": None, "rl": None, "label": label, "unit": unit}
            continue
        share = 100.0 * r / (p + r)               # 内部仍用它算仿真分（50 = 与基线持平）
        parts.append(share)
        detail[key] = {"plan": p, "rl": r, "label": label, "unit": unit,
                       "diff_pct": round(100.0 * (p - r) / r, 1) if r > 1e-9 else None}
    score_sim = round(sum(parts) / len(parts), 1) if parts else 50.0

    if q_mean_regret is None:
        score_q = 50.0
        q_note = "未取到 Q 值样本，按中性 50 计"
    else:
        score_q = round(100.0 * (1.0 - q_mean_regret), 1)
        q_note = "冻结网络对方案动作的平均 regret={:.3f}".format(q_mean_regret)
    score_total = round(0.6 * score_sim + 0.4 * score_q, 1)
    return {"sim": score_sim, "q": score_q, "total": score_total,
            "detail": detail, "q_note": q_note,
            "weights": {"sim": 0.6, "q": 0.4}}


def build_verdict(score, plan_metrics, rl_metrics, q_stats):
    """生成一句简短中文评价（前端直接展示，不做长篇分析）。"""
    sim = score["sim"]
    if sim >= 55:
        head = "方案整体优于 RL 基线"
    elif sim >= 48:
        head = "方案与 RL 基线基本持平"
    else:
        head = "方案整体弱于 RL 基线"

    diffs = []
    for key, label, unit in SIM_METRICS:
        item = score["detail"].get(key) or {}
        diff = item.get("diff_pct")
        if diff is None:
            continue
        diffs.append((abs(diff), "{} {}{}%".format(label.replace("平均", ""), ("+" if diff > 0 else ""), diff)))
    diffs.sort(reverse=True)
    tail = "；".join(x[1] for x in diffs[:2])
    q_part = ""
    if q_stats and q_stats.get("mean_regret") is not None:
        q_part = "，动作质量分 {}".format(score["q"])
    return "{}（{}）{}".format(head, tail or "指标无差异", q_part)


def worst_intersections(plan_metrics, rl_metrics, top=3):
    plan_aql = plan_metrics.get("per_intersection_AQL") or {}
    rl_aql = rl_metrics.get("per_intersection_AQL") or {}
    ids = set(plan_aql) | set(rl_aql)
    rows = []
    for iid in ids:
        p = plan_aql.get(iid)
        rows.append({"intersection_id": iid, "aql_plan": p, "aql_rl": rl_aql.get(iid)})
    rows.sort(key=lambda r: -(r["aql_plan"] if r["aql_plan"] is not None else -1))
    return rows[:top]


# ===========================================================================
# 4. 对外主入口
# ===========================================================================

def evaluate(schema, data_dir, flow_json, plan, episode_seconds, seed=2024,
             with_q=True, progress=None):
    """跑完两个臂并返回完整评估报告（pipeline 第④步直接调用）。"""
    plan_norm, warnings = normalize_plan(plan, schema)
    if not plan_norm:
        plan_norm = fallback_plan(schema)
    warnings = list(warnings)

    def engine_factory():
        return make_engine(schema, flow_json)[0]

    model = None
    if with_q:
        try:
            model = rl_core.load_frozen(schema)
        except Exception as e:            # noqa: BLE001 - Q 分可降级，不阻塞主流程
            warnings.append("冻结网络加载失败，跳过动作质量分：{}".format(str(e)[:120]))

    if progress is not None:
        progress("rl_arm_start", 0)
    rl_arm = run_rl_arm(schema, data_dir, episode_seconds, seed, engine_factory)
    if progress is not None:
        progress("rl_arm_done", 1)

    if progress is not None:
        progress("plan_arm_start", 0)
    plan_arm = run_plan_arm(schema, flow_json, lambda: make_engine(schema, flow_json),
                            plan_norm, episode_seconds, seed,
                            model=model,
                            progress=(lambda t, total: progress("plan_arm_progress", t / max(total, 1.0)))
                            if progress is not None else None)
    if progress is not None:
        progress("plan_arm_done", 1)

    q_stats = plan_arm.pop("q", None)
    score = score_arms(plan_arm["metrics"], rl_arm["metrics"],
                       (q_stats or {}).get("mean_regret"))
    verdict = build_verdict(score, plan_arm["metrics"], rl_arm["metrics"], q_stats)

    plan_arm.pop("raw", None)
    rl_raw = rl_arm.pop("raw", None)
    return {
        "episode_seconds": float(episode_seconds),
        "engine_kind": plan_arm.get("engine_kind"),
        "plan_arm": plan_arm,
        "rl_arm": rl_arm,
        "rl_raw_metrics": _pick_metrics(rl_raw or {}),
        "score": score,
        "q_stats": q_stats,
        "verdict": verdict,
        "worst_intersections": worst_intersections(plan_arm["metrics"], rl_arm["metrics"]),
        "plan_used": _summarize_plan(plan_norm, schema),
        "warnings": warnings,
        "scoring_note": SCORING_NOTE,
        "seed": seed,
    }


def _summarize_plan(plan_norm, schema=None):
    """
    把实际执行的方案压成前端易读的短结构（每口只给周期与相位序列）。
    light_phase 由 schema.selectable_phase_indices 反查（动作槽 → roadnet 相位下标）。
    """
    slot_to_phase = {}
    if schema is not None:
        for inter in schema.intersections:
            slot_to_phase[inter.inter_id] = list(inter.selectable_phase_indices)
    out = {}
    for iid, cycle in plan_norm.items():
        choices = slot_to_phase.get(iid) or []
        phases = []
        for s in cycle:
            slot = int(s["phase_slot"])
            light = int(choices[slot]) if 0 <= slot < len(choices) else None
            phases.append({"phase_slot": slot, "duration": int(s["duration"]),
                           "light_phase": light})
        out[iid] = {"cycle_seconds": sum(int(s["duration"]) for s in cycle), "phases": phases}
    return out
