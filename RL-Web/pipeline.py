# -*- coding: utf-8 -*-
"""
pipeline.py —— 五步链路编排（本文件是"胶水"，不含任何 RL 算法）

    ① 接收上传文件 → LLM 读取并解析内容（流式）
    ② 韦伯斯特法计算交通信号配时方案（webster.py，确定性公式；options["plan_source"]="llm" 可切回旧行为）
    ③ 上传的 json/csv → RL 模型可直接读取的输入格式（data_convert）
    ④ RL 模型读取转换后数据 + 配时方案 → 打分 / 效果评价（rl_scorer）
    ⑤ 输出精简优化建议（流式）

说明：第③步的转换在内部其实是先算的——因为第②步的流量比、相位槽与时长档位都要基于
真实可执行的动作空间，否则方案无法执行。但对页面展示仍按 ①→②→③→④→⑤ 的顺序推送进度。
"""
from __future__ import print_function

import json
import os
import re
import time

import data_convert
import rl_core
import rl_scorer
import webster
from deepseek_client import DeepSeekClient, DeepSeekError, extract_json

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

PARSE_SYSTEM = (
    "你是交通数据分析助手。只根据给出的文件内容与结构摘要作答，"
    "不要编造数字；输出 3~5 行中文要点，每行不超过 30 字，不要客套话。"
)
PLAN_SYSTEM = (
    "你是城市交通信号配时智能体。只能使用给定的交叉口 id、相位槽与时长档位，"
    "严禁编造不存在的相位或时长；只输出 JSON，不要任何解释文字。"
)
ADVICE_SYSTEM = (
    "你是交通信号优化顾问。用中文给最多 3 条可执行建议，每条不超过 25 字，"
    "直接说改什么（哪个路口、相位顺序或绿时怎么调），不要讲原理、不要客套。"
)


class PipelineError(RuntimeError):
    pass


# ===========================================================================
# 主入口
# ===========================================================================

def run_pipeline(files, api_key=None, model=None, base_url=None, options=None,
                 emit=None, llm=None):
    """
    参数
        files   : [{'name': str, 'content': str}] 上传文件（文本）
        api_key : DeepSeek Key（None 或空 = 本地无 LLM 模式，用内置对照方案）
        llm     : 可注入的 LLM 客户端（自检用），需实现 chat() 与 stream_chat()
        emit    : callable(event dict)，后端据此写 SSE
    返回：完整结果 dict（同时也会以 {'type': 'result'} 事件推给前端）
    """
    emit = emit or (lambda event: None)
    options = options or {}
    warnings = []
    started = time.time()

    run_id = options.get("run_id") or (time.strftime("%Y%m%d-%H%M%S") + "-" +
                                       ("%06x" % (int(time.time() * 1000) % 0xFFFFFF)))
    run_dir = os.path.join(RUNS_DIR, run_id)
    data_dir = os.path.join(run_dir, "data")
    _ensure_dir(data_dir)
    _archive_uploads(run_dir, files)

    # ---------------- 预处理：结构识别 + 转换（结果在第③步展示） ----------------
    emit(_stage(1, "start", "接收并结构化解析上传文件"))
    try:
        conv = data_convert.convert(files, data_dir, options)
    except Exception as e:                 # noqa: BLE001 - 数据问题要变成可读提示
        raise PipelineError("数据转换失败：{}: {}".format(type(e).__name__, e))
    if not conv.get("ok"):
        raise PipelineError("转换后的数据无法被原 RL 模型读取：{}".format(conv.get("error")))
    schema = conv["schema"]
    horizon = float(conv.get("episode_cap") or schema.flow.horizon_seconds or 3600.0)
    episode_seconds = _clamp_episode(options.get("episode_seconds"), horizon)
    seed = int(options.get("seed") or 2024)

    for f in conv.get("files") or []:
        emit(_log(1, "{} → {}（{}）".format(f["name"], f["detected"], f.get("used_as") or f.get("note"))))
    for w in conv.get("warnings") or []:
        emit(_log(1, w))
        warnings.append(w)

    plan_context = data_convert.build_plan_context(schema)
    flow_json = _load_flow(data_dir)
    digest = data_convert.demand_digest(flow_json, schema)
    emit(_log(1, "路网 {} 个受控交叉口；车流 {} 辆；仿真时长 {}s"
              .format(plan_context["n_intersections"], digest.get("n_vehicles", 0),
                      int(episode_seconds))))

    client = llm if llm is not None else _make_client(api_key, model, base_url)
    llm_used = client is not None
    llm_info = {"used": llm_used, "model": (model or "deepseek-chat"), "elapsed": {}}

    # ---------------- ① LLM 解析文件内容（流式） ----------------
    parse_summary = ""
    if llm_used:
        t0 = time.time()
        keep_summary = options.get("keep_llm_trace", True)
        try:
            parse_summary = _stage_parse(client, files, conv, digest, emit,
                                         keep=keep_summary)
        except DeepSeekError as e:
            raise PipelineError(str(e))
        llm_info["elapsed"]["parse"] = round(time.time() - t0, 2)
    else:
        parse_summary = ("未配置 API Key：跳过 LLM 解析，使用内置链路。"
                         "路网 {} 个交叉口，车流 {} 辆。"
                         .format(plan_context["n_intersections"], digest.get("n_vehicles", 0)))
        warnings.append("未配置 API Key：跳过 LLM 解析与建议，配时方案仍由韦伯斯特法计算")
        emit(_log(1, parse_summary))
    emit(_stage(1, "done", "解析完成"))

    # ---------------- ② 韦伯斯特法计算配时方案 ----------------
    # 说明：本次改造把"方案怎么来"从 LLM 自由生成改为韦伯斯特经典公式确定性计算，
    #       LLM 仍负责①解析与⑤建议。options["plan_source"]="llm" 可切回旧行为（默认 webster）。
    emit(_stage(2, "start", "韦伯斯特法计算信号配时方案"))
    plan_raw, plan_source = None, "webster"
    webster_out = None
    if options.get("plan_source") == "llm" and llm_used:
        t0 = time.time()
        plan_raw = _stage_plan(client, parse_summary, plan_context, digest, conv, emit)
        plan_source = "llm"
        llm_info["elapsed"]["plan"] = round(time.time() - t0, 2)
    try:
        webster_out = webster.compute(schema, flow_json, conv.get("flow_stats"), options)
    except Exception as e:                  # noqa: BLE001 - 配时计算失败要变成可读提示
        raise PipelineError("韦伯斯特配时计算失败：{}: {}".format(type(e).__name__, e))
    if plan_source == "webster":
        plan_raw = webster_out["plan"]
        for w in webster_out.get("warnings") or []:
            warnings.append(w)
        for row in webster_out["intersections"]:
            emit(_log(2, webster.brief(row)))
        ws = webster_out["summary"]
        emit(_log(2, "周期 {:.1f}~{:.1f}s（均值 {:.1f}s）；下限截断 {} 个、上限截断 {} 个、过饱和 {} 个；"
                     "Y {:.3f}~{:.3f}"
                  .format(ws["cycle_min"], ws["cycle_max"], ws["cycle_mean"],
                          ws["n_clamped_min"], ws["n_clamped_max"], ws["n_saturated"],
                          ws["Y_min"], ws["Y_max"])))
    emit(_stage(2, "done", "方案就绪"))

    # ---------------- ③ 数据转换结果（已落盘，做一次核对） ----------------
    emit(_stage(3, "start", "转换上传数据为 RL 输入格式"))
    emit(_log(3, "写入目录：" + data_dir))
    for key, value in sorted((conv.get("sources") or {}).items()):
        emit(_log(3, "{} = {}".format(key, value)))
    if conv.get("flow_stats"):
        fs = conv["flow_stats"]
        emit(_log(3, "车流 {} 条，时间 {:.0f}~{:.0f}s，路线 {} 种"
                  .format(fs["n_entries"], fs["t_min"], fs["t_max"], fs["n_unique_routes"])))
    emit(_stage(3, "done", "转换完成"))

    # ---------------- ④ RL 打分 + 效果评价 ----------------
    emit(_stage(4, "start", "RL 模型打分与效果评价"))
    try:
        report = rl_scorer.evaluate(
            schema, data_dir, flow_json, plan_raw or {}, episode_seconds, seed=seed,
            with_q=bool(options.get("with_q", True)),
            progress=lambda stage, value: emit(_progress(4, stage, value)),
        )
    except Exception as e:                 # noqa: BLE001
        raise PipelineError("RL 打分失败：{}: {}".format(type(e).__name__, e))
    warnings.extend(report.get("warnings") or [])
    m_rl = report["rl_arm"].get("metrics") or {}
    m_plan = report["plan_arm"].get("metrics") or {}
    att_ok = (m_rl.get("throughput") or 0) > 0 and (m_plan.get("throughput") or 0) > 0
    if att_ok:                       # 有车完成行程时用 ATT 作对照，否则用排队
        basis = "ATT 基线 {:.1f}s / 方案 {:.1f}s".format(m_rl.get("ATT") or 0, m_plan.get("ATT") or 0)
    else:
        basis = "AQL 基线 {:.2f} / 方案 {:.2f}".format(m_rl.get("AQL") or 0, m_plan.get("AQL") or 0)
    emit(_log(4, "{} → 综合 {} 分".format(basis, report["score"]["total"])))
    emit(_stage(4, "done", "打分完成"))

    # ---------------- ⑤ 精简优化建议（流式） ----------------
    emit(_stage(5, "start", "生成精简优化建议"))
    advice = ""
    if llm_used:
        t0 = time.time()
        try:
            advice = _stage_advice(client, report, plan_context, emit)
        except DeepSeekError as e:
            raise PipelineError(str(e))
        llm_info["elapsed"]["advice"] = round(time.time() - t0, 2)
    else:
        advice = _rule_advice(report, webster_out)
        emit(_token(5, advice))
    emit(_stage(5, "done", "建议已生成"))

    # ---------------- 汇总 ----------------
    result = {
        "run_id": run_id,
        "run_dir": run_dir,
        "elapsed_seconds": round(time.time() - started, 2),
        "episode_seconds": float(episode_seconds),
        "engine_kind": report.get("engine_kind"),
        "files": conv.get("files"),
        "conversion": {
            "sources": conv.get("sources"),
            "dims": conv.get("dims"),
            "flow_stats": conv.get("flow_stats"),
            "warnings": conv.get("warnings") or [],
        },
        "parse_summary": parse_summary,
        "plan": {
            "source": plan_source,
            "notes": (plan_raw or {}).get("notes") if isinstance(plan_raw, dict) else None,
            "used": report.get("plan_used"),
            "intersections": len(report.get("plan_used") or {}),
        },
        # 韦伯斯特法的完整配时明细（相位方案 / 周期 / 绿黄红 / 绿信比），前端按路口展示
        "webster": webster_out,
        "score": report.get("score"),
        "verdict": report.get("verdict"),
        "arms": {"rl": report.get("rl_arm"), "plan": report.get("plan_arm")},
        "worst_intersections": report.get("worst_intersections"),
        "q_stats": report.get("q_stats"),
        "advice": advice.strip(),
        "warnings": _dedup(warnings),
        "llm": llm_info,
        "scoring_note": report.get("scoring_note"),
    }
    emit({"type": "result", "data": result})
    return result


# ===========================================================================
# 各步骤实现
# ===========================================================================

def _make_client(api_key, model, base_url):
    if not api_key or not str(api_key).strip():
        return None
    return DeepSeekClient(api_key=api_key, model=model, base_url=base_url)


def _stage_parse(client, files, conv, digest, emit, keep=True, max_chars=1400):
    """① 把文件内容摘录 + 结构摘要给 LLM，让它用 3~5 行说清数据是什么。"""
    parts = []
    for item in (files or [])[:6]:
        name = item.get("name")
        text = (item.get("content") or "")
        excerpt = text[:max_chars] + ("…（已截断）" if len(text) > max_chars else "")
        parts.append("### 文件 {}（{} 字节）\n{}".format(name, len(text), excerpt))
    if not parts:
        parts.append("（本次未上传文件，使用的是内置 demo 数据）")
    structure = {
        "识别结果": [{"文件": f["name"], "类型": f["detected"], "用途": f.get("used_as")}
                 for f in (conv.get("files") or [])],
        "数据来源": conv.get("sources"),
        "维度": conv.get("dims"),
        "车流概况": conv.get("flow_stats"),
        "主要进口道": (digest or {}).get("top_entry_roads"),
        "警告": conv.get("warnings"),
    }
    user = ("下面是用户上传的数据内容与系统自动识别的结构摘要。\n\n{}\n\n"
            "结构摘要（JSON）：\n{}\n\n"
            "请输出 3~5 行中文要点：这是什么数据、路网规模、流量规模与主要流向、"
            "时间跨度、是否可直接用于 RL 配时评估。不要复述 JSON。"
            .format("\n\n".join(parts), json.dumps(structure, ensure_ascii=False)[:3000]))
    messages = [{"role": "system", "content": PARSE_SYSTEM},
                {"role": "user", "content": user}]
    text = ""
    for chunk in client.stream_chat(messages, temperature=0.1, max_tokens=320):
        if chunk["type"] == "delta":
            text += chunk["content"]
            emit(_token(1, chunk["content"]))
    return text.strip()


def _stage_plan(client, parse_summary, plan_context, digest, conv, emit, max_chars=1):
    """
    ② 让 LLM 在"真实可执行空间"里出方案，返回 dict（含 default_cycle / intersections）。
    失败时抛 DeepSeekError / 返回 None（由上层回退内置方案）。
    """
    compact = {
        "受控交叉口数": plan_context["n_intersections"],
        "时长档位_秒": plan_context["duration_options"],
        "交叉口": [
            {"id": row["intersection_id"],
             "相位槽": [{"slot": s["slot"],
                       "放行": "/".join(s["flows"][:3])}
                      for s in row["slots"]]}
            for row in plan_context["intersections"]
        ],
        "说明": plan_context["note"],
    }
    user = (
        "【数据解析要点】\n{}\n\n"
        "【流量概况】{} 辆，主要进口：{}\n\n"
        "【可执行动作空间】\n{}\n\n"
        "【任务】给出一个绿波/均衡型定时配时方案：\n"
        "1) 只能使用上面的 slot 与时长档位；\n"
        "2) 每个交叉口 2~4 个相位，周期 60~120s，单相位不短于 10s；\n"
        "3) 干道方向（东西向主路）可给更长绿时；\n"
        "4) 大多数交叉口可用 default_cycle 统一给，只在关键路口单独覆盖。\n\n"
        "【只输出 JSON】格式：\n"
        '{{"default_cycle":[{{"phase_slot":0,"duration":30}},{{"phase_slot":1,"duration":30}}],'
        '"intersections":{{"intersection_1_1":[{{"phase_slot":0,"duration":35}},'
        '{{"phase_slot":1,"duration":25}}]}},"notes":"不超过40字"}}'
        .format(parse_summary[:800],
                (digest or {}).get("n_vehicles", 0),
                ", ".join("{}:{}".format(r["road"], r["vehicles"])
                          for r in (digest or {}).get("top_entry_roads", [])[:4]) or "未知",
                json.dumps(compact, ensure_ascii=False)[:6000]))
    messages = [{"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": user}]
    res = client.chat(messages, temperature=0.2, max_tokens=1600, json_mode=True)
    content = res.get("content") or ""
    emit(_log(2, "LLM 返回 {} 字，正在校验…".format(len(content))))
    plan, err = extract_json(content, expect="object")
    if plan is None:
        emit(_log(2, "JSON 解析失败（{}），改用内置对照方案".format(err)))
        return None
    return plan


def _stage_advice(client, report, plan_context, emit):
    """⑤ 依据打分结果给最多 3 条可执行建议，流式输出。"""
    worst = report.get("worst_intersections") or []
    worst_txt = "；".join(
        "{}：排队 {} 辆（RL 基线 {}）".format(w["intersection_id"], w["aql_plan"], w["aql_rl"])
        for w in worst) or "无明显拥堵口"
    detail = report["score"]["detail"]
    metrics_txt = "；".join(
        "{} 方案 {} vs 基线 {}".format(v["label"], v["plan"], v["rl"])
        for v in detail.values() if v.get("plan") is not None)
    user = (
        "【评分】总分 {total}（仿真分 {sim}，动作质量分 {q}）\n"
        "【指标】{metrics}\n"
        "【最差路口】{worst}\n"
        "【当前方案】{plan}\n\n"
        "请给最多 3 条建议，每条一行、以「- 」开头、不超过 25 字，"
        "要具体到路口和相位/绿时怎么改。"
        .format(total=report["score"]["total"], sim=report["score"]["sim"], q=report["score"]["q"],
                metrics=metrics_txt or "无", worst=worst_txt,
                plan=_plan_brief(report.get("plan_used")))
    )
    messages = [{"role": "system", "content": ADVICE_SYSTEM},
                {"role": "user", "content": user}]
    text = ""
    for chunk in client.stream_chat(messages, temperature=0.3, max_tokens=260):
        if chunk["type"] == "delta":
            text += chunk["content"]
            emit(_token(5, chunk["content"]))
    return text.strip()


def _plan_brief(plan_used):
    """把执行方案压成一行，避免提示词过长。"""
    if not plan_used:
        return "无"
    cycles = {}
    for iid, item in plan_used.items():
        key = (item["cycle_seconds"], tuple((p["phase_slot"], p["duration"]) for p in item["phases"]))
        cycles.setdefault(key, []).append(iid)
    out = []
    for (cycle_seconds, phases), ids in sorted(cycles.items(), key=lambda kv: -len(kv[1]))[:3]:
        out.append("周期{}s：{} → 用于 {} 个口（如 {}）".format(
            cycle_seconds, "+".join("{}/{}s".format(s, d) for s, d in phases),
            len(ids), ids[0]))
    return "；".join(out)


def _rule_advice(report, webster_out=None):
    """无 LLM 时的规则建议（同样是精简输出）。"""
    worst = (report.get("worst_intersections") or [])[:2]
    lines = []
    for w in worst:
        plan_aql, rl_aql = w.get("aql_plan"), w.get("aql_rl")
        if plan_aql is None:
            continue
        gap = "" if rl_aql is None else "（基线 {} 辆）".format(rl_aql)
        lines.append("- {}：排队 {} 辆{}，建议延长主路直行绿时 5~10s".format(
            w["intersection_id"], plan_aql, gap))
    if not lines:
        lines.append("- 各路口排队均衡，可维持当前相位顺序")
    summary = (webster_out or {}).get("summary") or {}
    params = (webster_out or {}).get("params") or {}
    cycle = summary.get("cycle_mean") or 60.0
    yellow = params.get("yellow_seconds") or 3.0
    lines.append("- 周期统一 {:.0f}s、黄灯 {:.0f}s，绿时按流量比（绿信比）分配".format(cycle, yellow))
    return "\n".join(lines)


# ===========================================================================
# 杂项
# ===========================================================================

def _clamp_episode(value, horizon):
    horizon = max(60.0, float(horizon or 3600.0))
    if value in (None, "", 0):
        value = min(900.0, horizon)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = min(900.0, horizon)
    value = max(60.0, min(value, horizon))
    return float(int(value / 10.0) * 10) or 60.0


def _load_flow(data_dir):
    for name in sorted(os.listdir(data_dir)):
        if name.startswith("flow") and name.endswith(".json"):
            with open(os.path.join(data_dir, name), "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("flows") if isinstance(data, dict) else data
    raise PipelineError("[数据缺失] 转换目录里没有 flow*.json")


def _archive_uploads(run_dir, files):
    """把上传原文存一份，方便复现（文件名做安全化处理）。"""
    if not files:
        return
    up = os.path.join(run_dir, "uploads")
    _ensure_dir(up)
    for item in files:
        name = os.path.basename(str(item.get("name") or "upload.txt"))
        name = re.sub(r"[^0-9A-Za-z._\u4e00-\u9fa5-]", "_", name)[:80] or "upload.txt"
        try:
            with open(os.path.join(up, name), "w", encoding="utf-8") as f:
                f.write(item.get("content") or "")
        except OSError:
            pass


def _ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _dedup(items):
    out, seen = [], set()
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _stage(stage, status, message):
    return {"type": "stage", "stage": stage, "status": status, "message": message}


def _log(stage, message):
    return {"type": "log", "stage": stage, "message": message}


def _token(stage, content):
    return {"type": "token", "stage": stage, "content": content}


def _progress(stage, name, value):
    return {"type": "progress", "stage": stage, "name": name, "value": round(float(value), 3)}
