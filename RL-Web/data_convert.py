# -*- coding: utf-8 -*-
"""
data_convert.py —— 第③步：把上传的 json / csv 自动转换成「原有 RL 模型可直接读取」的输入格式

原有 RL 的输入规范（见 ../RL-API/data_schema.py，本文件不修改该规范）：
    data_dir/
        roadnet*.json                 CityFlow 路网（intersections / roads）
        flow*.json                    CityFlow 车流（顶层数组 或 {"flows": [...]}）
        signal_inter_<inter_id>.txt   每个交叉口的相位序列（动作掩码来源）
支持的上传类型（自动识别，识别不出就明确报告而不是瞎猜）：
    A. CityFlow 路网 json           → roadnet*.json
    B. CityFlow 车流 json           → flow*.json
    C. 车流表 csv                   → 转成 B（列名支持 startTime/endTime/interval/route/vehicle 及中文别名）
    D. 路段需求表 csv/json          → 按 roadnet 图做路由推导后转成 B
    E. 其他（例如排队/状态观测表）  → 只做结构摘要交给 LLM，不参与数值转换
转换产物写入 <runs>/<run_id>/data/，再用原有 data_schema.build_schema() 校验维度。
"""
from __future__ import print_function

import csv
import io
import json
import os
import shutil

import rl_core

# ---------------------------------------------------------------------------
# 列名别名（大小写/空格/下划线归一化后匹配）
# ---------------------------------------------------------------------------
TIME_START = ("starttime", "start", "begintime", "begin", "t0", "开始时间", "起始时间", "发车时间", "时间段开始")
TIME_END = ("endtime", "end", "t1", "结束时间", "终止时间", "时间段结束")
INTERVAL = ("interval", "step", "间隔", "发车间隔", "时间间隔")
ROUTE = ("route", "routes", "path", "roads", "路径", "路线", "路段序列", "行驶路径")
VEHICLE = ("vehicle", "vehicles", "count", "num", "number", "volume", "flow", "流量", "车辆数", "车次", "车辆")
ROAD = ("road", "roadid", "road_id", "link", "linkid", "路段", "路段id", "道路")
LANE = ("lane", "laneid", "lane_id", "车道", "车道id")
INTER_ID = ("intersection", "intersectionid", "inter_id", "intersection_id", "交叉口", "路口")


def _norm(s):
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _pick(header_map, names):
    for n in names:
        key = n.replace("_", "")
        if key in header_map:
            return header_map[key]
    return None


def _header_map(headers):
    return dict((_norm(h), h) for h in headers)


def _to_float(value, default=None):
    try:
        if value is None:
            return default
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


# ===========================================================================
# 1. 单文件识别
# ===========================================================================

def detect_file(name, text):
    """
    返回 {'kind', 'detail'}：kind ∈ roadnet / flow / csv_flow / road_demand / unknown。
    只做结构判断，不做任何数值改写。
    """
    lower = (name or "").lower()
    if lower.endswith(".json"):
        try:
            obj = json.loads(text)
        except ValueError as e:
            return {"kind": "unknown", "detail": "JSON 解析失败：{}".format(e)}
        if isinstance(obj, dict) and "intersections" in obj and "roads" in obj:
            return {"kind": "roadnet", "detail": "CityFlow 路网（{} 个交叉口 / {} 条路段）"
                    .format(len(_as_list(obj["intersections"])), len(_as_list(obj["roads"]))),
                    "obj": obj}
        entries = None
        if isinstance(obj, list):
            entries = obj
        elif isinstance(obj, dict) and isinstance(obj.get("flows"), list):
            entries = obj["flows"]
        if entries:
            first = entries[0] if isinstance(entries[0], dict) else {}
            if "route" in first and ("startTime" in first or "start_time" in first):
                return {"kind": "flow", "detail": "CityFlow 车流（{} 条 flow 条目）".format(len(entries)),
                        "obj": entries}
            if _pick(_header_map(first.keys()), ROAD + LANE) and _pick(_header_map(first.keys()), VEHICLE):
                return {"kind": "road_demand", "detail": "路段/车道需求表（{} 行）".format(len(entries)),
                        "obj": entries}
        return {"kind": "unknown", "detail": "已解析 JSON，但不是路网/车流结构（键：{}）"
                .format(list(obj.keys())[:8] if isinstance(obj, dict) else "list")}
    # --- csv ---
    try:
        rows = _read_csv(text)
    except Exception as e:               # noqa: BLE001
        return {"kind": "unknown", "detail": "CSV 解析失败：{}".format(e)}
    if not rows:
        return {"kind": "unknown", "detail": "CSV 没有数据行"}
    header_map = _header_map(rows[0].keys())
    if _pick(header_map, ROUTE) and _pick(header_map, TIME_START):
        return {"kind": "csv_flow", "detail": "车流表（{} 行，含 route + 时间列）".format(len(rows)),
                "obj": rows, "header": header_map}
    if _pick(header_map, ROAD + LANE) and _pick(header_map, VEHICLE):
        return {"kind": "road_demand", "detail": "路段/车道需求表（{} 行）".format(len(rows)),
                "obj": rows, "header": header_map}
    return {"kind": "unknown", "detail": "未识别的 CSV 表头：{}".format(list(rows[0].keys())[:8]),
            "obj": rows, "header": header_map}


def _as_list(obj):
    return list(obj.values()) if isinstance(obj, dict) else list(obj)


def _read_csv(text):
    stream = io.StringIO(text.lstrip("\ufeff"))
    # 自动判断分隔符（逗号 / 分号 / 制表符）
    sample = text[:2048]
    delimiter = ","
    for cand in (";", "\t"):
        if sample.count(cand) > sample.count(delimiter):
            delimiter = cand
    reader = csv.DictReader(stream, delimiter=delimiter)
    rows = []
    for row in reader:
        if row and any((v or "").strip() for v in row.values() if isinstance(v, str)):
            rows.append(dict((k.strip(), (v or "").strip()) for k, v in row.items() if k))
    return rows


# ===========================================================================
# 2. 各类型 → CityFlow flow 条目
# ===========================================================================

def csv_flow_to_entries(rows, header, default_vehicle):
    """C. 车流表 csv → CityFlow flow 条目（route 支持 "road_a->road_b" 或 "road_a,road_b"）。"""
    c_start = _pick(header, TIME_START)
    c_end = _pick(header, TIME_END)
    c_interval = _pick(header, INTERVAL)
    c_route = _pick(header, ROUTE)
    c_vehicle = _pick(header, VEHICLE)
    entries = []
    for row in rows:
        route = _split_route(row.get(c_route))
        if not route:
            continue
        start = _to_float(row.get(c_start), 0.0)
        end = _to_float(row.get(c_end), start if c_end is None else start + 1.0)
        interval = _to_float(row.get(c_interval), 1.0) if c_interval else 1.0
        count = _to_float(row.get(c_vehicle), 1.0) if c_vehicle else 1.0
        if interval is None or interval <= 0:
            interval = 1.0
        if count is None or count <= 0:
            count = 1.0
        entries.append({
            "vehicle": dict(default_vehicle),
            "route": route,
            "interval": interval,
            "startTime": start,
            "endTime": end,
            "vehicle_count": int(round(count)),
        })
    return entries


def _split_route(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    for sep in ("->", "→", ">", "|", ";", "，", "/"):
        if sep in text:
            return [p.strip() for p in text.split(sep) if p.strip()]
    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]
    if " " in text:
        return [p.strip() for p in text.split() if p.strip()]
    return [text]


def road_demand_to_entries(rows, header, roadnet_parsed, default_vehicle, horizon):
    """
    D. 路段/车道需求表 → 按路网拓扑推导路线后生成 flow 条目。

    需求解释：每条路段上「表里的车辆数」按 startT..endT 均匀发车；
    路线 = 从该路段出发，在每个交叉口优先直行、其次右转、最后左转，直到驶出边界。
    """
    c_road = _pick(header, ROAD)
    c_lane = _pick(header, LANE)
    c_count = _pick(header, VEHICLE)
    c_start = _pick(header, TIME_START)
    c_end = _pick(header, TIME_END)
    roads = roadnet_parsed["road_map"]
    graph = RouteBuilder(roadnet_parsed)
    entries = []
    skipped = []
    for row in rows:
        rid = row.get(c_road) if c_road else None
        if not rid and c_lane:
            lane = row.get(c_lane) or ""
            rid = lane.rsplit("_", 1)[0] if "_" in lane else lane
        if not rid or rid not in roads:
            skipped.append(rid)
            continue
        count = _to_float(row.get(c_count), 0.0) or 0.0
        if count <= 0:
            continue
        t0 = _to_float(row.get(c_start), 0.0) if c_start else 0.0
        t1 = _to_float(row.get(c_end), horizon) if c_end else horizon
        if t1 is None or t1 <= t0:
            t1 = t0 + 1.0
        route = graph.build(rid)
        if not route:
            skipped.append(rid)
            continue
        n = max(1, int(round(count)))
        span = max(1.0, t1 - t0)
        for k in range(n):
            t = round(t0 + span * k / float(n), 3)
            entries.append({
                "vehicle": dict(default_vehicle),
                "route": route,
                "interval": 1.0,
                "startTime": t,
                "endTime": t,
            })
    return entries, skipped


class RouteBuilder(object):
    """按路网几何推导一条「从该路段出发直到驶出边界」的路线（直行优先）。"""

    def __init__(self, roadnet_parsed):
        self.roads = roadnet_parsed["road_map"]
        self.inter_map = roadnet_parsed["inter_map"]
        self.out_roads = {}
        for rid, road in self.roads.items():
            self.out_roads.setdefault(road["startIntersection"], []).append(rid)
        for iid in self.out_roads:
            self.out_roads[iid].sort()

    def _vec(self, rid):
        road = self.roads[rid]
        a = self.inter_map[road["startIntersection"]]["point"]
        b = self.inter_map[road["endIntersection"]]["point"]
        dx, dy = float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"])
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        return dx / norm, dy / norm

    def build(self, start_rid, max_len=14):
        route = [start_rid]
        cur = start_rid
        while len(route) < max_len:
            end = self.roads[cur]["endIntersection"]
            if self.inter_map.get(end, {}).get("virtual"):
                break
            candidates = [r for r in self.out_roads.get(end, []) if r != cur]
            if not candidates:
                break
            direction = self._vec(cur)
            best, best_score = None, None
            for cand in candidates:
                d = self._vec(cand)
                dot = direction[0] * d[0] + direction[1] * d[1]
                if best_score is None or dot > best_score:
                    best, best_score = cand, dot
            if best is None:
                break
            route.append(best)
            cur = best
        return route


# ===========================================================================
# 3. 转换主流程
# ===========================================================================

def convert(files, run_data_dir, options=None):
    """
    把上传文件转换成 RL 输入目录。

    参数
        files        : [{'name': str, 'content': str}]（文本内容，前端已按 utf-8 读取）
        run_data_dir : 本次运行的 data 目录（会在其中写 roadnet*/flow*/signal_inter_*.txt）
        options      : {'prefer_uploaded_roadnet': bool}
    返回 dict：转换报告（给前端展示 + 给 LLM 做上下文）
    """
    options = options or {}
    os.makedirs(run_data_dir, exist_ok=True)
    builtin = rl_core.data_dir()

    report = {"files": [], "warnings": [], "sources": {}}
    roadnet_obj = None
    roadnet_from = None
    flow_entries = None
    flow_from = None
    flow_kind = None

    # --- 逐个文件识别 ---
    for item in files or []:
        name = item.get("name") or "unnamed"
        text = item.get("content") or ""
        det = detect_file(name, text)
        row = {"name": name, "size": len(text), "detected": det["kind"], "note": det["detail"]}
        report["files"].append(row)
        if det["kind"] == "roadnet" and roadnet_obj is None:
            roadnet_obj = det["obj"]
            roadnet_from = name
            row["used_as"] = "路网"
        elif det["kind"] == "flow" and flow_entries is None:
            flow_entries = det["obj"]
            flow_from = name
            flow_kind = "cityflow"
            row["used_as"] = "车流"
        elif det["kind"] == "csv_flow" and flow_entries is None:
            flow_entries = det["obj"]
            flow_from = name
            flow_kind = "csv_flow"
            row["used_as"] = "车流（csv 转换）"
        elif det["kind"] == "road_demand" and flow_entries is None:
            flow_entries = det["obj"]
            flow_from = name
            flow_kind = "road_demand"
            row["used_as"] = "车流（需求表推算）"
        else:
            row["used_as"] = "仅用于 LLM 解析（未参与数值转换）"

    # --- 路网 ---
    if roadnet_obj is not None:
        path = os.path.join(run_data_dir, "roadnet_uploaded.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(roadnet_obj, f, ensure_ascii=False)
        report["sources"]["roadnet"] = "上传：" + roadnet_from
    else:
        src = _find_and_copy(builtin, "roadnet", run_data_dir, "roadnet_builtin.json")
        report["sources"]["roadnet"] = "内置：{}".format(os.path.basename(src))
        report["warnings"].append("未上传路网，沿用内置路网 {}".format(os.path.basename(src)))
    roadnet_path = _find_pattern(run_data_dir, ("roadnet*.json", "*roadnet*.json"))
    parsed = rl_core.core("data_schema").parse_roadnet(json.load(open(roadnet_path, encoding="utf-8")))

    # --- 信号相位文件（动作掩码来源，保持原有口径不变）---
    n_signal = 0
    for fname in sorted(os.listdir(builtin)):
        if fname.startswith("signal_inter_") and fname.endswith(".txt"):
            shutil.copyfile(os.path.join(builtin, fname), os.path.join(run_data_dir, fname))
            n_signal += 1
    report["sources"]["signal_plan"] = "内置相位文件 {} 个（动作掩码来源，不随 LLM 方案改变）".format(n_signal)

    # --- 车流 ---
    default_vehicle = _default_vehicle(builtin)
    if flow_entries is None:
        src = _find_and_copy(builtin, "flow", run_data_dir, "flow_builtin.json")
        report["sources"]["flow"] = "内置：{}".format(os.path.basename(src))
        report["warnings"].append("未上传车流数据，沿用内置车流 {}".format(os.path.basename(src)))
        flow_kind = "builtin"
    else:
        horizon = 3600.0
        if flow_kind == "csv_flow":
            det = [f for f in report["files"] if f["name"] == flow_from][0]
            entries = csv_flow_to_entries(flow_entries, det.get("header") or
                                          _header_map(flow_entries[0].keys()), default_vehicle)
            warnings_extra = []
            flow_entries = _explode_counts(entries, warnings_extra)
            report["warnings"].extend(warnings_extra)
        elif flow_kind == "road_demand":
            det = [f for f in report["files"] if f["name"] == flow_from][0]
            if isinstance(flow_entries, list) and flow_entries and isinstance(flow_entries[0], dict):
                header = det.get("header") or _header_map(flow_entries[0].keys())
            else:
                header = {}
            entries, skipped = road_demand_to_entries(
                flow_entries, header, parsed, default_vehicle, horizon)
            if skipped:
                report["warnings"].append(
                    "需求表中有 {} 行路段不在路网内，已跳过（示例：{}）"
                    .format(len(skipped), skipped[:3]))
            flow_entries = entries
        else:
            flow_entries = _explode_counts(flow_entries, report["warnings"])

        # 丢掉路线里含未知路段的条目（例如 csv 里拼错的路段 id），避免整体校验失败
        flow_entries, dropped = _drop_unknown_routes(flow_entries, parsed["road_map"])
        if dropped:
            report["warnings"].append(
                "已丢弃 {} 条路线无效（含未知路段或路段数 < 2）的车流条目，示例：{}"
                .format(len(dropped), [" -> ".join(d[:4]) for d in dropped[:2]]))

        if not flow_entries:
            src = _find_and_copy(builtin, "flow", run_data_dir, "flow_builtin.json")
            report["sources"]["flow"] = "内置：{}（上传内容转换后为空，已回退）".format(
                os.path.basename(src))
            report["warnings"].append("上传的车流数据转换后没有任何有效条目，已回退到内置车流")
            flow_kind = "builtin"
        else:
            path = os.path.join(run_data_dir, "flow_uploaded.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(flow_entries, f, ensure_ascii=False)
            report["sources"]["flow"] = "上传：{}（{}）".format(flow_from, flow_kind)

    # --- 校验：用原有 data_schema 解析，并核对与冻结模型的维度 ---
    try:
        schema = rl_core.build_schema(run_data_dir)
        report["ok"] = True
        report["dims"] = {
            "n_intersections": schema.dims.n_intersections,
            "state_dim_max": schema.dims.state_dim_max,
            "n_selectable_phases_max": schema.dims.n_selectable_phases_max,
            "n_duration_actions": schema.dims.n_duration_actions,
        }
        report["flow_stats"] = {
            "n_entries": schema.flow.n_flows,
            "t_min": round(float(schema.flow.t_min), 2),
            "t_max": round(float(schema.flow.t_max), 2),
            "horizon_seconds": round(float(schema.flow.horizon_seconds), 2),
            "n_unique_routes": schema.flow.n_unique_routes,
            "route_len_min": schema.flow.route_len_min,
            "route_len_max": schema.flow.route_len_max,
        }
        report["episode_cap"] = float(schema.flow.horizon_seconds)
    except Exception as e:               # noqa: BLE001 - 校验失败必须给出可读原因
        report["ok"] = False
        report["error"] = "{}: {}".format(type(e).__name__, e)
        schema = None

    # --- 路网维度与冻结模型不兼容时：退回内置路网重试一次 ---
    if schema is not None:
        try:
            rl_core.load_frozen(schema)
        except Exception as e:           # noqa: BLE001
            msg = str(e)
            if roadnet_obj is not None:
                report["warnings"].append(
                    "上传路网与冻结 RL 模型的维度不兼容（{}），已回退内置路网做打分".format(msg[:120]))
                for fname in os.listdir(run_data_dir):
                    if fname.startswith("roadnet"):
                        os.remove(os.path.join(run_data_dir, fname))
                src = _find_and_copy(builtin, "roadnet", run_data_dir, "roadnet_builtin.json")
                report["sources"]["roadnet"] = "内置：{}（上传路网维度不兼容已回退）".format(
                    os.path.basename(src))
                try:
                    schema = rl_core.build_schema(run_data_dir)
                    rl_core.load_frozen(schema)
                    report["ok"] = True
                except Exception as e2:  # noqa: BLE001
                    report["ok"] = False
                    report["error"] = "{}: {}".format(type(e2).__name__, e2)
            else:
                report["ok"] = False
                report["error"] = "冻结模型与数据维度不匹配：{}".format(msg)

    report["data_dir"] = os.path.abspath(run_data_dir)
    report["schema"] = schema
    return report


def _drop_unknown_routes(entries, road_map, min_len=2):
    """过滤掉路线里含未知路段、或路段数小于 2 的车流条目（原有 data_schema 要求 route 长度 >= 2）。"""
    kept, dropped = [], []
    for e in entries:
        route = list(e.get("route") or [])
        if len(route) < min_len or any(r not in road_map for r in route):
            dropped.append(route)
            continue
        kept.append(e)
    return kept, dropped


def _explode_counts(entries, warnings):
    """
    把带 vehicle_count 的条目展开成 CityFlow 原生条目（startTime==endTime 表示发一辆车）。

    这样转换产物始终是「顶层数组 + 标准字段」，与原 data/flow.json 完全同构。
    规则：count 辆车在 [startTime, endTime) 内均匀发车；区间为零时长时集中在 startTime。
    """
    out = []
    for src in entries:
        e = dict(src)
        count = int(e.pop("vehicle_count", 1) or 1)
        count = max(1, count)
        start = float(e.get("startTime", 0))
        end = float(e.get("endTime", start))
        if end <= start:
            times = [start]
        elif count == 1:
            times = [start]                    # 单辆车：沿用原区间含义（首个时刻发车）
        else:
            span = end - start
            times = [round(start + span * i / float(count), 3) for i in range(count)]
        for t in times:
            item = dict(e)
            item["startTime"] = t
            item["endTime"] = t
            out.append(item)
        if count > 1 and end <= start:
            warnings.append("车流条目 {} 的区间长度为 0，{} 辆车集中在 t={}s 发车"
                            .format(e.get("route"), count, start))
    return out


def _default_vehicle(builtin_dir):
    """取内置 flow 里的 vehicle 参数作为转换时的默认车辆属性。"""
    try:
        flow_path = _find_pattern(builtin_dir, ("flow*.json", "*flow*.json"))
        with open(flow_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else data.get("flows") or []
        for e in entries:
            if isinstance(e.get("vehicle"), dict):
                return dict(e["vehicle"])
    except Exception:                    # noqa: BLE001
        pass
    return {"length": 5.0, "width": 2.0, "maxPosAcc": 2.0, "maxNegAcc": 4.5,
            "usualPosAcc": 2.0, "usualNegAcc": 4.5, "minGap": 2.5,
            "maxSpeed": 11.111, "headwayTime": 2}


def _find_pattern(dir_path, patterns):
    import glob
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(dir_path, pat)))
        if hits:
            return hits[0]
    raise RuntimeError("[数据缺失] {} 下找不到匹配 {} 的文件".format(dir_path, patterns))


def _find_and_copy(src_dir, kind, dst_dir, dst_name):
    patterns = ("roadnet*.json", "*roadnet*.json") if kind == "roadnet" else ("flow*.json", "*flow*.json")
    src = _find_pattern(src_dir, patterns)
    dst = os.path.join(dst_dir, dst_name)
    shutil.copyfile(src, dst)
    return src


# ===========================================================================
# 4. 给 LLM 用的「真实结构摘要」（第①/②步提示词的事实来源）
# ===========================================================================

_COMPASS = ("东", "北", "西", "南")


def _compass(dx, dy):
    """方向向量 → 中文方位。已核对 roadnet_4X4.json 的 go_straight/turn_left 语义：
    该路网符合 CityFlow 常规坐标系，+x 为东、+y 为北（(0,0) 口的 turn_left 从"向东行"转向 +y）。"""
    if abs(dx) >= abs(dy):
        return "东" if dx > 0 else "西"
    return "北" if dy > 0 else "南"


def _build_tables(schema):
    """一次性构建「坐标 / 路段 / roadLink」索引（每次调用独立，避免复用过期缓存）。"""
    points, roads, links = {}, {}, {}
    for inter in _as_list(schema.roadnet.get("intersections") or []):
        iid = inter.get("id")
        pt = inter.get("point") or {}
        points[iid] = (float(pt.get("x", 0.0)), float(pt.get("y", 0.0)))
        if inter.get("virtual"):
            continue
        links[iid] = dict(
            (k, {"index": k, "start": rl["startRoad"], "end": rl["endRoad"],
                 "type": rl.get("type")})
            for k, rl in enumerate(inter.get("roadLinks") or []))
    for road in _as_list(schema.roadnet.get("roads") or []):
        roads[road["id"]] = road
    return points, roads, links


# 对外别名：webster.py 需要同一套坐标/路段/roadLink 表来折算各相位流量
build_tables = _build_tables


def build_plan_context(schema, max_intersections=16):
    """
    构造配时方案的「可执行事实」：
        - 受控交叉口 + 每个可选相位槽的真实放行流向（中文描述，由 roadLinks 与坐标推导）
        - 持续时间档位（原有 config.DURATION_OPTIONS）
    LLM 只能在这个范围内出方案，避免给出无法执行的动作。
    """
    cfg = rl_core.core("config")
    points, roads, links = _build_tables(schema)
    turn_cn = {"go_straight": "直行", "turn_left": "左转", "turn_right": "右转"}

    rows = []
    for sp in schema.intersections[:max_intersections]:
        x, y = points.get(sp.inter_id, (0.0, 0.0))
        slots = []
        for slot, phase_index in enumerate(sp.selectable_phase_indices):
            seen, uniq = set(), []
            for link_idx in sp.phases[phase_index].available_road_links:
                link = links.get(sp.inter_id, {}).get(link_idx)
                if not link:
                    continue
                start_road = roads.get(link["start"]) or {}
                end_road = roads.get(link["end"]) or {}
                a = points.get(start_road.get("startIntersection"), (x, y))
                b = points.get(end_road.get("endIntersection"), (x, y))
                approach = _compass(a[0] - x, a[1] - y)      # 车辆从哪个方向驶来（上游端点相对本口）
                leave = _compass(b[0] - x, b[1] - y)         # 驶向哪个方向（下游端点相对本口）
                label = "{}→{}{}".format(approach, leave, turn_cn.get(link.get("type"), ""))
                if label not in seen:
                    seen.add(label)
                    uniq.append(label)
            slots.append({"slot": slot, "light_phase": int(phase_index), "flows": uniq[:6]})
        rows.append({"intersection_id": sp.inter_id, "slots": slots,
                     "neighbors": list(sp.neighbors),
                     "n_incoming_lanes": len(sp.incoming_lanes)})

    return {
        "n_intersections": len(schema.intersections),
        "intersections": rows,
        "duration_options": [int(x) for x in cfg.DURATION_OPTIONS],
        "note": "phase_slot 是动作槽下标（0 起），light_phase 是 roadnet 中的相位下标",
    }


def demand_digest(flow_json, schema, top=8):
    """上传车流的概况摘要（车辆数、时间跨度、主要进口道路），给 LLM 判断潮汐方向。"""
    entries = expand_entries(flow_json)
    if not entries:
        return {"n_vehicles": 0}
    total = 0
    t_min, t_max = None, None
    by_entry_road = {}
    for e in entries:
        route = e.get("route") or []
        if not route:
            continue
        count = e.get("_count", 1)
        total += count
        start = float(e.get("startTime", 0))
        end = float(e.get("endTime", start))
        t_min = start if t_min is None else min(t_min, start)
        t_max = end if t_max is None else max(t_max, end)
        by_entry_road[route[0]] = by_entry_road.get(route[0], 0) + count
    top_roads = sorted(by_entry_road.items(), key=lambda kv: -kv[1])[:top]
    return {
        "n_vehicles": total,
        "t_min": t_min,
        "t_max": t_max,
        "horizon_seconds": (t_max or 0) - (t_min or 0),
        "top_entry_roads": [{"road": r, "vehicles": c} for r, c in top_roads],
    }


def expand_entries(flow_json):
    """把 flow（含 vehicle_count 的中间产物）统一成带 _count 的条目列表。"""
    if isinstance(flow_json, dict):
        entries = flow_json.get("flows") or []
    else:
        entries = flow_json or []
    out = []
    for e in entries:
        item = dict(e)
        try:
            item["_count"] = int(item.pop("vehicle_count", 1) or 1)
        except (TypeError, ValueError):
            item["_count"] = 1
        out.append(item)
    return out
