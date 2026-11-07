# -*- coding: utf-8 -*-
"""
data_schema.py —— ./data 数据集读取 + 严格字段校验 + 运行时维度推导（全工程唯一权威）

对应论文4.2节：状态空间、动作空间、邻居关系的**特征来源**全部以本文件解析出的
真实数据字段为准。

【本文件承担的三项强制职责】
1. 【只读】只以 'r' 模式打开 ./data 下文件，绝不写入 / 生成 / 覆盖 data 目录内容。
2. 【数据合法性校验】需求方硬性约束：
   - ./data 不存在 → 抛异常并给出提示；
   - 论文4.2节状态空间/动作空间所依赖的必要字段若在真实数据中缺失
     → 抛 ValueError 并**打印缺失字段名**，程序终止；
   - 任何情况下都不做「自动填充 / 虚构补全」。
3. 【动态维度推导】模型输入维度、动作维度、邻居关系维度全部在运行时由真实数据推导，
   禁止在 config 或模型里写死（例如本数据 state_dim=69、rel_pos 类别=4、可选相位=4，
   这些都必须在运行时算出来，而不是常量）。

【与 CityFlow C++ 引擎的口径对齐】（保证 Python 侧算出的静态信息与仿真引擎一致）
- 车道 id 命名：`<road_id>_<lane_index>`（roadnet.h:323-324 Lane::getId）
- 车道长度：复刻 Road::initLanesPoints()（roadnet.cpp:456-505）：
  两端按交叉口 width 内缩，再按车道宽度做横向偏移，最后取折线长度作为车道长度。
- 排队判定：速度 < 0.1（engine.cpp:641）
- 相位切换：engine.set_tl_phase(inter_id, phaseIndex)，phaseIndex 为
  trafficLight.lightphases 的**下标**（engine.cpp:719, trafficlight.cpp:39）
- 仿真流文件格式：CityFlow 要求 flow 文件顶层为**数组**（engine.cpp:114）
"""

# 兼容 Python 3.8（Ubuntu conda 环境常见版本）：启用注解延迟求值
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config


# ===========================================================================
# 0. 异常与严格字段访问工具
# ===========================================================================

class DataValidationError(ValueError):
    """数据校验失败（继承 ValueError，满足需求方「缺失必要字段直接抛 ValueError」）。"""


def _fmt_missing(ctx: str, missing: Sequence[str]) -> str:
    return "[数据校验失败] {} 缺失必要字段: {}".format(ctx, list(missing))


def require_keys(obj: Dict[str, Any], keys: Sequence[str], ctx: str) -> None:
    """严格字段访问：obj 必须同时包含 keys 中所有键，否则抛 ValueError 并列出缺失字段。"""
    if not isinstance(obj, dict):
        raise DataValidationError(
            "[数据校验失败] {} 期望 dict，实际为 {}".format(ctx, type(obj).__name__))
    missing = [k for k in keys if k not in obj]
    if missing:
        raise DataValidationError(_fmt_missing(ctx, missing))


def get_typed(obj: Dict[str, Any], key: str, ctx: str, types, allow_none: bool = False):
    """严格类型访问：缺失 → ValueError；类型不符 → ValueError。"""
    if key not in obj:
        raise DataValidationError(_fmt_missing(ctx, [key]))
    val = obj[key]
    if val is None:
        if allow_none:
            return None
        raise DataValidationError(
            "[数据校验失败] {}.{} 值为 null，但本工程需要该字段的真实取值".format(ctx, key))
    if not isinstance(val, types):
        raise DataValidationError(
            "[数据校验失败] {}.{} 类型应为 {}，实际为 {}".format(
                ctx, key, types, type(val).__name__))
    return val


# --- 论文4.2节所依赖的必要字段清单（缺失即终止） ---------------------------

REQUIRED_ROADNET_TOP = ("intersections", "roads")
REQUIRED_INTERSECTION_COMMON = ("id", "point", "virtual", "roads", "roadLinks")
REQUIRED_INTERSECTION_SIGNAL = ("width", "trafficLight")
REQUIRED_ROAD_KEYS = ("id", "points", "lanes", "startIntersection", "endIntersection")
REQUIRED_LANE_KEYS = ("width", "maxSpeed")
REQUIRED_ROADLINK_KEYS = ("type", "startRoad", "endRoad", "laneLinks")
REQUIRED_LANELINK_KEYS = ("startLaneIndex", "endLaneIndex")
REQUIRED_TRAFFICLIGHT_KEYS = ("lightphases",)
REQUIRED_LIGHTPHASE_KEYS = ("time", "availableRoadLinks")

# CityFlow 流文件条目字段（engine.cpp:116-150 逐一读取，缺一不可）
REQUIRED_FLOW_ITEM_KEYS = ("vehicle", "route", "interval", "startTime", "endTime")
REQUIRED_FLOW_VEHICLE_KEYS = (
    "length", "width", "maxPosAcc", "maxNegAcc",
    "usualPosAcc", "usualNegAcc", "minGap", "maxSpeed", "headwayTime",
)

# 运行时引擎（CityFlow）必须提供的车辆信息字段：三者缺一，五元组里的
# seg1/seg2/seg3（依赖 distance）与 行驶车辆（依赖 speed）就无法由真实数据得到。
REQUIRED_ENGINE_VEHICLE_FIELDS = ("running", "distance", "speed")

# 论文4.2节 单车道五元组顺序（顺序即状态向量排布，禁止调换）
LANE_FEATURE_NAMES = ("sq", "sv", "seg1", "seg2", "seg3")

# 动作空间：相位动作 + 相位持续时间动作（双输出）
PHASE_ACTION_NAME = "phase"
DURATION_ACTION_NAME = "duration"

# CityFlow roadLink 合法类型（roadnet.cpp typeMap）；出现未知类型则引擎会直接报错，
# 因此这里提前拦截，避免训练中途才失败。
VALID_ROADLINK_TYPES = ("go_straight", "turn_left", "turn_right")

# 相位是否「放行」的判定：可用 roadLink 中至少含一个直行或左转
_GREEN_LINK_TYPES = ("go_straight", "turn_left")


# ===========================================================================
# 1. 数据文件发现（路径判断：./data 不存在 → 终止）
# ===========================================================================

def resolve_data_files(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    在 data_dir 下发现 路网 json / 需求 json / 定时配时 txt。
    任何一项找不到都抛 ValueError（绝不自动生成或回退到虚构数据）。
    """
    data_dir = os.path.abspath(data_dir or config.DATA_DIR)
    if not os.path.exists(data_dir):
        raise DataValidationError(
            "[路径错误] 数据集目录不存在: {}\n"
            "        请把 CityFlow 杭州数据（roadnet*.json / flow*.json / "
            "signal_inter_*.txt）放入该目录后重试；\n"
            "        若数据放在别处，可设置环境变量 RL2_DATA_DIR 或使用 --data-dir 参数。"
            .format(data_dir))
    if not os.path.isdir(data_dir):
        raise DataValidationError("[路径错误] 数据集路径不是目录: {}".format(data_dir))

    def _find(patterns: Sequence[str], what: str) -> List[str]:
        found: List[str] = []
        for pat in patterns:
            found.extend(sorted(glob.glob(os.path.join(data_dir, pat))))
        # 去重（不同 pattern 可能命中同一文件）并排除明显不属于该类别的文件
        uniq, seen = [], set()
        for p in found:
            rp = os.path.realpath(p)
            if rp not in seen:
                seen.add(rp)
                uniq.append(p)
        if not uniq:
            raise DataValidationError(
                "[数据缺失] 在 {} 下未找到{}（匹配模式 {}）。".format(data_dir, what, list(patterns)))
        return uniq

    roadnets = _find(config.ROADNET_PATTERNS, "CityFlow 路网文件")
    flows = _find(config.FLOW_PATTERNS, "CityFlow 交通需求文件")
    plans = _find(config.SIGNAL_PLAN_PATTERNS, "信号配时 txt 文件")

    if len(roadnets) > 1:
        raise DataValidationError(
            "[数据歧义] 发现多个路网文件，请只保留一个: {}".format(roadnets))
    if len(flows) > 1:
        raise DataValidationError(
            "[数据歧义] 发现多个需求文件，请只保留一个: {}".format(flows))

    files = {
        "data_dir": data_dir,
        "roadnet": roadnets[0],
        "flow": flows[0],
        "signal_plans": plans,
    }
    # 统计目录内全部文件（用于自检报告，证明「只读且看清了数据」）
    all_files = []
    for name in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            all_files.append({"name": name, "size": os.path.getsize(p)})
    files["all_files"] = all_files
    return files


def read_json(path: str) -> Any:
    """只读方式加载 json；解析失败直接抛异常（不吞错）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise DataValidationError(
            "[数据解析失败] 不是合法 JSON: {} ({})".format(path, e))


# ===========================================================================
# 2. 静态结构：车道 / 相位 / 交叉口 / 邻居关系
# ===========================================================================

@dataclass
class LaneSpec:
    """单条车道的静态属性（论文4.2节「单车道五元组」的承载对象）。"""
    lane_id: str            # CityFlow 口径：<road_id>_<lane_index>
    road_id: str
    lane_index: int
    length: float           # 与引擎一致的车道长度（m）
    width: float
    max_speed: float
    end_intersection: str   # 该车道的下游交叉口 id（可能为 virtual）
    incoming_of: Optional[str]  # 若下游是受控交叉口，则为该交叉口 id（即「驶入车道」）


@dataclass
class PhaseSpec:
    """信号相位（trafficLight.lightphases 的一项）。"""
    phase_index: int
    time: float
    available_road_links: List[int]
    link_types: List[str]
    is_green: bool          # 含直行或左转 → 放行相位；仅含右转 → 全红/清空相位


@dataclass
class IntersectionSpec:
    """受控交叉口（virtual=False）的静态结构。"""
    inter_id: str
    x: float
    y: float
    width: float
    incoming_roads: List[str]
    outgoing_roads: List[str]
    incoming_lanes: List[LaneSpec]
    phases: List[PhaseSpec]
    selectable_phase_indices: List[int]     # 动作掩码允许的相位（运行时由真实数据推导）
    neighbors: List[str]                    # 唯一邻居（受控交叉口）
    rel_offset: Dict[str, Tuple[float, float]]   # 邻居 → 相对位置 (dx, dy)
    rel_class: Dict[str, int]               # 邻居 → 相对位置 one-hot 类别下标
    rel_classes: List[Tuple[float, float]]  # 本交叉口邻居的相对位置类别（有序）
    state_dim: int                          # 该交叉口状态向量维度（运行时推导）
    n_phases: int                           # lightphases 数量（当前相位 one-hot 维度）

    @property
    def n_incoming_lanes(self) -> int:
        return len(self.incoming_lanes)

    @property
    def n_selectable_phases(self) -> int:
        return len(self.selectable_phase_indices)

    def phase_block_start(self) -> int:
        """状态向量中「当前相位 one-hot」块的起始下标。"""
        return self.n_incoming_lanes * len(LANE_FEATURE_NAMES)


@dataclass
class SignalPlan:
    """定时配时方案（signal_inter_*.txt）：(时间, 相位索引) 事件序列。"""
    inter_id: str
    path: str
    events: List[Tuple[float, int]]
    phases_used: List[int]
    t_end: float
    green_durations: Dict[int, float]
    cycle_seconds: Optional[float]
    n_lines: int


@dataclass
class FlowSpec:
    """交通需求（flow*.json）概要。"""
    path: str
    is_cityflow_array: bool     # True = 顶层数组（CityFlow engine.cpp:114 要求的格式）
    n_flows: int
    interval: float
    t_min: float
    t_max: float
    horizon_seconds: float
    vehicle_fields: List[str]
    route_len_min: int
    route_len_max: int
    n_unique_routes: int
    start_time_field: str
    end_time_field: str


@dataclass
class Dims:
    """
    运行时推导出的全部维度（模型/回放池/动作掩码一律以此为唯一来源）。
    config.py 中不定义任何维度，正是为了强制走这里。
    """
    n_intersections: int
    state_dim_per_inter: Dict[str, int]
    state_dim_max: int
    n_incoming_lanes_per_inter: Dict[str, int]
    n_phases_per_inter: Dict[str, int]
    n_selectable_phases_per_inter: Dict[str, int]
    n_selectable_phases_max: int
    neighbor_degree_max: int
    n_rel_pos_classes: int
    n_rel_class_groups: int
    n_duration_actions: int
    lane_feature_dim: int
    lane_feature_names: Tuple[str, ...]


@dataclass
class DataSchema:
    """./data 的完整解析结果：所有下游模块唯一的数据入口。"""
    data_files: Dict[str, Any]
    roadnet: Dict[str, Any]
    flow: FlowSpec
    intersections: List[IntersectionSpec]
    inter_by_id: Dict[str, IntersectionSpec]
    rel_classes: List[Tuple[float, float]]
    rel_class_group: List[int]          # 相对位置类别 → 分组（0=东西向, 1=南北向）
    signal_plans: Dict[str, SignalPlan]
    dims: Dims
    action_mask_source: str
    all_road_ids: List[str]
    fields_report: Dict[str, Any] = field(default_factory=dict)

    # --- 供下游使用的便捷方法 ---
    def controlled_ids(self) -> List[str]:
        return [it.inter_id for it in self.intersections]

    def action_mask(self, inter_id: str) -> List[bool]:
        """该交叉口的相位动作掩码（长度 = 该交叉口可选相位个数，恒为 True，
        掩码的意义在于异构交叉口之间用 None 补齐；见 env/model 中的 padding 说明）。"""
        n = self.inter_by_id[inter_id].n_selectable_phases
        return [True] * n

    def read_only_notice(self) -> str:
        return ("数据目录 {} 全程只读：本工程不写入、不生成、不覆写其中任何文件。"
                .format(self.data_files["data_dir"]))


# ===========================================================================
# 3. 车道长度：复刻 CityFlow Road::initLanesPoints()（roadnet.cpp:456-505）
# ===========================================================================

def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _vec_len(v):
    return math.hypot(v[0], v[1])


def _vec_unit(v):
    n = _vec_len(v)
    if n < 1e-12:
        raise DataValidationError("[数据校验失败] 路网出现零长度路段向量，无法计算车道长度")
    return (v[0] / n, v[1] / n)


def _polyline_length(points: Sequence[Tuple[float, float]]) -> float:
    return sum(_vec_len(_vec_sub(points[i + 1], points[i])) for i in range(len(points) - 1))


def _lane_geometry_lengths(road: Dict[str, Any], inter_map: Dict[str, Any]) -> List[float]:
    """
    按 CityFlow 的口径计算一条路上各车道的长度（返回按 lane_index 排序的列表）。

    步骤（与 roadnet.cpp:456-505 一致）：
      1) 若起点交叉口非 virtual：P0 ← P0 + unit(P1-P0) * width_start
      2) 若终点交叉口非 virtual：Pn ← Pn - unit(Pn-Pn-1) * width_end
      3) 对每条车道，按横向偏移 (dmin+dmax)/2 平移折线，长度取折线弧长
    对于只有 2 个点的直线路段，步骤 3 的横向偏移不改变长度，等价于「路段长度减去两端
    交叉口宽度」。
    """
    ctx = "road[{}]".format(road.get("id"))
    pts = [(float(p["x"]), float(p["y"])) for p in road["points"]]
    if len(pts) < 2:
        raise DataValidationError(
            "[数据校验失败] {} 的 points 少于 2 个点，无法确定车道走向".format(ctx))

    start_id = road["startIntersection"]
    end_id = road["endIntersection"]
    if start_id not in inter_map:
        raise DataValidationError(
            "[数据校验失败] {} 的 startIntersection='{}' 在 intersections 中不存在"
            .format(ctx, start_id))
    if end_id not in inter_map:
        raise DataValidationError(
            "[数据校验失败] {} 的 endIntersection='{}' 在 intersections 中不存在"
            .format(ctx, end_id))

    start_virtual = bool(inter_map[start_id].get("virtual", False))
    end_virtual = bool(inter_map[end_id].get("virtual", False))

    if not start_virtual:
        w = float(inter_map[start_id]["width"])
        u = _vec_unit(_vec_sub(pts[1], pts[0]))
        pts[0] = (pts[0][0] + u[0] * w, pts[0][1] + u[1] * w)
    if not end_virtual:
        w = float(inter_map[end_id]["width"])
        u = _vec_unit(_vec_sub(pts[-1], pts[-2]))
        pts[-1] = (pts[-1][0] - u[0] * w, pts[-1][1] - u[1] * w)

    lane_lengths: List[float] = []
    dsum = 0.0
    for lane in road["lanes"]:
        require_keys(lane, REQUIRED_LANE_KEYS, "{}.lanes[?]".format(ctx))
        lw = float(lane["width"])
        dmin, dmax = dsum, dsum + lw
        mid = (dmin + dmax) / 2.0
        lane_points: List[Tuple[float, float]] = []
        for j in range(len(pts)):
            if j == 0:
                u = _vec_unit(_vec_sub(pts[1], pts[0]))
            elif j + 1 == len(pts):
                u = _vec_unit(_vec_sub(pts[j], pts[j - 1]))
            else:
                u1 = _vec_unit(_vec_sub(pts[j + 1], pts[j]))
                u2 = _vec_unit(_vec_sub(pts[j], pts[j - 1]))
                u = _vec_unit((u1[0] + u2[0], u1[1] + u2[1]))
            # C++ 中 v = -u.normal()；Vector::normal() = (-y, x)，故 v = (u.y, -u.x)
            v = (u[1], -u[0])
            lane_points.append((pts[j][0] + v[0] * mid, pts[j][1] + v[1] * mid))
        length = _polyline_length(lane_points)
        if not (length > 0):
            raise DataValidationError(
                "[数据校验失败] {} 第 {} 条车道长度为 {}，路网几何异常"
                .format(ctx, len(lane_lengths), length))
        lane_lengths.append(length)
        dsum += lw
    return lane_lengths


# ===========================================================================
# 4. 路网解析
# ===========================================================================

def parse_roadnet(roadnet: Dict[str, Any]) -> Dict[str, Any]:
    """校验并解析路网 json，返回结构化结果（不修改入参）。"""
    require_keys(roadnet, REQUIRED_ROADNET_TOP, "roadnet(顶层)")
    inter_list = get_typed(roadnet, "intersections", "roadnet", list)
    road_list = get_typed(roadnet, "roads", "roadnet", list)
    if not inter_list or not road_list:
        raise DataValidationError("[数据校验失败] roadnet 的 intersections / roads 不能为空")

    inter_map: Dict[str, Dict[str, Any]] = {}
    for i, inter in enumerate(inter_list):
        ctx = "roadnet.intersections[{}]".format(i)
        require_keys(inter, REQUIRED_INTERSECTION_COMMON, ctx)
        iid = inter["id"]
        if iid in inter_map:
            raise DataValidationError("[数据校验失败] 交叉口 id 重复: {}".format(iid))
        is_virtual = bool(get_typed(inter, "virtual", ctx, bool))
        if not is_virtual:
            require_keys(inter, REQUIRED_INTERSECTION_SIGNAL, ctx)
        pt = inter["point"]
        require_keys(pt, ("x", "y"), ctx + ".point")
        inter_map[iid] = inter

    road_map: Dict[str, Dict[str, Any]] = {}
    lane_lengths: Dict[str, List[float]] = {}
    for i, road in enumerate(road_list):
        ctx = "roadnet.roads[{}]".format(i)
        require_keys(road, REQUIRED_ROAD_KEYS, ctx)
        rid = road["id"]
        if rid in road_map:
            raise DataValidationError("[数据校验失败] 路段 id 重复: {}".format(rid))
        for ln, lane in enumerate(road["lanes"]):
            require_keys(lane, REQUIRED_LANE_KEYS, "{}.lanes[{}]".format(ctx, ln))
            if float(lane["maxSpeed"]) <= 0 or float(lane["width"]) <= 0:
                raise DataValidationError(
                    "[数据校验失败] {}.lanes[{}] 的 width/maxSpeed 必须为正数".format(ctx, ln))
        lane_lengths[rid] = _lane_geometry_lengths(road, inter_map)
        road_map[rid] = road

    # 受控交叉口：roadLinks / trafficLight 深度校验
    road_link_types: Dict[str, List[str]] = {}
    for inter in inter_list:
        if inter.get("virtual", False):
            continue
        ctx = "intersection[{}]".format(inter["id"])
        rls = get_typed(inter, "roadLinks", ctx, list)
        types: List[str] = []
        for k, rl in enumerate(rls):
            rctx = "{}.roadLinks[{}]".format(ctx, k)
            require_keys(rl, REQUIRED_ROADLINK_KEYS, rctx)
            t = rl["type"]
            if t not in VALID_ROADLINK_TYPES:
                raise DataValidationError(
                    "[数据校验失败] {} type='{}' 不是 CityFlow 合法 roadLink 类型 {}"
                    .format(rctx, t, list(VALID_ROADLINK_TYPES)))
            if rl["startRoad"] not in road_map:
                raise DataValidationError(
                    "[数据校验失败] {} startRoad='{}' 不在 roads 中".format(rctx, rl["startRoad"]))
            if rl["endRoad"] not in road_map:
                raise DataValidationError(
                    "[数据校验失败] {} endRoad='{}' 不在 roads 中".format(rctx, rl["endRoad"]))
            for m, ll in enumerate(rl["laneLinks"]):
                require_keys(ll, REQUIRED_LANELINK_KEYS,
                             "{}.laneLinks[{}]".format(rctx, m))
                si, ei = ll["startLaneIndex"], ll["endLaneIndex"]
                n_start = len(road_map[rl["startRoad"]]["lanes"])
                n_end = len(road_map[rl["endRoad"]]["lanes"])
                if not (0 <= si < n_start) or not (0 <= ei < n_end):
                    raise DataValidationError(
                        "[数据校验失败] {}.laneLinks[{}] 车道下标越界 "
                        "(startLaneIndex={}, endLaneIndex={})".format(rctx, m, si, ei))
            types.append(t)
        tl = get_typed(inter, "trafficLight", ctx, dict)
        require_keys(tl, REQUIRED_TRAFFICLIGHT_KEYS, ctx + ".trafficLight")
        phases = get_typed(tl, "lightphases", ctx + ".trafficLight", list)
        if not phases:
            raise DataValidationError(
                "[数据校验失败] {} 的 lightphases 为空，无法构造相位动作空间".format(ctx))
        for pi, ph in enumerate(phases):
            pctx = "{}.trafficLight.lightphases[{}]".format(ctx, pi)
            require_keys(ph, REQUIRED_LIGHTPHASE_KEYS, pctx)
            avail = ph["availableRoadLinks"]
            if not isinstance(avail, list):
                raise DataValidationError(
                    "[数据校验失败] {}.availableRoadLinks 应为 list".format(pctx))
            for a in avail:
                if not isinstance(a, int) or isinstance(a, bool):
                    raise DataValidationError(
                        "[数据校验失败] {}.availableRoadLinks 元素应为 int，实际 {}"
                        .format(pctx, a))
                if not (0 <= a < len(rls)):
                    raise DataValidationError(
                        "[数据校验失败] {}.availableRoadLinks 含越界下标 {}（roadLinks 长度 {}）"
                        .format(pctx, a, len(rls)))
        road_link_types[inter["id"]] = types

    return {
        "inter_map": inter_map,
        "road_map": road_map,
        "lane_lengths": lane_lengths,
        "road_link_types": road_link_types,
    }


def _build_lane_specs(road: Dict[str, Any], lengths: List[float],
                      inter_map: Dict[str, Any]) -> List[LaneSpec]:
    """构造一条路上全部车道的 LaneSpec（lane_id 与 CityFlow 命名一致）。"""
    rid = road["id"]
    end_id = road["endIntersection"]
    end_virtual = bool(inter_map[end_id].get("virtual", False))
    incoming_of = None if end_virtual else end_id
    out: List[LaneSpec] = []
    for k, lane in enumerate(road["lanes"]):
        out.append(LaneSpec(
            lane_id="{}_{}".format(rid, k),   # roadnet.h:323-324 Lane::getId
            road_id=rid,
            lane_index=k,
            length=float(lengths[k]),
            width=float(lane["width"]),
            max_speed=float(lane["maxSpeed"]),
            end_intersection=end_id,
            incoming_of=incoming_of,
        ))
    return out


def build_intersection_specs(parsed: Dict[str, Any], signal_plans: Dict[str, SignalPlan],
                             action_mask_source: str) -> List[IntersectionSpec]:
    """
    由路网 + 定时配时方案构造受控交叉口静态结构，并**运行时推导**每交叉口状态维度。

    动作掩码来源（两者都是真实数据，不含任何虚构相位）：
      - "signal_plan"  : 定时方案 txt 中实际出现的相位（交集于路网绿相位）
      - "roadnet_green": 路网中所有含直行/左转的绿相位
    """
    inter_map = parsed["inter_map"]
    road_map = parsed["road_map"]
    lane_lengths = parsed["lane_lengths"]
    road_link_types = parsed["road_link_types"]

    if action_mask_source not in ("signal_plan", "roadnet_green"):
        raise DataValidationError(
            "[配置错误] ACTION_MASK_SOURCE 只能为 'signal_plan' 或 'roadnet_green'，"
            "实际为 {!r}".format(action_mask_source))

    controlled = [inter for inter in inter_map.values() if not inter.get("virtual", False)]
    controlled.sort(key=lambda z: z["id"])   # 稳定顺序，保证实验可复现

    specs: List[IntersectionSpec] = []
    for inter in controlled:
        iid = inter["id"]
        ctx = "intersection[{}]".format(iid)
        width = float(inter["width"])
        x, y = float(inter["point"]["x"]), float(inter["point"]["y"])

        # --- 驶入/驶出路段：由 roads 字段 + 各路段两端交叉口判定 ---
        incoming_roads, outgoing_roads = [], []
        for rid in inter["roads"]:
            if rid not in road_map:
                raise DataValidationError(
                    "[数据校验失败] {}.roads 引用了不存在路段 '{}'".format(ctx, rid))
            road = road_map[rid]
            if road["endIntersection"] == iid:
                incoming_roads.append(rid)
            elif road["startIntersection"] == iid:
                outgoing_roads.append(rid)

        incoming_roads.sort()
        outgoing_roads.sort()

        # --- 驶入车道（论文4.2节：状态由各驶入车道的五元组构成）---
        incoming_lanes: List[LaneSpec] = []
        for rid in incoming_roads:
            incoming_lanes.extend(_build_lane_specs(road_map[rid], lane_lengths[rid], inter_map))
        if not incoming_lanes:
            raise DataValidationError(
                "[数据校验失败] {} 没有驶入车道，无法构造论文4.2节要求的单车道五元组状态"
                .format(ctx))

        # --- 相位集合 ---
        phases: List[PhaseSpec] = []
        types = road_link_types[iid]
        for pi, ph in enumerate(inter["trafficLight"]["lightphases"]):
            avail = list(ph["availableRoadLinks"])
            link_types = [types[a] for a in avail]
            is_green = any(t in _GREEN_LINK_TYPES for t in link_types)
            phases.append(PhaseSpec(
                phase_index=pi, time=float(ph["time"]),
                available_road_links=avail, link_types=link_types, is_green=is_green))

        green_idx = [p.phase_index for p in phases if p.is_green]
        if not green_idx:
            raise DataValidationError(
                "[数据校验失败] {} 的 lightphases 中找不到任何含直行/左转的放行相位，"
                "无法构造相位动作空间".format(ctx))

        if action_mask_source == "signal_plan":
            if iid not in signal_plans:
                raise DataValidationError(
                    "[数据缺失] 交叉口 {} 缺少定时配时文件 {}（动作掩码来源设为 signal_plan）。\n"
                    "        请把该 txt 放入 ./data，或把 config.ACTION_MASK_SOURCE "
                    "改为 'roadnet_green'。".format(
                        iid, config.SIGNAL_PLAN_FILE_PREFIX + iid + config.SIGNAL_PLAN_FILE_SUFFIX))
            used = set(signal_plans[iid].phases_used)
            selectable = [p for p in green_idx if p in used]
            if not selectable:
                raise DataValidationError(
                    "[数据校验失败] 交叉口 {} 的定时方案相位集合 {} 与路网放行相位集合 {} 无交集，"
                    "无法生成动作掩码".format(iid, sorted(used), green_idx))
        else:
            selectable = list(green_idx)

        # --- 邻居交叉口与相对位置（论文4.2节 邻居关系特征融合模块的输入 r(x,y)）---
        neighbor_offsets: Dict[str, Tuple[float, float]] = {}
        for rid in list(incoming_roads) + list(outgoing_roads):
            road = road_map[rid]
            other = road["startIntersection"] if road["endIntersection"] == iid else road["endIntersection"]
            if other == iid:
                continue
            other_inter = inter_map[other]
            if other_inter.get("virtual", False):
                continue     # 虚拟边界交叉口不是受控邻居
            dx = float(other_inter["point"]["x"]) - x
            dy = float(other_inter["point"]["y"]) - y
            neighbor_offsets[other] = (dx, dy)

        neighbors = sorted(neighbor_offsets.keys())
        state_dim = len(incoming_lanes) * len(LANE_FEATURE_NAMES) + len(phases)

        specs.append(IntersectionSpec(
            inter_id=iid, x=x, y=y, width=width,
            incoming_roads=incoming_roads, outgoing_roads=outgoing_roads,
            incoming_lanes=incoming_lanes, phases=phases,
            selectable_phase_indices=selectable,
            neighbors=neighbors,
            rel_offset=neighbor_offsets,
            rel_class={},                    # 全局类别表建好后回填
            rel_classes=[],
            state_dim=state_dim,
            n_phases=len(phases),
        ))
    return specs


def build_relative_position_table(specs: List[IntersectionSpec]) -> Tuple[
        List[Tuple[float, float]], List[int]]:
    """
    由真实路网几何推导**相对位置 one-hot 的类别表** [r(x,y)]（论文4.2节）。

    - 类别 = 受控邻居间出现的所有唯一 (dx, dy)，按字典序排序，类别数自动推导；
    - 同时给出每个类别的分组（0=东西向, 1=南北向），供 NS 模块「分组 softmax」使用：
      东西向 = 邻居在 x 方向偏移；南北向 = 邻居在 y 方向偏移。
      若数据中出现既非东西也非南北的对角邻居，则论文的两分组结构无法定义 → 直接报错。
    """
    offsets = set()
    for sp in specs:
        for _, off in sp.rel_offset.items():
            offsets.add((round(off[0], 6), round(off[1], 6)))
    if not offsets:
        raise DataValidationError(
            "[数据校验失败] 路网中不存在任何受控邻居关系（交叉口之间无相邻路段），"
            "无法构造论文4.2节邻居信息融合模块所需的 r(x,y)")
    rel_classes = sorted(offsets)

    groups: List[int] = []
    for dx, dy in rel_classes:
        if abs(dy) < 1e-9 and abs(dx) > 1e-9:
            groups.append(0)        # 东西向
        elif abs(dx) < 1e-9 and abs(dy) > 1e-9:
            groups.append(1)        # 南北向
        else:
            raise DataValidationError(
                "[数据校验失败] 邻居相对位置 (dx={}, dy={}) 既不是东西向也不是南北向，"
                "论文4.2节 NS 模块的「东西/南北」两分组 softmax 无法定义".format(dx, dy))

    for sp in specs:
        sp.rel_classes = list(rel_classes)
        sp.rel_class = {n: rel_classes.index(
            (round(sp.rel_offset[n][0], 6), round(sp.rel_offset[n][1], 6)))
            for n in sp.neighbors}
    return rel_classes, groups


# ===========================================================================
# 5. 交通需求解析
# ===========================================================================

def parse_flow(flow_json: Any, path: str, road_ids: Sequence[str]) -> FlowSpec:
    """
    校验 flow 数据。

    CityFlow 要求流文件顶层为数组（engine.cpp:114 `if (!root.IsArray())`），本数据集的
    ./data/flow.json 正是该数组格式：每项含 vehicle / route / interval / startTime / endTime。
    为兼容历史数据，这里也接受 {interval, routes, flows} 字典格式（此时由 env 在
    output/generated 下派生 CityFlow 数组文件，绝不改动 ./data）。
    """
    road_id_set = set(road_ids)
    if isinstance(flow_json, list):
        items = flow_json
        is_array = True
    elif isinstance(flow_json, dict):
        require_keys(flow_json, ("flows",), "flow(顶层 dict)")
        items = flow_json["flows"]
        is_array = False
        if not isinstance(items, list) or not items:
            raise DataValidationError("[数据校验失败] flow 的 flows 字段必须是非空数组")
    else:
        raise DataValidationError(
            "[数据校验失败] flow 文件顶层应为数组（CityFlow 要求）或含 flows 的字典，"
            "实际为 {}".format(type(flow_json).__name__))

    t_min, t_max = None, None
    intervals = set()
    veh_fields = set()
    route_lens = []
    routes = set()
    for i, item in enumerate(items):
        ctx = "flow[{}]".format(i)
        require_keys(item, REQUIRED_FLOW_ITEM_KEYS, ctx)
        veh = item["vehicle"]
        require_keys(veh, REQUIRED_FLOW_VEHICLE_KEYS, ctx + ".vehicle")
        veh_fields.update(veh.keys())
        route = item["route"]
        if not isinstance(route, list) or not route:
            raise DataValidationError(
                "[数据校验失败] {}.route 必须是非空路段 id 数组".format(ctx))
        for r in route:
            if not isinstance(r, str):
                raise DataValidationError(
                    "[数据校验失败] {}.route 元素应为字符串路段 id".format(ctx))
            if r not in road_id_set:
                raise DataValidationError(
                    "[数据校验失败] {}.route 引用了路网中不存在的路段 '{}'"
                    .format(ctx, r))
        route_lens.append(len(route))
        routes.add(tuple(route))
        iv = float(item["interval"])
        if iv <= 0:
            raise DataValidationError("[数据校验失败] {}.interval 必须为正数".format(ctx))
        intervals.add(round(iv, 9))
        st, et = float(item["startTime"]), float(item["endTime"])
        t_min = st if t_min is None else min(t_min, st)
        t_max = et if t_max is None else max(t_max, et)

    if len(intervals) > 1:
        # CityFlow 支持每条 flow 独立 interval，这里仅提示不报错
        pass
    main_interval = sorted(intervals)[0]
    horizon = float(t_max) + max(main_interval, config.SIM_INTERVAL)
    return FlowSpec(
        path=path,
        is_cityflow_array=is_array,
        n_flows=len(items),
        interval=float(main_interval),
        t_min=float(t_min if t_min is not None else 0.0),
        t_max=float(t_max if t_max is not None else 0.0),
        horizon_seconds=horizon,
        vehicle_fields=sorted(veh_fields),
        route_len_min=min(route_lens) if route_lens else 0,
        route_len_max=max(route_lens) if route_lens else 0,
        n_unique_routes=len(routes),
        start_time_field="startTime",
        end_time_field="endTime",
    )


# ===========================================================================
# 6. 定时配时 txt 解析
# ===========================================================================

def parse_signal_plan(path: str, inter_id: str) -> SignalPlan:
    """解析 `time,phaseIndex` 逐行形式的定时配时文件（本数据为 15s 绿 + 5s 黄的结构）。"""
    events: List[Tuple[float, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            parts = s.replace("\t", " ").split(",") if "," in s else s.split()
            if len(parts) < 2:
                raise DataValidationError(
                    "[数据校验失败] 配时文件 {} 第 {} 行无法解析为 'time,phase': {!r}"
                    .format(path, ln, s))
            try:
                t = float(parts[0])
                p = int(float(parts[1]))
            except ValueError:
                raise DataValidationError(
                    "[数据校验失败] 配时文件 {} 第 {} 行数值解析失败: {!r}".format(path, ln, s))
            events.append((t, p))
    if not events:
        raise DataValidationError("[数据校验失败] 配时文件为空: {}".format(path))
    times = [e[0] for e in events]
    if any(times[i + 1] <= times[i] for i in range(len(times) - 1)):
        raise DataValidationError(
            "[数据校验失败] 配时文件 {} 的时间序列非严格递增，无法作为时间轴使用".format(path))
    phases_used = sorted(set(e[1] for e in events))

    # 绿色时长：同一相位在时间轴上连续出现的区间长度（同一相位出现两次 = 绿+黄/清空段）
    green_dur: Dict[int, float] = {}
    seg_start = events[0][0]
    for i in range(1, len(events)):
        if events[i][1] != events[i - 1][1]:
            dur = events[i][0] - seg_start
            green_dur[events[i - 1][1]] = green_dur.get(events[i - 1][1], 0.0) + dur
            seg_start = events[i][0]

    # 周期：第一次回到首个相位的时刻
    cycle = None
    for i in range(1, len(events)):
        if events[i][1] == events[0][1] and events[i][0] > events[0][0]:
            cycle = events[i][0] - events[0][0]
            break
    return SignalPlan(
        inter_id=inter_id, path=path, events=events, phases_used=phases_used,
        t_end=times[-1], green_durations=green_dur, cycle_seconds=cycle,
        n_lines=len(events),
    )


# ===========================================================================
# 7. 总入口：build_schema
# ===========================================================================

def build_schema(data_dir: Optional[str] = None,
                 action_mask_source: Optional[str] = None,
                 verbose: bool = True) -> DataSchema:
    """
    读取并校验 ./data，返回 DataSchema（含运行时推导的全部维度）。

    任何必要字段缺失都会在此抛 ValueError 并打印缺失字段名，程序立即终止，
    绝不使用虚构特征继续训练。
    """
    action_mask_source = action_mask_source or config.ACTION_MASK_SOURCE
    files = resolve_data_files(data_dir)
    roadnet = read_json(files["roadnet"])
    flow_json = read_json(files["flow"])

    # --- 定时配时文件：文件名 ↔ 交叉口 id ---
    plans: Dict[str, SignalPlan] = {}
    for p in files["signal_plans"]:
        base = os.path.basename(p)
        if base.startswith(config.SIGNAL_PLAN_FILE_PREFIX) and base.endswith(config.SIGNAL_PLAN_FILE_SUFFIX):
            iid = base[len(config.SIGNAL_PLAN_FILE_PREFIX):-len(config.SIGNAL_PLAN_FILE_SUFFIX)]
        else:
            iid = os.path.splitext(base)[0]
        if iid in plans:
            raise DataValidationError(
                "[数据歧义] 交叉口 {} 对应多个配时文件: {} 与 {}".format(
                    iid, plans[iid].path, p))
        plans[iid] = parse_signal_plan(p, iid)

    parsed = parse_roadnet(roadnet)
    specs = build_intersection_specs(parsed, plans, action_mask_source)
    rel_classes, rel_groups = build_relative_position_table(specs)
    flow = parse_flow(flow_json, files["flow"], list(parsed["road_map"].keys()))

    # --- 配时方案的相位下标必须能在路网 lightphases 中找到（否则引擎行为不可解释）---
    for sp in specs:
        plan = plans.get(sp.inter_id)
        if plan is None:
            continue
        bad = [p for p in plan.phases_used if not (0 <= p < sp.n_phases)]
        if bad:
            raise DataValidationError(
                "[数据校验失败] 交叉口 {} 的配时文件使用了越界相位下标 {}（路网相位总数 {}）"
                .format(sp.inter_id, bad, sp.n_phases))

    inter_by_id = {sp.inter_id: sp for sp in specs}
    n_sel = [sp.n_selectable_phases for sp in specs]
    n_lanes = [sp.n_incoming_lanes for sp in specs]
    n_phases = [sp.n_phases for sp in specs]
    degree = [len(sp.neighbors) for sp in specs]

    dims = Dims(
        n_intersections=len(specs),
        state_dim_per_inter={sp.inter_id: sp.state_dim for sp in specs},
        state_dim_max=max(sp.state_dim for sp in specs),
        n_incoming_lanes_per_inter={sp.inter_id: sp.n_incoming_lanes for sp in specs},
        n_phases_per_inter={sp.inter_id: sp.n_phases for sp in specs},
        n_selectable_phases_per_inter={sp.inter_id: sp.n_selectable_phases for sp in specs},
        n_selectable_phases_max=max(n_sel),
        neighbor_degree_max=max(degree),
        n_rel_pos_classes=len(rel_classes),
        n_rel_class_groups=len(set(rel_groups)),
        n_duration_actions=len(config.DURATION_OPTIONS),
        lane_feature_dim=len(LANE_FEATURE_NAMES),
        lane_feature_names=tuple(LANE_FEATURE_NAMES),
    )

    schema = DataSchema(
        data_files=files,
        roadnet=roadnet,
        flow=flow,
        intersections=specs,
        inter_by_id=inter_by_id,
        rel_classes=list(rel_classes),
        rel_class_group=list(rel_groups),
        signal_plans=plans,
        dims=dims,
        action_mask_source=action_mask_source,
        all_road_ids=sorted(parsed["road_map"].keys()),
    )
    schema.fields_report = build_fields_report(schema, parsed)
    if verbose:
        print_fields_report(schema)
    return schema


# ===========================================================================
# 8. 字段清单 + 维度报告（train_rl_aux.py 启动时打印）
# ===========================================================================

def build_fields_report(schema: DataSchema, parsed: Dict[str, Any]) -> Dict[str, Any]:
    """汇总「数据中实际存在哪些字段 + 推导出哪些维度」，可落盘为 json 供审计。"""
    roadnet = schema.roadnet
    inter_list = roadnet["intersections"]
    road_list = roadnet["roads"]
    sig = [x for x in inter_list if not x.get("virtual", False)]
    vir = [x for x in inter_list if x.get("virtual", False)]
    sample_inter = sig[0] if sig else {}
    sample_road = road_list[0] if road_list else {}
    sample_lane = (sample_road.get("lanes") or [{}])[0]
    sample_rl = (sample_inter.get("roadLinks") or [{}])[0]
    sample_phase = ((sample_inter.get("trafficLight") or {}).get("lightphases") or [{}])[0]

    lane_len_all = [ln.length for sp in schema.intersections for ln in sp.incoming_lanes]
    plan = next(iter(schema.signal_plans.values()), None)

    return {
        "data_dir": schema.data_files["data_dir"],
        "data_files": schema.data_files["all_files"],
        "roadnet_file": os.path.basename(schema.data_files["roadnet"]),
        "flow_file": os.path.basename(schema.data_files["flow"]),
        "n_signal_plan_files": len(schema.data_files["signal_plans"]),
        "roadnet_fields": {
            "top_level_keys": sorted(roadnet.keys()),
            "n_intersections": len(inter_list),
            "n_controlled_intersections": len(sig),
            "n_virtual_intersections": len(vir),
            "n_roads": len(road_list),
            "intersection_keys": sorted(sample_inter.keys()),
            "road_keys": sorted(sample_road.keys()),
            "lane_keys": sorted(sample_lane.keys()),
            "roadLink_keys": sorted(sample_rl.keys()),
            "trafficLight_keys": sorted((sample_inter.get("trafficLight") or {}).keys()),
            "lightphase_keys": sorted(sample_phase.keys()),
            "roadLink_types_present": sorted(set(
                rl["type"] for x in sig for rl in x.get("roadLinks", []))),
        },
        "flow_fields": {
            "is_cityflow_top_level_array": schema.flow.is_cityflow_array,
            "n_flows": schema.flow.n_flows,
            "interval": schema.flow.interval,
            "start_time_range": [schema.flow.t_min, schema.flow.t_max],
            "horizon_seconds": schema.flow.horizon_seconds,
            "unique_routes": schema.flow.n_unique_routes,
            "route_len_min": schema.flow.route_len_min,
            "route_len_max": schema.flow.route_len_max,
            "vehicle_fields": schema.flow.vehicle_fields,
        },
        "signal_plan_fields": None if plan is None else {
            "example_file": os.path.basename(plan.path),
            "format": "time,phaseIndex 逐行",
            "n_lines": plan.n_lines,
            "t_end": plan.t_end,
            "phases_used": plan.phases_used,
            "cycle_seconds": plan.cycle_seconds,
            "green_durations": plan.green_durations,
        },
        "derived_dims": {
            "n_intersections": schema.dims.n_intersections,
            "state_dim_per_intersection": schema.dims.state_dim_per_inter,
            "state_dim_max": schema.dims.state_dim_max,
            "lane_feature_dim": schema.dims.lane_feature_dim,
            "lane_feature_names": list(schema.dims.lane_feature_names),
            "n_incoming_lanes_per_intersection": schema.dims.n_incoming_lanes_per_inter,
            "n_lightphases_per_intersection": schema.dims.n_phases_per_inter,
            "n_selectable_phases_per_intersection": schema.dims.n_selectable_phases_per_inter,
            "n_selectable_phases_max": schema.dims.n_selectable_phases_max,
            "neighbor_degree_max": schema.dims.neighbor_degree_max,
            "n_rel_pos_classes": schema.dims.n_rel_pos_classes,
            "n_rel_class_groups": schema.dims.n_rel_class_groups,
            "n_duration_actions": schema.dims.n_duration_actions,
            "rel_pos_classes": [list(c) for c in schema.rel_classes],
            "rel_class_group": schema.rel_class_group,
            "incoming_lane_length_range_m": [min(lane_len_all), max(lane_len_all)],
        },
        "per_intersection": [{
            "id": sp.inter_id,
            "n_incoming_lanes": sp.n_incoming_lanes,
            "n_phases": sp.n_phases,
            "selectable_phase_indices": sp.selectable_phase_indices,
            "state_dim": sp.state_dim,
            "neighbors": sp.neighbors,
            "rel_class": sp.rel_class,
        } for sp in schema.intersections],
        "action_mask_source": schema.action_mask_source,
        "read_only_notice": schema.read_only_notice(),
    }


def print_fields_report(schema: DataSchema) -> None:
    """把「检测到的全部可用字段 + 数据维度」打印到控制台。"""
    r = schema.fields_report
    line = "=" * 78
    print(line)
    print("[数据校验通过] 目录（只读）: {}".format(r["data_dir"]))
    print(line)
    print("发现文件 {} 个：".format(len(r["data_files"])))
    for f in r["data_files"]:
        print("   - {:<44} {:>10} bytes".format(f["name"], f["size"]))

    rn = r["roadnet_fields"]
    print("\n[路网 roadnet] 顶层字段 = {}".format(rn["top_level_keys"]))
    print("   交叉口 {} 个（受控 {} / 虚拟边界 {}），路段 {} 条"
          .format(rn["n_intersections"], rn["n_controlled_intersections"],
                  rn["n_virtual_intersections"], rn["n_roads"]))
    print("   intersection 字段 = {}".format(rn["intersection_keys"]))
    print("   road 字段         = {}".format(rn["road_keys"]))
    print("   lane 字段         = {}".format(rn["lane_keys"]))
    print("   roadLink 字段     = {} （类型集合 {}）"
          .format(rn["roadLink_keys"], rn["roadLink_types_present"]))
    print("   trafficLight 字段 = {}".format(rn["trafficLight_keys"]))
    print("   lightphase 字段   = {}".format(rn["lightphase_keys"]))

    fl = r["flow_fields"]
    print("\n[需求 flow] 顶层数组格式 = {}，flow 条数 = {}，interval = {}"
          .format(fl["is_cityflow_top_level_array"], fl["n_flows"], fl["interval"]))
    print("   时间范围 = {} ~ {} s，推导 episode 时长 = {} s"
          .format(fl["start_time_range"][0], fl["start_time_range"][1], fl["horizon_seconds"]))
    print("   唯一路径 {} 条（长度 {}~{}），vehicle 字段 = {}"
          .format(fl["unique_routes"], fl["route_len_min"], fl["route_len_max"],
                  fl["vehicle_fields"]))

    sp = r["signal_plan_fields"]
    if sp is not None:
        print("\n[定时配时 txt] 例：{}，格式 {}，行数 {}，周期 {} s"
              .format(sp["example_file"], sp["format"], sp["n_lines"], sp["cycle_seconds"]))
        print("   使用相位 = {}，各相位绿灯时长 = {}"
              .format(sp["phases_used"], sp["green_durations"]))

    d = r["derived_dims"]
    print("\n[运行时推导维度]（模型/回放/动作掩码的唯一维度来源，config 中不写死）")
    print("   交叉口数            = {}".format(d["n_intersections"]))
    print("   单交叉口状态维度    = {} （= 驶入车道数 x {} + 相位 one-hot）"
          .format(d["state_dim_per_intersection"], d["lane_feature_dim"]))
    print("   状态维度上界        = {}".format(d["state_dim_max"]))
    print("   驶入车道数/交叉口   = {}".format(d["n_incoming_lanes_per_intersection"]))
    print("   路网相位总数/交叉口 = {}".format(d["n_lightphases_per_intersection"]))
    print("   可选相位动作数      = {} （来源 {}）"
          .format(d["n_selectable_phases_per_intersection"], r["action_mask_source"]))
    print("   最大邻居数          = {}".format(d["neighbor_degree_max"]))
    print("   相对位置 one-hot 维度 = {} 类别 {} 组"
          .format(d["n_rel_pos_classes"], d["n_rel_class_groups"]))
    print("   相对位置类别        = {}".format(d["rel_pos_classes"]))
    print("   持续时间动作数      = {}".format(d["n_duration_actions"]))
    print("   驶入车道长度范围(m) = {}".format(d["incoming_lane_length_range_m"]))
    print("\n[约束] {}".format(r["read_only_notice"]))
    print(line)
