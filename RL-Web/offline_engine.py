# -*- coding: utf-8 -*-
"""
offline_engine.py —— 离线近似仿真引擎（CityFlow 不可用时的本地替代品）

═══════════════════════════════════════════════════════════════════════════
这个文件**不参与 RL 算法**，它只是「仿真器的替代实现」：
    原有代码里 CityFlow 引擎负责提供 (车道车辆数/排队数/车辆位置/信号灯) 这些
    底层事实，RL 环境 env_cityflow.py 再从这些事实里算状态、奖励和指标。
    本机没有编译 cityflow（见 README「运行环境」），因此这里用一个纯 Python 的
    离散排队近似模型，实现 env_cityflow.py 实际会调用的 11 个引擎方法：
        reset / set_random_seed / set_tl_phase / next_step /
        get_lane_vehicle_count / get_lane_waiting_vehicle_count /
        get_lane_vehicles / get_vehicle_info / get_vehicles(True) /
        get_vehicle_count / get_average_travel_time
    → 状态空间、动作掩码、奖励、ATT/AWT/AQL 仍然 100% 由原有 RL 代码计算。

为什么不用 tools/selfcheck.py 的 MockEngine：
    MockEngine 完全不读 flow 数据（每 6 秒给每条驶入车道塞一辆车），
    所以你上传的 csv/json 需求数据不会影响任何指标 → 打分与上传内容无关。
    本引擎按 flow 的 (startTime, endTime, interval, route, vehicle) 注入车辆，
    车辆按路网真实几何与信号相位的 availableRoadLinks 通行，
    因此「换一份流量数据 / 换一套配时方案」都会真实改变 ATT/AWT/AQL。

近似口径（与 CityFlow 的差异，已明确标注，不冒充精确仿真）：
    1) 车辆按所在车道 maxSpeed 匀速前进，不做加减速曲线，只做「跟驰 + 停车线」约束；
    2) 车辆在车道末端（停车线）等待，只有当前相位的 availableRoadLinks 含其
       (startRoad, endRoad) 转向时才可进入下一路段；
    3) 不考虑车道间的换道、横向干扰与转弯减速；
    4) 排队判定 = 速度 < config.QUEUE_SPEED_THRESHOLD（与原 env 采样口径一致）。
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import print_function

import json
import math

import rl_core

# 车辆几何默认值（flow 条目自带 vehicle 字典时以其为准）
DEFAULT_VEH_LENGTH = 5.0
DEFAULT_MIN_GAP = 2.5
DEFAULT_MAX_SPEED = 11.111
_EPS = 1e-6


def _as_list(obj):
    """roadnet 里 intersections / roads 既可能是 list 也可能是 dict，统一成 list。"""
    if isinstance(obj, dict):
        return list(obj.values())
    return list(obj)


def expand_flows(flow_json):
    """
    把 CityFlow flow（json 数组 或 {"flows": [...]}）展开成「(发车时刻, 路线)」序列。

    CityFlow 语义：在 [startTime, endTime) 区间内每 interval 秒发一批车；
    startTime == endTime 时恰好发 1 辆（本项目内置 flow.json 就是这种情况）。
    """
    if isinstance(flow_json, dict):
        entries = flow_json.get("flows") or flow_json.get("flow") or []
    else:
        entries = flow_json or []
    out = []
    for entry in entries:
        route = entry.get("route") or []
        if not isinstance(route, list) or not route:
            continue
        start = float(entry.get("startTime", 0))
        end = float(entry.get("endTime", start))
        interval = float(entry.get("interval", 1) or 1)
        veh = entry.get("vehicle", 1)
        if isinstance(veh, dict):
            count = 1
        else:
            try:
                count = max(1, int(round(float(veh))))
            except (TypeError, ValueError):
                count = 1
        times = []
        if end <= start:
            times = [start]
        else:
            t = start
            guard = 0
            while t < end - _EPS and guard < 200000:
                times.append(t)
                t += max(interval, _EPS)
                guard += 1
        for t in times:
            for _ in range(count):
                out.append((t, list(route)))
    out.sort(key=lambda x: x[0])
    return out


class OfflineEngine(object):
    """env_cityflow.CityFlowTrafficEnv 可接受的 engine 对象（鸭子类型）。"""

    def __init__(self, schema, flow_json=None, verbose=False):
        self.schema = schema
        self.verbose = verbose
        # 【原有核心】路网几何/车道长度直接复用 data_schema.parse_roadnet 的结果，
        # 保证「车道长度」与 RL 状态空间里 seg1/seg2/seg3 的分段口径完全一致。
        parsed = rl_core.core("data_schema").parse_roadnet(schema.roadnet)
        self.inter_map = parsed["inter_map"]
        self.road_map = parsed["road_map"]
        self.lane_lengths = parsed["lane_lengths"]

        if flow_json is None:
            flow_json = schema.flow_raw if hasattr(schema, "flow_raw") else None
        if flow_json is None:
            with open(schema.data_files["flow"], "r", encoding="utf-8") as f:
                flow_json = json.load(f)
        self.arrivals = expand_flows(flow_json)

        self._build_network()
        self.reset()

    # ------------------------------------------------------------------
    # 网络索引
    # ------------------------------------------------------------------
    def _build_network(self):
        self.lanes = {}            # lane_id -> dict
        self.lane_ids_by_road = {}
        for rid, road in self.road_map.items():
            n = len(road["lanes"])
            lengths = self.lane_lengths[rid]
            ids = []
            for k in range(n):
                lane_id = "{}_{}".format(rid, k)
                lane = road["lanes"][k]
                self.lanes[lane_id] = {
                    "lane_id": lane_id,
                    "road_id": rid,
                    "index": k,
                    "length": float(lengths[k]),
                    "max_speed": float(lane.get("maxSpeed", DEFAULT_MAX_SPEED)),
                    "end_inter": road["endIntersection"],
                    "start_inter": road["startIntersection"],
                }
                ids.append(lane_id)
            self.lane_ids_by_road[rid] = ids
        self.all_lane_ids = sorted(self.lanes.keys())

        # 相位 → 允许通行的 roadLink 下标；roadLink 下标 → (startRoad, endRoad, laneLink 映射)
        self.phase_links = {}       # iid -> [set(link_idx), ...]
        self.link_info = {}         # iid -> [ {start,end,lanelink:{si:ei}, type} ]
        self.move_index = {}        # (iid, startRoad, endRoad) -> [link_idx, ...]
        for iid, inter in self.inter_map.items():
            if inter.get("virtual"):
                continue
            rls = inter.get("roadLinks") or []
            infos = []
            for k, rl in enumerate(rls):
                lane_map = {}
                for ll in rl.get("laneLinks") or []:
                    lane_map[int(ll["startLaneIndex"])] = int(ll["endLaneIndex"])
                infos.append({
                    "index": k,
                    "start": rl["startRoad"],
                    "end": rl["endRoad"],
                    "lane_map": lane_map,
                    "type": rl.get("type"),
                })
                self.move_index.setdefault(
                    (iid, rl["startRoad"], rl["endRoad"]), []).append(k)
            self.link_info[iid] = infos
            tl = inter.get("trafficLight") or {}
            self.phase_links[iid] = [set(int(x) for x in (ph.get("availableRoadLinks") or []))
                                     for ph in (tl.get("lightphases") or [])]

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def reset(self):
        self.t = 0.0
        self.phase = dict((iid, 0) for iid in self.phase_links)
        self.vehicles = {}          # vid -> dict
        self.lane_veh = dict((lid, []) for lid in self.all_lane_ids)
        self.finished = []          # 已完成车辆的行程时间
        self._veh_seq = 0
        self._arr_ptr = 0
        self._inject_until(0.0)
        return None

    def set_random_seed(self, seed):
        self.seed = seed

    def set_tl_phase(self, iid, phase):
        if iid not in self.phase_links:
            return
        self.phase[iid] = int(phase)

    def get_current_time(self):
        return self.t

    # ------------------------------------------------------------------
    # 车辆注入
    # ------------------------------------------------------------------
    def _inject_until(self, t_now):
        while self._arr_ptr < len(self.arrivals) and self.arrivals[self._arr_ptr][0] <= t_now + _EPS:
            _, route = self.arrivals[self._arr_ptr]
            self._arr_ptr += 1
            self._spawn(route)

    def _spawn(self, route):
        first = route[0]
        lane_ids = self.lane_ids_by_road.get(first)
        if not lane_ids:
            return                       # 路线首段不在路网中：跳过（与 CityFlow 拒收一致）
        lane_id = self._least_loaded(lane_ids)
        self._veh_seq += 1
        vid = "v{}".format(self._veh_seq)
        self.vehicles[vid] = {
            "id": vid,
            "route": route,
            "index": 0,
            "lane": lane_id,
            "distance": 0.0,
            "speed": 0.0,
            "enter": self.t,
            "length": DEFAULT_VEH_LENGTH,
            "min_gap": DEFAULT_MIN_GAP,
        }
        self.lane_veh[lane_id].append(vid)

    def _least_loaded(self, lane_ids):
        best = lane_ids[0]
        best_n = len(self.lane_veh[best])
        for lid in lane_ids[1:]:
            n = len(self.lane_veh[lid])
            if n < best_n:
                best, best_n = lid, n
        return best

    # ------------------------------------------------------------------
    # 单步推进
    # ------------------------------------------------------------------
    def next_step(self):
        self.t += 1.0
        self._inject_until(self.t)
        for lane_id in self.all_lane_ids:
            vids = self.lane_veh[lane_id]
            if not vids:
                continue
            lane = self.lanes[lane_id]
            # 按 distance 从大到小（车头在前）处理，跟驰约束就是对前车留间距
            vids.sort(key=lambda v: -self.vehicles[v]["distance"])
            front_limit = lane["length"]
            for vid in vids:
                v = self.vehicles[vid]
                if v is None:
                    continue
                at_stop_line = v["distance"] >= lane["length"] - 0.5
                if at_stop_line:
                    if not self._try_cross(v, lane):
                        v["speed"] = 0.0
                        v["distance"] = min(v["distance"], lane["length"] - 0.1)
                    else:
                        continue         # 已驶入下一路段：本车道不再处理
                else:
                    gap = front_limit - v["distance"]
                    move = min(lane["max_speed"], max(0.0, gap))
                    if move > 0.5:
                        v["distance"] = min(lane["length"], v["distance"] + move)
                        v["speed"] = move
                    else:
                        v["speed"] = 0.0
                front_limit = v["distance"] - (v["length"] + v["min_gap"])

    def _try_cross(self, v, lane):
        """在停车线处尝试驶入下一路段（受当前相位 availableRoadLinks 约束）。"""
        route = v["route"]
        nxt_index = v["index"] + 1
        if nxt_index >= len(route):
            self._finish(v)              # 路线走完：视为到达终点
            return True
        iid = lane["end_inter"]
        if self.inter_map.get(iid, {}).get("virtual"):
            self._finish(v)              # 边界虚拟交叉口：驶出路网
            return True
        next_road = route[nxt_index]
        keys = self.move_index.get((iid, lane["road_id"], next_road))
        if not keys:
            self._finish(v)              # 该转向在本路网不存在（数据异常）：直接离场
            return True
        allowed = self.phase_links.get(iid, [])
        cur_phase = self.phase.get(iid, 0)
        open_links = allowed[cur_phase] if 0 <= cur_phase < len(allowed) else set()
        if not open_links.intersection(keys):
            return False                 # 红灯：停车线前排队
        info = self.link_info[iid][keys[0]]
        target_lane_idx = info["lane_map"].get(lane["index"])
        if target_lane_idx is None and info["lane_map"]:
            target_lane_idx = min(info["lane_map"].values())
        target_lane_ids = self.lane_ids_by_road.get(next_road) or []
        if not target_lane_ids:
            self._finish(v)
            return True
        if target_lane_idx is not None and target_lane_idx < len(target_lane_ids):
            candidate = target_lane_ids[target_lane_idx]
        else:
            candidate = self._least_loaded(target_lane_ids)
        target_len = self.lanes[candidate]["length"]
        occ = self.lane_veh[candidate]
        if len(occ) * (DEFAULT_VEH_LENGTH + DEFAULT_MIN_GAP) >= target_len:
            return False                 # 下游路段已满：无法驶入
        for other in occ:                # 入口处要有足够空间
            d = self.vehicles[other]["distance"]
            if d < DEFAULT_VEH_LENGTH + DEFAULT_MIN_GAP:
                return False
        self.lane_veh[v["lane"]].remove(v["id"])
        v["lane"] = candidate
        v["index"] = nxt_index
        v["distance"] = 0.0
        v["speed"] = min(v["speed"], self.lanes[candidate]["max_speed"])
        self.lane_veh[candidate].append(v["id"])
        return True

    def _finish(self, v):
        self.finished.append(max(0.0, self.t - v["enter"]))
        lane_vids = self.lane_veh.get(v["lane"])
        if lane_vids and v["id"] in lane_vids:
            lane_vids.remove(v["id"])
        self.vehicles.pop(v["id"], None)

    # ------------------------------------------------------------------
    # 引擎查询接口（env_cityflow 实际调用的 11 个方法）
    # ------------------------------------------------------------------
    def get_lane_vehicle_count(self):
        return dict((lid, float(len(vids))) for lid, vids in self.lane_veh.items())

    def get_lane_waiting_vehicle_count(self):
        threshold = float(rl_core.core("config").QUEUE_SPEED_THRESHOLD)
        out = {}
        for lid, vids in self.lane_veh.items():
            out[lid] = float(sum(1 for vid in vids
                                 if self.vehicles[vid]["speed"] < threshold))
        return out

    def get_lane_vehicles(self):
        return dict((lid, list(vids)) for lid, vids in self.lane_veh.items())

    def get_vehicles(self, include_waiting=True):
        return list(self.vehicles.keys())

    def get_vehicle_count(self):
        return len(self.vehicles)

    def get_vehicle_info(self, vid):
        """CityFlow 语义：正常行驶返回 running/distance/speed/drivable/... 全部为字符串。"""
        v = self.vehicles.get(vid)
        if v is None:
            return {"running": "0"}
        lane_id = v["lane"]
        lane = self.lanes[lane_id]
        return {
            "running": "1",
            "distance": "{:.3f}".format(v["distance"]),
            "speed": "{:.3f}".format(v["speed"]),
            "drivable": lane_id,
            "road": lane["road_id"],
            "intersection": "",
            "route": v["route"],
        }

    def get_average_travel_time(self):
        if not self.finished:
            return 0.0
        return float(sum(self.finished)) / float(len(self.finished))

    def close(self):
        self.vehicles = {}

    def __repr__(self):
        return "<OfflineEngine t={} veh={} lanes={}>".format(
            self.t, len(self.vehicles), len(self.all_lane_ids))


def make_engine(schema, flow_json=None, prefer_cityflow=True, verbose=False):
    """
    引擎工厂：优先用真实 CityFlow（若本机已装），否则退回离线近似引擎。

    返回 (engine, kind)，kind ∈ {"cityflow", "offline"}，前端会显示当前引擎类型。
    """
    if prefer_cityflow:
        try:
            cf = rl_core.core("cityflow")
            cfg = rl_core.core("env_cityflow").build_cityflow_config(schema)
            engine = cf.Engine(cfg, thread_num=1)
            return engine, "cityflow"
        except Exception:                # noqa: BLE001 - 没装 cityflow 就走离线引擎
            pass
    return OfflineEngine(schema, flow_json=flow_json, verbose=verbose), "offline"
