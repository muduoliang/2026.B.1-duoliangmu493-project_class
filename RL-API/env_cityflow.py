# -*- coding: utf-8 -*-
"""
env_cityflow.py —— CityFlow 仿真环境封装（论文4.2节 RL 辅助模型的仿真侧）

对应论文：王之颢《城市交通信号控制大语言模型方法研究与系统实现》第四章 4.2 节
         「融合邻居信息的动态交通信号控制强化学习算法」

【本文件在论文4.2节中的位置】
论文 DQN 总体框架的五个模块之一是「环境」。本文件实现：
  1) 状态采集  —— 严格按论文4.2节 状态空间定义：
       单车道五元组 (排队车辆数 sq, 行驶车辆 sv, seg1, seg2, seg3)
       交叉口状态 = 各驶入车道五元组拼接 ⊕ 当前相位 one-hot
     其中：
       - sq 直接取 CityFlow 的 get_lane_waiting_vehicle_count()（速度 < 0.1 m/s，
         engine.cpp:641），**不使用数据中不存在的任何观测量**；
       - sv = 该车道车辆总数 − 排队数；
       - seg1/seg2/seg3 = 按车辆距车道终点的剩余距离分段统计（边界见 config.SEG_BAND_EDGES_M），
         剩余距离 = data_schema 复刻出的真实车道长度 − CityFlow get_vehicle_info() 的
         "distance"（当前 lane 上已行驶距离）；
       - 排除 "running" == "0"（已到达终点、等待销毁）的车辆（vehicle.cpp:435-457）。
  2) 动作执行  —— 双输出动作：相位动作（离散，取值来自真实数据推导的动作掩码）+
                 相位持续时间动作（离散，config.DURATION_OPTIONS）；
                 通过 engine.set_tl_phase() 施加（engine.cpp:719，要求 config 中
                 rlTrafficLight=true，否则 CityFlow 会忽略该调用）。
  3) 奖励计算  —— R_target = R_self + η · (Σ R_nbr / k)（论文4.2节奖励函数），
                 R_self 为该交叉口驶入车道排队车辆数之和的负值；k 为真实邻居数。

【异步决策机制（与论文双输出动作一致的关键设计）】
论文中每个交叉口自行输出「相位 + 相位持续时间」。因此本环境采用**异步事件驱动**：
  - 每个交叉口有各自的决策时刻 decision_time；
  - 某交叉口执行动作 (a_p, a_d) 后，其相位保持 a_d 秒，a_d 秒后该交叉口再次决策；
  - 仿真时间推进到「最早的下一个决策时刻」，期间其他交叉口保持各自相位不动。
这样每个交叉口的持续时间动作都被**真实执行**（不做任何多数投票/取均值之类的伪造），
转移元组 (S, a_p, a_d, R, S') 的 S' 恰好是该交叉口下一次决策时刻的状态。

【硬性约束（需求方要求，本文件严格遵守）】
- 状态维度、动作维度、邻居维度**全部在运行时由 ./data 推导**（唯一来源 data_schema），
  本文件不出现任何写死的特征维度常量；
- 运行时校验 CityFlow 必须提供的真实字段（REQUIRED_ENGINE_VEHICLE_FIELDS）：
  一旦引擎返回的字典缺少 distance/speed 等字段 → 直接抛 ValueError 终止，
  **绝不填充虚构数据**；
- ./data 只读；CityFlow 运行配置等生成物一律写到 output/generated/。
"""

# 兼容 Python 3.8
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import config
from data_schema import (
    DataSchema,
    DataValidationError,
    IntersectionSpec,
    LANE_FEATURE_NAMES,
    REQUIRED_ENGINE_VEHICLE_FIELDS,
    require_keys,
)

# --- CityFlow 导入（缺失时给出清晰的安装提示，而不是晦涩的 ImportError） ---
try:
    import cityflow  # type: ignore
    _CITYFLOW_IMPORT_ERROR: Optional[BaseException] = None
except BaseException as _e:  # pragma: no cover - 环境相关
    cityflow = None  # type: ignore
    _CITYFLOW_IMPORT_ERROR = _e


CITYFLOW_INSTALL_HINT = (
    "无法导入 cityflow。请在 Ubuntu（WSL2）conda 环境 cityflow 中安装：\n"
    "  sudo apt update && sudo apt install -y build-essential cmake\n"
    "  git clone https://github.com/cityflow-project/CityFlow.git\n"
    "  cd CityFlow && git submodule update --init --recursive\n"
    "  pip install .\n"
    "或改用 Docker 镜像 cityflowproject/cityflow:latest。\n"
    "原始异常: {!r}"
)

_EPS_TIME = 1e-9


# ===========================================================================
# 运行时维度快照（全部来自 data_schema，不含任何写死维度）
# ===========================================================================

@dataclass
class EnvSpec:
    """环境的维度/动作规格快照：模型、回放池、训练脚本一律从这里取维度。"""
    inter_ids: List[str]
    state_dim_per_inter: List[int]          # 与 inter_ids 同序
    state_dim_max: int
    n_intersections: int
    neighbor_degree_max: int
    n_rel_pos_classes: int
    rel_class_group: List[int]
    n_rel_class_groups: int
    n_selectable_phases_per_inter: List[int]
    n_selectable_phases_max: int
    n_duration_actions: int
    duration_options: Tuple[int, ...]
    lane_feature_dim: int

    def summary(self) -> str:
        return (
            "EnvSpec(交叉口={n} 各交叉口状态维度={sd}(上界{dm}) 最大邻居={k} "
            "相对位置类别={rel}({g}组) 相位动作数={ph}(上界{pm}) 持续时间动作数={nd})"
        ).format(n=self.n_intersections, sd=self.state_dim_per_inter, dm=self.state_dim_max,
                 k=self.neighbor_degree_max, rel=self.n_rel_pos_classes,
                 g=self.n_rel_class_groups, ph=self.n_selectable_phases_per_inter,
                 pm=self.n_selectable_phases_max, nd=self.n_duration_actions)


# ===========================================================================
# CityFlow 运行配置生成（写在 output/generated 下，绝不动 ./data）
# ===========================================================================

def materialize_cityflow_flow_file(schema: DataSchema, generated_dir: str) -> str:
    """
    返回可交给 CityFlow 的 flowFile 路径。

    CityFlow 要求 flow 文件顶层是**数组**（engine.cpp:114 `if (!root.IsArray()) throw`）。
    若 ./data 里的 flow 已是数组（本数据集即如此），直接返回原路径（只读使用）；
    若是历史字典格式（{interval, routes, flows}），则在 output/generated 下派生一个
    CityFlow 兼容数组文件——派生文件只读自 ./data，绝不回写 ./data。
    """
    if schema.flow.is_cityflow_array:
        return schema.data_files["flow"]

    with open(schema.data_files["flow"], "r", encoding="utf-8") as f:
        raw = json.load(f)
    require_keys(raw, ("flows",), "flow(顶层 dict)")

    # 历史格式的 route 可能是「路段 id 数组」或「routes 表的下标」
    routes_table = raw.get("routes")
    items: List[Dict[str, Any]] = []
    for i, item in enumerate(raw["flows"]):
        ctx = "flow.flows[{}]".format(i)
        require_keys(item, ("vehicle", "route", "interval", "startTime", "endTime"), ctx)
        route = item["route"]
        if isinstance(route, int) and isinstance(routes_table, list):
            route = routes_table[route]
        if isinstance(route, list) and route and isinstance(route[0], dict):
            route = [r["id"] for r in route]      # 兼容 [{id:...}] 形式
        items.append({
            "vehicle": item["vehicle"],
            "route": route,
            "interval": item["interval"],
            "startTime": item["startTime"],
            "endTime": item["endTime"],
        })

    os.makedirs(generated_dir, exist_ok=True)
    out_path = os.path.join(generated_dir, "flow_cityflow_array.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    return out_path


def build_cityflow_config(schema: DataSchema, generated_dir: Optional[str] = None,
                          seed: Optional[int] = None,
                          interval: Optional[float] = None,
                          thread_num: Optional[int] = None) -> str:
    """
    生成 CityFlow 运行配置 json，返回其路径。

    必需键依据 engine.cpp:50-76（缺一即抛 JsonTypeError）：
        interval / rlTrafficLight / seed / dir / roadnetFile / flowFile / saveReplay
    说明：
      - dir 与文件名为**字符串拼接**，这里用 dir="" + 绝对路径，避免路径歧义；
      - rlTrafficLight=true 是 RL 控制的必要条件：否则 set_tl_phase 不生效
        （engine.cpp:719-725），且引擎会按配时文件自动推进相位（engine.cpp:583-586）。
    """
    generated_dir = generated_dir or config.GENERATED_DIR
    seed = int(config.SEED if seed is None else seed)
    interval = float(config.SIM_INTERVAL if interval is None else interval)
    if not (0.2 <= interval <= 1.5):
        print("[警告] SIM_INTERVAL={} 不在 CityFlow 建议区间 [0.2, 1.5] 内".format(interval))

    flow_file = materialize_cityflow_flow_file(schema, generated_dir)
    cfg = {
        "interval": interval,
        "rlTrafficLight": True,
        "seed": seed,
        "dir": "",
        "roadnetFile": os.path.abspath(schema.data_files["roadnet"]),
        "flowFile": os.path.abspath(flow_file),
        "saveReplay": False,
        "laneChange": False,
    }
    os.makedirs(generated_dir, exist_ok=True)
    path = os.path.join(generated_dir, "cityflow_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return path


# ===========================================================================
# 环境主体
# ===========================================================================

class CityFlowTrafficEnv:
    """
    CityFlow 多交叉口 RL 环境（每个受控交叉口是一个决策智能体，共享一套网络参数）。

    典型训练循环（异步决策）：
        obs = env.reset()                                  # 初始待决策交叉口的状态
        while True:
            batch = env.build_transition_inputs(obs)       # 定长张量（含 padding/mask）
            a_p, a_d = 网络(batch)                          # 双输出动作
            obs, transitions, rewards, done, info = env.step(a_p, a_d)
            for tr in transitions: buffer.push(tr)         # 元组 (S, a_p, a_d, R, S')
            if done: break
    """

    def __init__(self,
                 schema: DataSchema,
                 episode_seconds: Optional[float] = None,
                 warmup_seconds: Optional[float] = None,
                 seed: Optional[int] = None,
                 thread_num: int = 1,
                 generated_dir: Optional[str] = None,
                 engine: Any = None,
                 verbose: bool = True):
        self.schema = schema
        self.dims = schema.dims
        self.verbose = verbose

        # ---- 维度全部来自运行时推导（data_schema.Dims） ----
        self.spec = EnvSpec(
            inter_ids=schema.controlled_ids(),
            state_dim_per_inter=[sp.state_dim for sp in schema.intersections],
            state_dim_max=self.dims.state_dim_max,
            n_intersections=self.dims.n_intersections,
            neighbor_degree_max=self.dims.neighbor_degree_max,
            n_rel_pos_classes=self.dims.n_rel_pos_classes,
            rel_class_group=list(schema.rel_class_group),
            n_rel_class_groups=self.dims.n_rel_class_groups,
            n_selectable_phases_per_inter=[sp.n_selectable_phases for sp in schema.intersections],
            n_selectable_phases_max=self.dims.n_selectable_phases_max,
            n_duration_actions=self.dims.n_duration_actions,
            duration_options=tuple(config.DURATION_OPTIONS),
            lane_feature_dim=self.dims.lane_feature_dim,
        )
        if self.spec.neighbor_degree_max < 1:
            raise DataValidationError(
                "[数据校验失败] 路网中不存在邻居交叉口关系，无法构造论文4.2节邻居信息融合模块")

        self.inter_ids = list(self.spec.inter_ids)
        self.inter_index = {iid: i for i, iid in enumerate(self.inter_ids)}
        self.n = len(self.inter_ids)

        # ---- 邻居索引表（论文4.2节 NR/NS 模块的输入组织） ----
        self.neighbor_index: List[np.ndarray] = []
        self.neighbor_rel: List[np.ndarray] = []
        self.neighbor_group: List[np.ndarray] = []
        for sp in schema.intersections:
            idx = np.asarray([self.inter_index[x] for x in sp.neighbors], dtype=np.int64)
            rel = np.asarray([sp.rel_class[x] for x in sp.neighbors], dtype=np.int64)
            grp = np.asarray([schema.rel_class_group[r] for r in rel], dtype=np.int64)
            self.neighbor_index.append(idx)
            self.neighbor_rel.append(rel)
            self.neighbor_group.append(grp)

        # ---- 动作空间：相位动作槽 → 真实 lightphases 下标（掩码来源为真实数据） ----
        self.phase_choices: List[List[int]] = [list(sp.selectable_phase_indices)
                                               for sp in schema.intersections]
        # ---- 状态布局索引 ----
        self.incoming_lane_ids: List[List[str]] = [
            [ln.lane_id for ln in sp.incoming_lanes] for sp in schema.intersections]
        self.incoming_lane_lengths: List[np.ndarray] = [
            np.asarray([ln.length for ln in sp.incoming_lanes], dtype=np.float64)
            for sp in schema.intersections]
        self.phase_block_start: List[int] = [sp.phase_block_start() for sp in schema.intersections]
        self.n_phases_per_inter: List[int] = [sp.n_phases for sp in schema.intersections]

        # ---- 仿真时间设置（episode 时长由真实需求数据推导） ----
        self.episode_seconds = float(
            episode_seconds if episode_seconds is not None
            else (config.EPISODE_SECONDS if config.EPISODE_SECONDS is not None
                  else schema.flow.horizon_seconds + config.EPISODE_DRAIN_SECONDS))
        if self.episode_seconds <= 0:
            raise DataValidationError(
                "[数据校验失败] 推导出的 episode 时长非正: {}（请检查 flow 的 startTime/endTime）"
                .format(self.episode_seconds))
        self.warmup_seconds = float(
            config.WARMUP_SECONDS if warmup_seconds is None else warmup_seconds)
        self.seed = int(config.SEED if seed is None else seed)
        self.interval = float(config.SIM_INTERVAL)

        # ---- 引擎 ----
        self._owns_engine = engine is None
        self.generated_dir = generated_dir or config.GENERATED_DIR
        if self._owns_engine:
            if cityflow is None:
                raise DataValidationError(
                    "[环境缺失] " + CITYFLOW_INSTALL_HINT.format(_CITYFLOW_IMPORT_ERROR))
            cfg_path = build_cityflow_config(schema, self.generated_dir, self.seed,
                                             self.interval, thread_num)
            self.config_path = cfg_path
            self.engine = cityflow.Engine(cfg_path, thread_num=thread_num)
        else:
            # 允许外部注入引擎（自检/离线测试用 MockEngine 走这条路径）
            self.engine = engine
            self.config_path = None

        # ---- 运行时状态 ----
        self.sim_time: float = 0.0
        self._cur_phase: List[int] = [self.phase_choices[i][0] for i in range(self.n)]
        self._decision_time: List[float] = [0.0] * self.n
        self._pending: List[Optional[Dict[str, Any]]] = [None] * self.n
        self._seen_vehicle_ids: set = set()
        self._inputs: Optional[Dict[str, np.ndarray]] = None      # 最近一次观测的定长张量
        self._queue_per_inter: Dict[str, float] = {}
        self._self_reward = np.zeros(self.n, dtype=np.float32)
        self._n_active_now: int = 0
        self._n_seen_total: int = 0
        self.metrics: Optional[Any] = None                        # eval_metric.EpisodeMetrics
        self._episode_decisions: int = 0
        self._episode_steps: int = 0
        self._last_band_counts = [0, 0, 0]

        if self.verbose:
            print("[env] {}".format(self.spec.summary()))
            print("[env] episode 时长 = {:.0f}s（由 ./data 需求数据推导），仿真粒度 = {}s，"
                  "随机种子 = {}".format(self.episode_seconds, self.interval, self.seed))
            if self.config_path:
                print("[env] CityFlow 运行配置（只读使用 ./data）: {}".format(self.config_path))

    # ------------------------------------------------------------------
    # 基础封装：引擎调用集中在此，便于统一做字段校验与错误提示
    # ------------------------------------------------------------------

    def _call_engine(self, method: str, *args, **kwargs):
        fn = getattr(self.engine, method, None)
        if fn is None:
            # CityFlow 的步进方法绑定名为 next_step（src/cityflow.cpp:16），
            # 少数旧接口写作 step；两者都不存在即环境不兼容。
            if method == "next_step" and hasattr(self.engine, "step"):
                fn = getattr(self.engine, "step")
            else:
                raise DataValidationError(
                    "[引擎接口缺失] CityFlow Engine 不存在方法 '{}'，请确认安装的是官方 "
                    "cityflow 包（在 CityFlow 源码根目录执行 pip install .）".format(method))
        try:
            return fn(*args, **kwargs)
        except DataValidationError:
            raise
        except BaseException as e:
            raise DataValidationError(
                "[引擎调用失败] engine.{}{} → {!r}".format(method, tuple(args), e))

    def _vehicle_info(self, vehicle_id: str) -> Dict[str, str]:
        """
        获取单车信息并做**运行时字段校验**（需求方硬性约束）。

        CityFlow 语义（vehicle.cpp:435-457）：
          - 车辆已到达终点、等待销毁时，getInfo 只返回 {"running": "0"}；
          - 正常行驶时返回 running/distance/speed/drivable/road/intersection/route。
        因此：running == "0" 时只校验 running 字段存在；running == "1" 时必须存在
        distance / speed 等字段，否则无法计算五元组中的 sv/seg1/seg2/seg3
        → 立即抛 ValueError 并打印缺失字段名（绝不虚构补全）。
        """
        info = self._call_engine("get_vehicle_info", vehicle_id)
        if not isinstance(info, dict):
            raise DataValidationError(
                "[引擎字段校验失败] get_vehicle_info({}) 返回类型 {}，期望 dict（string→string）"
                .format(vehicle_id, type(info).__name__))
        if "running" not in info:
            raise DataValidationError(
                "[引擎字段校验失败] get_vehicle_info({}) 缺少字段 'running'，"
                "现有字段 = {}".format(vehicle_id, sorted(info.keys())))
        if str(info["running"]) == "0":
            return info
        missing = [f for f in REQUIRED_ENGINE_VEHICLE_FIELDS if f not in info]
        if missing:
            raise DataValidationError(
                "[引擎字段校验失败] get_vehicle_info({}) 缺少必要字段 {}，"
                "论文4.2节状态空间（行驶车辆/seg1/seg2/seg3）无法由真实数据计算，程序终止；"
                "现有字段 = {}".format(vehicle_id, missing, sorted(info.keys())))
        return info

    # ------------------------------------------------------------------
    # 状态采集（论文4.2节 状态空间）
    # ------------------------------------------------------------------

    def _observe(self) -> Dict[str, np.ndarray]:
        """
        采集全部受控交叉口的状态向量（维度由 ./data 推导，逐交叉口动态）。

        状态向量排布（论文4.2节）：
            [ lane_0: (sq, sv, seg1, seg2, seg3), lane_1: (...), ... , 当前相位 one-hot ]
        其中 lane 顺序 = data_schema 解析 ./data/roadnet 得到的「驶入路段字典序 + 车道序」，
        相位 one-hot 维度 = 该交叉口 trafficLight.lightphases 的真实长度。
        """
        lane_count = self._call_engine("get_lane_vehicle_count")
        lane_wait = self._call_engine("get_lane_waiting_vehicle_count")
        lane_vehicles = self._call_engine("get_lane_vehicles")
        for name, d in (("get_lane_vehicle_count", lane_count),
                        ("get_lane_waiting_vehicle_count", lane_wait),
                        ("get_lane_vehicles", lane_vehicles)):
            if not isinstance(d, dict) or not d:
                raise DataValidationError(
                    "[引擎字段校验失败] {}(...) 返回空或非 dict（实际 {}），"
                    "无法采集论文4.2节要求的车道级状态".format(name, type(d).__name__))

        edges = tuple(float(x) for x in config.SEG_BAND_EDGES_M)
        obs: Dict[str, np.ndarray] = {}
        self._queue_per_inter = {}
        n_band_total = [0, 0, 0]

        for i, sp in enumerate(self.schema.intersections):
            vec = np.zeros(sp.state_dim, dtype=np.float32)
            q_inter = 0.0
            for k, lane in enumerate(sp.incoming_lanes):
                lid = lane.lane_id
                if lid not in lane_wait or lid not in lane_count:
                    raise DataValidationError(
                        "[引擎字段校验失败] 引擎未返回车道 '{}' 的车辆数/排队数；"
                        "该车道由 ./data/roadnet 推导得到，与仿真引擎不一致，程序终止。"
                        .format(lid))
                sq = float(lane_wait[lid])
                total = float(lane_count[lid])
                sv = max(0.0, total - sq)

                # --- seg1/seg2/seg3：按车辆距车道终点的剩余距离分段（论文4.2节）---
                seg = [0.0, 0.0, 0.0]
                for vid in lane_vehicles.get(lid, ()):   # 仅统计真实存在于该车道的车辆
                    info = self._vehicle_info(vid)
                    if str(info.get("running", "0")) != "1":
                        continue                          # 已到达终点，不计入观测
                    drivable = info.get("drivable")
                    if drivable is not None and str(drivable) != lid:
                        # 车辆已驶入 laneLink / 下一段可行驶单元，其 distance 不再属于本车道
                        continue
                    try:
                        dist = float(info["distance"])
                        _spd = float(info["speed"])       # 存在性已在 _vehicle_info 校验
                    except (TypeError, ValueError):
                        raise DataValidationError(
                            "[引擎字段校验失败] 车辆 {} 的 distance/speed 非数值: {!r}"
                            .format(vid, info))
                    remain = float(lane.length) - dist    # 距车道终点剩余距离
                    if remain <= edges[0]:
                        seg[0] += 1.0
                    elif remain <= edges[1]:
                        seg[1] += 1.0
                    else:
                        seg[2] += 1.0
                n_band_total[0] += seg[0]
                n_band_total[1] += seg[1]
                n_band_total[2] += seg[2]

                base = k * len(LANE_FEATURE_NAMES)
                vec[base + 0] = sq
                vec[base + 1] = sv
                vec[base + 2] = seg[0]
                vec[base + 3] = seg[1]
                vec[base + 4] = seg[2]
                q_inter += sq

            # --- 当前相位 one-hot（维度 = 该交叉口真实 lightphases 数）---
            ph = int(self._cur_phase[i])
            if not (0 <= ph < sp.n_phases):
                raise DataValidationError(
                    "[状态构造失败] 交叉口 {} 当前相位下标 {} 超出真实相位总数 {}"
                    .format(sp.inter_id, ph, sp.n_phases))
            vec[self.phase_block_start[i] + ph] = 1.0
            obs[sp.inter_id] = vec
            self._queue_per_inter[sp.inter_id] = q_inter

        self._last_band_counts = n_band_total
        return obs

    def _refresh_observation(self) -> Dict[str, np.ndarray]:
        """观测 + 定长张量组织（模型输入）一次完成，供状态采集与转移构造共用。"""
        obs = self._observe()
        self._inputs = self.build_transition_inputs(obs)
        self._last_obs = obs
        return obs

    # ------------------------------------------------------------------
    # 奖励计算（论文4.2节奖励函数）
    # ------------------------------------------------------------------

    def _self_rewards_now(self) -> np.ndarray:
        """R_self = −(该交叉口全部驶入车道的排队车辆数之和)（论文4.2节）。"""
        r = np.zeros(self.n, dtype=np.float32)
        for i, sp in enumerate(self.schema.intersections):
            r[i] = -float(self._queue_per_inter.get(sp.inter_id, 0.0))
        return r

    def _target_rewards(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        R_target = R_self + η * (Σ R_nbr / k)      （论文4.2节 奖励函数）

        k = 该交叉口真实邻居数（由 ./data 路网结构数出来，不是常数）。
        数值稳定处理：k = 0 时退化为 R_target = R_self（论文公式中 Σ/k 在 k=0 无定义，
        此处明确按退化处理，不做任何虚构邻居补全）。
        """
        r_self = self._self_rewards_now()
        r_target = r_self.copy()
        for i in range(self.n):
            nb = self.neighbor_index[i]
            if nb.size > 0:
                r_target[i] = r_self[i] + config.ETA * float(np.mean(r_self[nb]))
        self._self_reward = r_self
        return r_self, r_target

    # ------------------------------------------------------------------
    # 指标采样
    # ------------------------------------------------------------------

    def _sample_metrics(self) -> None:
        """每个仿真秒采样一次：排队总数、等待车辆数、在途车辆数、ATT。"""
        lane_wait = self._call_engine("get_lane_waiting_vehicle_count")
        if not isinstance(lane_wait, dict) or not lane_wait:
            raise DataValidationError(
                "[引擎字段校验失败] get_lane_waiting_vehicle_count(...) 返回空或非 dict，"
                "无法统计 AQL/AWT")
        queue_sum = float(sum(lane_wait.values()))
        veh_ids = self._call_engine("get_vehicles", True)
        if isinstance(veh_ids, (list, tuple, set)):
            self._seen_vehicle_ids.update(str(v) for v in veh_ids)
        n_active = int(self._call_engine("get_vehicle_count"))
        att_now = float(self._call_engine("get_average_travel_time"))
        self.metrics.observe(self.sim_time, queue_sum, queue_sum, n_active,
                             len(self._seen_vehicle_ids), att_now)
        self._n_active_now = n_active
        self._n_seen_total = len(self._seen_vehicle_ids)

    # ------------------------------------------------------------------
    # episode 生命周期
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, np.ndarray]:
        """
        重置仿真与统计；返回**初始待决策交叉口**的状态（t=0 时全部交叉口都需要决策）。
        """
        from eval_metric import EpisodeMetrics

        self._call_engine("reset")
        self._call_engine("set_random_seed", self.seed)
        self.sim_time = 0.0
        self._episode_decisions = 0
        self._episode_steps = 0
        self._seen_vehicle_ids = set()
        self.metrics = EpisodeMetrics(interval=self.interval)

        # 明确设定初始相位（取该交叉口第一个真实可选相位），保证状态 one-hot 确定可复现
        for i, iid in enumerate(self.inter_ids):
            ph = self.phase_choices[i][0]
            self._call_engine("set_tl_phase", iid, ph)
            self._cur_phase[i] = ph

        # 预热：不计入指标（config.WARMUP_SECONDS 默认 0，论文未给出该设置）
        if self.warmup_seconds > 0:
            if self.verbose:
                print("[env] 预热 {}s（不计入指标统计）".format(self.warmup_seconds))
            for _ in range(max(1, int(round(self.warmup_seconds / self.interval)))):
                self._call_engine("next_step")
                self.sim_time += self.interval
            self.metrics = EpisodeMetrics(interval=self.interval)   # 清空预热期统计
            self.sim_time = 0.0                                     # 指标时间轴从 0 开始
            self._seen_vehicle_ids = set()

        self._decision_time = [self.sim_time] * self.n
        self._pending = [None] * self.n
        self._refresh_observation()
        return self._pending_obs()

    def _pending_ids(self) -> List[int]:
        """当前待决策的交叉口下标（尚未执行动作、且决策时刻已到）。"""
        return [i for i in range(self.n)
                if self._pending[i] is None and self._decision_time[i] <= self.sim_time + _EPS_TIME]

    def _pending_obs(self) -> Dict[str, np.ndarray]:
        """待决策交叉口的观测子集（形状仍为各自真实的 state_dim）。"""
        obs = getattr(self, "_last_obs", None) or self._refresh_observation()
        return {self.inter_ids[i]: obs[self.inter_ids[i]] for i in self._pending_ids()}

    def pending_ids(self) -> List[str]:
        """供训练脚本查询：下一批需要决策的交叉口 id。"""
        return [self.inter_ids[i] for i in self._pending_ids()]

    def decision_batch(self) -> Dict[str, Any]:
        """
        抽取**当前待决策交叉口**对应的定长输入张量（直接喂给 model_net.DQNNetwork）。

        与 build_transition_inputs() 的区别：后者返回全部交叉口的定长张量（训练/自检用），
        本方法只取待决策交叉口的行（异步决策下每步待决策的交叉口是子集）。
        返回键 = own/own_mask/nbr/nbr_mask/rel/group/phase_avail + inter_ids + row_index。
        这些张量的维度全部来自 ./data 推导，不含任何写死维度。
        """
        if self._inputs is None:
            raise DataValidationError("[状态未初始化] 请先调用 env.reset() 再取决策输入")
        pend = self._pending_ids()
        if not pend:
            raise DataValidationError(
                "[状态校验失败] 当前没有待决策交叉口，无法构造决策输入（仿真时刻 {}）"
                .format(self.sim_time))
        idx = np.asarray(pend, dtype=np.int64)
        out: Dict[str, Any] = {k: v[idx].copy() for k, v in self._inputs.items()}
        out["inter_ids"] = [self.inter_ids[i] for i in pend]
        out["row_index"] = idx
        return out

    def is_done(self) -> bool:
        """公开的 episode 结束判定（供评测脚本/定时配时基线使用）。"""
        return self._is_done()

    def _any_decision_due(self) -> bool:
        for i in range(self.n):
            if self._pending[i] is not None and self._decision_time[i] <= self.sim_time + _EPS_TIME:
                return True
        return False

    def _is_done(self) -> bool:
        """episode 结束条件：达到由真实需求数据推导的时长；或需求结束且路网已排空。"""
        if self.sim_time >= self.episode_seconds - _EPS_TIME:
            return True
        if (self.sim_time >= self.schema.flow.t_max) and self._n_active_now == 0:
            return True
        return False

    def step(self,
             phase_actions: Dict[str, int],
             duration_actions: Dict[str, int]
             ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]],
                        Dict[str, float], bool, Dict[str, Any]]:
        """
        执行一步：对**当前待决策交叉口**施加双输出动作，然后推进仿真到下一个决策时刻。

        phase_actions    : {inter_id: 相位动作槽下标}（槽位由 action_masks() 定义）
        duration_actions : {inter_id: 持续时间动作下标}（对应 config.DURATION_OPTIONS）
        返回 (obs_next, transitions, rewards, done, info)
            obs_next    : 下一批待决策交叉口的观测
            transitions : 本步**已闭合**的经验元组列表（S, a_p, a_d, R, S'），直接入回放池
            rewards     : {inter_id: R_target}（与 transitions 对应）
        """
        pending = self._pending_ids()
        if not pending:
            raise DataValidationError(
                "[动作校验失败] 当前没有待决策交叉口，step() 被重复调用（仿真时刻 {}）"
                .format(self.sim_time))
        want = set(self.inter_ids[i] for i in pending)
        if set(phase_actions.keys()) != want or set(duration_actions.keys()) != want:
            raise DataValidationError(
                "[动作校验失败] 动作必须恰好覆盖当前待决策交叉口 {}；"
                "实际相位动作 {} 个、持续时间动作 {} 个".format(sorted(want),
                                                                 len(phase_actions),
                                                                 len(duration_actions)))

        # --- 1) 施加双输出动作：设置相位，并把该相位保持 duration 秒 ---
        for i in pending:
            iid = self.inter_ids[i]
            slot = int(phase_actions[iid])
            choices = self.phase_choices[i]
            if not (0 <= slot < len(choices)):
                raise DataValidationError(
                    "[动作校验失败] 交叉口 {} 的相位动作槽 {} 越界（该交叉口真实可选相位 {} 个）"
                    .format(iid, slot, len(choices)))
            dur_idx = int(duration_actions[iid])
            if not (0 <= dur_idx < len(config.DURATION_OPTIONS)):
                raise DataValidationError(
                    "[动作校验失败] 交叉口 {} 的持续时间动作下标 {} 越界（真实集合 {}）"
                    .format(iid, dur_idx, config.DURATION_OPTIONS))
            light_phase = choices[slot]
            if light_phase != self._cur_phase[i]:
                # engine.cpp:719：set_tl_phase 要求 CityFlow config 中 rlTrafficLight=true
                self._call_engine("set_tl_phase", iid, light_phase)
                self._cur_phase[i] = light_phase

            duration = float(config.DURATION_OPTIONS[dur_idx])
            # 记录待闭合的转移：S 的快照 + 动作 + 该动作保持的时长
            self._pending[i] = {
                "own": self._inputs["own"][i].copy(),
                "own_mask": self._inputs["own_mask"][i].copy(),
                "nbr": self._inputs["nbr"][i].copy(),
                "nbr_mask": self._inputs["nbr_mask"][i].copy(),
                "rel": self._inputs["rel"][i].copy(),
                "group": self._inputs["group"][i].copy(),
                "phase_avail": int(self._inputs["phase_avail"][i]),
                "phase_action": slot,
                "duration_action": dur_idx,
                "t_start": self.sim_time,
            }
            self._decision_time[i] = self.sim_time + duration
            self._episode_decisions += 1

        # --- 2) 推进仿真直到「最早的下一个决策时刻」或 episode 结束 ---
        while True:
            self._advance_one_second()
            if self._is_done():
                break
            if self._any_decision_due():
                break

        # --- 3) 刷新观测并闭合到期的转移 ---
        self._refresh_observation()
        transitions, rewards = self._close_due_transitions(done_flag=self._is_done())
        self.metrics.observe_decision(self._queue_per_inter)
        self._episode_steps += 1
        done = self._is_done()
        if done:
            # episode 结束时，把仍未到期的转移按「截断」闭合（S' = 终局状态，done=1），
            # 保证经验不丢失；这是标准的 episode 截断处理，不伪造任何状态。
            more, more_r = self._close_due_transitions(done_flag=True, force=True)
            transitions.extend(more)
            rewards.update(more_r)

        info = {
            "sim_time": self.sim_time,
            "n_closed": len(transitions),
            "n_active": self._n_active_now,
            "n_total_seen": self._n_seen_total,
            "queue_per_inter": dict(self._queue_per_inter),
            "phase_light_index": list(self._cur_phase),
            "band_counts": list(self._last_band_counts),
        }
        return self._pending_obs(), transitions, rewards, done, info

    def _close_due_transitions(self, done_flag: bool, force: bool = False
                               ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        闭合到期转移：为每个「动作保持时间已到」的交叉口生成 (S, a_p, a_d, R, S') 元组。

        force=True 时（episode 结束）强制闭合所有未到期转移。
        """
        r_self, r_target = self._target_rewards()
        transitions: List[Dict[str, Any]] = []
        rewards: Dict[str, float] = {}
        for i in range(self.n):
            pend = self._pending[i]
            if pend is None:
                continue
            due = self._decision_time[i] <= self.sim_time + _EPS_TIME
            if not (due or force):
                continue
            iid = self.inter_ids[i]
            tr = {
                "own": pend["own"], "own_mask": pend["own_mask"],
                "nbr": pend["nbr"], "nbr_mask": pend["nbr_mask"],
                "rel": pend["rel"], "group": pend["group"],
                "phase_avail": pend["phase_avail"],
                "phase_action": pend["phase_action"],
                "duration_action": pend["duration_action"],
                "reward": float(r_target[i]),
                "done": 1.0 if done_flag else 0.0,
                "next_own": self._inputs["own"][i].copy(),
                "next_own_mask": self._inputs["own_mask"][i].copy(),
                "next_nbr": self._inputs["nbr"][i].copy(),
                "next_nbr_mask": self._inputs["nbr_mask"][i].copy(),
                "next_rel": self._inputs["rel"][i].copy(),
                "next_group": self._inputs["group"][i].copy(),
                "next_phase_avail": int(self._inputs["phase_avail"][i]),
                "inter_index": i,
                "sim_time": float(pend["t_start"]),
                "self_reward": float(r_self[i]),
            }
            transitions.append(tr)
            rewards[iid] = float(r_target[i])
            self._pending[i] = None
            self._decision_time[i] = self.sim_time
        return transitions, rewards

    def _advance_one_second(self) -> None:
        """推进一个仿真粒度（config.SIM_INTERVAL），并采样一次指标。"""
        # 注意：CityFlow pybind 绑定名为 next_step（src/cityflow.cpp:16）
        self._call_engine("next_step")
        self.sim_time += self.interval
        self._sample_metrics()

    def advance(self, seconds: float) -> None:
        """
        保持当前相位不变，推进 seconds 秒仿真（按 config.SIM_INTERVAL 取整）。

        供「定时配时基线」评测使用（readme 中 FixedTime 对照）：基线直接回放
        ./data/signal_inter_*.txt 中真实存在的相位序列，不引入任何虚构动作。
        """
        for _ in range(max(1, int(round(float(seconds) / self.interval)))):
            self._advance_one_second()

    def apply_light_phases(self, light_phase_by_inter: Dict[str, int]) -> None:
        """
        直接按「trafficLight.lightphases 下标」设置相位（供定时配时基线使用）。

        与动作槽的区别：step() 里的相位动作是「可选相位动作槽」（由动作掩码定义），
        本方法用于直接回放 ./data 中 signal_inter_*.txt 的真实相位下标。
        """
        for iid, ph in light_phase_by_inter.items():
            if iid not in self.inter_index:
                raise DataValidationError(
                    "[动作校验失败] apply_light_phases 收到未知交叉口 id: {}".format(iid))
            i = self.inter_index[iid]
            ph = int(ph)
            if not (0 <= ph < self.n_phases_per_inter[i]):
                raise DataValidationError(
                    "[动作校验失败] 交叉口 {} 的 lightphases 下标 {} 越界（真实相位总数 {}）"
                    .format(iid, ph, self.n_phases_per_inter[i]))
            if ph != self._cur_phase[i]:
                self._call_engine("set_tl_phase", iid, ph)
                self._cur_phase[i] = ph

    # ------------------------------------------------------------------
    # 动作掩码 / 模型输入组织（维度全部运行时推导）
    # ------------------------------------------------------------------

    def action_masks(self) -> Dict[str, np.ndarray]:
        """
        异构交叉口相位动作掩码（论文4.2节：动作掩码适配 T 型/网格交叉口）。

        本实现中每个交叉口的合法相位动作数 = 该交叉口真实可选相位个数（来自 ./data），
        掩码恒为 True；异构性通过「定长张量 + 有效位掩码」表达：
            n_selectable_phases_max = max(真实可选相位个数)
        模型相位头输出 n_selectable_phases_max 维打分，掩码外的槽位在 softmax/argmax 前
        填 −inf，从而保证不会选出该交叉口不存在的相位。
        """
        masks: Dict[str, np.ndarray] = {}
        for i, iid in enumerate(self.inter_ids):
            n = len(self.phase_choices[i])
            m = np.zeros(self.spec.n_selectable_phases_max, dtype=bool)
            m[:n] = True
            masks[iid] = m
        return masks

    def build_transition_inputs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        把 dict 形式的观测整理成模型所需定长张量（padding + 有效位掩码）。

        返回（维度全部来自运行时推导，不含写死常量）：
            own         [n_inter, D_max]        本交叉口状态（不足 D_max 的部分补 0）
            own_mask    [n_inter, D_max]        1=真实维度，0=padding
            nbr         [n_inter, K_max, D_max] 邻居状态
            nbr_mask    [n_inter, K_max]        1=真实邻居，0=不存在邻居（论文4.2节邻居掩码）
            rel         [n_inter, K_max]        邻居相对位置 one-hot 类别下标 r(x,y)
            group       [n_inter, K_max]        邻居分组（0=东西向, 1=南北向），供分组 softmax
            phase_avail [n_inter]               该交叉口真实可选相位动作数
        """
        dmax = self.spec.state_dim_max
        kmax = self.spec.neighbor_degree_max
        own = np.zeros((self.n, dmax), dtype=np.float32)
        own_mask = np.zeros((self.n, dmax), dtype=np.float32)
        nbr = np.zeros((self.n, kmax, dmax), dtype=np.float32)
        nbr_mask = np.zeros((self.n, kmax), dtype=np.float32)
        rel = np.zeros((self.n, kmax), dtype=np.int64)
        grp = np.zeros((self.n, kmax), dtype=np.int64)
        phase_avail = np.zeros((self.n,), dtype=np.int64)

        for i, sp in enumerate(self.schema.intersections):
            v = np.asarray(obs[sp.inter_id], dtype=np.float32)
            if v.shape[0] != sp.state_dim:
                raise DataValidationError(
                    "[状态维度校验失败] 交叉口 {} 观测维度 {} 与 ./data 推导维度 {} 不一致"
                    .format(sp.inter_id, v.shape[0], sp.state_dim))
            own[i, :v.shape[0]] = v
            own_mask[i, :v.shape[0]] = 1.0
            nb = self.neighbor_index[i]
            for j in range(int(nb.size)):
                jj = int(nb[j])
                nv = np.asarray(obs[self.inter_ids[jj]], dtype=np.float32)
                nbr[i, j, :nv.shape[0]] = nv
                nbr_mask[i, j] = 1.0
                rel[i, j] = int(self.neighbor_rel[i][j])
                grp[i, j] = int(self.neighbor_group[i][j])
            rel[i, int(nb.size):] = 0       # padding 位置用 0 号类别占位（掩码已置 0）
            grp[i, int(nb.size):] = 0
            phase_avail[i] = len(self.phase_choices[i])
        return {
            "own": own, "own_mask": own_mask, "nbr": nbr, "nbr_mask": nbr_mask,
            "rel": rel, "group": grp, "phase_avail": phase_avail,
        }

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------

    def episode_metrics(self) -> Dict[str, float]:
        """episode 结束时汇总 ATT/AWT/AQL（含 per-intersection AQL）。"""
        att_final = float(self._call_engine("get_average_travel_time"))
        m = self.metrics.finalize(att_final=att_final,
                                  n_total_override=len(self._seen_vehicle_ids))
        m["n_decisions"] = float(self._episode_decisions)
        return m

    def close(self) -> None:
        """释放资源（CityFlow Engine 由 Python GC 管理，这里只做显式标记）。"""
        self.engine = None if self._owns_engine else self.engine

    def __repr__(self) -> str:
        return "<CityFlowTrafficEnv {}>".format(self.spec.summary())
