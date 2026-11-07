# -*- coding: utf-8 -*-
"""
tools/selfcheck.py —— 离线自检（不依赖真实 CityFlow 引擎）

在没有安装 cityflow / 没有 WSL 的机器上也能验证：
  1) ./data 能否被正确解析；论文4.2节必要字段是否存在（缺失即失败，不虚构补全）；
  2) 运行时推导出的维度是否自洽（state_dim = 车道数×5 + 相位 one-hot 维度 …）；
  3) 动作掩码是否只包含真实存在的相位；邻居关系是否对称；相对位置类别是否成对出现；
  4) 模型能否用「真实数据推导出的维度」前向（形状、掩码、分组 softmax）；
  5) 回放池 push/sample 往返是否正确；
  6) 评价指标 ATT/AWT/AQL 口径是否正确；
  7) 用 MockEngine（内置的极简假引擎，仅用于链路自检）跑通
     env.reset() → decision_batch() → env.step() → 经验元组 (S,a,R,S') 全链路；
  8) 绘图链路（plot_metrics）：逐秒序列导出与 CSV/npz 往返、训练 4 曲线出图、
     测试评估 4 类图（时序/箱线/柱状/CDF）出图，以及「产物缺失必须报错、不许画占位图」的反证。
     matplotlib 不可用时这两项渲染检查记 SKIP（环境能力问题，不算失败；train_rl_aux.py
     里的绘图本身就是子进程调用，绘图环境崩溃不影响训练与权重落盘）。

注意：MockEngine 只用于「代码链路自检」，其数据是本地生成的占位车辆，
      **不能作为论文复现的实验结果**；论文级结果必须在装有 cityflow 的
      Ubuntu/WSL conda 环境里用 env_cityflow.CityFlowTrafficEnv 跑真实仿真。

运行：
    python tools/selfcheck.py
    python tools/selfcheck.py --data-dir ./data
    python tools/selfcheck.py --only plot      # 只跑绘图组（可在没有 torch 的解释器里跑）
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config                                          # noqa: E402
from data_schema import (DataSchema, DataValidationError, LANE_FEATURE_NAMES,  # noqa: E402
                         REQUIRED_ENGINE_VEHICLE_FIELDS, build_schema)

_RESULTS: List[tuple] = []


class CheckSkipped(Exception):
    """环境能力不足（例如 matplotlib 原生扩展损坏）导致该项无法验证 —— 记为 SKIP，不算失败。"""


def check(name: str, fn) -> bool:
    """执行单项检查，异常记为 FAIL 并打印原因（不中断后续检查）。"""
    try:
        detail = fn()
        _RESULTS.append((name, True, detail or ""))
        print("  [PASS] {}{}".format(name, (" → " + str(detail)) if detail else ""))
        return True
    except CheckSkipped as e:
        _RESULTS.append((name, None, str(e)))
        print("  [SKIP] {} → {}".format(name, e))
        return False
    except BaseException as e:      # noqa: BLE001 - 自检工具需要兜底
        _RESULTS.append((name, False, repr(e)))
        print("  [FAIL] {} → {!r}".format(name, e))
        return False


# ===========================================================================
# MockEngine：仅用于链路自检的极简假引擎（实现 CityFlow 官方 pybind 同名接口）
# ===========================================================================

class MockEngine:
    """
    极简假引擎：车辆按固定车头时距进入驶入车道；所在 road 在**当前真实相位**的
    availableRoadLinks 中时，队首车辆以 maxSpeed 前进并在到达车道终点后消失。

    目的仅是让 env_cityflow 的全部代码路径（状态采集/动作施加/奖励/转移）跑起来，
    产生的数值没有任何交通工程意义。
    """

    def __init__(self, schema: DataSchema, headway: float = 6.0):
        self.schema = schema
        self.headway = float(headway)
        self.t = 0.0
        self._link_start_road: Dict[str, List[str]] = {}
        for inter in schema.roadnet["intersections"]:
            if inter.get("virtual", False):
                continue
            self._link_start_road[str(inter["id"])] = [
                str(rl.get("startRoad")) for rl in (inter.get("roadLinks") or [])]
        self.reset()

    # --- 生命周期 ---
    def reset(self) -> None:
        self.t = 0.0
        self._veh_id = 0
        self._lanes: Dict[str, List[Dict[str, Any]]] = {}
        self._finished: List[float] = []
        self._enter_time: Dict[str, float] = {}
        self._cur_phase: Dict[str, int] = {}
        self._lane_road: Dict[str, str] = {}
        self._lane_len: Dict[str, float] = {}
        for sp in self.schema.intersections:
            for ln in sp.incoming_lanes:
                self._lanes[ln.lane_id] = []
                self._lane_road[ln.lane_id] = ln.road_id
                self._lane_len[ln.lane_id] = float(ln.length)
                self._cur_phase[sp.inter_id] = 0

    def set_random_seed(self, seed: int) -> None:
        self._seed = int(seed)

    def next_step(self) -> None:
        self.t += float(config.SIM_INTERVAL)
        # 1) 需求：每条驶入车道按车头时距生成车辆
        for sp in self.schema.intersections:
            for ln in sp.incoming_lanes:
                if (self.t % self.headway) < float(config.SIM_INTERVAL) - 1e-9:
                    self._veh_id += 1
                    vid = "mock_{}".format(self._veh_id)
                    self._lanes[ln.lane_id].append(
                        {"id": vid, "distance": 0.0, "speed": 0.0})
                    self._enter_time[vid] = self.t
        # 2) 放行：当前相位放行的 road → 该 road 上的驶入车道队首车辆前进
        for sp in self.schema.intersections:
            phase = self._cur_phase[sp.inter_id]
            avail = set()
            try:
                links = sp.phases[phase].available_road_links
                starts = self._link_start_road[sp.inter_id]
                avail = {starts[i] for i in links if 0 <= i < len(starts)}
            except Exception:
                avail = set()
            for ln in sp.incoming_lanes:
                queue = self._lanes[ln.lane_id]
                green = ln.road_id in avail
                for veh in queue:
                    if green:
                        veh["speed"] = 11.111
                        veh["distance"] += 11.111 * float(config.SIM_INTERVAL)
                    else:
                        veh["speed"] = 0.0
                still = []
                for veh in queue:
                    if veh["distance"] >= self._lane_len[ln.lane_id]:
                        self._finished.append(self.t - self._enter_time[veh["id"]])
                    else:
                        still.append(veh)
                self._lanes[ln.lane_id] = still

    # --- CityFlow 同名查询接口 ---
    def set_tl_phase(self, intersection_id: str, phase_id: int) -> None:
        self._cur_phase[str(intersection_id)] = int(phase_id)

    def get_current_time(self) -> float:
        return float(self.t)

    def get_lane_vehicle_count(self) -> Dict[str, int]:
        return {lid: len(v) for lid, v in self._lanes.items()}

    def get_lane_waiting_vehicle_count(self) -> Dict[str, int]:
        return {lid: sum(1 for v in vs if v["speed"] < 0.1) for lid, vs in self._lanes.items()}

    def get_lane_vehicles(self) -> Dict[str, List[str]]:
        return {lid: [v["id"] for v in vs] for lid, vs in self._lanes.items()}

    def get_vehicles(self, include_waiting: bool = False) -> List[str]:
        out = []
        for vs in self._lanes.values():
            out.extend(v["id"] for v in vs)
        return out

    def get_vehicle_count(self) -> int:
        return int(sum(len(v) for v in self._lanes.values()))

    def get_vehicle_info(self, vehicle_id: str) -> Dict[str, str]:
        for lid, vs in self._lanes.items():
            for v in vs:
                if v["id"] == vehicle_id:
                    return {
                        "running": "1",
                        "distance": "{:.6f}".format(v["distance"]),
                        "speed": "{:.6f}".format(v["speed"]),
                        "drivable": lid,
                        "road": self._lane_road[lid],
                        "intersection": "",
                        "route": "",
                    }
        return {"running": "0"}

    def get_average_travel_time(self) -> float:
        running = [self.t - self._enter_time[v["id"]]
                   for vs in self._lanes.values() for v in vs]
        allt = list(self._finished) + running
        return float(np.mean(allt)) if allt else 0.0


# ===========================================================================
# 各分组检查
# ===========================================================================

def check_schema(schema: DataSchema) -> None:
    print("\n[1] 数据解析与论文4.2节必要字段校验")

    def dims_consistent():
        for sp in schema.intersections:
            expect = sp.n_incoming_lanes * len(LANE_FEATURE_NAMES) + sp.n_phases
            if sp.state_dim != expect:
                raise AssertionError(
                    "交叉口 {} 状态维度 {} ≠ 车道数 {}×{} + 相位 one-hot {}".format(
                        sp.inter_id, sp.state_dim, sp.n_incoming_lanes,
                        len(LANE_FEATURE_NAMES), sp.n_phases))
        return "state_dim = 车道数×5 + 相位 one-hot，逐交叉口自洽"

    def masks_valid():
        for sp in schema.intersections:
            if not sp.selectable_phase_indices:
                raise AssertionError("交叉口 {} 可选相位集合为空".format(sp.inter_id))
            for p in sp.selectable_phase_indices:
                if not (0 <= p < sp.n_phases):
                    raise AssertionError(
                        "交叉口 {} 动作掩码含越界相位 {}".format(sp.inter_id, p))
        return "所有交叉口的动作掩码相位均存在于真实 lightphases"

    def neighbor_symmetry():
        bad = []
        for sp in schema.intersections:
            for nb in sp.neighbors:
                if sp.inter_id not in schema.inter_by_id[nb].neighbors:
                    bad.append((sp.inter_id, nb))
        if bad:
            raise AssertionError("邻居关系不对称: {}".format(bad))
        degs = sorted({len(sp.neighbors) for sp in schema.intersections})
        return "邻居关系对称，邻居度集合 = {}".format(degs)

    def rel_table():
        classes = schema.rel_classes
        if len(classes) != len(set(classes)):
            raise AssertionError("相对位置类别存在重复: {}".format(classes))
        return "相对位置类别 {} ，分组 {}".format(classes, schema.rel_class_group)

    def lanes_have_geometry():
        for sp in schema.intersections:
            for ln in sp.incoming_lanes:
                if not (ln.length > 0):
                    raise AssertionError("车道 {} 长度非正: {}".format(ln.lane_id, ln.length))
        return "所有驶入车道长度 > 0（由 roadnet 几何推算，与 CityFlow 口径一致）"

    check("cross-check 状态维度自洽", dims_consistent)
    check("cross-check 动作掩码 ⊆ 真实相位", masks_valid)
    check("cross-check 邻居关系对称性", neighbor_symmetry)
    check("cross-check 相对位置 one-hot 表", rel_table)
    check("cross-check 车道几何长度", lanes_have_geometry)

    def missing_field_raises():
        """反证：必要字段被人为删除时必须抛 ValueError（证明校验真的生效）。"""
        from data_schema import require_keys
        try:
            require_keys({"a": 1}, ("a", "banana_field"), "自检用例")
        except ValueError as e:
            if "banana_field" not in str(e):
                raise AssertionError("异常信息未包含缺失字段名: {}".format(e))
            return "缺失字段时正确抛出 ValueError 并打印字段名"
        raise AssertionError("必要字段缺失却没有抛异常")

    check("数据校验：缺失必要字段必须终止", missing_field_raises)
    _ = REQUIRED_ENGINE_VEHICLE_FIELDS     # 引擎侧字段要求（env_cityflow._vehicle_info 强制校验）


def check_env_mock(schema: DataSchema) -> None:
    print("\n[2] 环境链路自检（MockEngine，仅验证代码路径，不作为实验结果）")
    from env_cityflow import CityFlowTrafficEnv

    engine = MockEngine(schema)
    env = CityFlowTrafficEnv(schema, episode_seconds=200.0, engine=engine, verbose=False)
    state = {"transitions": 0, "steps": 0, "reward_min": 1e9, "reward_max": -1e9}

    def reset_and_observe():
        env.reset()
        nb = env.decision_batch()
        if nb["own"].shape[0] != len(nb["inter_ids"]):
            raise AssertionError("决策输入行数与待决策交叉口数不一致")
        return "reset 后待决策交叉口 {} 个，own 形状 {}".format(
            len(nb["inter_ids"]), tuple(nb["own"].shape))

    def run_episode():
        env.reset()
        while True:
            batch = env.decision_batch()
            n = len(batch["inter_ids"])
            assert batch["own"].shape == (n, env.spec.state_dim_max), batch["own"].shape
            assert batch["nbr"].shape == (n, env.spec.neighbor_degree_max,
                                          env.spec.state_dim_max), batch["nbr"].shape
            assert batch["own_mask"].sum() > 0
            for i in range(n):
                assert batch["phase_avail"][i] <= env.spec.n_selectable_phases_max
            # 随机（合法）动作：相位槽 + 持续时间档
            a_p, a_d = {}, {}
            for i, iid in enumerate(batch["inter_ids"]):
                a_p[iid] = int(np.random.randint(0, int(batch["phase_avail"][i])))
                a_d[iid] = int(np.random.randint(0, env.spec.n_duration_actions))
            _, transitions, rewards, done, info = env.step(a_p, a_d)
            for tr in transitions:
                need = ("own", "own_mask", "nbr", "nbr_mask", "rel", "group",
                        "phase_avail", "phase_action", "duration_action", "reward", "done",
                        "next_own", "next_own_mask", "next_nbr", "next_nbr_mask",
                        "next_rel", "next_group", "next_phase_avail")
                miss = [k for k in need if k not in tr]
                if miss:
                    raise AssertionError("经验元组缺少键: {}".format(miss))
                state["reward_min"] = min(state["reward_min"], tr["reward"])
                state["reward_max"] = max(state["reward_max"], tr["reward"])
            state["transitions"] += len(transitions)
            state["steps"] += 1
            if done:
                break
        m = env.episode_metrics()
        for k in ("ATT", "AWT", "AQL", "throughput"):
            if k not in m:
                raise AssertionError("episode_metrics 缺少指标 {}".format(k))
        return ("episode 结束：{} 步决策批次，{} 条经验，指标 ATT={:.1f}s AWT={:.1f}s "
                "AQL={:.1f}".format(state["steps"], state["transitions"], m["ATT"],
                                    m["AWT"], m["AQL"]))

    def reward_formula():
        """校验 R_target = R_self + η·mean(R_nbr) 的数值关系（论文4.2节奖励函数）。"""
        env.reset()
        batch = env.decision_batch()
        a_p = {iid: 0 for iid in batch["inter_ids"]}
        a_d = {iid: 0 for iid in batch["inter_ids"]}
        _, transitions, rewards, done, _ = env.step(a_p, a_d)
        if not transitions:
            raise AssertionError("首个决策批次没有产生经验元组")
        for tr in transitions:
            i = tr["inter_index"]
            nb = env.neighbor_index[i]
            r_self = tr["self_reward"]
            expect = r_self + (config.ETA * float(np.mean(env._self_reward[nb]))
                              if nb.size else 0.0)
            if abs(expect - tr["reward"]) > 1e-4:
                raise AssertionError(
                    "奖励公式不符: 期望 {:.4f} 实际 {:.4f}".format(expect, tr["reward"]))
        return "R_target = R_self + η·mean(R_nbr)，η={}".format(config.ETA)

    check("env：reset + 决策输入形状", reset_and_observe)
    check("env：完整 episode + 经验元组完整性", run_episode)
    check("env：奖励公式一致性", reward_formula)

    def engine_field_guard():
        """反证：引擎返回缺少必要的单车字段时，状态采集必须抛 ValueError。"""
        class BadEngine(MockEngine):
            def get_vehicle_info(self, vehicle_id):
                info = MockEngine.get_vehicle_info(self, vehicle_id)
                info.pop("distance", None)
                return info
        e2 = CityFlowTrafficEnv(schema, episode_seconds=60.0, engine=BadEngine(schema),
                                verbose=False)
        e2.reset()
        try:
            e2.decision_batch()
            for _ in range(10):
                e2.step({i: 0 for i in e2.pending_ids()},
                        {i: 0 for i in e2.pending_ids()})
        except ValueError as ex:
            if "distance" not in str(ex):
                raise AssertionError("异常信息未指出缺失字段: {}".format(ex))
            return "单车字段缺失时正确抛出 ValueError（含字段名）"
        raise AssertionError("引擎字段缺失却没有抛异常")

    check("env：引擎字段缺失必须终止", engine_field_guard)
    env.close()


def check_torch(schema: DataSchema) -> None:
    print("\n[3] 网络自检（维度全部取自运行时推导结果）")
    try:
        import torch
    except BaseException as e:      # noqa: BLE001
        print("  [SKIP] 未安装 torch（{}），跳过网络检查".format(e))
        return

    from model_net import (DQNNetwork, count_parameters, epsilon_greedy_select,
                           masked_softmax, ObservationEncoder, NeighborRelationFusion,
                           NeighborStrengthEvaluation)
    from env_cityflow import CityFlowTrafficEnv

    d = schema.dims
    model = DQNNetwork(state_dim=d.state_dim_max, n_rel_pos_classes=d.n_rel_pos_classes,
                       n_selectable_phases_max=d.n_selectable_phases_max,
                       n_duration_actions=d.n_duration_actions,
                       n_neighbor_groups=d.n_rel_class_groups,
                       hidden_dim=config.HIDDEN_DIM, n_heads=config.N_HEADS,
                       n_self_attn_layers=config.N_SELF_ATTN_LAYERS)

    def forward_shapes():
        b = 5
        own = torch.randn(b, d.state_dim_max)
        own_mask = torch.ones_like(own)
        nbr = torch.randn(b, d.neighbor_degree_max, d.state_dim_max)
        nbr_mask = torch.ones(b, d.neighbor_degree_max)
        nbr_mask[0, :] = 0.0                     # 第 0 个样本：完全没有邻居（边界情况）
        rel = torch.zeros(b, d.neighbor_degree_max, dtype=torch.long)
        group = torch.zeros(b, d.neighbor_degree_max, dtype=torch.long)
        out = model(own, own_mask, nbr, nbr_mask, rel, group)
        assert out["phase_q"].shape == (b, d.n_selectable_phases_max), out["phase_q"].shape
        assert out["duration_q"].shape == (b, d.n_duration_actions), out["duration_q"].shape
        assert torch.isfinite(out["phase_q"]).all() and torch.isfinite(out["duration_q"]).all(), \
            "存在 NaN/Inf（邻居全 padding 时应被掩码保护）"
        assert torch.isfinite(out["os_y"]).all(), "os_y 存在 NaN/Inf"
        return "phase_q {} duration_q {}，无邻居样本亦无 NaN，参数量 {}".format(
            tuple(out["phase_q"].shape), tuple(out["duration_q"].shape),
            count_parameters(model))

    def mask_groups():
        valid = torch.tensor([[1., 1., 0., 0.], [1., 0., 0., 0.]])
        scores = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
        p = masked_softmax(scores, valid, dim=-1)
        assert abs(float(p[0].sum()) - 1.0) < 1e-5 and float(p[0, 2]) == 0.0
        assert abs(float(p[1, 0]) - 1.0) < 1e-6
        return "掩码外概率恒为 0，掩码内归一化正确"

    def masked_argmax():
        n_dur = d.n_duration_actions
        q = torch.tensor([[1.0, 9.0, 9.0]])
        dur = torch.tensor([[0.0] + [5.0] * (n_dur - 1)])
        valid = torch.tensor([[True, False, False]])
        g = torch.Generator().manual_seed(0)
        for _ in range(50):
            # ε=1 纯随机探索时**也必须**只在掩码内取样（该交叉口只真实存在 1 个可选相位）
            a_p, a_d = epsilon_greedy_select(q, dur, valid, 1.0, g)
            assert int(a_p[0]) == 0, "ε=1 时选到了被掩码的相位: {}".format(int(a_p[0]))
            assert 0 <= int(a_d[0]) < n_dur, "持续时间动作越界"
        a_p, a_d = epsilon_greedy_select(q, dur, valid, 0.0, g)
        assert int(a_p[0]) == 0, "ε=0 时未选中唯一合法相位"
        assert int(a_d[0]) == 1, "ε=0 时未取持续时间最大值对应的动作"
        return "ε=1（随机）与 ε=0（贪心）均不会选出掩码外相位；持续时间动作在 {} 档内".format(n_dur)

    def sub_modules_isolated():
        enc = ObservationEncoder(d.state_dim_max, config.HIDDEN_DIM)
        y = enc(torch.randn(3, d.state_dim_max))
        assert y.shape == (3, config.HIDDEN_DIM)
        K = d.neighbor_degree_max
        nr = NeighborRelationFusion(d.n_rel_pos_classes, config.HIDDEN_DIM, config.N_HEADS,
                                    config.N_SELF_ATTN_LAYERS)
        oy = torch.randn(3, config.HIDDEN_DIM)                    # 本交叉口编码 Ox
        nx = torch.randn(3, K, config.HIDDEN_DIM)                 # 邻居编码 Oy（[B,K,h]）
        nm = torch.tensor([[1., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.]])
        rel = torch.zeros(3, K, dtype=torch.long)
        # 论文4.2节 邻居关系特征融合模块：CrossAtt(Linear(r(x,y)), Oy) + N 层自注意力
        or_y = nr(oy, nx, rel, nm)
        assert or_y.shape == (3, K, config.HIDDEN_DIM), or_y.shape
        assert torch.isfinite(or_y).all(), "NR 模块出现 NaN/Inf"
        ns = NeighborStrengthEvaluation(config.HIDDEN_DIM, config.N_HEADS,
                                        d.n_rel_class_groups)
        grp = torch.zeros(3, K, dtype=torch.long)
        # 论文4.2节 邻居作用强度评估模块：CrossAtt(Or_y, Ox) + MLP 评分 + 分组 softmax
        os_y, w = ns(oy, or_y, grp, nm)
        assert os_y.shape == (3, config.HIDDEN_DIM), os_y.shape
        assert w.shape == (3, K), w.shape
        assert torch.isfinite(w).all() and float(w.min()) >= -1e-6, "邻居权重异常"
        assert abs(float(w.sum(dim=1)[1]) - 1.0) < 1e-4, "存在有效邻居时权重和应为 1"
        assert abs(float(w.sum(dim=1)[2])) < 1e-6, "无有效邻居时权重应全为 0"
        return "NR 模块输出 {}；NS 模块输出 {}、邻居权重 {}（分组 softmax 归一化正确）".format(
            tuple(or_y.shape), tuple(os_y.shape), tuple(w.shape))

    check("net：前向形状与数值稳定", forward_shapes)
    check("net：掩码 softmax", mask_groups)
    check("net：ε-greedy 尊重动作掩码", masked_argmax)
    check("net：NR/NS 子模块独立前向", sub_modules_isolated)

    # --- 用真实维度的 env 决策输出直接喂网络（端到端形状对齐） ---
    def env_to_net_end2end():
        env = CityFlowTrafficEnv(schema, episode_seconds=120.0, engine=MockEngine(schema),
                                 verbose=False)
        env.reset()
        batch = env.decision_batch()
        out = model(torch.as_tensor(batch["own"]), torch.as_tensor(batch["own_mask"]),
                    torch.as_tensor(batch["nbr"]), torch.as_tensor(batch["nbr_mask"]),
                    torch.as_tensor(batch["rel"], dtype=torch.long),
                    torch.as_tensor(batch["group"], dtype=torch.long))
        n = len(batch["inter_ids"])
        assert out["phase_q"].shape[0] == n
        env.close()
        return "env 决策输入 → 网络输出对齐（{} 个交叉口）".format(n)

    check("net：env 决策输入端到端对齐", env_to_net_end2end)


def check_buffer_and_metrics(schema: DataSchema) -> None:
    print("\n[4] 回放池与指标自检")
    from replay_buffer import ReplayBuffer

    d = schema.dims
    buf = ReplayBuffer(capacity=64, state_dim_max=d.state_dim_max,
                       neighbor_degree_max=d.neighbor_degree_max,
                       n_rel_pos_classes=d.n_rel_pos_classes,
                       n_duration_actions=d.n_duration_actions, seed=0)

    def roundtrip():
        rng = np.random.default_rng(0)
        for k in range(40):
            buf.push({
                "own": rng.normal(size=d.state_dim_max).astype(np.float32),
                "own_mask": np.ones(d.state_dim_max, dtype=np.float32),
                "nbr": rng.normal(size=(d.neighbor_degree_max, d.state_dim_max)).astype(np.float32),
                "nbr_mask": np.ones(d.neighbor_degree_max, dtype=np.float32),
                "rel": np.zeros(d.neighbor_degree_max, dtype=np.int64),
                "group": np.zeros(d.neighbor_degree_max, dtype=np.int64),
                "phase_avail": np.int64(d.n_selectable_phases_max),
                "phase_action": np.int64(0),
                "duration_action": np.int64(1),
                "reward": np.float32(-1.5),
                "done": np.float32(0.0),
                # 元信息：交叉口下标与仿真时刻（ReplayBuffer.push 要求，用于诊断与按交叉口统计）
                "inter_index": np.int64(k % d.n_intersections),
                "sim_time": np.float32(k),
                "next_own": rng.normal(size=d.state_dim_max).astype(np.float32),
                "next_own_mask": np.ones(d.state_dim_max, dtype=np.float32),
                "next_nbr": rng.normal(size=(d.neighbor_degree_max, d.state_dim_max)).astype(np.float32),
                "next_nbr_mask": np.ones(d.neighbor_degree_max, dtype=np.float32),
                "next_rel": np.zeros(d.neighbor_degree_max, dtype=np.int64),
                "next_group": np.zeros(d.neighbor_degree_max, dtype=np.int64),
                "next_phase_avail": np.int64(d.n_selectable_phases_max),
            })
        if len(buf) != 40:
            raise AssertionError("回放池长度应为 40，实际 {}".format(len(buf)))
        b = buf.sample(8, rng=np.random.default_rng(1))
        for k in ("own", "next_own", "reward", "phase_action", "duration_action"):
            if k not in b:
                raise AssertionError("采样结果缺少键 {}".format(k))
        assert b["own"].shape == (8, d.state_dim_max), b["own"].shape
        assert b["nbr"].shape == (8, d.neighbor_degree_max, d.state_dim_max)
        if not np.allclose(b["reward"], -1.5):
            raise AssertionError("采样 reward 不正确")
        return "push 40 条 / sample 8 条 形状与取值正确"

    def overflow_and_guard():
        for k in range(60):
            buf.push({
                "own": np.zeros(d.state_dim_max, dtype=np.float32),
                "own_mask": np.ones(d.state_dim_max, dtype=np.float32),
                "nbr": np.zeros((d.neighbor_degree_max, d.state_dim_max), dtype=np.float32),
                "nbr_mask": np.zeros(d.neighbor_degree_max, dtype=np.float32),
                "rel": np.zeros(d.neighbor_degree_max, dtype=np.int64),
                "group": np.zeros(d.neighbor_degree_max, dtype=np.int64),
                "phase_avail": np.int64(1),
                "phase_action": np.int64(0),
                "duration_action": np.int64(0),
                "reward": np.float32(0.0),
                "done": np.float32(0.0),
                "inter_index": np.int64(0),
                "sim_time": np.float32(k),
                "next_own": np.zeros(d.state_dim_max, dtype=np.float32),
                "next_own_mask": np.ones(d.state_dim_max, dtype=np.float32),
                "next_nbr": np.zeros((d.neighbor_degree_max, d.state_dim_max), dtype=np.float32),
                "next_nbr_mask": np.zeros(d.neighbor_degree_max, dtype=np.float32),
                "next_rel": np.zeros(d.neighbor_degree_max, dtype=np.int64),
                "next_group": np.zeros(d.neighbor_degree_max, dtype=np.int64),
                "next_phase_avail": np.int64(1),
            })
        if len(buf) != 64:
            raise AssertionError("环形缓冲应保持容量 64，实际 {}".format(len(buf)))
        try:
            buf.sample(1000, rng=np.random.default_rng(2))
            raise AssertionError("超量采样没有报错")
        except ValueError:
            pass
        return "容量上限生效；超量采样正确抛 ValueError"

    def metrics_formula():
        from eval_metric import EpisodeMetrics
        m = EpisodeMetrics(interval=1.0)
        for t in range(1, 11):
            m.observe(float(t), queue_sum=4.0, waiting_sum=3.0, n_active=10,
                      n_seen_total=10, att_now=100.0 + t)
        out = m.finalize(att_final=111.0)
        if abs(out["AQL"] - 4.0) > 1e-6:
            raise AssertionError("AQL 应为 4.0，实际 {}".format(out["AQL"]))
        if abs(out["AWT"] - (3.0 * 10 / 10)) > 1e-6:
            raise AssertionError("AWT 应为 3.0，实际 {}".format(out["AWT"]))
        if abs(out["ATT"] - 111.0) > 1e-6:
            raise AssertionError("ATT 应取引擎终值 111.0，实际 {}".format(out["ATT"]))
        return "AQL=ΣQ/n、AWT=Σ(W·Δt)/N_total、ATT=引擎终值 三者口径正确"

    def relative_metric():
        from eval_metric import relative_improvement
        v = relative_improvement(500.0, 300.0)
        if abs(v - 40.0) > 1e-6:
            raise AssertionError("相对提升应为 40%，实际 {}".format(v))
        return "relative_improvement(500,300)=40.0%"

    check("buffer：push/sample 往返", roundtrip)
    check("buffer：容量上限与异常保护", overflow_and_guard)
    check("metric：ATT/AWT/AQL 口径", metrics_formula)
    check("metric：相对提升计算", relative_metric)


# ===========================================================================
# [5] 绘图自检（训练曲线 + 测试评估图；验证代码链路，数值为构造的真值样本）
# ===========================================================================

def _matplotlib_preflight(probe_png: str) -> "tuple":
    """
    用**子进程**试跑一次最小渲染，返回 (是否可用, 原因)。

    必须用子进程：matplotlib 原生扩展损坏时（缺 DLL / ABI 不匹配），
    `import matplotlib.pyplot` 会让当前进程以原生异常码直接退出（本机 Windows conda
    实测 -1066598273 / 0xC06D007F），进程内的 try/except 完全拦不住。
    """
    import subprocess
    code = (
        "import matplotlib; matplotlib.use('Agg');"
        "import matplotlib.pyplot as plt;"
        "fig = plt.figure(figsize=(1, 1)); plt.plot([0.0, 1.0], [1.0, 0.0]);"
        "fig.savefig(r'{png}', dpi=50); plt.close(fig); print('MPL_OK')"
    ).format(png=probe_png)
    log_path = os.path.splitext(probe_png)[0] + ".log"
    env_vars = dict(os.environ)
    env_vars["PYTHONIOENCODING"] = "utf-8"
    try:
        os.makedirs(os.path.dirname(probe_png), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            proc = subprocess.run([sys.executable, "-c", code], stdout=fh,
                                  stderr=subprocess.STDOUT, env=env_vars)
    except Exception as e:                                 # noqa: BLE001
        return False, "预检子进程启动失败 {}: {}".format(type(e).__name__, e)
    if proc.returncode != 0:
        lines = []
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                lines = [ln.strip() for ln in fh.read().strip().splitlines() if ln.strip()]
        except Exception:                                   # noqa: BLE001
            pass
        detail = lines[-1] if lines else "无输出"
        code = int(proc.returncode)
        shown = "{}".format(code) if 0 < code < 256 else "{} (0x{:08X})".format(code, code & 0xFFFFFFFF)
        return False, "子进程退出码 {}（{}；日志 {}）".format(shown, detail, log_path)
    if not (os.path.exists(probe_png) and os.path.getsize(probe_png) > 0):
        return False, "预检未生成图片文件（{}）".format(probe_png)
    return True, ""


def check_plotting(schema: DataSchema) -> None:
    """
    验证 plot_metrics.py 的绘图链路：
      a) 逐秒序列导出（series_from_metrics）与 npz 往返；
      b) 训练曲线图（读 train_log.csv/eval_log.csv）；
      c) 测试评估 4 类图（时序/箱线/柱状/CDF）；
      d) 反证：产物缺失时必须报错退出，而不是画出占位图。

    本组使用的数值是「按公式构造的真值样本」（仅用于验证绘图代码路径），
    不是任何实验结论；论文级图必须由 train_rl_aux.py 的真实运行产物生成。

    【为什么先做 matplotlib 子进程预检】
    matplotlib 装上了但原生扩展损坏时（缺 DLL / ABI 不匹配），导入 pyplot 会让
    整个进程以原生异常码直接退出（本机 Windows conda 实测 -1066598273 / 0xC06D007F），
    Python 的 try/except 拦不住。因此先用子进程做一次最小渲染试跑：
    预检失败时，纯 numpy 的检查照常执行，需要真正渲染的检查标记为 SKIP（不计入失败）。
    """
    print("\n[5] 绘图自检（plot_metrics.py；数据源必须是真实产物）")
    from eval_metric import (EpisodeMetrics, load_series, per_intersection_ids,  # noqa: F401
                             save_series, save_series_csv, series_from_metrics)

    out_root = os.path.join(config.OUTPUT_DIR, "_selfcheck")
    fig_dir = os.path.join(out_root, "figures")
    log_dir = os.path.join(out_root, "logs")
    rep_dir = os.path.join(out_root, "reports")
    for d in (fig_dir, log_dir, rep_dir):
        os.makedirs(d, exist_ok=True)

    try:
        import plot_metrics
    except Exception as e:
        raise RuntimeError("无法导入 plot_metrics（请先 pip install 'matplotlib>=3.5'）：{!r}".format(e))

    render_ok, render_reason = _matplotlib_preflight(os.path.join(out_root, "mpl_probe.png"))
    if render_ok:
        print("   [预检] matplotlib 渲染可用（子进程最小渲染试跑通过）")
    else:
        print("   [预检] matplotlib 渲染不可用：{}".format(render_reason))
        print("          需要渲染的检查将标记为 SKIP（纯 numpy 的检查照常执行）。")
        print("          修复：pip install --force-reinstall --no-cache-dir 'matplotlib>=3.5'")

    # --- a) 序列导出与 npz 往返 ---
    def series_roundtrip():
        m = EpisodeMetrics(interval=1.0)
        for t in range(1, 121):
            m.observe(float(t), queue_sum=float(3 + (t % 7)), waiting_sum=float(2 + (t % 5)),
                      n_active=50, n_seen_total=200 + t, att_now=100.0 + 0.5 * t)
            m.observe_decision({"intersection_1_1": float(t % 4), "intersection_2_2": 2.0})
        s = series_from_metrics(m)
        for k in ("t", "queue", "waiting", "active", "seen", "att", "delay",
                  "waited_vehicle_seconds"):
            if k not in s:
                raise AssertionError("序列缺少键 {}".format(k))
        if s["t"].size != 120 or abs(s["delay"][-1] - s["waited_vehicle_seconds"][-1]
                                     / s["seen"][-1]) > 1e-9:
            raise AssertionError("delay 口径应为 累计等待车辆秒数 / N_seen(t)")
        ids = per_intersection_ids(s)
        if ids != ["intersection_1_1", "intersection_2_2"]:
            raise AssertionError("per-intersection 序列应保留真实交叉口 id，实际 {}".format(ids))
        p = save_series(s, os.path.join(log_dir, "series", "selftest_series.npz"))
        back = load_series(p)
        if not np.allclose(back["queue"], s["queue"]):
            raise AssertionError("npz 往返后数据不一致")
        # 回归用例：train_rl_aux.py 曾因漏 import save_series_csv 导致 CSV 静默不落盘
        csv_p = save_series_csv(s, os.path.join(log_dir, "series", "selftest_series.csv"))
        with open(csv_p, "r", encoding="utf-8") as fh:
            rows = fh.read().strip().splitlines()
        head = rows[0].split(",")
        if "queue" not in head or "delay" not in head:
            raise AssertionError("序列 CSV 表头应含逐秒真实字段，实际 {}".format(head))
        if any("per_inter__" in c for c in head):
            raise AssertionError("序列 CSV 不应包含 per-inter 键（表头 {}）".format(head))
        if len(rows) - 1 != s["t"].size:
            raise AssertionError("序列 CSV 行数应等于仿真步数 {}，实际 {}".format(
                s["t"].size, len(rows) - 1))
        return ("序列键齐全（120 步）、delay 口径正确、per-inter id 保留、"
                "npz 往返一致、CSV 表头/行数正确")

    # --- b) 训练曲线图 ---
    def training_curves_figure():
        import csv as _csv
        tr = os.path.join(log_dir, "train_log.csv")
        ev = os.path.join(log_dir, "eval_log.csv")
        with open(tr, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["episode", "sim_seconds", "decisions", "epsilon", "avg_reward",
                        "ATT", "AWT", "AQL", "throughput", "loss_mean", "grad_updates",
                        "replay_size", "wall_seconds"])
            for ep in range(12):
                w.writerow([ep, 3600, 2300, 1.0 - ep * 0.08, -10.0 + ep,
                            600.0 - 12.0 * ep, 120.0 - 4.0 * ep, 30.0 - 1.2 * ep,
                            500 + 15 * ep, 0.5 / (ep + 1), 100 * ep, 2300 * (ep + 1), 30.0])
        with open(ev, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["episode", "ATT", "AWT", "AQL", "throughput", "avg_decision_reward"])
            for ep in (5, 11):
                w.writerow([ep, 600.0 - 12.0 * ep, 120.0 - 4.0 * ep, 30.0 - 1.2 * ep,
                            500 + 15 * ep, -9.0 + ep])
        files = plot_metrics.plot_training_curves(
            tr, os.path.join(fig_dir, "training_curves.png"), eval_log_path=ev,
            smooth=3, dpi=90, lang="auto", also_individual=True)
        sizes = [os.path.getsize(p) for p in files]
        if len(files) < 5 or min(sizes) < 3000:
            raise AssertionError("应生成 1 张拼图 + 4 张单图且文件非空，实际 {} 张 / {} 字节".format(
                len(files), sizes))
        return "2×2 拼图 + 4 张单图，最小 {} 字节".format(min(sizes))

    # --- c) 评估 4 类图 ---
    def evaluation_figures():
        m1, m2 = EpisodeMetrics(interval=1.0), EpisodeMetrics(interval=1.0)
        for t in range(1, 301):
            m1.observe(float(t), queue_sum=float(18 + 6 * np.sin(t / 20.0)),
                       waiting_sum=float(12 + 4 * np.cos(t / 15.0)), n_active=80,
                       n_seen_total=400 + 2 * t, att_now=500.0)
            m2.observe(float(t), queue_sum=float(11 + 4 * np.sin(t / 20.0)),
                       waiting_sum=float(7 + 3 * np.cos(t / 15.0)), n_active=80,
                       n_seen_total=400 + 2 * t, att_now=430.0)
            for iid in ("intersection_1_1", "intersection_2_2", "intersection_3_3"):
                m1.observe_decision({iid: float(10 + hash(iid) % 5)})
                m2.observe_decision({iid: float(6 + hash(iid) % 4)})
        series = {"Baseline-A": series_from_metrics(m1), "Policy-B": series_from_metrics(m2)}
        metrics = {"Baseline-A": {"ATT": 500.0, "AWT": 100.0, "AQL": 18.0, "throughput": 900.0},
                   "Policy-B": {"ATT": 430.0, "AWT": 70.0, "AQL": 11.0, "throughput": 1010.0}}
        figs = plot_metrics.plot_evaluation_report(
            series, metrics, fig_dir, prefix="eval", fmt="png", dpi=90, lang="auto",
            smooth=9, source_note="selfcheck（构造真值样本，非实验结果）", with_cdf=True)
        if sorted(figs.keys()) != ["bar", "boxplot", "cdf", "timeseries"]:
            raise AssertionError("评估图应包含 4 类，实际 {}".format(sorted(figs.keys())))
        sizes = {k: os.path.getsize(v) for k, v in figs.items()}
        if min(sizes.values()) < 3000:
            raise AssertionError("图片文件过小，疑似空图：{}".format(sizes))
        return "时序/箱线/柱状/CDF 四类图均生成（最小 {} 字节）".format(min(sizes.values()))

    # --- d) 评估图输入汇集（纯 numpy：验证 series / eval_log / run_report 的聚合逻辑） ---
    def eval_inputs_aggregation():
        import csv as _csv
        import json as _json
        run_root = os.path.join(out_root, "fake_run")
        r_log = os.path.join(run_root, "logs")
        r_rep = os.path.join(run_root, "reports")
        os.makedirs(os.path.join(r_log, "series"), exist_ok=True)
        os.makedirs(r_rep, exist_ok=True)

        mb, mr = EpisodeMetrics(interval=1.0), EpisodeMetrics(interval=1.0)
        for t in range(1, 61):
            mb.observe(float(t), queue_sum=20.0, waiting_sum=10.0, n_active=40,
                       n_seen_total=100 + t, att_now=500.0)
            mr.observe(float(t), queue_sum=12.0, waiting_sum=6.0, n_active=40,
                       n_seen_total=100 + t, att_now=420.0)
        save_series(series_from_metrics(mb),
                    os.path.join(r_log, "series", plot_metrics.SERIES_BASELINE))
        save_series(series_from_metrics(mr),
                    os.path.join(r_log, "series", "eval_ep007_series.npz"))
        csv_p = save_series_csv(series_from_metrics(mr),
                                os.path.join(r_log, "series", "eval_ep007_series.csv"))
        with open(csv_p, "r", encoding="utf-8") as fh:
            head = fh.readline().strip()
        if "queue" not in head or "per_inter__" in head:
            raise AssertionError("序列 CSV 表头应为逐秒真实字段且不含 per-inter 键，实际 {}".format(head))

        with open(os.path.join(r_log, "eval_log.csv"), "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["episode", "ATT", "AWT", "AQL", "throughput"])
            w.writerow([3, 480.0, 90.0, 17.0, 900.0])
            w.writerow([7, 420.0, 60.0, 12.0, 1010.0])       # ep7 的 ATT 最小 → 应被选为对比对象
        with open(os.path.join(r_rep, "run_report.json"), "w", encoding="utf-8") as f:
            _json.dump({"fixed_time_baseline": {"ATT": 500.0, "AWT": 100.0, "AQL": 20.0,
                                                "throughput": 880.0}}, f)

        series, metrics, note, missing = plot_metrics.build_eval_inputs(r_log, r_rep, verbose=False)
        if plot_metrics.LBL_BASELINE_KEY not in series:
            raise AssertionError("应识别基线逐秒序列 {}；missing={}".format(
                plot_metrics.SERIES_BASELINE, missing))
        rl_key = plot_metrics.LBL_RL_KEY.format(7)
        if rl_key not in series:
            raise AssertionError("应选 ATT 最优的评测 ep7；实际 tags={}".format(list(series)))
        if abs(metrics[rl_key]["ATT"] - 420.0) > 1e-9:
            raise AssertionError("RL 指标应取自 eval_log.csv 的真实行")
        if abs(metrics[plot_metrics.LBL_BASELINE_KEY]["AQL"] - 20.0) > 1e-9:
            raise AssertionError("基线指标应取自 run_report.json 的 fixed_time_baseline")
        return "基线 + 最优评测 ep7 的序列与指标均按真值汇集（来源 {}）".format(note)

    # --- e) 反证：产物缺失必须报错 ---
    def missing_artifacts_guard():
        empty = os.path.join(out_root, "empty_run")
        dirs = {"output": empty, "log": os.path.join(empty, "logs"),
                "report": os.path.join(empty, "reports"),
                "model": os.path.join(empty, "rl_model"),
                "generated": os.path.join(empty, "generated")}
        try:
            plot_metrics.generate_all(dirs, fmt="png", dpi=90, verbose=False)
            raise AssertionError("缺失 train_log.csv 时没有报错（可能画出占位图）")
        except FileNotFoundError as e:
            if "train_log.csv" not in str(e):
                raise AssertionError("报错信息应指出缺失 train_log.csv，实际 {}".format(e))
        return "train_log.csv 缺失时正确抛 FileNotFoundError，不生成占位图"

    def render_check(name: str, fn) -> None:
        """需要真正渲染的检查：matplotlib 预检失败时记 SKIP（环境问题，不算代码失败）。"""
        if not render_ok:
            _RESULTS.append((name, None, render_reason))
            print("  [SKIP] {} → {}".format(name, render_reason))
            return
        check(name, fn)

    check("plot：序列导出 / CSV / npz 往返", series_roundtrip)
    check("plot：评估图输入汇集（真值聚合）", eval_inputs_aggregation)
    check("plot：产物缺失必须报错（反证）", missing_artifacts_guard)
    render_check("plot：训练过程 4 曲线出图", training_curves_figure)
    render_check("plot：测试评估 4 类图出图", evaluation_figures)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="离线自检（不依赖真实 CityFlow 引擎）")
    p.add_argument("--data-dir", default=None, help="数据目录（默认 ./data）")
    p.add_argument("--only", default=None,
                   help="只跑某一组（组名片段）：schema / env / torch / buffer / plot")
    args = p.parse_args(argv)

    print("=" * 78)
    print("RL 辅助模型工程自检（对应论文第四章 4.2 节）")
    print("数据目录: {}".format(args.data_dir or config.DATA_DIR))
    print("=" * 78)

    try:
        schema = build_schema(args.data_dir, verbose=True)
    except ValueError as e:
        print("\n[数据校验失败，自检终止]\n{}".format(e))
        return 2

    # 分组注册：--only 可按组名片段选择（例如在没有 torch 的解释器里只跑 plot 组）
    groups = [
        ("schema", "数据解析与字段校验", check_schema),
        ("env", "环境 / 状态 / 动作掩码（MockEngine）", check_env_mock),
        ("torch", "网络前向与训练链路（PyTorch）", check_torch),
        ("buffer", "回放池与评价指标口径", check_buffer_and_metrics),
        ("plot", "绘图链路（matplotlib，环境不支持则 SKIP）", check_plotting),
    ]
    if args.only:
        sel = args.only.strip().lower()
        picked = [g for g in groups if sel in g[0]]
        if not picked:
            print("\n[参数错误] --only {} 未匹配任何组；可选：{}".format(
                args.only, ", ".join(g[0] for g in groups)))
            return 3
    else:
        picked = groups
    for _key, _title, fn in picked:
        fn(schema)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok is True)
    n_skip = sum(1 for _, ok, _ in _RESULTS if ok is None)
    n_fail = sum(1 for _, ok, _ in _RESULTS if ok is False)
    print("\n" + "=" * 78)
    print("自检结果: 通过 {} / 共 {} 项（跳过 {}，失败 {}）".format(
        n_pass, len(_RESULTS), n_skip, n_fail))
    if n_fail:
        print("失败项: {}".format([n for n, ok, _ in _RESULTS if ok is False]))
        print("=" * 78)
        return 1
    if n_skip:
        print("跳过项: {}".format([n for n, ok, _ in _RESULTS if ok is None]))
        print("跳过 = 该环境缺少能力（多为 matplotlib 原生扩展损坏），不代表代码有缺陷。")
        print("修复示例: pip install --force-reinstall --no-cache-dir 'matplotlib>=3.5'")
        print("提示：train_rl_aux.py 已把绘图放到子进程，绘图环境崩溃不影响训练与权重落盘。")
    print("已执行的检查全部通过。提示：MockEngine 仅用于链路自检，"
          "论文级结果请在 Ubuntu/WSL conda 环境用真实 CityFlow 运行 train_rl_aux.py。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
