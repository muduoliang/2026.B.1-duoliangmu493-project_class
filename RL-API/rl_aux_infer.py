# -*- coding: utf-8 -*-
"""
rl_aux_infer.py —— 冻结后的 RL 辅助模型「纯推理封装」，供 DeepSeek / 上层智能体调用。

论文对应：第四章 4.2 节（Our RL）。本文件不含任何训练逻辑：
    * 权重来自 freeze_rl.py 导出的冻结包（eval + requires_grad=False + no_grad）；
    * 所有维度、交叉口顺序、相位槽 → 真实相位映射，全部在运行时从 ./data 推导，
      并与冻结包中记录的训练时维度逐项校验（不一致直接抛异常，不降级不猜测）；
    * 对外只暴露 JSON 可序列化的 dict，便于直接作为 API 工具返回值。

决策口径（与训练时评测一致，论文 4.2 节）：
    相位动作   : 在「该交叉口真实可选相位」掩码内取 argmax Q
    持续时间动作: 在 7 档 {10,15,20,25,30,35,40}s 内取 argmax Q
    两个输出头共享同一份融合特征 F_{x,y}（邻居注意力 + 自注意力 + 分组 softmax）。

用法（工程根目录）：
    python rl_aux_infer.py --selftest              # 冻结包推理自检（不需要 CityFlow）
    python rl_aux_infer.py --demo --steps 3        # 真实 CityFlow 上跑 3 个决策步
    python rl_aux_infer.py --demo --steps 2 --mock # 无 CityFlow 时用 MockEngine 验证链路
    python rl_aux_infer.py --demo --steps 2 --json # 只输出机器可读 JSON

作为库使用：
    from rl_aux_infer import FrozenRLAuxPolicy
    policy = FrozenRLAuxPolicy()                  # 默认读 output/rl_model/frozen/rl_aux_frozen.pth
    policy.reset()
    result = policy.advance(steps=1)              # {"decisions": [...], "metrics": {...}, ...}
"""

from __future__ import print_function

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import config
from data_schema import build_schema
from env_cityflow import CityFlowTrafficEnv
from freeze_rl import (DIM_KEYS, FROZEN_FORMAT, check_dims, forward_check,
                       load_checkpoint, sha256_state_dict)
from model_net import DQNNetwork

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FROZEN = os.path.join(config.MODEL_DIR, "frozen", "rl_aux_frozen.pth")
_LONG_KEYS = ("rel", "group", "phase_avail")


def _json_num(v: Any) -> Any:
    """严格 JSON 安全化：非有限浮点（NaN/Inf）→ None，其余数值保留 4 位小数。

    注意：episode 刚开始时 env.episode_metrics() 的 ATT/AWT/AQL 还是 NaN；
    NaN 不是合法 JSON，若直接发给 DeepSeek API 会导致工具消息解析失败。
    """
    if isinstance(v, dict):
        return {str(k): _json_num(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_num(x) for x in v]
    if v is None or isinstance(v, (bool, str)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return None
    return round(f, 4)


# ===========================================================================
# 1. 冻结包加载（带完整性 + 维度一致性校验）
# ===========================================================================

def load_frozen(frozen_path: str, data_dir: Optional[str] = None,
                verbose: bool = True) -> Dict[str, Any]:
    """读取冻结包，并用「当前 ./data 推导维度」复核一遍（防止数据被替换后静默错配）。"""
    payload = load_checkpoint(frozen_path)
    if payload.get("format") != FROZEN_FORMAT:
        raise ValueError("[加载失败] 不是冻结包：{}（期望 format={}）"
                         .format(frozen_path, FROZEN_FORMAT))
    param_sha = sha256_state_dict(payload["model_state"])
    if param_sha != payload["param_sha256"]:
        raise ValueError("[加载失败] 冻结包参数哈希不一致：当前={} / 记录={}"
                         .format(param_sha[:16], payload["param_sha256"][:16]))
    schema = build_schema(data_dir or config.DATA_DIR, verbose=False)
    check_dims(payload["dims"], schema)
    if verbose:
        print("[冻结包] {}（权重哈希 {}，训练 episode {}，最优评测 ATT {:.2f}s）".format(
            os.path.abspath(frozen_path), param_sha[:12],
            payload["frozen_from"]["trained_episode"],
            payload["frozen_from"]["best_eval_att"]))
    return {"payload": payload, "schema": schema}


def build_network(dims: Dict[str, Any]) -> DQNNetwork:
    return DQNNetwork(
        state_dim=int(dims["state_dim_max"]),
        n_rel_pos_classes=int(dims["n_rel_pos_classes"]),
        n_selectable_phases_max=int(dims["n_selectable_phases_max"]),
        n_duration_actions=int(dims["n_duration_actions"]),
        n_neighbor_groups=int(dims["n_rel_class_groups"]),
        hidden_dim=int(dims["hidden_dim"]),
        n_heads=int(dims["n_heads"]),
        n_self_attn_layers=int(dims["n_self_attn_layers"]),
    )


# ===========================================================================
# 2. 冻结策略：一步决策 / 推进 / 指标
# ===========================================================================

class FrozenRLAuxPolicy(object):
    """
    冻结 RL 辅助模型的多交叉口推理封装（异步决策，与训练时 env.step 语义一致）。

    说明：CityFlow 每个交叉口按自己的动作时长推进，待决策的下一个时刻由环境内部决定；
    本类不改变这一语义，只负责「读状态 → 前向 → 掩码 argmax → 施加动作」。
    """

    def __init__(self,
                 frozen_path: Optional[str] = None,
                 data_dir: Optional[str] = None,
                 device: str = "cpu",
                 seed: Optional[int] = None,
                 thread_num: int = 1,
                 verbose: bool = True):
        self.frozen_path = frozen_path or DEFAULT_FROZEN
        loaded = load_frozen(self.frozen_path, data_dir, verbose=verbose)
        self.payload = loaded["payload"]
        self.schema = loaded["schema"]
        self.dims = self.payload["dims"]
        self.inter_ids = list(self.payload["inter_ids"])
        self.duration_options = [int(x) for x in self.payload["duration_options"]]

        self.device = device
        self.model = build_network(self.dims)
        self.model.load_state_dict(self.payload["model_state"], strict=True)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.seed = int(config.SEED if seed is None else seed)
        self.thread_num = int(thread_num)
        self.env = None
        self.n_steps = 0
        self.n_decisions = 0
        self._last_decisions = []
        if verbose:
            print("[策略] 已冻结：state_dim={}，可选相位≤{}，持续时间档={}，参数 {:,}".format(
                self.dims["state_dim_max"], self.dims["n_selectable_phases_max"],
                len(self.duration_options),
                int(sum(v.numel() for v in self.payload["model_state"].values()))))

    # ------------------------------------------------------------------
    # 环境生命周期
    # ------------------------------------------------------------------
    def create_env(self,
                   engine: Any = None,
                   episode_seconds: Optional[float] = None,
                   warmup_seconds: Optional[float] = None,
                   verbose: bool = False) -> CityFlowTrafficEnv:
        """创建（或替换）仿真环境。engine 传 None 时要求本机已安装 cityflow。"""
        self.env = CityFlowTrafficEnv(
            self.schema,
            episode_seconds=episode_seconds,
            warmup_seconds=warmup_seconds,
            seed=self.seed,
            thread_num=self.thread_num,
            engine=engine,
            verbose=verbose,
        )
        return self.env

    def _require_env(self) -> CityFlowTrafficEnv:
        if self.env is None:
            raise RuntimeError("[使用错误] 尚未创建仿真环境，请先调用 create_env()")
        return self.env

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        env = self._require_env()
        if seed is not None:
            self.seed = int(seed)
            env.seed = int(seed)
        env.reset()
        self.n_steps = 0
        self.n_decisions = 0
        self._last_decisions = []
        return self.status()

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:                # noqa: BLE001 - 关闭失败不影响主流程
                pass
            self.env = None

    # ------------------------------------------------------------------
    # 前向 + 掩码 argmax（论文 4.2 节双输出动作）
    # ------------------------------------------------------------------
    def _batch_to_torch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        t = {}
        for k in ("own", "own_mask", "nbr", "nbr_mask", "rel", "group", "phase_avail"):
            arr = np.asarray(batch[k])
            if k in _LONG_KEYS:
                t[k] = torch.as_tensor(arr.astype(np.int64), dtype=torch.long,
                                       device=self.device)
            else:
                t[k] = torch.as_tensor(arr.astype(np.float32), dtype=torch.float32,
                                       device=self.device)
        return t

    def decide(self) -> List[Dict[str, Any]]:
        """对当前待决策交叉口给出动作（不推进仿真）。"""
        env = self._require_env()
        batch = env.decision_batch()
        t = self._batch_to_torch(batch)
        with torch.no_grad():
            out = self.model(own=t["own"], own_mask=t["own_mask"], nbr=t["nbr"],
                             nbr_mask=t["nbr_mask"], rel=t["rel"], group=t["group"])

        n_phase_slots = int(out["phase_q"].size(1))
        slot_ids = torch.arange(n_phase_slots, device=self.device).unsqueeze(0)
        valid = slot_ids < t["phase_avail"].unsqueeze(1)          # 动作掩码（真实可选相位个数）
        masked_q = out["phase_q"].masked_fill(~valid, float("-inf"))
        phase_slot = masked_q.argmax(dim=1)
        dur_idx = out["duration_q"].argmax(dim=1)

        phase_q = out["phase_q"].detach().cpu().numpy()
        dur_q = out["duration_q"].detach().cpu().numpy()
        weights = out["nbr_weights"].detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()

        decisions = []
        for row, iid in enumerate(batch["inter_ids"]):
            sp = self.schema.inter_by_id[iid]
            choices = list(sp.selectable_phase_indices)
            slot = int(phase_slot[row].item())
            light_phase = int(choices[slot])
            nbr_w = {}
            for j, name in enumerate(list(sp.neighbors)):
                if j < weights.shape[1]:
                    nbr_w[name] = round(float(weights[row, j]), 4)
            dur_index = int(dur_idx[row].item())
            decisions.append({
                "intersection_id": iid,
                "phase_action_slot": slot,
                "light_phase": light_phase,
                "duration_action_index": dur_index,
                "duration_seconds": int(self.duration_options[dur_index]),
                "phase_q": [round(float(x), 4) for x in phase_q[row]],
                "phase_valid": [bool(x) for x in valid_np[row]],
                "duration_q": [round(float(x), 4) for x in dur_q[row]],
                "neighbor_weights": nbr_w,
                "queue_len": round(float(self._queue_of(iid)), 3),
            })
        self._last_decisions = decisions
        return decisions

    def _queue_of(self, iid: str) -> float:
        env = self._require_env()
        queue = getattr(env, "_queue_per_inter", None) or {}
        try:
            return float(queue.get(iid, 0.0))
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # 推进仿真
    # ------------------------------------------------------------------
    def step(self) -> Dict[str, Any]:
        env = self._require_env()
        decisions = self.decide()
        phase_actions = {d["intersection_id"]: d["phase_action_slot"] for d in decisions}
        duration_actions = {d["intersection_id"]: d["duration_action_index"] for d in decisions}
        _, _, rewards, done, _ = env.step(phase_actions, duration_actions)
        self.n_steps += 1
        self.n_decisions += len(decisions)
        return {
            "sim_time": round(float(env.sim_time), 3),
            "step": self.n_steps,
            "n_decisions": len(decisions),
            "decisions": decisions,
            "rewards": {k: round(float(v), 4) for k, v in rewards.items()},
            "done": bool(done),
            "metrics": self.metrics() if done else None,
        }

    def advance(self, steps: int = 1) -> Dict[str, Any]:
        """连续推进 N 个决策步（episode 结束会提前停止）。"""
        steps = max(1, int(steps))
        history = []
        done = False
        for _ in range(steps):
            res = self.step()
            history.append(res)
            done = bool(res["done"])
            if done:
                break
        return {
            "steps_executed": len(history),
            "done": done,
            "sim_time": history[-1]["sim_time"] if history else round(float(self.env.sim_time), 3),
            "decisions": [d for res in history for d in res["decisions"]],
            "rewards": history[-1]["rewards"] if history else {},
            "metrics": self.metrics(),
            "last_step": history[-1] if history else None,
        }

    # ------------------------------------------------------------------
    # 指标 / 状态（JSON 可序列化）
    # ------------------------------------------------------------------
    def metrics(self) -> Dict[str, Any]:
        env = self._require_env()
        return _json_num(env.episode_metrics() or {})

    def status(self) -> Dict[str, Any]:
        env = self._require_env()
        try:
            pending = list(env.pending_ids())
        except Exception:                    # noqa: BLE001 - 无待决策交叉口时可能抛
            pending = []
        return {
            "sim_time": round(float(env.sim_time), 3),
            "episode_seconds": round(float(env.episode_seconds), 3),
            "done": bool(env.is_done()),
            "pending_intersections": pending,
            "n_steps": self.n_steps,
            "n_decisions": self.n_decisions,
            "metrics": self.metrics(),
        }

    def describe(self) -> Dict[str, Any]:
        """冻结模型自描述（不含权重），便于服务注册/健康检查。"""
        return {
            "frozen_path": os.path.abspath(self.frozen_path),
            "format": self.payload["format"],
            "param_sha256": self.payload["param_sha256"],
            "trained_episode": self.payload["frozen_from"]["trained_episode"],
            "best_eval_att": self.payload["frozen_from"]["best_eval_att"],
            "dims": {k: int(self.dims[k]) for k in DIM_KEYS},
            "hidden_dim": int(self.dims["hidden_dim"]),
            "n_heads": int(self.dims["n_heads"]),
            "n_self_attn_layers": int(self.dims["n_self_attn_layers"]),
            "inter_ids": self.inter_ids,
            "duration_options": self.duration_options,
            "data_fingerprint": self.payload["data_fingerprint"]["combined_sha256"],
            "paper_section": self.payload["paper_section"],
        }


# ===========================================================================
# 3. CLI：自检 / 演示
# ===========================================================================

def _make_mock_engine(schema):
    """无 CityFlow 时的链路验证引擎（tools/selfcheck.py 的 MockEngine，数值无实验意义）。"""
    tools_dir = os.path.join(ROOT, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from selfcheck import MockEngine      # noqa: E402
    print("[演示] 使用 MockEngine：仅验证推理链路，ATT/AQL 等数值没有实验意义")
    return MockEngine(schema)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="冻结 RL 辅助模型的推理封装（论文 4.2 节）")
    p.add_argument("--frozen", default=DEFAULT_FROZEN, help="冻结包路径")
    p.add_argument("--data-dir", default=config.DATA_DIR, help="数据目录（只读）")
    p.add_argument("--device", default="cpu", help="推理设备（cpu / cuda）")
    p.add_argument("--selftest", action="store_true", help="只做冻结包 + 前向自检（不需要 CityFlow）")
    p.add_argument("--demo", action="store_true", help="创建环境并推进若干决策步")
    p.add_argument("--steps", type=int, default=3, help="--demo 的决策步数（默认 3）")
    p.add_argument("--episode-seconds", type=float, default=None, help="演示用 episode 时长（可选）")
    p.add_argument("--mock", action="store_true", help="用 MockEngine 代替 CityFlow（仅链路验证）")
    p.add_argument("--json", action="store_true", help="只输出机器可读 JSON")
    p.add_argument("--describe", action="store_true", help="打印冻结模型自描述后退出")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not args.json:
        print("=" * 78)
        print("RL 辅助模型（冻结版）推理封装")
        print("=" * 78)
    try:
        policy = FrozenRLAuxPolicy(args.frozen, args.data_dir, device=args.device,
                                   verbose=not args.json)
    except Exception as e:                    # noqa: BLE001 - 加载失败要给可读提示
        print("[失败] 冻结包加载失败：{}: {}".format(type(e).__name__, e))
        return 2

    if args.describe or args.selftest:
        check = forward_check(policy.model, policy.dims)
        info = policy.describe()
        info["forward_check"] = check
        if not check["finite"] or not check["deterministic"]:
            print("[失败] 前向自检异常：{}".format(check))
            return 2
        if args.json:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            print("[自检通过] 冻结模型可推理")
            print("  参数哈希 : {}".format(info["param_sha256"][:16]))
            print("  维度     : {}".format(info["dims"]))
            print("  前向     : {}".format(check))
            print("  训练来源 : episode {}，最优评测 ATT {:.2f}s".format(
                info["trained_episode"], info["best_eval_att"]))
        if args.selftest and not args.demo:
            return 0

    if not args.demo:
        print("[提示] 加 --demo --steps N 可在仿真环境里推进决策；加 --selftest 只做静态自检")
        return 0

    try:
        engine = _make_mock_engine(policy.schema) if args.mock else None
        policy.create_env(engine=engine, episode_seconds=args.episode_seconds,
                          verbose=not args.json)
    except Exception as e:                    # noqa: BLE001 - 缺 cityflow 时给安装提示
        print("[失败] 仿真环境创建失败：{}: {}".format(type(e).__name__, e))
        print("       若本机没有编译 cityflow：请在 WSL conda 环境运行，或先加 --mock 验证链路。")
        return 3

    try:
        policy.reset()
        result = policy.advance(steps=args.steps)
    finally:
        policy.close()

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("[推进完成] 步数={}，仿真时刻={}s，决策数={}，结束={}".format(
            result["steps_executed"], result["sim_time"], len(result["decisions"]),
            result["done"]))
        for d in result["decisions"][:4]:
            print("  - {}: 相位槽 {} → lightphase {}，持续 {}s".format(
                d["intersection_id"], d["phase_action_slot"], d["light_phase"],
                d["duration_seconds"]))
        if len(result["decisions"]) > 4:
            print("  ... 其余 {} 条决策见 --json 输出".format(len(result["decisions"]) - 4))
        m = result["metrics"]
        if m:
            print("[指标] ATT={:.3f}s  AWT={:.3f}s  AQL={:.3f}辆  完成车辆={}".format(
                m.get("ATT", float("nan")), m.get("AWT", float("nan")),
                m.get("AQL", float("nan")), m.get("throughput", float("nan"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
