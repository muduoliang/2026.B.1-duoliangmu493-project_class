# -*- coding: utf-8 -*-
"""
replay_buffer.py —— 经验回放缓冲区（论文4.2节 DQN 五大模块之一：Replay-Buffer）

对应论文：王之颢《城市交通信号控制大语言模型方法研究与系统实现》第四章 4.2 节
         论文给定容量 12000（config.REPLAY_CAPACITY），经验元组为 (S, a_p, a_d, R, S')。

【实现要点】
1. 内存预分配：容量 12000 × (状态 + 邻居) 用 numpy 定长数组存储，避免 Python 对象开销；
2. 维度全部来自运行时推导（state_dim_max / neighbor_degree_max / n_rel_pos_classes /
   n_duration_actions 由 env_cityflow.EnvSpec 提供），**本文件不写死任何特征维度**；
3. 每次 push 都校验形状，形状不符直接抛 ValueError（数据完整性约束）；
4. 邻居信息以 [K_max, D_max] 矩阵 + 有效位掩码存储，不存在的邻居一律为 0 且掩码为 0，
   绝不填充虚构特征。
"""

# 兼容 Python 3.8
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

import config


class ReplayBuffer:
    """
    定长环形经验回放池。

    一条经验（对应「某个交叉口的一次决策」）包含：
        own / own_mask                 当前状态 S_x
        nbr / nbr_mask / rel / group   当前状态里的邻居信息（论文4.2节 NR/NS 模块输入）
        phase_avail                    该交叉口真实可选相位动作数（异构掩码用）
        phase_action / duration_action 双输出动作 (a_p, a_d)
        reward                         论文4.2节奖励 R_target
        next_*                         下一状态 S'
        done                           该 episode 是否在该决策后结束
        inter_index / sim_time         元信息（交叉口下标、仿真时刻，便于诊断）
    """

    def __init__(self,
                 capacity: int = config.REPLAY_CAPACITY,
                 state_dim_max: int = 0,
                 neighbor_degree_max: int = 0,
                 n_rel_pos_classes: int = 0,
                 n_duration_actions: int = 0,
                 seed: Optional[int] = None):
        if capacity <= 0:
            raise ValueError("[回放池构造失败] capacity 必须为正，实际 {}".format(capacity))
        for name, v in (("state_dim_max", state_dim_max),
                        ("neighbor_degree_max", neighbor_degree_max),
                        ("n_rel_pos_classes", n_rel_pos_classes),
                        ("n_duration_actions", n_duration_actions)):
            if v <= 0:
                raise ValueError(
                    "[回放池构造失败] {}={} 非法：维度必须由 ./data 运行时推导后传入"
                    .format(name, v))

        self.capacity = int(capacity)
        self.state_dim_max = int(state_dim_max)
        self.K = int(neighbor_degree_max)
        self.n_rel = int(n_rel_pos_classes)
        self.n_dur = int(n_duration_actions)

        C, D, K = self.capacity, self.state_dim_max, self.K
        f32, i64 = np.float32, np.int64
        # --- 预分配 ---
        self.own = np.zeros((C, D), dtype=f32)
        self.own_mask = np.zeros((C, D), dtype=f32)
        self.nbr = np.zeros((C, K, D), dtype=f32)
        self.nbr_mask = np.zeros((C, K), dtype=f32)
        self.rel = np.zeros((C, K), dtype=i64)
        self.group = np.zeros((C, K), dtype=i64)
        self.phase_avail = np.zeros((C,), dtype=i64)

        self.next_own = np.zeros((C, D), dtype=f32)
        self.next_own_mask = np.zeros((C, D), dtype=f32)
        self.next_nbr = np.zeros((C, K, D), dtype=f32)
        self.next_nbr_mask = np.zeros((C, K), dtype=f32)
        self.next_rel = np.zeros((C, K), dtype=i64)
        self.next_group = np.zeros((C, K), dtype=i64)
        self.next_phase_avail = np.zeros((C,), dtype=i64)

        self.phase_action = np.zeros((C,), dtype=i64)
        self.duration_action = np.zeros((C,), dtype=i64)
        self.reward = np.zeros((C,), dtype=np.float32)
        self.done = np.zeros((C,), dtype=np.float32)
        self.inter_index = np.zeros((C,), dtype=i64)
        self.sim_time = np.zeros((C,), dtype=np.float32)

        self._size = 0
        self._pos = 0
        self._total_pushed = 0
        self._rng = np.random.default_rng(config.SEED if seed is None else seed)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def push(self, transition: Dict[str, Any]) -> None:
        """写入一条经验；形状/类型不符立即抛 ValueError。"""
        i = self._pos
        self._assign(self.own, i, transition, "own", (self.state_dim_max,))
        self._assign(self.own_mask, i, transition, "own_mask", (self.state_dim_max,))
        self._assign(self.nbr, i, transition, "nbr", (self.K, self.state_dim_max))
        self._assign(self.nbr_mask, i, transition, "nbr_mask", (self.K,))
        self._assign(self.rel, i, transition, "rel", (self.K,))
        self._assign(self.group, i, transition, "group", (self.K,))
        self._assign(self.phase_avail, i, transition, "phase_avail", ())
        self._assign(self.next_own, i, transition, "next_own", (self.state_dim_max,))
        self._assign(self.next_own_mask, i, transition, "next_own_mask", (self.state_dim_max,))
        self._assign(self.next_nbr, i, transition, "next_nbr", (self.K, self.state_dim_max))
        self._assign(self.next_nbr_mask, i, transition, "next_nbr_mask", (self.K,))
        self._assign(self.next_rel, i, transition, "next_rel", (self.K,))
        self._assign(self.next_group, i, transition, "next_group", (self.K,))
        self._assign(self.next_phase_avail, i, transition, "next_phase_avail", ())
        self._assign(self.phase_action, i, transition, "phase_action", ())
        self._assign(self.duration_action, i, transition, "duration_action", ())
        self._assign(self.reward, i, transition, "reward", ())
        self._assign(self.done, i, transition, "done", ())
        self._assign(self.inter_index, i, transition, "inter_index", ())
        self._assign(self.sim_time, i, transition, "sim_time", ())

        # --- 动作范围校验（真实动作空间约束）---
        if not (0 <= int(self.phase_action[i]) < int(self.phase_avail[i])):
            raise ValueError(
                "[回放池写入失败] phase_action={} 超出该交叉口真实可选相位动作数 {}"
                .format(int(self.phase_action[i]), int(self.phase_avail[i])))
        if not (0 <= int(self.duration_action[i]) < self.n_dur):
            raise ValueError(
                "[回放池写入失败] duration_action={} 超出真实持续时间档数 {}"
                .format(int(self.duration_action[i]), self.n_dur))
        if not (0 <= int(self.rel[i].max(initial=0)) < self.n_rel):
            raise ValueError("[回放池写入失败] rel 类别下标越界（真实类别数 {}）".format(self.n_rel))

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._total_pushed += 1

    def _assign(self, target: np.ndarray, i: int, tr: Dict[str, Any],
                key: str, shape) -> None:
        if key not in tr:
            raise ValueError("[回放池写入失败] 经验缺少字段 '{}'".format(key))
        arr = np.asarray(tr[key])
        if arr.shape != shape:
            raise ValueError(
                "[回放池写入失败] 字段 '{}' 形状 {} 与运行时推导的形状 {} 不一致"
                .format(key, arr.shape, shape))
        target[i] = arr

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def total_pushed(self) -> int:
        return self._total_pushed

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None
               ) -> Dict[str, np.ndarray]:
        """均匀随机采样一个批次，返回 numpy 数组字典（转 torch 由训练脚本负责）。"""
        if batch_size <= 0:
            raise ValueError("[回放池采样失败] batch_size 必须为正")
        if self._size < batch_size:
            raise ValueError(
                "[回放池采样失败] 当前样本数 {} < batch_size {}；"
                "config.STRICT_BATCH_FROM_FULL_BUFFER=True 时论文要求攒满一个 batch 再训练"
                .format(self._size, batch_size))
        rng = rng or self._rng
        idx = rng.integers(0, self._size, size=batch_size)
        return {
            "own": self.own[idx], "own_mask": self.own_mask[idx],
            "nbr": self.nbr[idx], "nbr_mask": self.nbr_mask[idx],
            "rel": self.rel[idx], "group": self.group[idx],
            "phase_avail": self.phase_avail[idx],
            "phase_action": self.phase_action[idx],
            "duration_action": self.duration_action[idx],
            "reward": self.reward[idx], "done": self.done[idx],
            "next_own": self.next_own[idx], "next_own_mask": self.next_own_mask[idx],
            "next_nbr": self.next_nbr[idx], "next_nbr_mask": self.next_nbr_mask[idx],
            "next_rel": self.next_rel[idx], "next_group": self.next_group[idx],
            "next_phase_avail": self.next_phase_avail[idx],
            "inter_index": self.inter_index[idx], "sim_time": self.sim_time[idx],
            "index": idx,
        }

    # ------------------------------------------------------------------
    # 存档 / 诊断
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """把回放池落盘（npz），便于中断续训。"""
        np.savez_compressed(
            path, size=self._size, pos=self._pos, total_pushed=self._total_pushed,
            own=self.own[:self._size], own_mask=self.own_mask[:self._size],
            nbr=self.nbr[:self._size], nbr_mask=self.nbr_mask[:self._size],
            rel=self.rel[:self._size], group=self.group[:self._size],
            phase_avail=self.phase_avail[:self._size],
            next_own=self.next_own[:self._size], next_own_mask=self.next_own_mask[:self._size],
            next_nbr=self.next_nbr[:self._size], next_nbr_mask=self.next_nbr_mask[:self._size],
            next_rel=self.next_rel[:self._size], next_group=self.next_group[:self._size],
            next_phase_avail=self.next_phase_avail[:self._size],
            phase_action=self.phase_action[:self._size],
            duration_action=self.duration_action[:self._size],
            reward=self.reward[:self._size], done=self.done[:self._size],
            inter_index=self.inter_index[:self._size], sim_time=self.sim_time[:self._size])

    def load(self, path: str) -> None:
        """从 npz 恢复回放池内容（维度必须与当前运行时推导一致）。"""
        with np.load(path) as z:
            n = int(z["size"])
            if n > self.capacity:
                raise ValueError(
                    "[回放池载入失败] 存档样本数 {} 超过当前容量 {}".format(n, self.capacity))
            for key in ("own", "own_mask", "nbr", "nbr_mask", "rel", "group",
                        "phase_avail", "next_own", "next_own_mask", "next_nbr",
                        "next_nbr_mask", "next_rel", "next_group", "next_phase_avail",
                        "phase_action", "duration_action", "reward", "done",
                        "inter_index", "sim_time"):
                arr = z[key]
                if arr.shape[1:] != getattr(self, key).shape[1:]:
                    raise ValueError(
                        "[回放池载入失败] 字段 '{}' 形状 {} 与当前推导形状 {} 不一致"
                        .format(key, arr.shape[1:], getattr(self, key).shape[1:]))
                getattr(self, key)[:n] = arr
            self._size = n
            self._pos = int(z["pos"]) % self.capacity
            self._total_pushed = int(z["total_pushed"])

    def stats(self) -> Dict[str, Any]:
        """回放池统计（写入训练日志，便于诊断）。"""
        if self._size == 0:
            return {"size": 0, "capacity": self.capacity, "reward_mean": None}
        return {
            "size": int(self._size),
            "capacity": int(self.capacity),
            "total_pushed": int(self._total_pushed),
            "reward_mean": float(self.reward[:self._size].mean()),
            "reward_min": float(self.reward[:self._size].min()),
            "reward_max": float(self.reward[:self._size].max()),
            "duration_action_hist": np.bincount(
                self.duration_action[:self._size], minlength=self.n_dur).tolist(),
            "phase_action_hist": np.bincount(
                self.phase_action[:self._size]).tolist(),
        }

    def __repr__(self) -> str:
        return ("<ReplayBuffer size={}/{} D_max={} K_max={} n_rel={} n_dur={}>"
                .format(self._size, self.capacity, self.state_dim_max, self.K,
                        self.n_rel, self.n_dur))
