# -*- coding: utf-8 -*-
"""
model_net.py —— 全部神经网络实现（论文4.2节：观测编码器 + NR 模块 + NS 模块 + 双输出头）

对应论文：王之颢《城市交通信号控制大语言模型方法研究与系统实现》第四章 4.2 节
         「融合邻居信息的动态交通信号控制强化学习算法」

【模块与论文的对应关系】
  1) 观测编码器            : 把单交叉口状态 Ox（维度由 ./data 运行时推导）编码为 h 维特征
  2) 邻居关系特征融合模块 NR : 线性层映射相对位置 r(x,y) → 与邻居观测 O_y 做交叉注意力
                              → N=2 层自注意力堆叠 → 得到邻居关系融合特征 O_r^y
  3) 邻居作用强度评估模块 NS : 以 O_r^y 为 Query、本交叉口特征 O_x 为 Key/Value 做交叉注意力
                              → 打分 MLP → **按东西/南北分组 softmax** → 加权聚合
                              → MLP → 邻居作用强度特征 O_s^y（不存在的邻居由掩码屏蔽）
  4) 特征融合 F_{x,y}       : F_{x,y} = MLP(O_s^y ⊕ O_x)
  5) 双输出头              : 相位选择网络（离散相位动作）+ 持续时间选择网络（7 档持续时间）

【硬性约束】所有输入维度（状态维度、邻居数上界、相对位置类别数、相位动作数、持续时间档数）
均由 env_cityflow.EnvSpec 在运行时从 ./data 推导后传入构造函数，**代码中没有任何写死的
特征维度常量**（h、注意力头数、自注意力层数属于论文给定的网络超参数，来自 config）。
"""

# 兼容 Python 3.8
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from data_schema import DataValidationError


# ===========================================================================
# 工具：掩码安全 softmax
# ===========================================================================

def masked_softmax(scores: torch.Tensor, valid: torch.Tensor,
                   dim: int = -1, fill: float = 0.0) -> torch.Tensor:
    """
    在 valid=0 的位置输出 fill（默认 0），valid=1 的位置做 softmax。
    若某行一个有效位都没有（理论上不会发生：每个交叉口至少 1 个真实相位），
    则返回全 fill，避免 NaN。
    """
    valid = valid.to(dtype=torch.bool)
    neg_inf = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~valid, neg_inf)
    probs = torch.softmax(masked, dim=dim)
    any_valid = valid.any(dim=dim, keepdim=True)
    probs = torch.where(any_valid, probs, torch.full_like(probs, fill))
    return probs


# ===========================================================================
# 1) 观测编码器
# ===========================================================================

class ObservationEncoder(nn.Module):
    """
    单交叉口观测编码器：Ox = Encoder(Sx)（论文4.2节 状态 → 特征提取）

    输入维度 state_dim 由 ./data 运行时推导后传入；
    padding 位（own_mask=0）不参与有效信息，但为保持批处理一致性仍进入线性层，
    由外部掩码在注意力/打分处屏蔽。
    """

    def __init__(self, state_dim: int, hidden_dim: int = config.HIDDEN_DIM,
                 n_layers: int = 2):
        super().__init__()
        if state_dim <= 0:
            raise DataValidationError(
                "[网络构造失败] state_dim={} 非法（必须由 ./data 推导出正维度）".format(state_dim))
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        layers = []
        in_dim = self.state_dim
        for _ in range(max(1, n_layers)):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = self.hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ===========================================================================
# 2) 邻居关系特征融合模块（NR）
# ===========================================================================

class NeighborRelationFusion(nn.Module):
    """
    邻居关系特征融合模块（论文4.2节）：

        O_r^y = SelfAtt^N( CrossAtt( Linear(r(x,y)), O_y ) )

    说明：
      - Linear(r(x,y))：r(x,y) 是邻居相对位置的 one-hot（类别数由 ./data 的真实几何推导，
        本数据为 4 类），这里用 nn.Embedding 实现等价的「one-hot × 线性层」映射；
      - CrossAtt(Q=相对位置嵌入, KV=邻居观测编码 O_y)：让「位置关系」去查询「邻居状态」；
      - SelfAtt 堆叠 N=2 层：在邻居集合内部交换信息（含残差 + LayerNorm）；
      - 不存在的邻居由 key_padding_mask 屏蔽（论文4.2节 邻居掩码）。
    """

    def __init__(self, n_rel_pos_classes: int, hidden_dim: int = config.HIDDEN_DIM,
                 n_heads: int = config.N_HEADS, n_self_attn_layers: int = config.N_SELF_ATTN_LAYERS):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise DataValidationError(
                "[网络构造失败] 隐藏维度 h={} 无法被注意力头数 {} 整除（论文 h=128, heads=8）"
                .format(hidden_dim, n_heads))
        self.n_rel_pos_classes = int(n_rel_pos_classes)
        self.hidden_dim = int(hidden_dim)
        self.rel_embed = nn.Embedding(self.n_rel_pos_classes, self.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(self.hidden_dim, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(self.hidden_dim)
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, n_heads, batch_first=True)
            for _ in range(max(1, int(n_self_attn_layers)))
        ])
        self.self_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(max(1, int(n_self_attn_layers)))
        ])
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim))
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, own_enc: torch.Tensor, nbr_enc: torch.Tensor,
                rel: torch.Tensor, nbr_valid: torch.Tensor) -> torch.Tensor:
        """
        own_enc   [B, h]            本交叉口观测编码 O_x（此处仅用于形状/对齐，NR 不使用）
        nbr_enc   [B, K, h]         邻居观测编码 O_y
        rel       [B, K] int64      邻居相对位置类别下标 r(x,y)
        nbr_valid [B, K] float/bool 邻居有效位（1=真实邻居，0=padding）
        返回      [B, K, h]         O_r^y
        """
        valid = nbr_valid.to(dtype=torch.bool)
        any_valid = valid.any(dim=1, keepdim=True)                 # [B,1]
        # 全 padding 的行（本数据不会出现）强制放行第 0 位，避免 MHA 全掩码产生 NaN
        key_pad = ~(valid | (~any_valid))
        # Linear(r(x,y))：相对位置 one-hot 的线性映射
        q = self.rel_embed(rel)                                    # [B, K, h]

        attn_out, _ = self.cross_attn(q, nbr_enc, nbr_enc, key_padding_mask=key_pad)
        or_y = self.cross_norm(q + attn_out)                       # 交叉注意力 + 残差

        for attn, norm in zip(self.self_attn_layers, self.self_norms):
            h1, _ = attn(or_y, or_y, or_y, key_padding_mask=key_pad)
            or_y = norm(or_y + h1)                                 # 自注意力堆叠（N=2）
        or_y = self.ffn_norm(or_y + self.ffn(or_y))                # 前馈 + 残差

        return or_y * valid.unsqueeze(-1).to(or_y.dtype)           # padding 位置清零


# ===========================================================================
# 3) 邻居作用强度评估模块（NS）
# ===========================================================================

class NeighborStrengthEvaluation(nn.Module):
    """
    邻居作用强度评估模块（论文4.2节）：

        z  = CrossAtt(Q = O_r^y, KV = O_x)        每个邻居相对「本交叉口」的作用打分依据
        s  = MLP(z)                               邻居作用强度原始打分
        α  = 分组 softmax(s)                      按邻居所处方向分组（东西向 / 南北向）归一化
        O_s^y = MLP( ⊕_{g} Σ_{j∈g} α_j · O_r^y_j )  分组加权聚合后拼接再映射

    分组依据：邻居相对位置类别的方向分组（data_schema.build_relative_position_table 由真实
    路网几何推导，本数据分为东西向/南北向 2 组）。分组数 n_groups 同样由数据推导后传入。
    不存在的邻居：打分置 −inf（softmax 后权重为 0），且聚合时权重清零（论文4.2节 邻居掩码）。
    """

    def __init__(self, hidden_dim: int = config.HIDDEN_DIM, n_heads: int = config.N_HEADS,
                 n_groups: int = 2):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise DataValidationError(
                "[网络构造失败] 隐藏维度 h={} 无法被注意力头数 {} 整除".format(hidden_dim, n_heads))
        self.hidden_dim = int(hidden_dim)
        self.n_groups = int(n_groups)
        if self.n_groups < 1:
            raise DataValidationError("[网络构造失败] 邻居分组数必须 >= 1")

        self.cross_attn = nn.MultiheadAttention(self.hidden_dim, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(self.hidden_dim)
        self.self_attn = nn.MultiheadAttention(self.hidden_dim, n_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(self.hidden_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1))
        self.out_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * self.n_groups, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim))

    def forward(self, own_enc: torch.Tensor, or_y: torch.Tensor,
                group: torch.Tensor, nbr_valid: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        own_enc   [B, h]           本交叉口观测编码 O_x
        or_y      [B, K, h]        NR 模块输出的邻居关系特征 O_r^y
        group     [B, K] int64     邻居分组（0=东西, 1=南北, ...，由数据推导）
        nbr_valid [B, K]           邻居有效位
        返回 (O_s^y [B, h], 邻居作用强度权重 [B, K])
        """
        valid = nbr_valid.to(dtype=torch.bool)
        B, K, h = or_y.shape

        # --- 交叉注意力：以邻居关系特征为 Query，本交叉口特征为 Key/Value ---
        kv = own_enc.unsqueeze(1)                                   # [B, 1, h]
        attn_out, _ = self.cross_attn(or_y, kv, kv)                 # 不带 key 掩码：KV 只有自身
        z = self.cross_norm(or_y + attn_out)

        # --- 邻居集合内自注意力（增强邻居间交互后再打分）---
        h2, _ = self.self_attn(z, z, z, key_padding_mask=~valid.any(dim=1, keepdim=True) | ~valid)
        z = self.self_norm(z + h2)

        scores = self.score_mlp(z).squeeze(-1)                      # [B, K] 邻居作用强度原始打分
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)

        # --- 分组 softmax：同一分组内归一化（论文4.2节 分组 softmax）---
        weights = torch.zeros_like(scores)
        for g in range(self.n_groups):
            in_g = valid & (group == g)
            if not bool(in_g.any()):
                continue
            # 组内 softmax：使用 masked_softmax 在组维度上归一化
            w = masked_softmax(scores, in_g, dim=-1, fill=0.0)
            weights = weights + w

        # --- 分组加权聚合 → 拼接 → MLP ---
        pooled = []
        for g in range(self.n_groups):
            w_g = weights * (group == g).to(weights.dtype)           # [B, K]
            agg = torch.einsum("bk,bkh->bh", w_g, or_y)              # Σ α_j · O_r^y_j
            pooled.append(agg)
        os_y = self.out_mlp(torch.cat(pooled, dim=-1))               # [B, h]
        return os_y, weights


# ===========================================================================
# 4) 完整 Q 网络（特征融合 + 双输出头）
# ===========================================================================

class DQNNetwork(nn.Module):
    """
    论文4.2节 DQN 的在线网络结构（目标网络与之同构）：

        O_x      = Encoder(S_x)
        O_r^y    = NR(Linear(r(x,y)), O_y)
        O_s^y    = NS(O_r^y, O_x)
        F_{x,y}  = MLP(O_s^y ⊕ O_x)
        Q_phase  = PhaseHead(F_{x,y})      维度 = 运行时推导的最大可选相位动作数
        Q_dur    = DurHead(F_{x,y})        维度 = config.DURATION_OPTIONS 档数

    Q(s, (a_p, a_d)) = Q_phase(a_p) + Q_dur(a_d)（双输出动作的因子化 Q 值）。
    """

    def __init__(self,
                 state_dim: int,
                 n_rel_pos_classes: int,
                 n_selectable_phases_max: int,
                 n_duration_actions: int,
                 n_neighbor_groups: int,
                 hidden_dim: int = config.HIDDEN_DIM,
                 n_heads: int = config.N_HEADS,
                 n_self_attn_layers: int = config.N_SELF_ATTN_LAYERS):
        super().__init__()
        # --- 维度校验：全部来自运行时推导，构造前先校验，避免"默默用错维度" ---
        if state_dim <= 0 or n_rel_pos_classes <= 0 or n_selectable_phases_max <= 0 \
                or n_duration_actions <= 0 or n_neighbor_groups <= 0:
            raise DataValidationError(
                "[网络构造失败] 维度非法：state_dim={}, n_rel_pos_classes={}, "
                "n_selectable_phases_max={}, n_duration_actions={}, n_neighbor_groups={}"
                .format(state_dim, n_rel_pos_classes, n_selectable_phases_max,
                        n_duration_actions, n_neighbor_groups))

        self.state_dim = int(state_dim)
        self.n_rel_pos_classes = int(n_rel_pos_classes)
        self.n_selectable_phases_max = int(n_selectable_phases_max)
        self.n_duration_actions = int(n_duration_actions)
        self.hidden_dim = int(hidden_dim)

        self.encoder = ObservationEncoder(self.state_dim, self.hidden_dim)
        self.nr_module = NeighborRelationFusion(
            self.n_rel_pos_classes, self.hidden_dim, n_heads, n_self_attn_layers)
        self.ns_module = NeighborStrengthEvaluation(
            self.hidden_dim, n_heads, n_neighbor_groups)

        # F_{x,y} = MLP(O_s^y ⊕ O_x)（论文4.2节 特征融合）
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True))

        # 相位选择网络 / 持续时间选择网络（论文4.2节 双输出）
        self.phase_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.n_selectable_phases_max))
        self.duration_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.n_duration_actions))

    # ------------------------------------------------------------------
    def forward(self,
                own: torch.Tensor,
                own_mask: torch.Tensor,
                nbr: torch.Tensor,
                nbr_mask: torch.Tensor,
                rel: torch.Tensor,
                group: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        own      [B, D_max]       本交叉口状态
        own_mask [B, D_max]       有效维度掩码（异构交叉口 padding）
        nbr      [B, K_max, D_max] 邻居状态
        nbr_mask [B, K_max]       邻居有效位（论文4.2节 邻居掩码）
        rel      [B, K_max]       相对位置 one-hot 类别下标 r(x,y)
        group    [B, K_max]       邻居方向分组
        返回 dict: phase_q [B, P_max], duration_q [B, D], os_y, or_y, weights
        """
        # padding 位清零：保证编码器输入不含"虚构"数值（掩码位置的信息一律为 0）
        own_clean = own * own_mask.to(own.dtype)
        nbr_clean = nbr * own_mask.unsqueeze(1).to(nbr.dtype)

        own_enc = self.encoder(own_clean)                                    # [B, h]
        B, K, _ = nbr_clean.shape
        nbr_enc = self.encoder(nbr_clean.reshape(B * K, -1)).reshape(B, K, self.hidden_dim)

        or_y = self.nr_module(own_enc, nbr_enc, rel, nbr_mask)               # [B, K, h]
        os_y, weights = self.ns_module(own_enc, or_y, group, nbr_mask)       # [B, h]

        fused = self.fusion(torch.cat([os_y, own_enc], dim=-1))              # [B, h] = F_{x,y}
        return {
            "phase_q": self.phase_head(fused),        # [B, P_max]
            "duration_q": self.duration_head(fused),  # [B, D]
            "os_y": os_y,
            "or_y": or_y,
            "nbr_weights": weights,
        }


# ===========================================================================
# 5) 动作选择（ε-greedy + 掩码）
# ===========================================================================

def _rand_on_device(shape, device, generator, dtype=torch.float32):
    """生成 [0,1) 均匀随机数：兼容 CPU 生成器 + GPU 张量的组合。

    训练脚本用一个固定的 CPU torch.Generator 保证可复现；而网络张量可能位于 cuda。
    torch.rand(..., device='cuda', generator=<cpu 生成器>) 会直接报
    "Expected a 'cuda' device type for generator but found 'cpu'"，
    因此这里先在生成器自己的设备上取样，再搬到目标设备（开销可忽略）。
    """
    if generator is None:
        return torch.rand(*shape, device=device, dtype=dtype)
    return torch.rand(*shape, generator=generator, dtype=dtype).to(device)


def _randint_on_device(high, shape, device, generator):
    """生成 [0, high) 整数随机数，设备兼容性同 _rand_on_device。"""
    if generator is None:
        return torch.randint(0, high, shape, device=device)
    return torch.randint(0, high, shape, generator=generator).to(device)


def epsilon_greedy_select(phase_q: torch.Tensor,
                          duration_q: torch.Tensor,
                          phase_valid: torch.Tensor,
                          epsilon: float,
                          generator: Optional[torch.Generator] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ε-greedy 双输出动作选择（论文4.2节 双输出动作：相位 + 持续时间）。

    - 相位动作：只在 phase_valid 为 True 的槽位中取值（异构交叉口掩码，论文4.2节要求）；
    - 以概率 ε 在合法相位槽中均匀随机探索，否则取掩码后 Q 值最大的相位；
    - 持续时间动作：同样以概率 ε 随机（在真实持续时间集合的 7 档中），否则取最大 Q。
    返回 (phase_action [B], duration_action [B])，均为 int64。
    """
    B = phase_q.shape[0]
    device = phase_q.device
    valid = phase_valid.to(dtype=torch.bool, device=device)
    if valid.dim() == 1:
        valid = valid.unsqueeze(0).expand(B, -1)

    rand = _rand_on_device((B,), device, generator)
    explore = rand < float(epsilon)

    neg_inf = torch.finfo(phase_q.dtype).min
    greedy_phase = (phase_q.masked_fill(~valid, neg_inf)).argmax(dim=-1)

    # 随机相位：在合法槽位里均匀采样
    rand_scores = _rand_on_device((B, phase_q.shape[1]), device, generator)
    rand_phase = rand_scores.masked_fill(~valid, neg_inf).argmax(dim=-1)

    phase_action = torch.where(explore, rand_phase, greedy_phase)
    greedy_dur = duration_q.argmax(dim=-1)
    rand_dur = _randint_on_device(duration_q.shape[1], (B,), device, generator)
    duration_action = torch.where(explore, rand_dur, greedy_dur)
    return phase_action.to(torch.int64), duration_action.to(torch.int64)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量（写入训练日志，便于与论文规模核对）。"""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
