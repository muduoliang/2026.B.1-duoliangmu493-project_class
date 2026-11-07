# -*- coding: utf-8 -*-
"""
rl_core.py —— 【原有 RL 核心代码的桥接层，本文件不实现任何算法】

═══════════════════════════════════════════════════════════════════════════
★ 重要标注：哪些是「原有 RL 核心代码」
    以下模块仍然留在 ../RL-API 目录里，**本次改造一行都没有改**，通过本文件
    动态加入 sys.path 后被直接 import 使用（所以你可以随时替换/更新模型文件，
    只要保持 ../RL-API 的模块接口不变即可）：

        ../RL-API/config.py        超参入口（含 DURATION_OPTIONS / SIM_INTERVAL 等）
        ../RL-API/data_schema.py   ★ 数据规范校验 + 维度推导（唯一维度来源）
        ../RL-API/env_cityflow.py  ★ RL 环境（状态空间/动作掩码/奖励/指标采样）
        ../RL-API/model_net.py     ★ 论文 4.2 节 DQN 网络（邻居融合 + 双输出）
        ../RL-API/eval_metric.py   ★ 评价指标 ATT/AWT/AQL 的计算口径
        ../RL-API/freeze_rl.py     ★ 冻结包格式与维度校验
        ../RL-API/rl_aux_infer.py  ★ 冻结策略推理封装（掩码 argmax 决策）

    本文件只做四件事：
        1) 定位 RL-API 目录（可用环境变量 RLAPI_HOME 覆盖）；
        2) 把该目录加入 sys.path；
        3) 懒加载上述模块并暴露少量薄封装；
        4) 加载「冻结模型」，供打分使用。
    换句话说：算法、状态构造、动作空间、指标公式全部来自原有代码。
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import print_function

import os
import sys
import threading

# ---------------------------------------------------------------------------
# 1. RL-API 目录定位（默认与本目录同级的 ../RL-API）
# ---------------------------------------------------------------------------
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RLAPI_HOME = os.path.abspath(os.path.join(WEB_DIR, os.pardir, "RL-API"))

_import_lock = threading.Lock()
_loaded = {}


def rlapi_home():
    """RL-API 根目录（可用环境变量 RLAPI_HOME 指向别的副本）。"""
    return os.path.abspath(os.environ.get("RLAPI_HOME") or DEFAULT_RLAPI_HOME)


def data_dir():
    """原有只读数据目录（roadnet_4X4.json / flow.json / signal_inter_*.txt）。"""
    return os.path.join(rlapi_home(), "data")


def frozen_path():
    """冻结模型包路径（可用环境变量 RL_WEB_FROZEN 覆盖，方便替换模型文件）。"""
    return os.environ.get("RL_WEB_FROZEN") or os.path.join(
        rlapi_home(), "output", "rl_model", "frozen", "rl_aux_frozen.pth")


def _ensure_path():
    home = rlapi_home()
    if not os.path.isdir(home):
        raise RuntimeError(
            "[RL 核心缺失] 找不到 RL-API 目录：{}\n"
            "请把本 RL-Web 放在 RL-API 同级目录，或设置环境变量 RLAPI_HOME 指向它。".format(home))
    if home not in sys.path:
        sys.path.insert(0, home)


def core(name):
    """懒加载并缓存原有 RL 核心模块（config / data_schema / env_cityflow / ...）。"""
    with _import_lock:
        if name in _loaded:
            return _loaded[name]
        _ensure_path()
        mod = __import__(name)
        _loaded[name] = mod
        return mod


# ---------------------------------------------------------------------------
# 2. 薄封装：数据规范 / 环境 / 网络（全部委托给原有代码）
# ---------------------------------------------------------------------------
def build_schema(dir_path, verbose=False):
    """【原有核心】data_schema.build_schema：解析 roadnet/flow/signal 并推导维度。"""
    return core("data_schema").build_schema(dir_path, verbose=verbose)


def make_env(schema, engine=None, episode_seconds=None, seed=None, verbose=False):
    """【原有核心】env_cityflow.CityFlowTrafficEnv：状态/奖励/指标都在原类里。"""
    env_mod = core("env_cityflow")
    return env_mod.CityFlowTrafficEnv(
        schema, episode_seconds=episode_seconds, seed=seed,
        engine=engine, verbose=verbose)


class FrozenModel(object):
    """
    冻结模型的加载封装。

    ★ 真正的前向/打分仍由原有代码执行：
        freeze_rl.load_checkpoint + freeze_rl.check_dims + model_net.DQNNetwork
    （这里复刻的只是 rl_aux_infer.FrozenRLAuxPolicy.__init__ 里那 6 行装配流程，
      因为网页流程需要「自己控制环境」，不能沿用 policy 内部那个 env。）
    """

    def __init__(self, schema, path=None):
        torch = core("torch")
        freeze = core("freeze_rl")
        model_net = core("model_net")

        self.path = os.path.abspath(path or frozen_path())
        self.payload = freeze.load_checkpoint(self.path)
        if self.payload.get("format") != freeze.FROZEN_FORMAT:
            raise RuntimeError("[模型加载失败] {} 不是冻结包（期望 format={}）"
                               .format(self.path, freeze.FROZEN_FORMAT))
        param_sha = freeze.sha256_state_dict(self.payload["model_state"])
        if param_sha != self.payload["param_sha256"]:
            raise RuntimeError("[模型加载失败] 冻结包参数哈希不一致，文件可能被改动")
        # 【原有核心】维度校验：数据与冻结模型的维度必须完全一致
        freeze.check_dims(self.payload["dims"], schema)

        dims = self.payload["dims"]
        self.dims = dims
        self.model = model_net.DQNNetwork(
            state_dim=int(dims["state_dim_max"]),
            n_rel_pos_classes=int(dims["n_rel_pos_classes"]),
            n_selectable_phases_max=int(dims["n_selectable_phases_max"]),
            n_duration_actions=int(dims["n_duration_actions"]),
            n_neighbor_groups=int(dims["n_rel_class_groups"]),
            hidden_dim=int(dims["hidden_dim"]),
            n_heads=int(dims["n_heads"]),
            n_self_attn_layers=int(dims["n_self_attn_layers"]),
        )
        self.model.load_state_dict(self.payload["model_state"], strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.param_sha256 = param_sha
        self.trained_episode = int(self.payload["frozen_from"]["trained_episode"])
        self.best_eval_att = float(self.payload["frozen_from"]["best_eval_att"])
        self._torch = torch

    # -- 前向：给定环境张量，返回两个动作头的 Q 值（原有网络，原样调用） --
    def forward(self, batch):
        t = self._torch
        long_keys = ("rel", "group", "phase_avail")
        kwargs = {}
        for k in ("own", "own_mask", "nbr", "nbr_mask", "rel", "group", "phase_avail"):
            arr = batch[k]
            if k in long_keys:
                kwargs[k] = t.as_tensor(arr.astype("int64"), dtype=t.long)
            else:
                kwargs[k] = t.as_tensor(arr.astype("float32"), dtype=t.float32)
        with t.no_grad():
            out = self.model(own=kwargs["own"], own_mask=kwargs["own_mask"],
                             nbr=kwargs["nbr"], nbr_mask=kwargs["nbr_mask"],
                             rel=kwargs["rel"], group=kwargs["group"])
        return {k: v.detach().cpu().numpy() for k, v in out.items()}

    def info(self):
        return {
            "path": self.path,
            "param_sha256": self.param_sha256,
            "trained_episode": self.trained_episode,
            "best_eval_att": round(self.best_eval_att, 3),
            "dims": {k: int(self.dims[k]) for k in (
                "state_dim_max", "n_selectable_phases_max", "n_duration_actions",
                "neighbor_degree_max", "n_rel_pos_classes", "n_rel_class_groups")},
            "hidden_dim": int(self.dims["hidden_dim"]),
            "n_heads": int(self.dims["n_heads"]),
        }


_frozen_cache = {}
_frozen_lock = threading.Lock()


def load_frozen(schema, path=None):
    """进程内缓存加载冻结模型（同一数据维度只加载一次）。"""
    key = (os.path.abspath(path or frozen_path()), tuple(sorted(
        (k, v) for k, v in schema.dims.state_dim_per_inter.items())))
    with _frozen_lock:
        if key not in _frozen_cache:
            _frozen_cache[key] = FrozenModel(schema, path=path)
        return _frozen_cache[key]


# ---------------------------------------------------------------------------
# 3. 运行环境探测（给前端「状态灯」用）
# ---------------------------------------------------------------------------
def probe():
    """返回各依赖是否可用，不抛异常（网页顶部状态条用）。"""
    out = {
        "rlapi_home": rlapi_home(),
        "rlapi_exists": os.path.isdir(rlapi_home()),
        "data_dir": data_dir(),
        "data_exists": os.path.isdir(data_dir()),
        "frozen_path": frozen_path(),
        "frozen_exists": os.path.isfile(frozen_path()),
        "python": sys.version.split()[0],
        "torch": None,
        "numpy": None,
        "cityflow": None,
        "error": None,
    }
    for name in ("numpy", "torch"):
        try:
            mod = core(name)
            out[name] = getattr(mod, "__version__", "OK")
        except Exception as e:               # noqa: BLE001 - 探测不抛异常
            out[name] = None
            out["error"] = out["error"] or "{}: {}".format(type(e).__name__, e)
    try:
        core("cityflow")
        out["cityflow"] = "installed"
    except Exception:                        # noqa: BLE001
        out["cityflow"] = None

    # 汇总给网页状态条/启动横幅用
    missing = []
    if not out["rlapi_exists"]:
        missing.append("RL-API 目录")
    if not out["data_exists"]:
        missing.append("RL-API/data")
    if not out["numpy"]:
        missing.append("numpy")
    if not out["torch"]:
        missing.append("torch")
    if not out["frozen_exists"]:
        missing.append("冻结模型 rl_aux_frozen.pth")
    out["missing"] = missing
    out["ok"] = not missing
    out["engine_hint"] = ("cityflow（真实仿真器）" if out["cityflow"]
                          else "offline（内置换算引擎，未安装 cityflow）")
    return out
