# -*- coding: utf-8 -*-
"""
freeze_rl.py —— 把训练好的「融合邻居信息的 RL 辅助模型」冻结成纯推理产物。

论文对应：第四章 4.2 节（Our RL，融合邻居信息的动态交通信号控制强化学习辅助模型）。
本脚本**不改变任何训练逻辑**，只做「导出 + 校验 + 记账」三件事。

一、冻结做了什么（训练态 vs 推理态）
    1. 只保留在线网络参数 model_state，丢弃目标网络 target_state、Adam 优化器状态、
       经验回放池 —— 推理只需要一个前向。
    2. model.eval()：本网络虽无 Dropout/BatchNorm，但 eval 仍是推理语义的唯一正确入口。
    3. requires_grad_(False) + torch.no_grad()：不再构建计算图，也不产生任何梯度。
    4. 把「运行时从 ./data 推导出的维度 + 交叉口顺序 + 持续时间集合 + 数据指纹」
       一起写进冻结包，加载端因此可以「先校验再推理」：./data 一旦变化就直接报错，
       而不是静默用错维度继续推理（延续本工程「维度只能来自真实数据」的硬约束）。
    5. 记录参数哈希与文件哈希：后续 DeepSeek API 服务可据此核对「线上跑的是不是同一份权重」。

二、产物（默认 output/rl_model/frozen/）
    rl_aux_frozen.pth   # 冻结权重 + 维度 + 数据指纹（推理唯一入口）
    manifest.json       # 人读清单：哈希、维度、校验结果、加载方式、来源
    rl_aux_frozen_ts.pt # 可选：TorchScript 版本（--torchscript）

三、用法（工程根目录，WSL 内 conda 环境）
    source /home/mdy493/cityflow/bin/activate
    cd /home/mdy493/CityFlow/RL-firture
    python freeze_rl.py                                  # 冻结 best 权重（推荐）
    python freeze_rl.py --ckpt output/rl_model/dqn_rl_aux_last.pth --force
    python freeze_rl.py --verify-only                    # 只校验已冻结产物
    python freeze_rl.py --torchscript                    # 额外导出 TorchScript

注意：本脚本只读 ./data，所有产物写入 output/ 下（README 的只读约束同样适用）。
"""

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import config
from data_schema import DataValidationError, build_schema
from model_net import DQNNetwork, count_parameters

FROZEN_FORMAT = "rl_aux_frozen/v1"
PAPER_SECTION = "第四章 4.2 节 融合邻居信息的动态交通信号控制强化学习辅助模型（Our RL）"

# 冻结包里必须与 ./data 推导值逐项一致的维度键（缺一不可，绝不"尽力匹配"）
DIM_KEYS = ("state_dim_max", "n_rel_pos_classes", "n_rel_class_groups",
            "n_selectable_phases_max", "n_duration_actions", "neighbor_degree_max")


# ===========================================================================
# 1. 基础工具：哈希 / 读取
# ===========================================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def sha256_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    """对参数做「键序固定」的字节级哈希：同一份权重在任何机器上都得到同一串。"""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        arr = v.detach().to(torch.float32).cpu().contiguous().numpy()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    读取训练权重。

    torch>=2.6 的 torch.load 默认 weights_only=True；本工程的 checkpoint 只含
    张量/字典/列表/数字，因此优先以最安全的方式读取，失败再退回兼容模式。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("[冻结失败] 找不到权重文件：{}".format(path))
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                       # torch < 1.13 不支持 weights_only 参数
        return torch.load(path, map_location="cpu")
    except Exception:                       # 老 ckpt 可能带非张量对象
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


# ===========================================================================
# 2. 从 checkpoint 重建网络（维度全部取自 checkpoint 里记录的运行时推导值）
# ===========================================================================

def build_model(ckpt: Dict[str, Any]) -> DQNNetwork:
    if "model_state" not in ckpt:
        raise DataValidationError(
            "[冻结失败] 权重文件缺少 model_state 键（现有键：{}）".format(sorted(ckpt.keys())))
    dims = ckpt.get("dims")
    if not isinstance(dims, dict):
        raise DataValidationError("[冻结失败] 权重文件缺少 dims 键，无法重建网络维度")
    missing = [k for k in DIM_KEYS if k not in dims]
    if missing:
        raise DataValidationError(
            "[冻结失败] 权重文件 dims 缺少维度字段：{}".format(", ".join(missing)))
    missing_model = [k for k in ("hidden_dim", "n_heads", "n_self_attn_layers") if k not in dims]
    if missing_model:
        raise DataValidationError(
            "[冻结失败] 权重文件 dims 缺少网络结构字段：{}".format(", ".join(missing_model)))

    model = DQNNetwork(
        state_dim=int(dims["state_dim_max"]),
        n_rel_pos_classes=int(dims["n_rel_pos_classes"]),
        n_selectable_phases_max=int(dims["n_selectable_phases_max"]),
        n_duration_actions=int(dims["n_duration_actions"]),
        n_neighbor_groups=int(dims["n_rel_class_groups"]),
        hidden_dim=int(dims["hidden_dim"]),
        n_heads=int(dims["n_heads"]),
        n_self_attn_layers=int(dims["n_self_attn_layers"]),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model


def freeze_model(model: DQNNetwork, device: str = "cpu") -> DQNNetwork:
    """冻结：eval + 关闭梯度 + 搬到目标设备 + 强制 CPU 参数（导出用）。"""
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for b in model.buffers():
        b.requires_grad_(False)
    # 冻结自检：任何一个参数还带梯度，就说明冻结没生效
    grad_on = [n for n, p in model.named_parameters() if p.requires_grad]
    if grad_on:
        raise RuntimeError("[冻结失败] 以下参数仍处于可训练状态：{}".format(grad_on))
    return model


# ===========================================================================
# 3. 冻结前的一致性校验：权重维度 vs 当前 ./data 推导维度
# ===========================================================================

def schema_dims(schema: Any) -> Dict[str, int]:
    d = schema.dims
    return {
        "state_dim_max": int(d.state_dim_max),
        "n_rel_pos_classes": int(d.n_rel_pos_classes),
        "n_rel_class_groups": int(d.n_rel_class_groups),
        "n_selectable_phases_max": int(d.n_selectable_phases_max),
        "n_duration_actions": int(d.n_duration_actions),
        "neighbor_degree_max": int(d.neighbor_degree_max),
    }


def check_dims(ckpt_dims: Dict[str, Any], schema: Any) -> Dict[str, Any]:
    """
    逐项比对「训练时维度」与「当前 ./data 推导维度」。

    不一致直接抛 DataValidationError（不降级、不猜测、不补全）——这是本工程的硬约束：
    观测维度只能来自真实数据，权重与数据对不上时必须终止，而不是静默跑出无意义动作。
    """
    want = schema_dims(schema)
    bad: List[str] = []
    for k in DIM_KEYS:
        got = int(ckpt_dims[k])
        if got != want[k]:
            bad.append("{}: 权重={} / 当前 ./data={}".format(k, got, want[k]))
    # 交叉口顺序也必须一致：策略输出与 inter_index 一一对应，顺序错了动作会张冠李戴
    trained_ids = ckpt_dims.get("inter_ids")
    data_ids = list(schema.controlled_ids())
    if trained_ids is not None and list(trained_ids) != data_ids:
        bad.append("交叉口顺序不一致：权重={} / 当前 ./data={}".format(trained_ids, data_ids))
    if bad:
        raise DataValidationError(
            "[冻结失败] 权重维度与当前 ./data 不一致，已终止（不构造虚构维度）：\n  - "
            + "\n  - ".join(bad)
            + "\n提示：请用当前 ./data 重新训练，或把训练时的整份 ./data 恢复到原样。")
    return {"expected_from_data": want,
            "checkpoint_dim_keys": list(ckpt_dims.keys())}


def fingerprint_data(schema: Any) -> Dict[str, Any]:
    """./data 指纹：只用「真实参与建模的文件」，逐个 sha256 + 组合哈希。"""
    entries: List[Tuple[str, str]] = []
    df = schema.data_files
    entries.append(("roadnet", str(df["roadnet"])))
    entries.append(("flow", str(df["flow"])))
    plans = df.get("signal_plans") or {}
    if isinstance(plans, dict):
        for iid in sorted(plans.keys()):
            entries.append(("signal:{}".format(iid), str(plans[iid])))
    else:
        for p in sorted(str(x) for x in plans):
            entries.append(("signal:{}".format(os.path.basename(p)), p))

    files: List[Dict[str, Any]] = []
    lines: List[str] = []
    for name, path in entries:
        size = os.path.getsize(path)
        digest = sha256_file(path)
        files.append({"name": name, "file": os.path.basename(path),
                      "sha256": digest, "bytes": size})
        lines.append("{}\0{}\0{}".format(name, digest, size))
    combined = sha256_bytes("\n".join(sorted(lines)).encode("utf-8"))
    return {"data_dir": str(df.get("data_dir", config.DATA_DIR)),
            "files": files, "combined_sha256": combined}


# ===========================================================================
# 4. 前向自检：形状 / 有限值 / 确定性
# ===========================================================================

def dummy_inputs(dims: Dict[str, Any], batch: int = 3, seed: int = 0):
    """构造一份纯几何占位张量：只用于验证网络能前向，不代表任何真实交通状态。"""
    g = torch.Generator().manual_seed(seed)
    B = batch
    K = int(dims["neighbor_degree_max"])
    D = int(dims["state_dim_max"])
    own = torch.rand(B, D, generator=g)
    own_mask = torch.ones(B, D)
    nbr = torch.rand(B, K, D, generator=g)
    nbr_mask = torch.zeros(B, K)
    nbr_mask[:, 0] = 1.0
    rel = torch.randint(0, int(dims["n_rel_pos_classes"]), (B, K), generator=g)
    group = torch.zeros(B, K, dtype=torch.long)
    return {"own": own, "own_mask": own_mask, "nbr": nbr,
            "nbr_mask": nbr_mask, "rel": rel, "group": group}


def forward_check(model: DQNNetwork, dims: Dict[str, Any]) -> Dict[str, Any]:
    model.eval()
    t = dummy_inputs(dims)
    with torch.no_grad():
        o1 = model(**t)
        o2 = model(**t)
    phase_q, dur_q = o1["phase_q"], o1["duration_q"]
    finite = bool(torch.isfinite(phase_q).all().item() and torch.isfinite(dur_q).all().item())
    same = bool(torch.equal(phase_q, o2["phase_q"]) and torch.equal(dur_q, o2["duration_q"]))
    shapes = {"phase_q": list(phase_q.shape), "duration_q": list(dur_q.shape),
              "os_y": list(o1["os_y"].shape), "nbr_weights": list(o1["nbr_weights"].shape)}
    return {"finite": finite, "deterministic": same, "shapes": shapes}


# ===========================================================================
# 5. 写出冻结包与清单
# ===========================================================================

def make_payload(model: DQNNetwork, ckpt: Dict[str, Any], schema: Any,
                 ckpt_path: str, ckpt_sha: str, data_fp: Dict[str, Any],
                 device: str) -> Dict[str, Any]:
    dims = ckpt["dims"]
    state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
    payload = {
        "format": FROZEN_FORMAT,
        "paper_section": PAPER_SECTION,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "frozen_from": {
            "checkpoint": os.path.abspath(ckpt_path),
            "checkpoint_sha256": ckpt_sha,
            "trained_episode": int(ckpt.get("episode", -1)),
            "grad_updates": int(ckpt.get("updates", 0)),
            "decisions": int(ckpt.get("decisions", 0)),
            "best_eval_att": float(ckpt.get("best_eval_att", float("nan"))),
        },
        # 推理端只需要这些：网络参数 + 维度 + 交叉口顺序 + 动作集合
        "model_state": state,
        "param_sha256": sha256_state_dict(state),
        "dims": {k: dims[k] for k in sorted(dims.keys())},
        "inter_ids": list(ckpt.get("inter_ids", list(schema.controlled_ids()))),
        "duration_options": [int(x) for x in config.DURATION_OPTIONS],
        "lane_feature_names": list(getattr(schema.dims, "lane_feature_names", ()) or ()),
        "data_fingerprint": data_fp,
        "settings": {k: v for k, v in (ckpt.get("settings") or {}).items()
                     if k in ("hidden_dim", "n_heads", "n_self_attn_layers", "eta",
                              "gamma", "batch_size", "capacity", "seed")},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "platform": platform.platform()},
        "device_at_freeze": device,
    }
    return payload


def save_frozen(payload: Dict[str, Any], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(payload, out_path)
    return os.path.getsize(out_path)


def export_torchscript(model: DQNNetwork, dims: Dict[str, Any], out_path: str) -> Optional[str]:
    """可选：导出 TorchScript（trace）。失败不影响冻结主流程，只提示原因。"""
    try:
        model_cpu = model.to("cpu").eval()
        t = dummy_inputs(dims, batch=1)
        with torch.no_grad():
            traced = torch.jit.trace(model_cpu, (t["own"], t["own_mask"], t["nbr"],
                                                 t["nbr_mask"], t["rel"], t["group"]),
                                     strict=False)
        torch.jit.save(traced, out_path)
        return out_path
    except Exception as e:                  # noqa: BLE001 - TorchScript 只是可选产物
        print("[提示] TorchScript 导出失败（不影响冻结）：{}: {}".format(type(e).__name__, e))
        return None


def write_manifest(path: str, payload: Dict[str, Any], frozen_path: str,
                   frozen_sha: str, frozen_bytes: int, verification: Dict[str, Any],
                   torchscript_path: Optional[str]) -> Dict[str, Any]:
    dims = payload["dims"]
    manifest = {
        "format": payload["format"],
        "paper_section": payload["paper_section"],
        "created_at": payload["created_at"],
        "frozen_file": {
            "path": os.path.abspath(frozen_path),
            "sha256": frozen_sha,
            "bytes": frozen_bytes,
        },
        "source": payload["frozen_from"],
        "network": {
            "param_count": int(sum(v.numel() for v in payload["model_state"].values())),
            "param_sha256": payload["param_sha256"],
            "state_dim_max": int(dims["state_dim_max"]),
            "n_selectable_phases_max": int(dims["n_selectable_phases_max"]),
            "n_duration_actions": int(dims["n_duration_actions"]),
            "neighbor_degree_max": int(dims["neighbor_degree_max"]),
            "n_rel_pos_classes": int(dims["n_rel_pos_classes"]),
            "n_rel_class_groups": int(dims["n_rel_class_groups"]),
            "hidden_dim": int(dims["hidden_dim"]),
            "n_heads": int(dims["n_heads"]),
            "n_self_attn_layers": int(dims["n_self_attn_layers"]),
        },
        "data_fingerprint": payload["data_fingerprint"],
        "verification": verification,
        "environment": payload["environment"],
        "torchscript": torchscript_path,
        "how_to_use": [
            "python rl_aux_infer.py --selftest          # 先自检冻结包（不需要 CityFlow）",
            "python rl_aux_infer.py --demo --steps 3    # 在真实 CityFlow 上跑几步",
            "python deepseek_agent.py                   # 交给 DeepSeek 工具调用",
        ],
        "notice": "冻结包只用于推理；不要再用它继续训练（训练请回到 train_rl_aux.py 的原始权重）。",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


# ===========================================================================
# 6. 校验已冻结产物（不重新冻结）
# ===========================================================================

def verify_frozen(frozen_path: str, data_dir: str, verbose: bool = True) -> Dict[str, Any]:
    if not os.path.isfile(frozen_path):
        raise FileNotFoundError("[校验失败] 找不到冻结包：{}（先运行 python freeze_rl.py）"
                                .format(frozen_path))
    payload = load_checkpoint(frozen_path)
    if payload.get("format") != FROZEN_FORMAT:
        raise DataValidationError("[校验失败] 冻结包格式不符：{}（期望 {}）"
                                  .format(payload.get("format"), FROZEN_FORMAT))

    schema = build_schema(data_dir, verbose=False)
    check_dims(payload["dims"], schema)

    model = DQNNetwork(
        state_dim=int(payload["dims"]["state_dim_max"]),
        n_rel_pos_classes=int(payload["dims"]["n_rel_pos_classes"]),
        n_selectable_phases_max=int(payload["dims"]["n_selectable_phases_max"]),
        n_duration_actions=int(payload["dims"]["n_duration_actions"]),
        n_neighbor_groups=int(payload["dims"]["n_rel_class_groups"]),
        hidden_dim=int(payload["dims"]["hidden_dim"]),
        n_heads=int(payload["dims"]["n_heads"]),
        n_self_attn_layers=int(payload["dims"]["n_self_attn_layers"]),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_model(model, "cpu")

    param_sha = sha256_state_dict(payload["model_state"])
    if param_sha != payload["param_sha256"]:
        raise DataValidationError(
            "[校验失败] 冻结包参数哈希不一致（文件被改动或损坏）：当前={} / 记录={}"
            .format(param_sha, payload["param_sha256"]))

    report = forward_check(model, payload["dims"])
    if not report["finite"] or not report["deterministic"]:
        raise DataValidationError("[校验失败] 冻结网络前向异常：{}".format(report))

    # 再次用「当前 ./data」核对交叉口顺序与动作集合
    if list(payload["inter_ids"]) != list(schema.controlled_ids()):
        raise DataValidationError(
            "[校验失败] 冻结包交叉口顺序与当前 ./data 不一致：{} vs {}"
            .format(payload["inter_ids"], schema.controlled_ids()))
    if [int(x) for x in payload["duration_options"]] != [int(x) for x in config.DURATION_OPTIONS]:
        raise DataValidationError("[校验失败] 冻结包持续时间集合与 config.DURATION_OPTIONS 不一致")

    result = {
        "frozen_file": os.path.abspath(frozen_path),
        "param_sha256": param_sha,
        "dims": {k: payload["dims"][k] for k in DIM_KEYS},
        "forward": report,
        "inter_ids": payload["inter_ids"],
        "data_fingerprint": payload["data_fingerprint"]["combined_sha256"],
        "source_episode": payload["frozen_from"]["trained_episode"],
        "best_eval_att": payload["frozen_from"]["best_eval_att"],
        "ok": True,
    }
    if verbose:
        print("[校验通过] 冻结包与当前 ./data 完全一致")
        print("  参数哈希 : {}".format(param_sha[:16]))
        print("  数据指纹 : {}".format(result["data_fingerprint"][:16]))
        print("  前向形状 : {}".format(report["shapes"]))
    return result


# ===========================================================================
# 7. 主流程
# ===========================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="冻结 RL 辅助模型（论文 4.2 节）为纯推理产物")
    p.add_argument("--ckpt", default=os.path.join(config.MODEL_DIR, "dqn_rl_aux_best.pth"),
                   help="训练权重路径（默认 output/rl_model/dqn_rl_aux_best.pth）")
    p.add_argument("--data-dir", default=config.DATA_DIR, help="数据目录（只读）")
    p.add_argument("--out-dir", default=os.path.join(config.MODEL_DIR, "frozen"),
                   help="冻结产物目录（默认 output/rl_model/frozen）")
    p.add_argument("--device", default="cpu", help="冻结时的设备（默认 cpu，便于导出）")
    p.add_argument("--force", action="store_true", help="覆盖已存在的冻结产物")
    p.add_argument("--verify-only", action="store_true", help="只校验已有冻结包，不重新冻结")
    p.add_argument("--torchscript", action="store_true", help="额外导出 TorchScript 版本")
    p.add_argument("--quiet", action="store_true", help="精简输出")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_frozen = os.path.join(args.out_dir, "rl_aux_frozen.pth")
    out_manifest = os.path.join(args.out_dir, "manifest.json")

    print("=" * 78)
    print("RL 辅助模型冻结（论文 {}）".format(PAPER_SECTION))
    print("=" * 78)

    if args.verify_only:
        try:
            verify_frozen(out_frozen, args.data_dir, verbose=not args.quiet)
        except (DataValidationError, FileNotFoundError) as e:
            print("[失败] {}".format(e))
            return 2
        return 0

    if os.path.exists(out_frozen) and not args.force:
        print("[提示] 冻结产物已存在：{}".format(out_frozen))
        print("       如需重新生成，请加 --force（或先删除该目录）")
        try:
            verify_frozen(out_frozen, args.data_dir, verbose=not args.quiet)
        except (DataValidationError, FileNotFoundError) as e:
            print("[失败] 已有产物校验不通过：{}".format(e))
            return 2
        return 0

    # ---------- 1) 数据校验 + 权重读取 ----------
    try:
        schema = build_schema(args.data_dir, verbose=False)
    except DataValidationError as e:
        print("[失败] ./data 校验不通过：{}".format(e))
        return 2
    ckpt_sha = sha256_file(args.ckpt)
    ckpt = load_checkpoint(args.ckpt)
    print("[权重] {}".format(os.path.abspath(args.ckpt)))
    print("       sha256={}  训练 episode={}  梯度更新={}  最优评测 ATT={:.2f}s"
          .format(ckpt_sha[:16], ckpt.get("episode", -1), ckpt.get("updates", 0),
                  float(ckpt.get("best_eval_att", float("nan")))))

    # ---------- 2) 维度一致性（关键：权重 vs 当前 ./data） ----------
    try:
        dim_report = check_dims(ckpt["dims"], schema)
    except DataValidationError as e:
        print("[失败] {}".format(e))
        return 2
    print("[维度] 权重与当前 ./data 一致：{}".format(
        ", ".join("{}={}".format(k, v) for k, v in schema_dims(schema).items())))

    # ---------- 3) 重建 + 冻结 + 自检 ----------
    model = build_model(ckpt)
    n_param = count_parameters(model)
    model = freeze_model(model, args.device)
    check = forward_check(model, ckpt["dims"])
    if not check["finite"] or not check["deterministic"]:
        print("[失败] 冻结网络前向异常：{}".format(check))
        return 2

    # ---------- 4) 数据指纹 + 冻结包 ----------
    data_fp = fingerprint_data(schema)
    payload = make_payload(model, ckpt, schema, args.ckpt, ckpt_sha, data_fp, args.device)
    frozen_bytes = save_frozen(payload, out_frozen)
    frozen_sha = sha256_file(out_frozen)

    # 回读校验：写入磁盘的这一份必须与内存中的参数逐位相同
    roundtrip = load_checkpoint(out_frozen)
    if sha256_state_dict(roundtrip["model_state"]) != payload["param_sha256"]:
        print("[失败] 冻结包回读后参数哈希不一致（磁盘写入损坏）")
        return 2

    ts_path = None
    if args.torchscript:
        ts_path = export_torchscript(model, ckpt["dims"],
                                     os.path.join(args.out_dir, "rl_aux_frozen_ts.pt"))

    verification = {
        "dims_match_data": True,
        "state_dict_roundtrip": True,
        "forward_finite": check["finite"],
        "forward_deterministic": check["deterministic"],
        "no_trainable_params": True,
        "shapes": check["shapes"],
        "dims_expected_from_data": dim_report["expected_from_data"],
    }
    manifest = write_manifest(out_manifest, payload, out_frozen, frozen_sha, frozen_bytes,
                              verification, ts_path)

    # ---------- 5) 结果汇总 ----------
    print("-" * 78)
    print("[冻结完成] {}".format(out_frozen))
    print("  权重字节   : {:,} B".format(frozen_bytes))
    print("  文件 sha256: {}".format(frozen_sha))
    print("  参数哈希   : {}".format(payload["param_sha256"]))
    print("  网络参数量 : {:,}（h={}，heads={}，N={}）".format(
        manifest["network"]["param_count"], manifest["network"]["hidden_dim"],
        manifest["network"]["n_heads"], manifest["network"]["n_self_attn_layers"]))
    print("  数据指纹   : {}（{} 个数据文件）".format(
        data_fp["combined_sha256"], len(data_fp["files"])))
    print("  前向自检   : 有限值={} 确定性={} 形状={}".format(
        check["finite"], check["deterministic"], check["shapes"]))
    print("  清单文件   : {}".format(out_manifest))
    if ts_path:
        print("  TorchScript: {}".format(ts_path))
    print("=" * 78)
    print("下一步：")
    print("  1) python rl_aux_infer.py --selftest        # 冻结包推理自检（不需要 CityFlow）")
    print("  2) python rl_aux_infer.py --demo --steps 3  # 真实 CityFlow 上跑几步")
    print("  3) python deepseek_agent.py --dry-run       # 验证 DeepSeek 工具调用链路")
    return 0


if __name__ == "__main__":
    sys.exit(main())
