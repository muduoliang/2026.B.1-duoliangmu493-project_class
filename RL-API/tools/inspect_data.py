# -*- coding: utf-8 -*-
"""
tools/inspect_data.py —— 数据集字段/维度检查工具（只读）

用途：在不启动 CityFlow、不做任何训练的情况下，把 ./data 中**实际存在**的字段、
数据类型、样例值和据此推导出的所有维度打印出来，供人工核对「模型输入特征是否
完全来自真实数据」。

运行：
    python tools/inspect_data.py
    python tools/inspect_data.py --data-dir ./data --json output/reports/data_fields_report.json

注意：本工具只读取 ./data，绝不写入或修改其中任何文件。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config                                     # noqa: E402
from data_schema import build_schema              # noqa: E402


def _type_of(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list[{}]".format(len(v))
    if isinstance(v, dict):
        return "dict{{{} keys}}".format(len(v))
    if v is None:
        return "null"
    return type(v).__name__


def _inventory(name: str, obj: Dict[str, Any], indent: str = "  ") -> None:
    print("{}{} 顶层字段:".format(indent, name))
    for k, v in obj.items():
        print("{}  - {:<24} {:<18} 例: {}".format(indent, k, _type_of(v), repr(v)[:70]))


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="检查 ./data 的字段与推导维度（只读）")
    p.add_argument("--data-dir", default=None, help="数据目录（默认 ./data）")
    p.add_argument("--json", default=None, help="把字段报告另存为 json（默认只打印）")
    args = p.parse_args(argv)

    print("=" * 78)
    print("数据集字段/维度检查（只读；不修改 ./data 下任何文件）")
    print("=" * 78)

    try:
        schema = build_schema(args.data_dir, verbose=False)
    except ValueError as e:
        print("\n[数据校验失败]\n{}".format(e))
        return 2

    # --- 1) 文件清单 ---
    print("\n[1] 使用的数据文件")
    for kind, path in schema.data_files.items():
        if isinstance(path, list):
            for x in path:
                print("  - {:<14} {}".format(kind, x))
        else:
            print("  - {:<14} {}".format(kind, path))

    # --- 2) 原始 json 顶层字段 ---
    print("\n[2] 原始字段清单")
    _inventory("roadnet", schema.roadnet)

    inter = schema.roadnet["intersections"]
    controlled = [x for x in inter if not x.get("virtual", False)]
    virtual = [x for x in inter if x.get("virtual", False)]
    print("  roadnet.intersections: 共 {} 个（受控 {} / 虚拟边界 {}）".format(
        len(inter), len(controlled), len(virtual)))
    if controlled:
        _inventory("  intersection(受控样例)", controlled[0], indent="    ")
        tl = controlled[0].get("trafficLight") or {}
        _inventory("  trafficLight", tl, indent="      ")
        lps = tl.get("lightphases") or []
        if lps:
            _inventory("  lightphases[0]", lps[0], indent="        ")
        rls = controlled[0].get("roadLinks") or []
        if rls:
            _inventory("  roadLinks[0]", rls[0], indent="        ")
            lls = rls[0].get("laneLinks") or []
            if lls:
                print("        laneLinks[0] = {!r}".format(lls[0]))

    roads = schema.roadnet["roads"]
    print("  roadnet.roads: 共 {} 条".format(len(roads)))
    if roads:
        _inventory("  road(样例)", roads[0], indent="    ")
        if roads[0].get("lanes"):
            _inventory("  lanes[0]", roads[0]["lanes"][0], indent="      ")

    # --- 3) 推导维度 ---
    d = schema.dims
    print("\n[3] 运行时推导出的维度（模型输入维度即来自这里，代码中无写死值）")
    print("  - 受控交叉口数 n_intersections      = {}".format(d.n_intersections))
    print("  - 单交叉口状态维度 state_dim         = {}（逐交叉口，见下）".format(
        d.state_dim_per_inter))
    print("  - 状态维度上界 state_dim_max         = {}".format(d.state_dim_max))
    print("  - 单车道特征 {}                = {}".format(
        list(d.lane_feature_names), d.lane_feature_dim))
    print("  - 每交叉口驶入车道数                 = {}".format(d.n_incoming_lanes_per_inter))
    print("  - 每交叉口真实相位总数(one-hot 维度)  = {}".format(d.n_phases_per_inter))
    print("  - 每交叉口可选相位动作数             = {}".format(d.n_selectable_phases_per_inter))
    print("  - 相位动作数上界                     = {}".format(d.n_selectable_phases_max))
    print("  - 最大邻居数 neighbor_degree_max     = {}".format(d.neighbor_degree_max))
    print("  - 相对位置 one-hot 类别数            = {}（类别 {}）".format(
        d.n_rel_pos_classes, schema.rel_classes))
    print("  - 邻居分组数（东西/南北）             = {}".format(d.n_rel_class_groups))
    print("  - 持续时间动作数                     = {}（集合 {}）".format(
        d.n_duration_actions, config.DURATION_OPTIONS))
    print("  - 动作掩码来源                       = {}".format(schema.action_mask_source))

    # --- 4) 需求/配时 ---
    f = schema.flow
    print("\n[4] 交通需求/配时（flow 与 signal_*.txt）")
    print("  - flow 顶层格式 CityFlow 数组        = {}".format(f.is_cityflow_array))
    print("  - 需求条数 n_flows                   = {}".format(f.n_flows))
    print("  - 车辆字段 vehicle_fields            = {}".format(f.vehicle_fields))
    print("  - 时间字段 {}~{} = {} ~ {} s".format(
        f.start_time_field, f.end_time_field, f.t_min, f.t_max))
    print("  - 推导 episode 时长 horizon_seconds  = {}".format(f.horizon_seconds))
    print("  - route 长度范围                     = {} ~ {}".format(
        f.route_len_min, f.route_len_max))
    print("  - 不同 route 条数                    = {}".format(f.n_unique_routes))
    for iid, plan in sorted(schema.signal_plans.items()):
        print("  - 配时文件 {:<26} 事件 {:>4} 行 使用相位 {} 周期 {}s 绿时长 {}".format(
            os.path.basename(plan.path), plan.n_lines, plan.phases_used,
            plan.cycle_seconds, plan.green_durations))

    # --- 5) 邻居结构（论文4.2节邻居模块的输入来源） ---
    print("\n[5] 邻居关系（论文4.2节 邻居关系特征融合/作用强度评估的输入）")
    for sp in schema.intersections[:6]:
        print("  - {:<22} 邻居 {:<52} 相对位置类别 {}".format(
            sp.inter_id,
            ",".join(sp.neighbors) if sp.neighbors else "(无)",
            {n: sp.rel_class[n] for n in sp.neighbors}))
    if len(schema.intersections) > 6:
        print("  ...（共 {} 个交叉口，完整信息见 json 报告）".format(len(schema.intersections)))

    print("\n" + "=" * 78)
    print(schema.read_only_notice())

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(schema.fields_report, fh, ensure_ascii=False, indent=2)
        print("字段报告已写入: {}".format(args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
