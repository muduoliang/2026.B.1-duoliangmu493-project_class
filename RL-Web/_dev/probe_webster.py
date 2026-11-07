# -*- coding: utf-8 -*-
"""
probe_webster.py —— 开发探针：核对韦伯斯特法的计算结果是否自洽、可手算复核。
用法：python _dev/probe_webster.py

检查项
    1. 周期落在 [60, 120]
    2. 每个相位：绿 + 黄 + 红 == 周期；绿信比 == 绿 / 周期
    3. Y == 各相位 y 之和；L == 相位数 ×（黄灯 + 启动损失）
    4. C0 = (1.5L + 5) / (1 - Y)，未截断时与输出一致
    5. 过饱和场景（把流量放大 50 倍）→ 周期取上限 120s 并标记 saturated
    6. 生成的 plan 相位槽与绿灯秒数合法
"""
from __future__ import print_function

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
sys.path.insert(0, ROOT)

import data_convert  # noqa: E402
import webster  # noqa: E402

DATA = os.path.join(ROOT, os.pardir, "RL-API", "data")


def load_builtin():
    with open(os.path.join(DATA, "roadnet_4X4.json"), encoding="utf-8") as f:
        roadnet = json.load(f)
    with open(os.path.join(DATA, "flow.json"), encoding="utf-8") as f:
        flow = json.load(f)
    files = [{"name": "roadnet_4X4.json", "content": json.dumps(roadnet)},
             {"name": "flow.json", "content": json.dumps(flow)}]
    tmp = os.path.join(HERE, "tmp_webster")
    conv = data_convert.convert(files, tmp, {})
    assert conv.get("ok"), conv.get("error")
    return conv, flow


def check(result, tag=""):
    tol = 0.15
    for row in result["intersections"]:
        c = row["cycle_seconds"]
        assert 60.0 - 1e-6 <= c <= 120.0 + 1e-6, (tag, row["intersection_id"], c)
        ys = sum(p["y"] for p in row["phases"])
        assert abs(ys - row["Y"]) < 1e-3, (tag, row["intersection_id"], ys, row["Y"])
        assert abs(row["L_seconds"] - row["n_phases"] * 6.0) < 1e-6, (tag, row["n_phases"])
        for p in row["phases"]:
            total = p["green"] + p["yellow"] + p["red"]
            assert abs(total - c) < tol, (tag, row["intersection_id"], p, total, c)
            if c > 0:
                assert abs(p["green_ratio"] - p["green"] / c) < 2e-3, (tag, p)
        if row["c0_seconds"] is not None and row["clamped"] is None:
            expect = (1.5 * row["L_seconds"] + 5.0) / (1.0 - row["Y"])
            assert abs(row["c0_seconds"] - expect) < 0.2, (tag, row["intersection_id"],
                                                           row["c0_seconds"], expect)
        for step in result["plan"]["intersections"][row["intersection_id"]]:
            assert 0 <= step["phase_slot"] < 4, step
            assert 1 <= step["duration"] <= 120, step


def main():
    conv, flow = load_builtin()
    schema = conv["schema"]
    res = webster.compute(schema, flow, conv.get("flow_stats"), {})
    check(res, "builtin")

    s = res["summary"]
    print("参数        : {}".format(json.dumps(res["params"], ensure_ascii=False)))
    print("需求        : {}".format(json.dumps(res["demand"], ensure_ascii=False)))
    print("汇总        : {}".format(json.dumps(s, ensure_ascii=False)))
    print("路口数      : {}".format(len(res["intersections"])))
    print()
    # 打印前两个路口，便于手算复核
    for row in res["intersections"][:2]:
        print("路口 {}｜周期 {:.1f}s｜C0={}｜Y={:.3f}｜L={:.1f}s｜黄灯 {}s".format(
            row["intersection_id"], row["cycle_seconds"],
            "无（过饱和）" if row["c0_seconds"] is None else "{:.1f}s".format(row["c0_seconds"]),
            row["Y"], row["L_seconds"], res["params"]["yellow_seconds"]))
        for p in row["phases"]:
            print("   相位{:<2} y={:<7.4f} q={:<8.1f} S={:<8.1f} 绿={:<5.1f} 黄={:<4.1f} "
                  "红={:<5.1f} 绿信比={:<6.3f} {}".format(
                      p["light_phase"], p["y"], p["volume"], p["saturation"], p["green"],
                      p["yellow"], p["red"], p["green_ratio"], "、".join(p["flows"])))
        print("   校验：Σ绿+Σ黄+Σ红 = {:.1f}s，周期 {:.1f}s".format(
            sum(p["green"] + p["yellow"] + p["red"] for p in row["phases"]), row["cycle_seconds"]))
        print()

    # 过饱和场景：把每条 flow 的车辆数放大（time 不变 → 流率 ×50）
    heavy = []
    for e in data_convert.expand_entries(flow):
        item = dict(e)
        item["vehicle_count"] = item.pop("_count", 1) * 50
        heavy.append(item)
    res2 = webster.compute(schema, {"flows": heavy}, conv.get("flow_stats"), {})
    check(res2, "heavy")
    print("过饱和场景   : Y={:.3f}，周期 {:.1f}~{:.1f}s，上限截断 {} 个，下限截断 {} 个".format(
        res2["summary"]["Y_max"], res2["summary"]["cycle_min"], res2["summary"]["cycle_max"],
        res2["summary"]["n_clamped_max"] + res2["summary"]["n_saturated"],
        res2["summary"]["n_clamped_min"]))
    assert res2["summary"]["Y_max"] > 0.95, res2["summary"]["Y_max"]
    assert res2["summary"]["cycle_max"] == 120.0, res2["summary"]["cycle_max"]

    # 低饱和场景：1/20 流量 → C0 低于下限被截断到 60s
    light = []
    for e in data_convert.expand_entries(flow):
        item = dict(e)
        item["vehicle_count"] = max(1, int(item.pop("_count", 1) / 20))
        light.append(item)
    res3 = webster.compute(schema, {"flows": light}, conv.get("flow_stats"), {})
    check(res3, "light")
    print("低饱和场景   : Y={:.3f}，周期 {:.1f}s，下限截断 {} 个".format(
        res3["summary"]["Y_max"], res3["summary"]["cycle_min"], res3["summary"]["n_clamped_min"]))
    assert res3["summary"]["n_clamped_min"] > 0, res3["summary"]
    print("\n全部通过：周期区间、绿/黄/红合计、绿信比、Y、L、C0 公式与 plan 合法性均自洽")
    print("渲染一遍单路口摘要：", webster.brief(res["intersections"][0]))


if __name__ == "__main__":
    main()
