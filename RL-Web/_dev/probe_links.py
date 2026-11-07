# -*- coding: utf-8 -*-
"""_dev/probe_links.py —— 核对 roadLink 的 startRoad/endRoad 语义与方位标签（只读）"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rl_core  # noqa: E402

IID = sys.argv[1] if len(sys.argv) > 1 else "intersection_1_1"


def main():
    import glob
    data = rl_core.data_dir()
    rn_path = sorted(glob.glob(os.path.join(data, "roadnet*.json")))[0]
    with open(rn_path, encoding="utf-8") as f:
        rn = json.load(f)
    points = dict((i["id"], (i["point"]["x"], i["point"]["y"])) for i in rn["intersections"])
    roads = dict((r["id"], r) for r in rn["roads"])
    inter = [i for i in rn["intersections"] if i["id"] == IID][0]
    x, y = points[IID]
    print("roadnet:", os.path.basename(rn_path))
    print("intersection", IID, "point=(%.1f, %.1f)" % (x, y))
    print("its roads field:", inter.get("roads"))
    print()
    print("roadLinks (start -> end), 及两端坐标:")
    for k, rl in enumerate(inter["roadLinks"]):
        s, e = roads[rl["startRoad"]], roads[rl["endRoad"]]
        sp = points[s["startIntersection"]]
        ep = points[e["endIntersection"]]
        sx, sy = points[s["endIntersection"]]
        ex, ey = points[e["startIntersection"]]
        print("  [%2d] %-8s %-16s(start=%.0f,%.0f -> end=%.0f,%.0f) -> %-16s(start=%.0f,%.0f -> end=%.0f,%.0f)"
              % (k, rl["type"], rl["startRoad"], sp[0], sp[1], sx, sy, rl["endRoad"], ex, ey, ep[0], ep[1]))
    print()
    print("路网中与该口相邻的 4 个点:")
    for iid, (px, py) in sorted(points.items()):
        if iid == IID:
            continue
        if abs(px - x) < 400 and abs(py - y) < 400:
            print("  %-18s (%.0f, %.0f)  dx=%+.0f dy=%+.0f" % (iid, px, py, px - x, py - y))
    # 相位 1..4 的流向标签复核
    import data_convert
    schema = rl_core.build_schema(data)
    ctx = data_convert.build_plan_context(schema)
    row = [r for r in ctx["intersections"] if r["intersection_id"] == IID][0]
    print()
    for slot in row["slots"]:
        print("  slot", slot["slot"], "light_phase", slot["light_phase"], "->", " / ".join(slot["flows"]))


if __name__ == "__main__":
    main()
