# -*- coding: utf-8 -*-
"""_dev/probe_convert.py —— 验证 data_convert 的三条路径（不参与产品逻辑）"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import data_convert  # noqa: E402
import rl_core       # noqa: E402

TMP = os.path.join(HERE, "tmp_convert")


def drop(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


def show(title, rep):
    print("=" * 78)
    print("[case]", title)
    print("ok =", rep.get("ok"), "| error =", rep.get("error"))
    print("sources:", rep.get("sources"))
    print("flow_stats:", rep.get("flow_stats"))
    print("dims:", rep.get("dims"))
    for w in rep.get("warnings") or []:
        print("warn:", w)
    for f in rep.get("files") or []:
        print("file:", f["name"], "->", f["detected"], "|", f.get("used_as") or f.get("note"))


def main():
    cfg = rl_core.core("config")
    print("DURATION_OPTIONS:", cfg.DURATION_OPTIONS)
    builtin = rl_core.data_dir()
    print("builtin data dir:", builtin)

    # ---------- case 1: 不上传 ----------
    d1 = os.path.join(TMP, "c1")
    drop(d1)
    rep1 = data_convert.convert([], d1)
    show("no upload (builtin fallback)", rep1)
    print("files in run dir:", sorted(os.listdir(d1)))

    if rep1.get("ok"):
        ctx = data_convert.build_plan_context(rep1["schema"])
        print("ctx n_intersections:", ctx["n_intersections"], "duration_options:", ctx["duration_options"])
        for row in ctx["intersections"][:2]:
            print(" ", row["intersection_id"], "nbrs=", row["neighbors"])
            for s in row["slots"]:
                print("    slot", s["slot"], "light_phase", s["light_phase"], "->", " / ".join(s["flows"]))
        # 所有口的槽位流向都非空？
        empty = [(r["intersection_id"], s["slot"]) for r in ctx["intersections"]
                 for s in r["slots"] if not s["flows"]]
        print("empty slots:", empty[:5], "count:", len(empty))

    # ---------- case 2: csv 车流表 ----------
    d2 = os.path.join(TMP, "c2")
    drop(d2)
    csv_text = (
        "startTime,endTime,interval,route,vehicle_count\n"
        "0,600,5,road_0_1_0->road_1_2_0,60\n"
        "0,600,5,road_4_1_0->road_3_2_0,60\n"
        "100,500,10,road_1_0_0->road_2_1_0,20\n"
        "0,300,5,road_not_exist->road_x,10\n"
    )
    rep2 = data_convert.convert([{"name": "demand.csv", "content": csv_text}], d2)
    show("csv flow table", rep2)
    if rep2.get("ok"):
        fp = [f for f in os.listdir(d2) if f.startswith("flow")]
        print("flow files:", fp)
        with open(os.path.join(d2, fp[0]), encoding="utf-8") as f:
            fl = json.load(f)
        print("first 3 entries:", json.dumps(fl[:3], ensure_ascii=False))

    # ---------- case 3: 路段需求表 ----------
    d3 = os.path.join(TMP, "c3")
    drop(d3)
    import glob
    import json as _json
    rn_path = sorted(glob.glob(os.path.join(builtin, "roadnet*.json")))[0]
    with open(rn_path, encoding="utf-8") as f:
        roadnet = _json.load(f)
    real_roads = [r["id"] for r in roadnet["roads"]][:4]
    demand_csv = "road,vehicles,startTime,endTime\n" + "\n".join(
        "{},{},0,{}".format(r, 30, 600) for r in real_roads)
    rep3 = data_convert.convert([{"name": "road_demand.csv", "content": demand_csv}], d3)
    show("road demand csv", rep3)
    if rep3.get("ok"):
        with open(os.path.join(d3, "flow_uploaded.json"), encoding="utf-8") as f:
            fl = json.load(f)
        print("n entries:", len(fl))
        print("first entry:", json.dumps(fl[0], ensure_ascii=False))
        routes = set(tuple(x["route"]) for x in fl)
        print("n unique routes:", len(routes))
        for r in list(routes)[:3]:
            print("  route:", " -> ".join(r))
        print("digest:", data_convert.demand_digest(fl, rep3["schema"]))

    # ---------- case 4: 同时上传 roadnet + flow（原样） ----------
    d4 = os.path.join(TMP, "c4")
    drop(d4)
    with open(os.path.join(builtin, "flow.json"), encoding="utf-8") as f:
        flow_text = f.read()
    rep4 = data_convert.convert([
        {"name": "my_roadnet.json", "content": _json.dumps(roadnet, ensure_ascii=False)},
        {"name": "my_flow.json", "content": flow_text},
    ], d4)
    show("roadnet + flow json", rep4)
    if rep4.get("ok"):
        print("digest:", data_convert.demand_digest(json.loads(flow_text), rep4["schema"]))

    print("=" * 78)
    print("TMP:", TMP)


if __name__ == "__main__":
    main()
