# -*- coding: utf-8 -*-
"""开发期探针：导出 roadnet / flow / signal plan 的真实字段结构。"""
import json
import os
import sys

RLAPI = r"E:\project-study\project1-LLM_traffic\RL-API"
sys.path.insert(0, RLAPI)

import config  # noqa: E402
import data_schema  # noqa: E402
from data_schema import build_schema  # noqa: E402

print("REQUIRED_ENGINE_VEHICLE_FIELDS =", getattr(data_schema, "REQUIRED_ENGINE_VEHICLE_FIELDS", "MISSING"))

schema = build_schema(config.DATA_DIR, verbose=False)
print("\n-- FlowSpec --")
print(json.dumps({k: (v if not isinstance(v, dict) else len(v)) for k, v in schema.flow.__dict__.items()},
                 ensure_ascii=False, default=str))

raw_flow = json.load(open(schema.data_files["flow"], encoding="utf-8"))
print("flow json type:", type(raw_flow).__name__, "len:", len(raw_flow))
print("flow[0]:", json.dumps(raw_flow[0], ensure_ascii=False))
routes = sorted({tuple(f["route"]) for f in raw_flow})
print("n unique routes:", len(routes), "route lens:", sorted({len(r) for r in routes}))
print("route samples:", [list(r) for r in routes[:4]])
print("startTime range:", min(f["startTime"] for f in raw_flow), max(f["endTime"] for f in raw_flow))
print("vehicle kinds:", {type(f["vehicle"]).__name__ for f in raw_flow},
      "interval:", sorted({f["interval"] for f in raw_flow})[:8])
print("n entries with startTime==endTime:", sum(1 for f in raw_flow if f["startTime"] == f["endTime"]))

rn = schema.roadnet
print("\n-- roadnet top keys --", sorted(rn.keys()))
print("intersections type:", type(rn["intersections"]).__name__, "roads type:", type(rn["roads"]).__name__)
inters = rn["intersections"]
inter_list = list(inters.values()) if isinstance(inters, dict) else inters
roads_raw = rn["roads"]
road_list = list(roads_raw.values()) if isinstance(roads_raw, dict) else roads_raw
byid = {x["id"]: x for x in inter_list}
roads_byid = {x["id"]: x for x in road_list}
print("n intersections:", len(inter_list), "n roads:", len(road_list))
iid0 = "intersection_2_2"
inter = byid[iid0]
print("inter keys:", sorted(inter.keys()))
print("inter point/virtual/width:", inter.get("point"), inter.get("virtual"), inter.get("width"))
tl = inter["trafficLight"]
print("trafficLight keys:", sorted(tl.keys()), "n lightphases:", len(tl["lightphases"]))
for k, ph in enumerate(tl["lightphases"][:4]):
    print("  phase%d keys=%s time=%s availLinks=%s" % (k, sorted(ph.keys()), ph.get("time"), ph.get("availableRoadLinks")))
print("roadLinks n:", len(inter["roadLinks"]))
for k, rl in enumerate(inter["roadLinks"][:4]):
    print("  roadLink%d keys=%s type=%s %s->%s laneLinks=%s" % (
        k, sorted(rl.keys()), rl.get("type"), rl.get("startRoad"), rl.get("endRoad"),
        [(ll.get("startLaneIndex"), ll.get("endLaneIndex")) for ll in rl.get("laneLinks", [])][:3]))
rid0 = sorted(roads_byid.keys())[0]
road = roads_byid[rid0]
print("road keys:", sorted(road.keys()), "sample road:", rid0, road.get("startIntersection"), "->", road.get("endIntersection"))
print("lanes n:", len(road["lanes"]), "lane keys:", sorted(road["lanes"][0].keys()),
      "length:", road["lanes"][0].get("length"), "maxSpeed:", road["lanes"][0].get("maxSpeed"))
virtuals = [x["id"] for x in inter_list if x.get("virtual")]
print("virtual intersections n:", len(virtuals), virtuals[:6])

print("\n-- signal plan parsed --")
sp = schema.signal_plans[iid0]
print(json.dumps({k: v for k, v in sp.__dict__.items()}, ensure_ascii=False, default=str)[:900])
print("\n-- IntersectionSpec sample --")
ispec = schema.inter_by_id[iid0]
print("incoming_roads:", ispec.incoming_roads, "outgoing_roads:", ispec.outgoing_roads)
print("phase0:", ispec.phases[0].__dict__)
print("selectable:", ispec.selectable_phase_indices)
print("first 3 incoming lanes:", [(ln.lane_id, ln.road_id, ln.lane_index, ln.length, ln.end_intersection, ln.incoming_of) for ln in ispec.incoming_lanes[:3]])
