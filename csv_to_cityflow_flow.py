# -*- coding: utf-8 -*-
"""
csv_to_cityflow_flow.py — 把 CTM 密度预测 CSV（demo_forecast.csv 格式）转换成
CityFlow 4x4 路网（roadnet_4X4.json）可直接加载的 flow.json 流量文件。

背景与映射假设
----------------
1. CSV 中的 link_id（AB/BA/AC/CA/BD/DB/CD/DC）来自 2x2 四节点路网：
   A(左上) B(右上) C(左下) D(右下)，密度单位 veh/km（所有车道合计）。
2. 4x4 CityFlow 路网有 16 条外部进口路段，按“象限 + 行驶方向”对应展开：
       AB(北排东行) -> road_0_1_0 / road_0_2_0      CD(南排东行) -> road_0_3_0 / road_0_4_0
       BA(北排西行) -> road_5_1_2 / road_5_2_2      DC(南排西行) -> road_5_3_2 / road_5_4_2
       AC(西列南行) -> road_1_0_1 / road_2_0_1      BD(东列南行) -> road_3_0_1 / road_4_0_1
       CA(西列北行) -> road_1_5_3 / road_2_5_3      DB(东列北行) -> road_3_5_3 / road_4_5_3
   每条 2x2 路段的流量均分到它对应的 2 条 4x4 进口。
3. 密度 -> 流量：q(vph) = k(veh/km) * v_free(km/h)，并封顶在 --capacity 处。
4. 时间轴：以 issue_time 到各 target_time_min 为一个区间桶（多 horizon 同目标时刻取均值），
   每个桶为每条进口生成若干条从该进口出发的随机行驶路径，流量平分，interval 由流量反推。

用法
----
python csv_to_cityflow_flow.py                        # 使用全部默认路径
python csv_to_cityflow_flow.py --column ridge_pred --horizon all
python csv_to_cityflow_flow.py --out D:/xx/flow_4x4.json --vfree 50 --capacity 3600
"""

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict

# ---------------- 默认路径（可被命令行覆盖） ----------------
DEFAULT_CSV = r"C:/Users/铎玚/Downloads/demo_forecast.csv"
DEFAULT_ROADNET = (r"C:/Users/铎玚/Desktop/source/hangzhou_4x4/sample-code/data/"
                   r"hangzhou_4x4_gudang_18041610_1h/roadnet_4X4.json")
DEFAULT_OUT = r"D:/Documents/qwen-agent/Efz212TyRZ/项目课/flow_4x4_demo_forecast.json"

# 2x2 有向路段 -> 4x4 进口路段（见文件头注释）
LINK_TO_ENTRIES = {
    "AB": ["road_0_1_0", "road_0_2_0"],   # 北排 东行（西进口）
    "CD": ["road_0_3_0", "road_0_4_0"],   # 南排 东行（西进口）
    "BA": ["road_5_1_2", "road_5_2_2"],   # 北排 西行（东进口）
    "DC": ["road_5_3_2", "road_5_4_2"],   # 南排 西行（东进口）
    "AC": ["road_1_0_1", "road_2_0_1"],   # 西列 南行（北进口）
    "BD": ["road_3_0_1", "road_4_0_1"],   # 东列 南行（北进口）
    "CA": ["road_1_5_3", "road_2_5_3"],   # 西列 北行（南进口）
    "DB": ["road_3_5_3", "road_4_5_3"],   # 东列 北行（南进口）
}

# 车辆参数与示例 flow.json 保持一致
VEHICLE = {
    "length": 5.0, "width": 2.0, "maxPosAcc": 2.0, "maxNegAcc": 4.5,
    "usualPosAcc": 2.0, "usualNegAcc": 4.5, "minGap": 2.5,
    "maxSpeed": 11.111, "headwayTime": 2,
}

ROAD_ID_RE = re.compile(r"^road_(\d+)_(\d+)_(\d+)$")


def load_roadnet(path):
    """解析路网：返回 路段->(起点,终点,方向索引)、路口->出边、虚拟路口集合。"""
    with open(path, encoding="utf-8") as f:
        net = json.load(f)
    virtual = {x["id"] for x in net["intersections"] if x.get("virtual")}
    roads, adj = {}, defaultdict(list)
    for r in net["roads"]:
        m = ROAD_ID_RE.match(r["id"])
        if not m:
            raise ValueError("无法解析路段编号命名: %s" % r["id"])
        roads[r["id"]] = (r["startIntersection"], r["endIntersection"], int(m.group(3)))
        adj[r["startIntersection"]].append(r["id"])
    for rid, (s, e, _) in roads.items():
        if e in virtual and s in virtual:
            raise ValueError("路网异常：两端都是虚拟路口 %s" % rid)
    return roads, adj, virtual


def gen_route(entry, roads, adj, virtual, seed, max_len=60):
    """从进口路段出发随机游走，直到驶出路网（到达虚拟路口），禁止原地掉头。"""
    rng = random.Random(seed)
    route = [entry]
    prev_inter, cur_inter = roads[entry][0], roads[entry][1]
    while cur_inter not in virtual and len(route) < max_len:
        last_dir = roads[route[-1]][2]
        cands = [rid for rid in adj[cur_inter] if roads[rid][1] != prev_inter]
        if not cands:
            break
        weights = [3.0 if roads[rid][2] == last_dir else 1.0 for rid in cands]
        nxt = rng.choices(cands, weights=weights, k=1)[0]
        prev_inter, cur_inter = cur_inter, roads[nxt][1]
        route.append(nxt)
    return route


def read_forecast(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV 无数据行")
    need = {"episode_id", "issue_step", "issue_time_min", "target_time_min",
            "horizon_min", "link_id", "current_density"}
    missing = need - set(rows[0].keys())
    if missing:
        raise ValueError("CSV 缺少必需列: %s" % sorted(missing))
    return rows


def build_timeline(rows, episode, column, horizon_mode, predict_cols):
    """返回 {link: [(time_min, density), ...] 升序} 以及各 link 的 issue_time_min。"""
    eps = sorted({r["episode_id"] for r in rows})
    if episode is None:
        episode = eps[0]
        if len(eps) > 1:
            print("[警告] CSV 含多个 episode，默认取 %s（可用 --episode 指定）" % episode)
    sub = [r for r in rows if r["episode_id"] == episode]
    if not sub:
        raise ValueError("episode %s 在 CSV 中不存在" % episode)
    if column not in predict_cols and column not in ("current_density", "actual_density"):
        raise ValueError("列 %s 不存在，可选: %s" % (column, predict_cols + ["current_density", "actual_density"]))

    issue_steps = sorted({r["issue_step"] for r in sub}, key=int)
    if len(issue_steps) > 1:
        print("[警告] 该 episode 有多个 issue_step，默认取最早的 %s" % issue_steps[0])
    sub = [r for r in sub if r["issue_step"] == issue_steps[0]]

    acc = defaultdict(lambda: defaultdict(list))  # link -> t -> [values]
    issue_t = {}
    for r in sub:
        link = r["link_id"].strip()
        t = float(r["target_time_min"])
        h = int(float(r["horizon_min"]))
        if horizon_mode != "all" and h != horizon_mode:
            continue
        acc[link][t].append(float(r[column]))
        issue_t.setdefault(link, float(r["issue_time_min"]))
        acc[link].setdefault("__cur__", [])
    timeline = {}
    for link, buckets in acc.items():
        pts = [(issue_t[link], float(next(x["current_density"] for x in sub if x["link_id"] == link)))]
        for t, vals in sorted((k, v) for k, v in buckets.items() if k != "__cur__"):
            pts.append((float(t), statistics.fmean(vals)))
        timeline[link] = pts
    return timeline, issue_t


def build_flow_file(timeline, roads, adj, virtual, args):
    records, stats = [], defaultdict(float)
    for link, pts in sorted(timeline.items()):
        entries = LINK_TO_ENTRIES.get(link)
        if entries is None:
            print("[警告] link %s 不在 2x2->4x4 映射表中，跳过" % link)
            continue
        buckets = [(pts[i][0], pts[i + 1][0], pts[i][1]) for i in range(len(pts) - 1)]
        buckets[-1] = (pts[-2][0], pts[-1][0] + args.tail_min, pts[-1][1])
        buckets[0] = (pts[0][0], pts[1][0], pts[0][1])  # issue 到首个预测时刻用当前密度
        for t0, t1, k in buckets:
            q = max(0.0, min(args.capacity, k * args.vfree))  # veh/h
            if q <= 0:
                continue
            st0, st1 = int(round(t0 * 60)), int(round(t1 * 60))
            if st1 <= st0:
                continue
            for entry in entries:
                q_entry = q / len(entries)
                n_routes = max(1, min(args.n_routes, int(round(q_entry * (t1 - t0) / 60.0))))
                for i in range(n_routes):
                    route = gen_route(entry, roads, adj, virtual,
                                      seed="%s|%s|%d|%d" % (args.seed, entry, st0, i))
                    interval = 3600.0 / (q_entry / n_routes)
                    records.append({
                        "vehicle": VEHICLE,
                        "route": route,
                        "interval": round(interval, 3),
                        "startTime": st0,
                        "endTime": st1,
                    })
                    stats[link] += (st1 - st0) / interval
    return records, stats


def validate(records, roads):
    """逐条校验 route 中相邻路段首尾相接、且路段存在于路网。"""
    bad = 0
    for rec in records:
        rt = rec["route"]
        for a, b in zip(rt, rt[1:]):
            if a not in roads or b not in roads or roads[a][1] != roads[b][0]:
                bad += 1
                break
    if bad:
        raise AssertionError("有 %d 条记录路径不连通，请检查路网解析" % bad)


def main():
    ap = argparse.ArgumentParser(description="CTM 密度预测 CSV -> CityFlow 4x4 flow.json")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--roadnet", default=DEFAULT_ROADNET)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--column", default="ridge_pred",
                    help="取值列: persistence_pred / ridge_pred / random_forest_pred / "
                         "hist_gradient_boosting_pred / current_density / actual_density")
    ap.add_argument("--horizon", default="all",
                    help="只取某个预测步长(分钟)，或 all=同一目标时刻多步长取均值")
    ap.add_argument("--episode", default=None)
    ap.add_argument("--vfree", type=float, default=50.0, help="自由流速度 km/h，q=k*v")
    ap.add_argument("--capacity", type=float, default=3600.0, help="单进口上限 veh/h")
    ap.add_argument("--n-routes", type=int, default=4, dest="n_routes",
                    help="每个时间桶内每条进口最多拆分成几条行驶路径")
    ap.add_argument("--tail-min", type=float, default=5.0, dest="tail_min",
                    help="最后一个预测时刻之后再延长多少分钟")
    ap.add_argument("--seed", default=42)
    args = ap.parse_args()
    args.horizon = "all" if args.horizon == "all" else int(args.horizon)

    rows = read_forecast(args.csv)
    predict_cols = [c for c in rows[0].keys() if c and c != "link_id"]
    timeline, issue_t = build_timeline(rows, args.episode, args.column, args.horizon, predict_cols)
    roads, adj, virtual = load_roadnet(args.roadnet)
    for link, entries in LINK_TO_ENTRIES.items():
        for e in entries:
            if e not in roads:
                raise ValueError("路网中缺少进口路段 %s（link %s 映射失败）" % (e, link))

    records, stats = build_flow_file(timeline, roads, adj, virtual, args)
    validate(records, roads)
    records.sort(key=lambda r: (r["startTime"], r["route"][0]))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("[\n")
        f.write(",\n".join(json.dumps(r, ensure_ascii=False) for r in records))
        f.write("\n]\n")

    t_min = min(r["startTime"] for r in records) / 60
    t_max = max(r["endTime"] for r in records) / 60
    print("已写出 %d 条记录 -> %s" % (len(records), args.out))
    print("取值列=%s  horizon=%s  v_free=%.0fkm/h  上限=%.0fvph" %
          (args.column, args.horizon, args.vfree, args.capacity))
    print("流量时间范围: %.1f ~ %.1f 分钟" % (t_min, t_max))
    print("预计生成车辆总数(约): %d" % round(sum(stats.values())))
    for link, n in sorted(stats.items()):
        print("  %-2s -> %-30s ~%.0f 辆" % (link, "/".join(LINK_TO_ENTRIES[link]), n))


if __name__ == "__main__":
    main()
