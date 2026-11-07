# -*- coding: utf-8 -*-
"""
probe_http.py —— 开发探针：对已启动的 app.py 做端到端 HTTP 验证（只读，不改 RL-API）。
用法：先启动 app.py，再 python _dev/probe_http.py [port]

覆盖
    1. GET / 与 /api/health
    2. POST /api/config  —— Key 必须只回掩码，绝不能回原文
    3. POST /api/run     —— 无文件（内置数据）
    4. POST /api/run     —— 上传路网 json + 路段需求 csv（验证第③步转换）
    5. 裸 socket 直连     —— 验证 SSE 是增量推送（不是攒完再一次性返回）
    6. POST /api/chat    —— 无 Key 时必须给可读错误而不是挂死
"""
from __future__ import print_function

import json
import os
import socket
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
BASE = "http://127.0.0.1:{}".format(PORT)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, os.pardir, "RL-API", "data")


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post_json(path, body, timeout=60):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def post_sse(path, body, timeout=180):
    """POST 并读完整流，返回 (status, [事件字典...], 事件行数, content-type)。"""
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
        ctype, status = r.headers.get("Content-Type"), r.status
    events, lines = [], 0
    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        lines += 1
        try:
            events.append(json.loads(line[5:].strip()))
        except ValueError:
            events.append({"type": "_bad_json"})
    return status, events, lines, ctype


def socket_stream_check(path="/api/run",
                        body=None, label="SSE 增量检查", show=6):
    """裸 socket：逐块读、记录每块到达时刻，证明服务端是边算边推。"""
    body = json.dumps(body or {"files": [], "options": {"episode_seconds": 120}}).encode("utf-8")
    head = ("POST {} HTTP/1.0\r\nHost: 127.0.0.1:{}\r\nContent-Type: application/json\r\n"
            "Content-Length: {}\r\n\r\n").format(path, PORT, len(body)).encode("ascii")
    s = socket.create_connection(("127.0.0.1", PORT), timeout=180)
    s.sendall(head + body)
    t0 = time.time()
    stamps, buf = [], ""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk.decode("utf-8", "replace")
        stamps.append((round((time.time() - t0) * 1000, 1), len(chunk)))
    s.close()
    data_frames = [ln for ln in buf.split("\n") if ln.strip().startswith("data:")]
    print("{}: 收到 {} 个 TCP 块 / {} 个 data 帧".format(label, len(stamps), len(data_frames)))
    print("  前 {} 块到达时刻(ms)：{}".format(
        show, [x[0] for x in stamps[:show]]))
    spread = (stamps[-1][0] - stamps[0][0]) if len(stamps) > 1 else 0.0
    print("  首块 {}ms → 末块 {}ms，跨度 {}ms（跨度>50ms 说明是边算边推）".format(
        stamps[0][0] if stamps else -1, stamps[-1][0] if stamps else -1, spread))
    return len(stamps), len(data_frames), spread


def summarize_result(result):
    sc = result.get("score") or {}
    print("  run_id={} engine={} elapsed={}s".format(
        result.get("run_id"), result.get("engine_kind"), result.get("elapsed_seconds")))
    print("  score sim={} q={} total={}".format(sc.get("sim"), sc.get("q"), sc.get("total")))
    print("  plan.source={} 路口={} 方案组数={}".format(
        result["plan"].get("source"), result["plan"].get("intersections"),
        len(set(json.dumps(v.get("phases")) for v in (result["plan"].get("used") or {}).values()))))
    print("  verdict={}".format(result.get("verdict")))
    print("  advice={}".format((result.get("advice") or "").replace("\n", " / ")[:150]))
    print("  worst={}".format([(w["intersection_id"], w.get("aql_plan"), w.get("aql_rl"))
                               for w in (result.get("worst_intersections") or [])]))
    print("  sources={} dims={}".format((result.get("conversion") or {}).get("sources"),
                                        (result.get("conversion") or {}).get("dims")))
    print("  run_dir 存在={}".format(os.path.isdir(result.get("run_dir") or "")))
    detail = sc.get("detail") or {}
    banned = [k for k in ("share", "note") for v in detail.values() if k in (v or {})]
    assert not banned, "指标对照仍含已删除字段: {}".format(banned)
    print("  detail={}".format({k: {kk: vv for kk, vv in (v or {}).items()
                                   if kk in ("plan", "rl", "label", "diff_pct")}
                               for k, v in detail.items()}))
    print("  detail 无 share/note 字段 : OK")

    # ---------- 韦伯斯特配时明细 ----------
    wb = result.get("webster") or {}
    rows = wb.get("intersections") or []
    assert rows, "结果缺少 webster.intersections"
    assert result["plan"].get("source") == "webster", result["plan"].get("source")
    bad = [r["intersection_id"] for r in rows
           if any(abs(ph["green"] + ph["yellow"] + ph["red"] - r["cycle_seconds"]) > 0.15
                  for ph in r["phases"])]
    assert not bad, "绿+黄+红 != 周期: {}".format(bad)
    assert all(60.0 - 1e-6 <= r["cycle_seconds"] <= 120.0 + 1e-6 for r in rows), "周期越界"
    first = rows[0]
    print("  webster.summary={}".format(wb.get("summary")))
    print("  webster {}｜周期 {}s（C0={}，{}）｜Y={}｜{}".format(
        first["intersection_id"], first["cycle_seconds"], first.get("c0_seconds"),
        first.get("clamped") or "未截断", first["Y"],
        " ".join("相位{} G{}/Y{}/R{} λ{}".format(p.get("light_phase"), p["green"], p["yellow"],
                                                 p["red"], p["green_ratio"])
                 for p in first["phases"])))
    print("  16 口周期分布={}".format(
        sorted(set(r["cycle_seconds"] for r in rows))))
    print("  warnings={}".format(result.get("warnings")))


def main():
    print("=" * 74)
    status, ctype, body = get("/")
    print("GET / -> {} {} {} bytes（前端单页）".format(status, ctype, len(body)))

    status, _, body = get("/api/health")
    health = json.loads(body.decode("utf-8"))
    rl = health.get("rl") or {}
    print("GET /api/health -> ok={} engine={} missing={}".format(
        health.get("ok"), rl.get("engine_hint"), rl.get("missing")))
    print("  python={} torch={} numpy={} frozen_exists={}".format(
        rl.get("python"), rl.get("torch"), rl.get("numpy"), rl.get("frozen_exists")))

    # ---------- Key 掩码 ----------
    fake = "sk-test-1234567890abcdef"
    status, cfg = post_json("/api/config", {"api_key": fake, "model": "deepseek-chat"})
    masked = (cfg.get("config") or {}).get("masked")
    print("POST /api/config -> masked={} has_key={}".format(
        masked, (cfg.get("config") or {}).get("has_key")))
    assert masked and fake not in json.dumps(cfg), "!! Key 泄露或掩码失效"
    post_json("/api/config", {"api_key": ""})
    print("  已清除测试 Key（前端始终拿不到原文）")

    # ---------- 无文件：内置数据 ----------
    print("-" * 74)
    status, events, lines, ctype = post_sse(
        "/api/run", {"files": [], "options": {"episode_seconds": 120, "seed": 2024, "with_q": True}})
    kinds = {}
    for e in events:
        kinds[e.get("type")] = kinds.get(e.get("type"), 0) + 1
    print("POST /api/run(内置数据) -> {} / {} 帧 / {} / {}".format(status, lines, ctype, kinds))
    print("  stage 序列={}".format([(e.get("stage"), e.get("status"))
                                  for e in events if e.get("type") == "stage"]))
    result = None
    for e in events:
        if e.get("type") == "result":
            result = e.get("data")
        elif e.get("type") == "error":
            print("  error: {}".format(e.get("message")))
    if result:
        summarize_result(result)
    else:
        print("  !! 未收到 result 事件")

    # ---------- 上传 json + csv ----------
    print("-" * 74)
    rn_path = None
    for name in ("roadnet_4X4.json", "roadnet.json"):
        cand = os.path.join(DATA, name)
        if os.path.isfile(cand):
            rn_path = cand
            break
    if rn_path:
        with open(rn_path, "rb") as f:
            roadnet_text = f.read().decode("utf-8")
        import re
        roads = sorted(set(re.findall(r'"id"\s*:\s*"(road_[0-9_]+)"', roadnet_text)))[:3]
        demand = "road,vehicles,startTime,endTime\n" + "\n".join(
            "{},{},0,300".format(r, 30) for r in roads)
        files = [{"name": "my_roadnet.json", "content": roadnet_text},
                 {"name": "my_demand.csv", "content": demand}]
        status, events, lines, ctype = post_sse(
            "/api/run", {"files": files, "options": {"episode_seconds": 120, "seed": 7}})
        result2 = None
        for e in events:
            if e.get("type") == "result":
                result2 = e.get("data")
            elif e.get("type") == "error":
                print("  error: {}".format(e.get("message")))
        print("POST /api/run(上传 json+csv) -> {} / {} 帧".format(status, lines))
        if result2:
            summarize_result(result2)
        else:
            print("  !! 未收到 result 事件")
    else:
        print("跳过上传用例：未找到 RL-API/data/roadnet_*.json")

    # ---------- 流式增量验证 ----------
    print("-" * 74)
    socket_stream_check(body={"files": [], "options": {"episode_seconds": 120}})

    # ---------- 无 Key 追问 ----------
    print("-" * 74)
    status, events, lines, _ = post_sse("/api/chat", {"messages": [{"role": "user", "content": "测试"}]})
    print("POST /api/chat(无 Key) -> {}".format(
        [{k: (v[:70] if isinstance(v, str) else v) for k, v in e.items()} for e in events]))
    print("=" * 74)


if __name__ == "__main__":
    main()
