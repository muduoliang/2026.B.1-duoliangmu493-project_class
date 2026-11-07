# -*- coding: utf-8 -*-
"""
probe_llm_pipeline.py —— 开发探针：用本地假 DeepSeek 服务跑「真实 LLM 路径」的全链路。
用法：python _dev/probe_llm_pipeline.py

与 pipeline_selftest.py 的区别：那里注入的是 MockLLM 对象，这里走真正的
DeepSeekClient（urllib + SSE 解析），只把 base_url 指向本地假服务，因此
①解析（流式）②方案（JSON 模式）⑤建议（流式）三段都真实经过 HTTP 与解析逻辑。
"""
from __future__ import print_function

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))

import pipeline  # noqa: E402

# 假服务返回的方案：与 pipeline 提示词要求的 JSON 结构一致
PLAN_JSON = {
    "default_cycle": [{"phase_slot": 0, "duration": 35},
                      {"phase_slot": 1, "duration": 35}],
    "intersections": {
        "intersection_1_4": [{"phase_slot": 0, "duration": 45},
                             {"phase_slot": 1, "duration": 25},
                             {"phase_slot": 2, "duration": 15}],
    },
    "notes": "干道东西向加长绿时，1_4 口加左转保护",
}
PARSE_TEXT = "路网 16 个控制口；流量集中在东西向主路；建议干道优先。"
ADVICE_TEXT = ("- intersection_1_4：东西直行绿时 +5s\n"
               "- intersection_4_4：左转并入直行\n"
               "- 周期统一 100s，保持 15s 转向保护")

CALLS = {"stream": 0, "json": 0}


class FakeDeepSeek(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        if body.get("stream"):
            CALLS["stream"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            pieces = [ADVICE_TEXT[i:i + 12] for i in range(0, len(ADVICE_TEXT), 12)]
            for piece in pieces:
                chunk = {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return

        CALLS["json"] += 1
        assert body.get("response_format") == {"type": "json_object"}, body.get("response_format")
        payload = json.dumps({
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": json.dumps(PLAN_JSON, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160},
        }, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeDeepSeek)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:{}".format(port)
    print("假 DeepSeek: {}".format(base))

    # 用内置路网 + 需求表 csv（不依赖真 Key）
    import glob
    import re
    builtin = os.path.join(os.path.dirname(HERE), os.pardir, "RL-API", "data")
    rn = sorted(glob.glob(os.path.join(builtin, "roadnet*.json")))[0]
    with open(rn, encoding="utf-8") as f:
        roadnet_text = f.read()
    roads = sorted(set(re.findall(r'"id"\s*:\s*"(road_[0-9_]+)"', roadnet_text)))[:3]
    demand = "road,vehicles,startTime,endTime\n" + "\n".join(
        "{},{},0,300".format(r, 30) for r in roads)
    files = [{"name": "roadnet_4x4.json", "content": roadnet_text},
             {"name": "demand.csv", "content": demand}]

    events = []

    def emit(ev):
        events.append(ev)

    t0 = time.time()
    report = pipeline.run_pipeline(files, api_key="sk-fake-key-0001", base_url=base,
                                   model="deepseek-chat", emit=emit,
                                   options={"episode_seconds": 120, "seed": 5, "with_q": True})
    elapsed = time.time() - t0
    kinds = {}
    for e in events:
        kinds[e.get("type")] = kinds.get(e.get("type"), 0) + 1
    tokens = [e for e in events if e.get("type") == "token"]
    stages = [(e.get("stage"), e.get("status")) for e in events if e.get("type") == "stage"]

    print("\n耗时 {:.2f}s | 事件 {} | stage 序列 {}".format(elapsed, kinds, stages))
    print("HTTP 调用：JSON 模式 {} 次 / 流式 {} 次".format(CALLS["json"], CALLS["stream"]))
    print("parse_summary : {}".format(report["parse_summary"][:80].replace("\n", " ")))
    print("plan.source   : {}".format(report["plan"]["source"]))
    print("plan.notes    : {}".format(report["plan"]["notes"]))
    print("plan 路口数    : {}".format(report["plan"]["intersections"]))
    print("score         : {}".format({k: report["score"].get(k) for k in ("sim", "q", "total")}))
    print("q_stats       : {}".format(report["q_stats"]))
    print("advice        : {}".format(report["advice"].replace("\n", " / ")))
    print("warnings      : {}".format(report["warnings"]))
    print("llm.elapsed   : {}".format(report["llm"].get("elapsed")))

    # ---------- 断言 ----------
    assert report["plan"]["source"] == "llm", report["plan"]["source"]
    assert report["plan"]["intersections"] == 16, report["plan"]["intersections"]
    assert report["plan"]["notes"] == PLAN_JSON["notes"], report["plan"]["notes"]
    used = report["plan"]["used"]
    # 45s 会按 config.DURATION_OPTIONS 就近吸附到 40s 档（原有档位约束，不是缺陷）
    assert used["intersection_1_4"]["cycle_seconds"] == 40 + 25 + 15, used["intersection_1_4"]
    assert [p["duration"] for p in used["intersection_1_4"]["phases"]] == [40, 25, 15]
    assert used["intersection_1_1"]["cycle_seconds"] == 70, used["intersection_1_1"]  # default 35+35
    groups = set(json.dumps(v["phases"]) for v in used.values())
    assert len(groups) == 2, "应恰好两组（统一周期 + 1_4 覆盖），实际 {}".format(len(groups))
    bullets = [x for x in report["advice"].split("\n") if x.strip()]
    assert 1 <= len(bullets) <= 3, bullets
    assert CALLS["json"] >= 1 and CALLS["stream"] >= 2, CALLS
    assert kinds.get("token", 0) >= 2, kinds           # ① 解析与 ⑤ 建议都应有流式 token
    assert not any("未配置 API Key" in w for w in report["warnings"]), report["warnings"]
    assert any(st == (5, "done") for st in stages), stages
    print("\n全部通过：①解析(流式) → ②方案(JSON) → ③转换 → ④打分 → ⑤建议(流式) 均走真实 HTTP 客户端")
    httpd.shutdown()


if __name__ == "__main__":
    main()
