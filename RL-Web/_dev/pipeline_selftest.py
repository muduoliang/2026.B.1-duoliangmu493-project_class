# -*- coding: utf-8 -*-
"""_dev/pipeline_selftest.py —— 用 mock LLM 跑通五步链路（不需要 API Key）"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pipeline  # noqa: E402


class MockLLM(object):
    """模拟 DeepSeek：② 返回 4 相位配时方案，①/⑤ 返回短文。"""

    def __init__(self, plan=None):
        self.plan = plan or {
            "default_cycle": [
                {"phase_slot": 0, "duration": 35},
                {"phase_slot": 1, "duration": 35},
                {"phase_slot": 2, "duration": 15},
                {"phase_slot": 3, "duration": 15},
            ],
            "intersections": {
                "intersection_1_4": [{"phase_slot": 0, "duration": 40},
                                     {"phase_slot": 1, "duration": 25},
                                     {"phase_slot": 2, "duration": 15},
                                     {"phase_slot": 3, "duration": 20}],
            },
            "notes": "干道东西向加长绿时，转向相位压缩",
        }

    def chat(self, messages, temperature=0.2, max_tokens=1200, tools=None, json_mode=False):
        return {"content": json.dumps(self.plan, ensure_ascii=False),
                "tool_calls": [], "finish_reason": "stop", "usage": {"total_tokens": 800}}

    def stream_chat(self, messages, temperature=0.2, max_tokens=1500, tools=None):
        system = messages[0]["content"]
        if "顾问" in system:
            text = ("- intersection_1_4：主路直行绿时 +5s\n"
                    "- intersection_1_1：左转相位并入直行\n"
                    "- 周期统一 100s，保持 15s 转向保护")
        else:
            text = ("数据为 CityFlow 4x4 网格路网，16 个受控交叉口。\n"
                    "车流按路段需求表推算，共 120 辆，6 分钟跨度。\n"
                    "主要进口集中在西侧边界路口。\n"
                    "可直接用于 RL 配时评估。")
        for i in range(0, len(text), 12):
            yield {"type": "delta", "content": text[i:i + 12]}
        yield {"type": "done", "content": text, "usage": {"total_tokens": 200},
               "finish_reason": "stop"}


def main():
    events = []

    def emit(event):
        events.append(event)
        t = event.get("type")
        if t == "stage":
            print("  [stage {}{}] {}".format(event["stage"], "/" + event["status"], event["message"]))
        elif t == "log":
            print("    · ({}): {}".format(event["stage"], event["message"]))
        elif t == "progress":
            print("    ~ ({}): {}={}".format(event["stage"], event["name"], event["value"]))
        elif t == "result":
            print("  [result] 收到最终结果")

    import glob
    builtin = os.path.join(os.path.dirname(HERE), os.pardir, "RL-API", "data")
    rn = sorted(glob.glob(os.path.join(builtin, "roadnet*.json")))[0]
    with open(rn, encoding="utf-8") as f:
        roadnet_text = f.read()
    import re
    roads = sorted(set(re.findall(r'"id"\s*:\s*"(road_[0-9_]+)"', roadnet_text)))[:3]
    demand_csv = "road,vehicles,startTime,endTime\n" + "\n".join(
        "{},{},0,600".format(r, 40) for r in roads)

    files = [{"name": "roadnet_4x4.json", "content": roadnet_text},
             {"name": "demand.csv", "content": demand_csv}]

    t0 = time.time()
    report = pipeline.run_pipeline(files, api_key="", llm=MockLLM(), emit=emit,
                                   options={"episode_seconds": 300})
    print()
    print("elapsed: %.2fs" % (time.time() - t0))
    print("run_id:", report["run_id"], "| engine:", report["engine_kind"])
    print("plan source:", report["plan"]["source"], "| 口数:", report["plan"]["intersections"],
          "| notes:", report["plan"]["notes"])
    print("score:", json.dumps(report["score"], ensure_ascii=False))
    print("verdict:", report["verdict"])
    print("advice:", report["advice"].replace("\n", " / "))
    print("q_stats:", report["q_stats"])
    print("warnings:", json.dumps(report["warnings"], ensure_ascii=False))
    print("arms rl:", json.dumps(report["arms"]["rl"]["metrics"], ensure_ascii=False)[:220])
    print("arms plan:", json.dumps(report["arms"]["plan"]["metrics"], ensure_ascii=False)[:220])
    print("event counts:", {k: sum(1 for e in events if e.get("type") == k)
                            for k in ("stage", "log", "token", "progress", "result")})

    # 无 LLM 模式（纯本地对照）
    print()
    print("=== 无 LLM 模式 ===")
    report2 = pipeline.run_pipeline(files, api_key="", llm=None, emit=emit,
                                    options={"episode_seconds": 200})
    print("plan source:", report2["plan"]["source"], "| score:", report2["score"]["total"])
    print("advice:", report2["advice"].replace("\n", " / "))


if __name__ == "__main__":
    main()
