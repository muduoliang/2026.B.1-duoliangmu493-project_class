# -*- coding: utf-8 -*-
"""_dev/probe_score.py —— 验证 rl_scorer 两臂评估 + 打分（不参与产品逻辑）"""
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import data_convert  # noqa: E402
import rl_core       # noqa: E402
import rl_scorer     # noqa: E402

TMP = os.path.join(HERE, "tmp_score")


def main():
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP)
    rep = data_convert.convert([], TMP)
    print("convert ok:", rep.get("ok"), "| flow_stats:", rep.get("flow_stats"))
    schema = rep["schema"]
    flow_path = [os.path.join(TMP, f) for f in os.listdir(TMP) if f.startswith("flow")][0]
    with open(flow_path, encoding="utf-8") as f:
        flow_json = json.load(f)

    plan = {
        "default_cycle": [
            {"phase_slot": 0, "duration": 30},
            {"phase_slot": 1, "duration": 30},
        ],
        "intersections": {
            "intersection_1_1": [
                {"phase_slot": 0, "duration": 25},
                {"phase_slot": 1, "duration": 25},
                {"phase_slot": 3, "duration": 15},
            ],
            "intersection_2_2": [{"phase_slot": 7, "duration": 12}],   # 越界相位，应被夹取
        },
    }

    t0 = time.time()
    report = rl_scorer.evaluate(schema, TMP, flow_json, plan, episode_seconds=600, seed=2024,
                                progress=lambda stage, v: print("  progress:", stage, v))
    print("elapsed: %.2fs" % (time.time() - t0))
    print()
    print("engine_kind:", report["engine_kind"])
    print("score:", json.dumps(report["score"], ensure_ascii=False))
    print("verdict:", report["verdict"])
    print("q_stats:", report["q_stats"])
    print("warnings:", report["warnings"])
    print()
    for arm in ("rl_arm", "plan_arm"):
        a = report[arm]
        print(arm, "metrics:", json.dumps(a["metrics"], ensure_ascii=False))
        print("   n_decisions:", a["n_decisions"], "sim_time:", a["sim_time"], "wall:", a["wall_seconds"])
    print()
    print("worst:", json.dumps(report["worst_intersections"], ensure_ascii=False))
    print("plan_used sample:", json.dumps(
        dict(list(report["plan_used"].items())[:2]), ensure_ascii=False))
    print("n plan intersections:", len(report["plan_used"]))


if __name__ == "__main__":
    main()
