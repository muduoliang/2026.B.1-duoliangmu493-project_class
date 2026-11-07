# -*- coding: utf-8 -*-
"""开发期探针：验证离线引擎能让上传的 flow 真正影响指标（不再是恒 0 排队）。"""
import os
import sys
import time

WEB = r"E:\project-study\project1-LLM_traffic\RL-Web"
sys.path.insert(0, WEB)

import rl_core  # noqa: E402
import offline_engine  # noqa: E402


def run(kind, seconds=600.0):
    schema = rl_core.build_schema(rl_core.data_dir())
    engine = offline_engine.OfflineEngine(schema)
    env = rl_core.make_env(schema, engine=engine, episode_seconds=seconds, seed=20240501)
    env.reset()
    frozen = rl_core.load_frozen(schema)
    steps = 0
    t0 = time.time()
    while not env.is_done():
        batch = env.decision_batch()
        out = frozen.forward(batch)
        phase_q = out["phase_q"]
        dur_q = out["duration_q"]
        pa, da = {}, {}
        for row, iid in enumerate(batch["inter_ids"]):
            avail = int(batch["phase_avail"][row])
            if kind == "rl":
                valid = [phase_q[row, k] for k in range(avail)]
                slot = max(range(avail), key=lambda k: valid[k])
                d = int(dur_q[row].argmax())
            else:                       # 固定「全部用 0 号相位 + 40s」的恶劣方案
                slot = 0
                d = 6
            pa[iid] = slot
            da[iid] = d
        env.step(pa, da)
        steps += 1
        if steps > 400:
            break
    m = env.episode_metrics()
    print("[%s] steps=%d %.2fs sim=%.0fs ATT=%.2f AWT=%.2f AQL=%.3f throughput=%.1f n_total=%s"
          % (kind, steps, time.time() - t0, env.sim_time, m["ATT"], m["AWT"], m["AQL"],
             m["throughput"], m["n_total_vehicles"]))
    worst = sorted(m["per_intersection_AQL"].items(), key=lambda kv: -kv[1])[:3]
    print("    worst AQL:", [(k, round(v, 2)) for k, v in worst],
          "engine veh=%d finished=%d" % (len(engine.vehicles), len(engine.finished)))
    return m


if __name__ == "__main__":
    print("numpy from:", rl_core.core("numpy").__file__)
    run("rl", seconds=600.0)
    run("fixed_bad", seconds=600.0)
