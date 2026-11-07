# -*- coding: utf-8 -*-
"""开发期探针：验证 RL-API 核心模块在本地解释器下可加载、可推进 episode。"""
import json
import os
import sys
import time

RLAPI = r"E:\project-study\project1-LLM_traffic\RL-API"
sys.path.insert(0, RLAPI)

from rl_aux_infer import FrozenRLAuxPolicy, _make_mock_engine  # noqa: E402

t0 = time.time()
pol = FrozenRLAuxPolicy(verbose=True)
print("[probe] load frozen: %.2fs" % (time.time() - t0))

eng = _make_mock_engine(pol.schema)
env = pol.create_env(engine=eng, episode_seconds=300.0, verbose=False)
pol.reset()
print("[probe] episode_seconds=%.1f n_inter=%d" % (env.episode_seconds, env.n))

t1 = time.time()
for k in range(6):
    r = pol.step()
    print("[probe] step=%d t=%.1f n_dec=%d done=%s" % (k, r["sim_time"], r["n_decisions"], r["done"]))
print("[probe] 6 steps cost %.2fs" % (time.time() - t1))
print("[probe] metrics =", json.dumps(pol.metrics(), ensure_ascii=False)[:400])

# --- 关键验证：给「指定动作」打分（LLM 方案打分需要 Q 值，而不是 argmax） ---
import torch  # noqa: E402

env.reset()
batch = env.decision_batch()
t = pol._batch_to_torch(batch)
with torch.no_grad():
    out = pol.model(own=t["own"], own_mask=t["own_mask"], nbr=t["nbr"],
                    nbr_mask=t["nbr_mask"], rel=t["rel"], group=t["group"])
print("[probe] out keys:", list(out.keys()))
print("[probe] phase_q shape=%s duration_q shape=%s" % (tuple(out["phase_q"].shape), tuple(out["duration_q"].shape)))
rows = [d["intersection_id"] for d in pol._last_decisions] or batch["inter_ids"]
print("[probe] inter_ids:", batch["inter_ids"][:4], "...")
# 指定动作为「0 号相位槽 / 2 号时长档」，取该动作的 Q 值用于打分
q_sel = out["phase_q"][:, 0]
q_max = out["phase_q"].masked_fill(
    torch.arange(out["phase_q"].size(1)).unsqueeze(0) >= t["phase_avail"].unsqueeze(1),
    float("-inf")).max(dim=1).values
print("[probe] q(phase0)=%.3f q(best)=%.3f gap_mean=%.3f"
      % (float(q_sel.mean()), float(q_max.mean()), float((q_max - q_sel).mean())))
print("[probe] OK")
