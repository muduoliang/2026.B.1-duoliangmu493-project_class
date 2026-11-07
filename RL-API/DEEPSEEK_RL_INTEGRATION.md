# 冻结 RL 辅助模型并接入 DeepSeek API —— 一步一步指南

> 论文对应：第四章 4.2 节「融合邻居信息的动态交通信号控制强化学习辅助模型（Our RL）」
> 本指南只做两件事：**把训练好的权重冻结成推理包**，**把该推理包作为工具接给 DeepSeek**。
> 不做 LLM 的 SFT/DPO 微调，不做前端可视化，不修改 `data/` 下任何文件。

---

## 0. 前置检查（1 分钟）

```bash
source /home/mdy493/cityflow/bin/activate
cd /home/mdy493/CityFlow/RL-firture          # 你的工程根目录

ls output/rl_model/dqn_rl_aux_best.pth       # 训练产物必须在
ls data/roadnet_4X4.json data/flow.json      # 数据必须只读存在
python -c "import torch, cityflow; print(torch.__version__)"
```

判断标准：

| 现象 | 含义 |
| --- | --- |
| `dqn_rl_aux_best.pth` 存在且约 8.7 MB | 训练完成（本工程实测 8,758,840 B） |
| 导入 `cityflow` 成功 | 可跑真实仿真；否则只能 `--mock` 验证链路 |
| 导入 `torch` 失败 | 先激活 conda 环境 `cityflow` |

---

## 1. 冻结模型（`freeze_rl.py`）

### 1.1 一条命令

```bash
python freeze_rl.py
```

默认行为：读 `output/rl_model/dqn_rl_aux_best.pth` → 校验维度与 `./data` 一致 →
`eval()` + 冻结参数 → 导出 `output/rl_model/frozen/rl_aux_frozen.pth` + `manifest.json`。

可选项：

```bash
python freeze_rl.py --ckpt output/rl_model/dqn_rl_aux_last.pth   # 换权重
python freeze_rl.py --device cuda                                 # 在 GPU 上导出
python freeze_rl.py --force                                       # 覆盖已有冻结包
python freeze_rl.py --verify-only                                 # 只校验已导出的包
python freeze_rl.py --torchscript                                 # 额外导出 TorchScript（可选）
```

### 1.2 冻结包做了什么（四层保护）

1. **加载即校验**：`weights_only=True` 只反序列化张量，不执行 checkpoint 里的任意对象；
2. **维度一致性**：把权重里的 `state_dim_max / n_rel_pos_classes / n_rel_class_groups /
   n_selectable_phases_max / n_duration_actions / neighbor_degree_max` 与**当前 `./data`
   重新推导的维度**逐项比对，不一致直接抛异常（数据被替换时不会「静默错配」）；
3. **参数哈希**：对全部参数张量做 sha256，写入 manifest；加载时重算比对，防止文件被改动；
4. **前向自检**：用固定随机张量跑一次前向，断言输出有限值、且两次结果完全一致（无 dropout/BN 随机性）。

### 1.3 预期输出（本工程实测）

```
[冻结完成] .../output/rl_model/frozen/rl_aux_frozen.pth
  权重字节   : 2,187,301 B
  参数哈希   : 6218c9ae652e5634...
  网络参数量 : 540,684（h=128，heads=8，N=2）
  数据指纹   : 1de13e51362e08f5...（18 个数据文件）
  前向自检   : 有限值=True 确定性=True 形状={'phase_q': [3, 4], 'duration_q': [3, 7], ...}
```

> 校验失败会直接抛异常并打印差异项——这是**预期行为**，绝不要绕过。

---

## 2. 推理自检（`rl_aux_infer.py`）

### 2.1 不需要仿真器的静态自检

```bash
python rl_aux_infer.py --selftest
```

通过时应看到 `[自检通过] 冻结模型可推理`。

### 2.2 在真实 CityFlow 上跑几步

```bash
python rl_aux_infer.py --demo --steps 3
```

输出每个待决策交叉口的：相位槽（`phase_action_slot`）、真实相位（`light_phase`）、
持续时间动作下标与秒数、相位 Q 值、时长 Q 值、邻居注意力权重、当前队列长度。

### 2.3 没有 CityFlow 时验证链路

```bash
python rl_aux_infer.py --demo --steps 2 --mock
```

`--mock` 使用 `tools/selfcheck.py` 的 MockEngine，**只验证代码链路，指标没有实验意义**。

---

## 3. 配置 DeepSeek（`SDK-test.py`）

### 3.1 安装 SDK

```bash
pip install "openai>=1.0"
```

### 3.2 设置 API Key（不要写进代码/仓库）

```bash
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
# 可选：export DEEPSEEK_MODEL=deepseek-chat
# 可选：export DEEPSEEK_BASE_URL=https://api.deepseek.com
```

想永久生效：把 `export` 行写进 `~/.bashrc`，然后 `source ~/.bashrc`。

### 3.3 连通性测试（先排除网络/Key/额度问题）

```bash
python SDK-test.py
python SDK-test.py --tools      # 额外验证 function calling（本工程 agent 依赖它）
```

预期：`[成功] 对话接口正常`，并打印耗时与 token 用量。

---

## 4. 工具链路演练（不花钱、不联网）

```bash
python deepseek_agent.py --dry-run --mock
```

这一步把「模型要调用的 3 个工具」完整跑一遍：

```
rl_aux_status()            → 仿真时刻 / 待决策交叉口 / 累计决策数 / ATT·AWT·AQL·throughput
rl_aux_advance({"steps":2})→ 冻结模型前向 + 施加动作，返回每交叉口的相位/时长/Q 值
rl_aux_status()            → 动作执行后的新状态与指标
```

看到 `[演练] 工具链路 OK` 说明：**冻结权重、状态解析、动作执行、指标输出**四段都通了。

---

## 5. 真实接入 DeepSeek

```bash
python deepseek_agent.py \
  --question "请推进 5 个决策步，汇总当前信号控制效果，指出最拥堵的两个交叉口，并说明模型的动作偏好。"
```

运行机制（function calling 循环）：

1. DeepSeek 读到 `SYSTEM_PROMPT`（明确「指标只能来自工具、禁止编造」）；
2. 模型决定调用 `rl_aux_status` / `rl_aux_advance` / `rl_aux_reset`；
3. 本进程用**冻结的 RL 模型 + CityFlow** 真实执行，返回严格 JSON（NaN 已转 `null`）；
4. 工具结果以 `role=tool` 回灌，模型继续推理或给出最终中文结论；
5. 会话逐条写入 `output/logs/deepseek_session.jsonl`（含工具调用参数与 token 用量）。

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--model deepseek-chat` | 模型名（默认 `deepseek-chat`） |
| `--max-turns 8` | 工具调用最大轮数 |
| `--temperature 0.2` | 降低随机性，结论更稳定 |
| `--mock` | 无 CityFlow 时用 MockEngine（数值无意义） |
| `--dry-run` | 完全不联网，只演练工具链路 |
| `--quiet` | 精简输出 |

### 5.1 作为库接进你自己的系统

```python
from rl_aux_infer import FrozenRLAuxPolicy

policy = FrozenRLAuxPolicy()            # 默认加载 output/rl_model/frozen/rl_aux_frozen.pth
policy.create_env()                     # 真实 CityFlow；无仿真器时传 engine=MockEngine(...)
policy.reset()
result = policy.advance(steps=1)        # {"decisions": [...], "metrics": {...}, ...}
policy.close()
```

`deepseek_agent.ToolRuntime` 可单独复用：`runtime.handle_json(name, arguments_json)`。

---

## 6. 边界与安全

| 事项 | 要求 |
| --- | --- |
| `data/` | **只读**；冻结包里的数据指纹会随数据变化而变，加载时报不匹配 |
| 权重 | 冻结后 `requires_grad=False`，推理全程 `torch.no_grad()`，不会再更新 |
| API Key | 只从环境变量 `DEEPSEEK_API_KEY` 读取，不写入代码、不进 git |
| 指标 | 只能来自工具；系统提示词已禁止模型编造 ATT/AWT/AQL |
| 训练 | 本链路不含训练代码，训练请用 `train_rl_aux.py` |

---

## 7. 排错表

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: cityflow` | 当前不是 WSL conda 环境 | `source /home/mdy493/cityflow/bin/activate`，确认已编译安装 cityflow |
| `[加载失败] 不是冻结包` | 传了训练 checkpoint 而不是冻结包 | 用 `--frozen output/rl_model/frozen/rl_aux_frozen.pth` |
| `[维度不一致]` | 数据目录换了 / 数据被改 | 检查 `--data-dir`；**不要**手改维度绕过 |
| `冻结包参数哈希不一致` | 文件被改动或复制损坏 | `python freeze_rl.py --force` 重新导出 |
| `[缺少 API Key]` | 未设置环境变量 | `export DEEPSEEK_API_KEY=sk-...` |
| 401 / 402 / 429 | Key 无效 / 余额不足 / 限流 | 检查 Key、充值、加重试或换时段 |
| 工具消息解析失败 | 历史版本输出过 NaN | 本版本已把 NaN 转 `null` 并禁用 `allow_nan` |
| DeepSeek 不调用工具 | 模型不支持 tools | 用 `deepseek-chat`，并跑 `python SDK-test.py --tools` 验证 |
| `PermissionError: [WinError 5]` | Windows 侧无写权限（本工程在 WSL 运行可避免） | 在 WSL 内运行，或提升该目录写权限 |

---

## 8. 验收清单

- [ ] `python freeze_rl.py` 输出 `[冻结完成]`，`frozen/rl_aux_frozen.pth` 约 2.19 MB
- [ ] `python rl_aux_infer.py --selftest` 输出 `[自检通过]`
- [ ] `python rl_aux_infer.py --demo --steps 3` 能打印 16 个交叉口的相位/时长/Q 值
- [ ] `python SDK-test.py --tools` 输出 `[成功] function calling 可用`
- [ ] `python deepseek_agent.py --dry-run` 输出工具链路 OK
- [ ] `python deepseek_agent.py` 得到中文结论，且结论中的指标与工具返回一致
- [ ] `output/logs/deepseek_session.jsonl` 有完整会话记录
