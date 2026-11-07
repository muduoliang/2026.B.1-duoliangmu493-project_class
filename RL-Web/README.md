# RL-Web · 交通信号配时 LLM + RL 本地调试台

把 `../RL-API` 里已有的 Python 业务链路（文件读取 → DeepSeek 调用 → 配时生成 → RL 打分评价）封装成本地网页应用：
**Python 后端代理 + 单页 HTML 前端**，API Key 只存在后端内存，前端永远拿不到原文。

> **原有算法代码零改动**：`../RL-API` 一个文件都没动，RL-Web 只是通过 `sys.path` 引用它。
> 想换模型/换数据，见第 6 节。

---

## 1. 一行启动

```bat
run.bat
```

`run.bat` 会自动挑一个带 torch 的解释器（默认 `E:\project\2025.A.01\anaconde\envs\cityflow\python.exe`，找不到回退 `python`），
然后打开浏览器访问 <http://127.0.0.1:8000>。

手动启动：

```bat
python app.py                 :: 默认 127.0.0.1:8000
python app.py --port 8010     :: 换端口
set RL_WEB_PORT=8010 & run.bat
```

已实测环境：Python 3.8.20 / numpy 1.24.4 / torch 2.4.1（**不需要 openai、不需要 cityflow**）。
前端零依赖（原生 JS，无 CDN），断网可用。

---

## 2. 目录与职责

| 文件                   | 职责                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------- |
| `app.py`             | 后端 HTTP 服务（标准库 `ThreadingHTTPServer`），SSE 流式推送，Key 只存内存                                |
| `pipeline.py`        | 五步链路编排（①解析 ②配时 ③转换 ④打分 ⑤建议）                                                            |
| `deepseek_client.py` | DeepSeek `chat/completions` 代理：流式 SSE 解析 + JSON 抽取（urllib 实现，py3.8 兼容）                 |
| `data_convert.py`    | 第③步：上传的 json/csv → 原有 RL 输入格式（`roadnet_*.json` / `flow_*.json` / `signal_inter_*.txt`） |
| `rl_core.py`         | **桥接层**：加载原有 RL 模块与冻结模型，本身不实现算法                                                        |
| `rl_scorer.py`       | 双臂评估（LLM 方案 vs RL 基线）+ 打分 + 简短评价（应用层口径，见第 7 节）                                         |
| `offline_engine.py`  | **仿真器替代品**：本机没装 cityflow 时顶替真实引擎（不改算法，见第 8 节）                                          |
| `web/index.html`     | 单页前端（配置栏 / 上传 / 五步进度 / 结果区 / 复制 / 追问）                                                  |
| `runs/<run_id>/`     | 每次运行的转换产物与上传原文（可随时整个删掉）                                                                |
| `_dev/`              | 开发探针：`probe_http.py`（HTTP 冒烟）、`pipeline_selftest.py`（MockLLM 全链路）、`probe_score.py` 等   |

---

## 3. 使用步骤

1. **配置（第 1 栏）**：粘贴 DeepSeek API Key → 点「保存」。页面只回显掩码（如 `sk-a***9f2c`），输入框立即清空。
   不填 Key 也能跑：LLM 的两步（①解析、⑤建议）会跳过，**配时方案仍由韦伯斯特法正常算**，用于验证链路是否通。
2. **上传数据（第 2 栏）**：`.json` 与 `.csv` 可单选、可同时多选；不传则用 `RL-API/data` 的内置路网与车流。
3. **演示参数（第 3 栏）**：仿真时长（120/300/600/900 秒，默认 600）、随机种子、是否计入 Q 值分。
4. **点「开始运行」**：右侧五步进度条 + 流式日志，跑完自动出结果。
5. **结果区**：综合得分 / 仿真指标分 / 动作质量分 → 一句话评价 → 指标对照表（指标 / 方案 / RL 基线 / 相对差异）→ **韦伯斯特配时（每路口一块：周期、相位、绿/黄/红、绿信比）** → 排队最高路口 → 精简优化建议。
   右上「复制结果」把整段结果复制成纯文本；下方「追问」可对本次结果继续提问（流式输出）。

---

## 4. 五步链路 ↔ 代码映射

| 步 | 做什么 | 代码位置 | 推送的前端事件 |
| --- | --- | --- | --- |
| ① | 接收上传文件，LLM 读取解析内容 | `pipeline._stage_parse` | `stage(1)` + `log` + `token`（流式要点） |
| ② | **韦伯斯特法**计算配时方案（周期、相位绿/黄/红、绿信比） | `webster.compute`（确定性公式，见第 4.1 节） | `stage(2)` + 每路口一行摘要 |
| ③ | json/csv 自动转成 RL 可直接读的输入格式 | `data_convert.convert` | `stage(3)` + 每条转换明细 |
| ④ | RL 模型读转换后数据 + 韦伯斯特方案，打分与评价 | `rl_scorer.evaluate`（双臂 + Q 值对齐） | `stage(4)` + `progress` |
| ⑤ | 输出精简优化建议 | `pipeline._stage_advice` | `stage(5)` + `token`（流式） |

> 实现细节：第③步的转换在内部**先**执行（因为第②步要按真实车道数与动作空间算流量比、算饱和度），
> 但对前端仍严格按 ①→②→③→④→⑤ 推送进度，界面表现与需求顺序一致。
> 转换结果、上传原文都落在 `runs/<run_id>/`，方便复现。
>
> 第②步默认**不再由 LLM 自由给方案**（需求要求「严格使用韦伯斯特法」）。若想切回旧的 LLM 生成方案，
> 请求里加 `"options": {"plan_source": "llm"}` 即可（此时代码仍会算一份韦伯斯特方案用于对照展示）。

---

## 4.1 第②步：韦伯斯特法配时（本次改造的核心）

实现文件：`webster.py`（应用层新增，**不修改** `RL-API` 里任何算法与模型）。

**输入**：第③步已经落盘的 `flow_uploaded.json`（或内置 `flow.json`）+ 路网车道数。
把每条车流的路线相邻路段 `(road_i → road_{i+1})` 经 roadnet 的 `roadLinks` 映射到「交叉口 + 相位槽 + 进口道路」，
按 `3600 / 流量时间跨度` 折算成小时流量 `q`，取每个相位各进口中流量比最大的那条作为关键车道组：

```
y_j = max_进口( q_进口 / (S0 × n_车道) )        S0 = 1800 veh/h/车道（可改）
Y   = Σ y_j                                    （各相位流量比之和）
L   = n_相位 × (yellow + start_lost)           （yellow=3s、start_lost=3s → 每相位 6s）
C0  = (1.5L + 5) / (1 - Y)                     （韦伯斯特最优周期，Y≥0.95 视为过饱和）
C   = min(max(C0, 60), 120)                    （需求：周期限制 60~120s）
g_j = (C - L) × y_j / Y                        （有效绿灯按流量比分配）
λ_j = g_j / C                                  （绿信比）
红_j = C - g_j - yellow
```

**最短绿灯保护**：`min_green=5s`（不足 5s 的相位抬到 5s，其余相位按比例让出；设 `min_green=0` 即纯公式）。
相位方案本身是交叉口的固定配置，**不会因为某相位暂时没车就被删掉**（该相位只拿 5s 保护绿），
这样输出的相位编号与 `signal_inter_*.txt` 的动作掩码始终一致。

**输出字段**（`result.webster`，前端逐路口渲染）：

| 字段 | 含义 |
| --- | --- |
| `params` | 本次用到的饱和流量、黄灯、启动损失、周期上下界、`min_green`、公式串 |
| `demand` | 车辆数、转向动作数、可映射比例、时间跨度、折算系数 |
| `intersections[]` | 每路口：`cycle_seconds`（采用周期）、`c0_seconds`（原始计算值）、`clamped`（`min`/`max`/`saturated`/null）、`Y`、`L_seconds`、`green_total`、`min_green_used` |
| `intersections[].phases[]` | 每相位：`slot`、`light_phase`、`flows`（中文流向，最多 4 条）、`volume`、`lanes`、`saturation`、`y`、**`green`、`yellow`、`red`、`green_ratio`** |
| `summary` | 全路网周期区间/均值、上下限截断个数、过饱和个数、Y 区间 |
| `plan` | 可直接喂给 RL 的方案 `{"intersections": {iid: [{phase_slot, duration}]}}` |

> 说明：RL 的动作空间只认 `config.DURATION_OPTIONS`（10/15/20/25/30/35/40s），
> 打分仿真时绿灯就近取整到档位（例：18.3s → 20s），**周期以本表为准**，页面底部也标了这句。

---

## 5. ★ 原有 RL 核心代码标注（替换模型时看这里）

### 5.1 未改动、原样引用的原有核心（`../RL-API/`）

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `RL-API/env_cityflow.py` | ★RL 环境：状态空间、动作掩码、相位推进、奖励、指标采样 | **未改动** |
| `RL-API/rl_aux_infer.py` | ★冻结策略装配（加载 checkpoint → 校验 sha256/dims → 建网 → 加载权重） | **未改动** |
| `RL-API/model_net.py` | ★DQN 网络（`forward(own, own_mask, nbr, nbr_mask, rel, group)`） | **未改动** |
| `RL-API/data_schema.py` | 路网/车流/信号相位解析、维度与车道几何长度 | **未改动** |
| `RL-API/config.py` | 仿真与动作参数（`DURATION_OPTIONS`、`SIM_INTERVAL` 等） | **未改动** |
| `RL-API/eval_metric.py` | ATT / AWT / AQL / throughput 统计与 finalize | **未改动** |
| `RL-API/deepseek_agent.py` | 原 DeepSeek 工具调用代理（function calling 版） | **未改动**（网页版由 `deepseek_client.py` 承接，为满足「Key 只在后端 + 流式」） |

### 5.2 RL-Web 新增的部分（都是应用层，不含算法）

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `rl_core.py` | 桥接 | 用 `sys.path` 引入上面 7 个模块；复刻 `rl_aux_infer` 的 6 行装配动作得到 `FrozenModel`，**不修改任何超参/结构** |
| `offline_engine.py` | 仿真器替代 | 实现 env 需要的 11 个引擎方法，替代未安装的 cityflow |
| `rl_scorer.py` | 编排 + 打分口径 | 调用原 `env`/`model` 跑两个臂，把结果换算成 0~100 分与一句话评价 |
| `data_convert.py` | 数据转换 | 把上传的 json/csv 写成原有格式文件，再用 `data_schema` 校验维度 |
| `pipeline.py` / `app.py` / `web/index.html` | 应用层 | 编排、HTTP/SSE、界面 |

### 5.3 怎么换成你的新模型

```bat
:: 1) 换冻结权重（不动原目录里的文件）
set RL_WEB_FROZEN=E:\path\to\new_frozen.pth
:: 2) 或者直接覆盖原路径（默认值）
::    ../RL-API/output/rl_model/frozen/rl_aux_frozen.pth
```

* 冻结包自带 `format` / `dims` / `sha256_state_dict`，维度不符时原代码会明确报
  `[维度校验失败]` 或 `sha256 不一致`，**这是原有保护，不要绕过**。
* 如果新模型的**相位槽数或时长档位变了**，必须同时更新 `../RL-API/data/signal_inter_*.txt`（动作掩码来源）与
  `../RL-API/config.py` 的 `DURATION_OPTIONS`；否则维度校验会失败。
* 如果 RL-API 放在别处：`set RLAPI_HOME=D:\path\to\RL-API`。

---

## 6. 后端接口

| 方法 | 路径 | 说明 | 流式 |
| --- | --- | --- | --- |
| GET | `/` | 单页前端 | – |
| GET | `/api/health` | 环境自检：RL-API 目录、冻结模型、numpy/torch/cityflow、Key 掩码 | – |
| POST | `/api/config` | 保存 Key / 模型（`{api_key, model}`，传空字符串即清除）；响应只含掩码 | – |
| POST | `/api/run` | 五步链路。请求 `{files:[{name,content}], options:{episode_seconds,seed,with_q}}` | ✅ SSE |
| POST | `/api/chat` | DeepSeek 对话代理。请求 `{messages:[...], context:"本轮结果摘要"}` | ✅ SSE |

SSE 事件类型：`stage`（第几步/开始或结束）、`log`、`token`（LLM 流式文本）、`progress`、`result`（完整结果 JSON）、`done`、`error`。
前端用 `fetch + ReadableStream` 解析（`EventSource` 不支持 POST）。

安全边界（本地调试定位）：Key 只在 `app.py` 的 `CONFIG`（进程内存）里；`/api/config` 与 `/api/health` 只返回
`mask_key()` 结果；日志与错误信息经过 `deepseek_client.redact()`；不做鉴权/并发控制/HTTPS。

---

## 7. 打分口径（应用层新增，不改 RL 核心）

```
score_sim   = 100 * mean_over_metrics( rl_metric / (plan_metric + rl_metric) )    # 50 = 与 RL 基线持平
score_q     = 100 * (1 - mean_regret)                                            # regret=(Q_best-Q_plan)/(Q_best-Q_worst)
score_total = 0.6 * score_sim + 0.4 * score_q
```

* **两个臂**：`plan_arm`（施加第②步的配时方案，默认即韦伯斯特方案；`plan_source=llm` 时施加 LLM 方案）与
  `rl_arm`（冻结模型 `argmax` 决策）。同样时长、同样种子、同一流量。
* **指标来源**：原 `env_cityflow.episode_metrics()` → `eval_metric.finalize()`（ATT/AWT/AQL/throughput…）。
* **不计入的情况**（避免 0 分母把分拉偏）：某指标为 0 或缺失、或者短窗口内没有车辆完成行程导致 ATT 无效 → 该指标在表里显示「—」。
* **指标对照表只保留四列**：指标 / 方案 / RL 基线 / 相对差异（`diff_pct`）。按需求已删除「份额」「说明」两项，接口里也不再返回这两个字段。
* **注**：原 `env_cityflow._sample_metrics` 把同一个 `queue_sum` 同时计入「排队」与「等待」，所以 AWT 与 AQL 的差异百分比通常相同
  —— 这是**原有口径**，为满足「不改动 RL 打分评价核心逻辑」故意保留。

---

## 8. 上传数据格式（自动识别，不识别就回退内置并提示）

| 类型 | 例子 | 识别依据 |
| --- | --- | --- |
| 路网 json | `{"intersections":[...],"roads":[...]}`（CityFlow roadnet） | 顶层含 `intersections` + `roads` |
| 车流 json | `[{"vehicle":{...},"route":["road_x","road_y"],"startTime":0,"endTime":10,"interval":1}]` | 顶层是数组且含 `route`/`startTime` |
| 车流 csv | `route,startTime,endTime,vehicle_count`<br>`road_0_1_0>road_1_1_0,0,600,20` | 列名含 route + start/end |
| 路段需求 csv | `road,vehicles,startTime,endTime`<br>`road_0_1_0,40,0,600` | 只有路段没有完整路线 → 自动按直行优先推到边界 |

转换产物（`runs/<run_id>/`）：`roadnet_*.json`、`flow_*.json`、`signal_inter_*.txt`（从原目录复制，动作掩码来源）、`llm_plan.json`（仅 `plan_source=llm` 时）。

* `vehicle_count` 会把 N 辆车在 `[startTime, endTime)` 内均匀发车。
* 路段需求表缺路线时用 `RouteBuilder` 直行优先推路线（长度上限 14）。
* 路由里出现路网中不存在的路段 → 该条丢弃并在结果里给 warning。
* **方案不会写回 `signal_inter_*.txt`**：那是动作掩码/维度来源，改它会改变 `n_selectable_phases_max` 导致冻结模型维度失配。
  韦伯斯特方案在内存里通过 `env.step(phase_slot, duration_idx)` 施加，明细随结果返回（`result.webster`）。

---

## 9. 已知限制（都是本地调试定位下的取舍）

1. **没有安装 cityflow** → 用 `offline_engine.OfflineEngine` 顶替真实仿真器：
   简化跟驰（无变道、车队按车道序、停车线按当前相位放行、下游满则排队），指标口径与 env 完全一致。
   装了 cityflow 后 `make_engine` 会自动改用真引擎（`/api/health` 会显示）。
2. **AWT / AQL 同源**（见第 7 节），两者差异百分比常相同。
3. **短时长 ATT 可能无效**：演示窗口内没有车辆跑完全程时 ATT 不计入（600 秒基本够用）。
4. **Q 值幅度小**：冻结模型相邻动作 Q 差约 0.02，所以用「按 valid 区间归一化的 regret」而不是裸 Q 值比较。
5. **单进程单任务**：一次一个 run，不做队列/并发/鉴权；`runs/` 会累积，随时可删。
6. **LLM 方案可行性由后端兜底**：槽位越界夹取、时长按最近档位落格，异常一律回退内置方案而不是报错中断。

---

## 10. 本次实测记录

| 场景 | 结果 |
| --- | --- |
| `/api/health` | `ok=true`，python 3.8.20 / torch 2.4.1 / numpy 1.24.4 / 冻结模型存在 / 引擎 = offline |
| Key 掩码 | 写入 `sk-test-1234567890abcdef` → 响应只有 `sk-t***cdef`；原文不出后端 |
| `_dev/probe_webster.py`（配时公式三场景） | 全部通过：周期∈[60,120]、`绿+黄+红=周期`、`λ=g/C`、`Y=Σy`、`L=6s×相位`、未截断时 `C0=(1.5L+5)/(1-Y)`；内置数据（Y=0.049~0.161）全口 C0<60 → 取 60s；流量×50 → 过饱和取 120s；流量÷20 → 60s |
| 内置数据跑五步（120s） | 0.86~0.93s；`plan.source=webster`；`sim=54.1 / q=51.5 / total=53.1`；16 口周期全 60s，`intersection_1_1` = 18.3/7.7/5.0/5.0s（λ 0.305/0.128/0.083/0.083） |
| 上传 roadnet json + 需求 csv（120s） | 0.65s；`sources={roadnet:上传, flow:上传(road_demand), signal_plan:内置}`；`plan.source=webster`，`sim=58.0 / q=52.7 / total=55.9` |
| 指标对照字段 | 只剩 `plan / rl / label / diff_pct`；探针断言 `share`、`note` 两个键不存在（已通过） |
| SSE 增量 | 裸 socket 收到 48 个 TCP 块 / 46 个 data 帧，首块 1.8ms → 末块 890ms（边算边推，非攒完再发） |
| 无 Key 追问 | 返回 `{"type":"error","message":"未配置 API Key：请先在配置栏保存 Key"}`，不挂死 |
| 无 Key 跑完整链路 | 跳过 LLM 解析与建议，配时方案仍按韦伯斯特法计算（warning 明示） |
| `_dev/pipeline_selftest.py`（MockLLM 全链路） | 3.15s；`sim=55.1 / q=50.4 / total=53.2` |
| `_dev/probe_llm.py`（本地假 DeepSeek，不需要真 Key） | 非流式 / 流式 / HTTP 401 错误 / JSON 抽取 / 脱敏 全部通过；`[DONE]` 与 usage 帧解析正确 |
| `_dev/probe_llm_pipeline.py`（真实 `DeepSeekClient` + 假服务） | 2.79s；①解析(流式)②方案(JSON 模式)⑤建议(流式) 全走真实 HTTP、`plan.source=llm`、16 口、两组周期 |
| `_dev/probe_score.py`（600s 双臂） | 2.56s；`sim=46.6 / q=50.9 / total=48.3` |

复跑：

```bat
python _dev\probe_webster.py                :: 只验配时公式（不需要 Key、不跑仿真）
python _dev\pipeline_selftest.py            :: 不用 Key，MockLLM 走完整链路
python _dev\probe_llm.py                    :: 假 DeepSeek 服务验证客户端（流式/错误/JSON）
python _dev\probe_llm_pipeline.py           :: 真实客户端 + 全链路（仍是假服务，不烧 token）
python app.py --port 8077                   :: 另开一个窗口
python _dev\probe_http.py 8077              :: HTTP 冒烟（含 SSE 增量与 Key 掩码检查）
```

---

## 11. 常见问题

* **页面打不开**：确认 `app.py` 窗口还在，且端口没被占用（`set RL_WEB_PORT=8010` 换端口）。
* **提示「未配置 API Key」**：第 1 栏保存 Key 后会自动变为 `已保存 sk-***`；刷新页面后需要重新保存（Key 只存后端内存，不落盘）。
* **`/api/health` 显示缺 torch**：`run.bat` 选错了解释器，用 `set RL_WEB_PYTHON=...\python.exe` 指定带 torch 的那个。
* **跑出「LLM 未给出可执行配时」**：模型返回的 JSON 不含可执行 slots，已自动回退内置方案；把 Key 配好重跑即可。
* **分数总是 50 上下**：演示窗口短、流量小时两臂差异本就小；把「演示仿真时长」调到 600~900 秒再试。
