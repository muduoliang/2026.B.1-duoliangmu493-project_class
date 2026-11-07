# 融合邻居信息的动态交通信号控制强化学习辅助模型（Our RL）

> 硕士论文《城市交通信号控制大语言模型方法研究与系统实现》
> **第四章 4.2 节**：融合邻居信息的动态交通信号控制强化学习算法 —— 完整工程复现

本工程只复现 **4.2 节的 RL 辅助模型**：CityFlow 仿真环境 + DQN 双输出动作网络 +
邻居关系特征融合模块（NR）+ 邻居作用强度评估模块（NS）。
**不含** LLM 两阶段微调（SFT/DPO）、**不含**智能体系统、**不含**前端可视化。

---

## 0. 目录结构与文件功能

```
RL-2/
├── data/                        # 【只读】原始数据集，全程禁止修改/生成/覆写
│   ├── roadnet_4X4.json         # 路网：32 交叉口（16 受控 + 16 虚拟边界）、80 条路 × 3 车道
│   ├── flow.json                # 交通需求：2983 条（CityFlow 数组格式），0~3599 s，interval=1.0
│   └── signal_inter_intersection_{i}_{j}.txt   # 16 份定时配时方案（time,phaseIndex 逐行）
├── config.py                    # 全部超参数集中管理；论文数值一一对应；【不含任何数据维度】
├── data_schema.py               # 数据读取 + 合法性校验 + 维度自动推导（唯一维度来源）
├── env_cityflow.py              # CityFlow 环境封装：状态采集 / 动作执行 / 奖励计算
├── model_net.py                 # 观测编码器 + NR 模块 + NS 模块 + 相位/持续时间双输出网络
├── replay_buffer.py             # 经验回放池（定长环形，numpy 预分配）
├── eval_metric.py               # ATT / AWT / AQL 指标计算与基线对比（含逐秒序列导出）
├── plot_metrics.py              # 绘图：训练过程 4 曲线 + 测试评估 4 类图（只读运行产物）
├── train_rl_aux.py              # 主训练脚本（数据校验 → 建环境 → 建网 → 基线 → 训练 → 落盘 → 出图）
├── requirements.txt             # 依赖清单（Ubuntu/WSL + conda 环境 cityflow）
├── readme.md                    # 本文件
├── tools/
│   ├── inspect_data.py          # 只读数据体检：打印全部字段、类型、样例与推导维度
│   ├── selfcheck.py             # 离线自检 24 项（无 cityflow 也能跑，含 MockEngine 全链路）
│   ├── smoke_train.py           # 离线冒烟训练：用 MockEngine 跑通 train_rl_aux.py 全流程
│   └── plot_offline.py          # 离线重绘全部图表（换格式/换分辨率，不重新训练）
├── cityflow-github/CityFlow/    # CityFlow 源码（用于 WSL 下源码编译安装）
└── output/                      # 【全部产物】运行后自动创建
    ├── rl_model/                # 模型权重：dqn_rl_aux_last.pth / _best.pth / _ep{N}.pth、replay_buffer.npz
    ├── logs/                    # train_log.csv（逐 episode 指标）、eval_log.csv（贪心评测）
    │   └── series/              # 基线/各次评测 episode 的逐秒真实序列（*_series.npz + .csv）
    ├── figures/                 # 训练曲线 + 测试评估图（含 figures_index.json 图表索引）
    ├── reports/                 # data_fields_report.json（字段与维度报告）、run_report.json（运行报告）
    └── generated/               # 运行时生成的 CityFlow config.json（不写回 data/）
```

| 文件 | 功能说明 | 对应论文 |
| --- | --- | --- |
| `config.py` | 超参数集中管理：γ=0.8、Adam lr=1e-3、回放容量 12000、batch=3000、η=0.2、注意力头 8、h=128、N=2、持续时间档 {10..40}、100 episode × 100 epoch。**有意不写任何输入维度**，维度一律运行时从 `./data` 解析 | 4.2 节 |
| `data_schema.py` | 解析 roadnet / flow / signal 配时；校验论文 4.2 节所需字段是否存在，缺失即抛 `ValueError` 并打印字段名；自动推导状态维度、相位动作数、邻居数、相对位置类别、持续时间动作数 | 4.2 节 状态/动作空间 |
| `env_cityflow.py` | 封装 CityFlow：逐秒状态采集（五元组 + 相位 one-hot）、`set_tl_phase` 执行相位动作、保持时长执行持续时间动作、奖励 `R_target=R_self+η·ΣR_nbr/k`、动作掩码、经验元组打包 | 4.2 节 环境与奖励 |
| `model_net.py` | 观测编码器；**邻居关系特征融合模块**：`Linear(r(x,y))` → 交叉注意力 → N 层自注意力；**邻居作用强度评估模块**：交叉注意力 → MLP 评分 → 分组 softmax → 加权聚合；相位选择网络 + 持续时间选择网络 | 4.2 节 NR/NS 模块、DQN 框架 |
| `replay_buffer.py` | 容量 12000 的环形经验回放池，`(S,a_p,a_d,R,S')` 及邻居张量整体存取，支持 npz 落盘/续训 | 4.2 节 经验回放 |
| `eval_metric.py` | ATT（平均旅行时间）、AWT（平均等待时间）、AQL（平均排队长度）、throughput 与相对提升计算；导出逐秒真实序列（排队长度 / 等待车辆数 / 在途车辆 / 每车平均延误 / 各交叉口排队采样）供绘图 | 4.2 节 评价指标 |
| `plot_metrics.py` | 绘图模块：①训练过程 4 条曲线（Episode Reward / 平均延误 / 平均排队长度 / 吞吐量）；②测试评估 4 类图（时序折线、箱线图、柱状图、CDF）。**只读 `output/` 下的真实运行产物**，产物缺失直接报错，不生成占位数据 | 4.2 节 实验分析 |
| `train_rl_aux.py` | 主训练脚本：数据校验 → 建环境 → 建网/回放池 → FixedTime 基线 → 训练（ε-greedy 双动作 + 每 episode 100 轮 epoch 梯度更新）→ 落盘权重/日志/逐秒序列/报告 → 出图 | 4.2 节 训练设置 |

---

## 1. 环境准备（Ubuntu / WSL2）

```bash
# 1) 激活预先创建好的 conda 环境
source /home/mdy493/cityflow/bin/activate

# 2) 安装 Python 依赖（numpy + torch）
cd /path/to/RL-2
pip install -r requirements.txt

# 3) 安装 CityFlow 仿真器（必须源码编译，PyPI 上没有可直接 pip 安装的发行版）
sudo apt update && sudo apt install -y build-essential cmake
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow
git submodule update --init --recursive     # 必需：拉取 extern/pybind11、extern/rapidjson
pip install .                               # 在 cityflow 这个 conda 环境内执行
python -c "import cityflow; print(cityflow.__version__)"
```

> 本工程目录内已附带一份 CityFlow 源码（`cityflow-github/CityFlow`），可直接
> `cd cityflow-github/CityFlow && git submodule update --init --recursive && pip install .`。
> 官方安装说明见 <https://cityflow.readthedocs.io/en/latest/install.html>（Windows 宿主建议使用 WSL 或 Docker 镜像 `cityflowproject/cityflow:latest`）。
> GPU 环境按需安装 CUDA 版 torch；纯 CPU 也能跑，`train_rl_aux.py` 会自动选择设备。

---

## 2. 数据集放置说明

`./data/` 必须包含（文件名支持通配匹配，见 `config.py:ROADNET_PATTERNS / FLOW_PATTERNS / SIGNAL_PLAN_PATTERNS`）：

| 文件 | 必需性 | 说明 |
| --- | --- | --- |
| `roadnet_*.json` | 必需 | 路网；提供交叉口、路段、车道、`trafficLight.lightphases`、`roadLinks` |
| `flow_*.json` / `flow.json` | 必需 | 交通需求；CityFlow 数组格式 `[{vehicle,route,interval,startTime,endTime}, ...]` |
| `signal_inter_<intersection_id>.txt` | 必需 | 定时配时方案（`time,phaseIndex` 逐行），用于动作掩码与 FixedTime 基线 |

> 🔴 **强制约束：禁止修改 `./data` 下任何原始数据集文件，本工程对 data 目录只做只读读取，
> 不写入、不生成、不覆写任何文件。** 运行期产生的一切文件（含 CityFlow 运行配置）都写到
> `output/generated/`、`output/rl_model/`、`output/logs/`、`output/reports/`。

---

## 3. 训练启动

```bash
source /home/mdy493/cityflow/bin/activate
cd /path/to/RL-2

# 正式训练（论文 4.2 节超参：100 episode × 100 epoch）
python train_rl_aux.py

# 常用可选参数
python train_rl_aux.py --device cuda --thread-num 4        # 指定设备与仿真线程（默认自动选 cuda）
python train_rl_aux.py --episodes 10 --eval-every 2        # 快速试跑（--eval-every 默认 5）
python train_rl_aux.py --resume output/rl_model/dqn_rl_aux_last.pth   # 续训到 --episodes（默认 100）总数
python train_rl_aux.py --smoke                             # 极小规模冒烟（2 episode × 2 epoch，600 s）
python train_rl_aux.py --no-baseline                       # 跳过 FixedTime 基线评测

# 耗时标定（推荐先跑一次，用真实 wall_seconds 估算全量时间）
python train_rl_aux.py --episodes 2 --no-baseline          # 第 2 个 episode 才是稳态（含 100 次梯度更新）
# 然后看 output/logs/train_log.csv 的 wall_seconds 列：
# 全量耗时 ≈ 稳态 wall_seconds × episode 数 + (episode 数 / eval_every) × 单次评测耗时 + 基线耗时

# 绘图相关参数（训练结束自动出图；默认 png / 150 dpi / 中文字体自动检测）
python train_rl_aux.py --plot-every 10                     # 每 10 个 episode 刷新一次训练曲线（长训练可中途看图）
python train_rl_aux.py --plot-format pdf --plot-dpi 300     # 论文用矢量图 + 高分辨率
python train_rl_aux.py --plot-lang en                       # WSL 缺中文字体时强制英文标注
python train_rl_aux.py --no-plots                           # 完全不绘图（无 matplotlib 的机器）
python train_rl_aux.py --no-cdf --no-individual-plots       # 只出 3 类评估图 / 只出 2×2 拼图
```

### 图表重绘（不重新训练）

```bash
# 从 output/ 下已有的真实产物重绘全部图表（换格式、换分辨率、换语言）
python tools/plot_offline.py                                # 等价于训练结束时的默认出图
python tools/plot_offline.py --fmt pdf --dpi 300            # 论文矢量图
python tools/plot_offline.py --output-root ./output         # 指定产物根目录
# 退出码：0 成功；4 缺 matplotlib；5 缺少训练产物（train_log.csv）；6 绘图异常；7 没生成任何图
```

无 CityFlow 的机器上可先做离线验证（不需要 cityflow 扩展包，但需要 numpy + torch）：

```bash
python tools/selfcheck.py        # 离线自检 24 项：数据校验 / 环境链路 / 网络 / 回放池与指标 / 绘图
python tools/selfcheck.py --only plot   # 只跑绘图组（可在没有 torch 的解释器里验证出图链路）
python tools/inspect_data.py     # 只读打印 ./data 的全部字段、类型、样例与自动推导的维度
python tools/smoke_train.py      # MockEngine 冒烟训练，验证训练全链路与产物落盘（数值无实验意义）
```

> 自检里的绘图两项需要真正渲染（matplotlib）。若该解释器的 matplotlib 原生扩展损坏
> （Windows 上导入 `matplotlib.pyplot` 可能直接以原生异常码杀掉进程，`try/except` 拦不住），
> 这两项会标记为 **SKIP** 而不是 FAIL，其余纯 numpy 检查照常执行；这不是代码缺陷。
> 修复：`pip install --force-reinstall --no-cache-dir 'matplotlib>=3.5'`；或换一个能渲染的解释器
> 只跑绘图组：`python tools/selfcheck.py --only plot`。
> 训练脚本本身已把绘图放在**子进程**里，因此绘图环境崩溃**不会**影响训练结果与权重落盘。

### 输出路径

| 产物 | 路径 |
| --- | --- |
| 模型权重 | `output/rl_model/dqn_rl_aux_last.pth`（最后一轮）、`dqn_rl_aux_best.pth`（贪心评测 ATT 最优）、`dqn_rl_aux_ep{N}.pth`（每 `--save-every` 个 episode） |
| 回放池快照 | `output/rl_model/replay_buffer.npz` |
| 训练日志 | `output/logs/train_log.csv`：`episode, sim_seconds, decisions, epsilon, avg_reward, ATT, AWT, AQL, throughput, loss_mean, grad_updates, replay_size, wall_seconds` |
| 评测日志 | `output/logs/eval_log.csv`：贪心策略评测的 ATT/AWT/AQL/throughput |
| **逐秒真实序列** | `output/logs/series/baseline_fixedtime_series.npz`（FixedTime 基线）与 `eval_ep{N}_series.npz`（每次贪心评测），同名 `.csv` 便于 Excel 分析；键：`t/queue/waiting/active/seen/att/waited_vehicle_seconds/delay/per_inter__<交叉口id>` |
| **训练过程图** | `output/figures/training_curves.<ext>`（2×2 拼图）+ `training_curve_{reward,delay,queue,throughput}.<ext>`（4 张单图） |
| **测试评估图** | `output/figures/eval_{timeseries,boxplot,bar,cdf}.<ext>`（时序折线 / 箱线 / 柱状 / CDF） |
| 图表索引 | `output/figures/figures_index.json`：每张图的路径、数据来源文件、缺失项说明 |
| 字段与维度报告 | `output/reports/data_fields_report.json` |
| 运行报告 | `output/reports/run_report.json`：含 `runtime_dimensions`、`fixed_time_baseline`、`best_eval_ATT`、相对提升百分比、`figures` 图表清单 |
| CityFlow 运行配置 | `output/generated/cityflow_config.json`（指向 `./data` 的绝对路径，只读引用） |

### 图表说明（`plot_metrics.py`）

所有图表**只读取 `output/` 下的真实运行产物**（`train_log.csv`、`eval_log.csv`、`logs/series/*.npz`、
`run_report.json`），不构造任何数据；产物缺失直接报错退出（退出码 5），绝不画占位曲线。

| 图 | 数据来源（真实字段） | 看什么 |
| --- | --- | --- |
| `training_curve_reward` | `train_log.csv: avg_reward`（每 episode 决策奖励均值） | 回合奖励是否随 episode 上升并趋于平稳 |
| `training_curve_delay` | `train_log.csv: AWT`（平均等待时间=平均延误，逐秒 W(t) 累积 / N_total），并叠画 `ATT` | 延误是否持续下降；AWT 与 ATT 是否同步改善 |
| `training_curve_queue` | `train_log.csv: AQL`（逐秒排队长度时间平均） | 平均排队长度是否下降（论文 4.2 节主对比指标之一） |
| `training_curve_throughput` | `train_log.csv: throughput`（完成行程车辆数） | 通行量是否随训练上升（不能靠牺牲通行量换低延误） |
| `eval_timeseries` | 各 tag 的 `series.npz: t / delay / queue`（逐秒，`t` 为真实仿真秒） | 延误/排队长度随仿真步数的波动：RL 曲线应整体低于 FixedTime 且高峰削平 |
| `eval_boxplot` | `series.npz: queue`（逐秒排队分布）、`per_inter__*`（各交叉口排队采样） | 分布中位数与离散程度：RL 应中位数更低、箱体更短（更稳定） |
| `eval_bar` | `eval_log.csv` 最优 episode + `run_report.json: fixed_time_baseline` | ATT/AWT/AQL/throughput 四指标柱状对比，柱上标注相对 FixedTime 的改善率 |
| `eval_cdf`（可选） | 同上逐秒序列 | 累积分布：RL 曲线整体左移说明「长时间高排队」的比例被压缩；标注 P50/P90 |

> 图上的延迟/排队曲线可选滑动平均（`--plot-smooth`、`--plot-ts-smooth`），原始逐 episode/逐秒点仍以浅色叠加显示，
> 便于区分「真实波动」与「趋势」。中文标注需系统字体（Windows 字体经 `/mnt/c/Windows/Fonts/` 自动发现），
> 找不到中文字体时自动切换英文标注（`--plot-lang en` 可强制）。

---

## 4. 论文参数对照表（严格对齐 4.2 节）

| 项目 | 论文 4.2 节 | 本工程取值 | 位置 |
| --- | --- | --- | --- |
| 仿真环境 | CityFlow | CityFlow（`interval=1.0`，`rlTrafficLight=true`） | `env_cityflow.py` |
| 网络基础架构 | DQN + 经验回放 + 目标网络 | 同 | `model_net.py` / `train_rl_aux.py` |
| 回放缓冲区容量 | 12000 | 12000 | `config.REPLAY_CAPACITY` |
| batch_size | 3000 | 3000 | `config.BATCH_SIZE` |
| 折扣因子 γ | 0.8 | 0.8 | `config.GAMMA` |
| 优化器 | Adam，lr=1e-3 | Adam，lr=1e-3 | `config.LEARNING_RATE` |
| 奖励平衡系数 η | 0.2 | 0.2 | `config.ETA` |
| 注意力头数 | 8 | 8 | `config.N_HEADS` |
| 隐藏维度 h | 128 | 128 | `config.HIDDEN_DIM` |
| 自注意力堆叠层数 N | 2 | 2 | `config.N_SELF_ATTN_LAYERS` |
| 相位持续时间集合 | {10,15,20,25,30,35,40} s | 同（7 档动作） | `config.DURATION_OPTIONS` |
| 训练设置 | 每 episode 100 轮 epoch，共 100 episode | 同 | `config.EPOCHS_PER_EPISODE / TOTAL_EPISODES` |
| 状态 | 单车道五元组 (排队车辆数, 行驶车辆, seg1, seg2, seg3) ⊕ 当前相位 one-hot ⊕ 邻居相对位置 one-hot | 完全同构，维度由数据推导 | `env_cityflow.py` / `model_net.py` |
| 奖励 | $R_{target}=R_{self}+\eta\frac{\sum R_{nbr}}{k}$ | 同（$R=-\text{驶入车道排队数}$，k=0 时退化为 $R_{self}$） | `env_cityflow.py` |
| 邻居融合 | 邻居关系特征融合模块 + 邻居作用强度评估模块 | 同（交叉注意力 / 自注意力 / MLP 评分 / 分组 softmax / 邻居掩码） | `model_net.py` |
| 评价指标 | ATT、AQL（本文补 AWT 便于观察等待时间） | ATT / AWT / AQL / throughput | `eval_metric.py` |

> 论文未明确、本工程自行补充并在 `config.py:describe()` 中标注来源的参数：ε 初值/终值/衰减步数
> （1.0 → 0.05，按智能体决策次数线性衰减 200000 步）、目标网络同步间隔（200 次梯度更新）、
> 仿真预热时长（0 s）、数据分段边界（seg1<100 m ≤ seg2<200 m ≤ seg3）。

### 运行时自动推导的真实维度（来自当前 `./data`）

| 项目 | 取值 | 来源 |
| --- | --- | --- |
| 受控交叉口数 | 16 | roadnet 中 `virtual=false` 的交叉口 |
| 每交叉口驶入车道数 | 12（4 进口 × 3 车道） | roadnet 几何 |
| 路网相位总数 / 可选相位动作数 | 9 / 4 | `trafficLight.lightphases` / `signal_inter_*.txt` |
| **状态维度** | **69 = 12 × 5 + 9** | 五元组 × 12 车道 ⊕ 9 维相位 one-hot |
| 最大邻居数 | 4（角 2 / 边 3 / 内部 4） | 交叉口间距 (800, 600) 推导 |
| 相对位置 one-hot 类别 | 4 类（2 组：东西 / 南北） | 邻居相对坐标去重 |
| 持续时间动作数 | 7 | 论文动作集合 |
| 驶入车道长度范围 | 570 ~ 785 m | roadnet 几何（与 CityFlow 车道长度口径一致） |
| 需求规模 | 2983 条，interval 1.0，0 ~ 3599 s，horizon 3600 s | flow 数据 |
| 网络参数量（h=128, heads=8, N=2） | 540,684 | `model_net.count_parameters` |

---

## 5. 数据校验：成功 / 失败现象

**成功**（`python train_rl_aux.py` 或 `tools/selfcheck.py` 启动阶段）：

```
[数据校验通过] 目录（只读）: .../data
发现文件 18 个：flow.json / roadnet_4X4.json / signal_inter_intersection_1_1.txt ...
[路网 roadnet] 顶层字段 = ['intersections', 'roads'] ...
[需求 flow] 顶层数组格式 = True，flow 条数 = 2983，interval = 1.0 ...
[运行时推导维度]（模型/回放/动作掩码的唯一维度来源，config 中不写死）
   单交叉口状态维度 = 69（= 驶入车道数 x 5 + 相位 one-hot）
   最大邻居数 = 4 ；相对位置 one-hot 维度 = 4 类别 2 组 ；持续时间动作数 = 7
[数据检查] 受控交叉口 16 个；论文4.2节状态五元组 sq/sv/seg1/seg2/seg3；每交叉口驶入车道数 [12]；最大邻居数 4；相位动作数 [4]（上界 4）；持续时间动作数 7
```

同时把完整字段/维度报告写入 `output/reports/data_fields_report.json`。

**失败**（任一论文 4.2 节必需字段缺失时立即终止，**绝不虚构补全**）：

```
ValueError: [数据校验失败] roadnet/intersections[0] 缺少必需字段 'trafficLight'；
           论文4.2节状态/动作空间依赖该字段，禁止虚构补全。
```
程序打印缺失字段名并以退出码 2 结束（`DataValidationError` → `main()` 返回 2）。

校验覆盖点：
* `./data` 目录不存在 → 直接抛异常终止（`data_schema.resolve_data_files`）；
* roadnet 顶层 / intersection / road / lane / roadLink / lightphase 必需字段；
* flow 每条需求的 `vehicle / route / interval / startTime / endTime` 与 `vehicle` 子字段；
* 运行期 CityFlow 单车查询返回的 `running / distance / speed` 字段（缺失即抛 `ValueError`）；
* 论文五元组字段 `sq / sv / seg1 / seg2 / seg3` 与实际车道数量的一致性；
* 配时文件相位下标越界、动作掩码不在真实 `lightphases` 内 → 抛异常。

> 自检中已包含两条**反证用例**：人为删除 roadnet 必需字段、人为删除引擎单车字段，
> 两者都必须抛 `ValueError` 且异常信息含字段名 —— 这是"数据约束真的生效"的证据。

---

## 6. 如何判断复现是否符合论文预期

1. **训练日志趋势**（`output/logs/train_log.csv`）
   * `AQL`（平均排队长度）与 `AWT`（平均等待时间）随 episode 上升应**单调下降并趋于平稳**；
   * `ATT`（平均旅行时间）应下降并稳定，不再出现大幅震荡；
   * `avg_reward` 为负值（奖励 = 负排队数），应逐步**向 0 方向抬升**；
   * `throughput`（完成车辆数）应上升或保持高位，说明并未靠"少放车"换取低排队。
2. **与数据自带定时方案对比**：脚本启动时用 `./data` 里真实的 `signal_inter_*.txt` 跑
   FixedTime 基线，`output/reports/run_report.json` 里给出
   `relative_improvement_vs_fixed_time_pct = (FixedTime − Our RL) / FixedTime × 100%`。
   论文中杭州 HZ-1 场景 FixedTime ATT 497.87 s、Our RL 285.53 s（约 42.6% 下降）；
   HZ-2 场景 418.31 s → 317.71 s（约 24.0% 下降）。**注意：论文场景与本 `./data`
   并非同一份数据（论文未公开该数据集），因此绝对数值不可直接对齐，
   应对齐的是"相对 FixedTime 的下降幅度"与"指标收敛趋势"这两条定性结论。**
3. **收敛速度与稳定性**：DQN + 目标网络下 loss 应在数个 episode 内趋于稳定；
   若 `loss_mean` 持续发散或 ATT 抖动剧烈，优先检查 `--lr`、`batch_size` 与 ε 衰减步数。
4. **消融实验（论文 4.2 节两大模块的验证，本工程可直接扩展）**：
   * 去掉邻居信息：`ETA=0` 且网络不输入邻居张量（`nbr_mask` 全 0）→ 退化为独立单交叉口 DQN；
   * 只用 NR 模块：NS 模块替换为等权平均聚合；
   * 只用 NS 模块：NR 模块替换为直接线性映射；
   * 邻居作用强度改均匀权重（关掉分组 softmax）→ 验证"作用强度评估"是否真的有效；
   * 动作退化：固定持续时间（只用相位动作）→ 验证双输出动作的增益。
   预期结论应与论文一致：完整模型（NR + NS + 双输出）ATT/AQL 最低，去掉邻居信息后明显劣化。
5. **复现性**：固定 `config.SEED=20240501`、`--thread-num 1` 时，同机同版本的多次运行
   指标应基本一致（CityFlow 多线程会引入轻微非确定性）。

---

## 7. 实现说明与已知边界

* **状态采集完全来自数据**：`sq` 取 `engine.get_lane_waiting_vehicle_count()`（速度 < 0.1 m/s 的车辆数），
  `sv = 车道车辆数 − sq`，`seg1/2/3` 由 `engine.get_vehicle_info(v)["distance"]` 与该车道
  **真实几何长度**（由 roadnet 的 points 折线按 CityFlow 口径推算）计算剩余距离后分段。
  车道集合、车道数、相位维数、邻居关系、相对位置类别全部由 `./data` 解析得到，
  `config.py` 内不含任何维度常量。
* **异构交叉口动作掩码**：可选相位集合优先取 `signal_inter_<id>.txt` 中实际使用的相位
  （`config.ACTION_MASK_SOURCE="signal_plan"`）；若改用 `"roadnet_green"`，
  则取 roadnet 中含绿灯通行权的相位。T 型 / 网格交叉口的相位差异通过
  `phase_valid` 掩码与 `phase_avail`（每交叉口可选相位个数）在网络输出端逐样本裁剪，
  越界槽位永不参与 argmax 与梯度。
* **异步决策**：论文双输出动作为"相位 + 持续时间"，各交叉口自选时长意味着决策时刻天然不同步。
  本实现采用事件驱动的异步推进：每个交叉口动作后相位保持 $a_d$ 秒，仿真推进到最早到达的
  决策时刻，期间其他交叉口保持各自相位；`S'` 恰为该交叉口下一次决策时刻的状态。
  这样每个经验元组的动作都是"该交叉口真实执行过的动作"，不做跨交叉口的时长投票或平均。
* **Bellman 口径**：双头 Q 值取 $Q(S,a_p,a_d)=Q_{phase}(S,a_p)+Q_{dur}(S,a_d)$，
  目标 $y=R+\gamma(1-done)\left(\max Q_{phase}+\max Q_{dur}\right)$，预测与目标同口径，
  相位头 argmax 前用 `masked_fill(~valid, -1e9)` 保证掩码外动作不被选中。
* **batch 保护**：论文 batch_size=3000 与回放容量 12000 的比例在训练早期无法满足，
  `config.STRICT_BATCH_FROM_FULL_BUFFER=True` 时缓冲区样本不足则跳过梯度更新（不降级采样，
  避免小批量噪声污染 DQN 目标）；`--smoke` 会自动把 batch/capacity 压到可跑规模。
* **论文 PDF 为加密扫描件**（无文字层），本工程的论文规格来自用户提供并经数据核对的 4.2 节描述；
  所有论文未明确处均在 `config.py:describe()` 中标为"本实现补充"。

---

## 8. 常见问题

| 现象 | 处理 |
| --- | --- |
| `ImportError: No module named 'cityflow'` | 按第 1 节源码编译安装（记得 `git submodule update --init --recursive`），并确认在 `cityflow` conda 环境内 |
| `CUDA out of memory` | 降低 `--batch-size`（如 1024）或 `--hidden-dim`，或用 `--device cpu` |
| 训练太慢 | 提高 `--thread-num`（CityFlow 仿真线程），或先用 `--episodes 5 --epochs-per-episode 10` 验证链路 |
| 想核对数据里到底有哪些字段 | `python tools/inspect_data.py` |
| 想确认代码没被改坏 | `python tools/selfcheck.py`（应为 `通过 24 / 共 24 项`；绘图环境损坏时显示 `通过 22 / 共 24 项（跳过 2）`，属环境问题非代码缺陷） |
| 自检/训练里图出不来，提示「matplotlib 渲染失败」 | 该解释器的 matplotlib 原生扩展损坏（Windows 上常见，导入 `pyplot` 直接杀进程）。修复：`pip install --force-reinstall --no-cache-dir 'matplotlib>=3.5'`；也可用另一个装了 numpy+matplotlib 的解释器执行 `python tools/plot_offline.py --output-root ./output` 重绘（绘图不需要 torch/cityflow） |
| `output/generated/cityflow_config.json` 需要手动改吗 | 不需要，脚本自动生成并指向 `./data` 的绝对路径（只读引用） |
