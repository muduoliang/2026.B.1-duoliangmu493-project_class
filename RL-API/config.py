# -*- coding: utf-8 -*-
"""
config.py —— 全项目超参数集中管理（唯一超参入口，方便调参）

对应论文：《城市交通信号控制大语言模型方法研究与系统实现》王之颢
         第四章 4.2 节「融合邻居信息的动态交通信号控制强化学习算法（RL 辅助模型）」

【本文件的两条硬性约定】
1. 本文件【只放超参数与路径】，**绝不放任何数据集特征维度**（state_dim / n_lanes /
   n_phases / n_rel_pos ... 一律不在此定义）。所有输入维度都由 data_schema.py
   在运行时读取 ./data 后自动推导（见 data_schema.DataSchema.dims）。
   这样做的原因：论文中的维度是理论描述，真实可用维度必须以 ./data 实际字段为准。
2. 论文原文明确定义的量，注释里标「论文4.2节」并给出来源章节，禁止臆改。

【只读约束】本工程对 ./data 目录只做读取，不写入、不生成、不覆盖任何文件；
所有产物统一写到 ./output/。
"""

import os

# ---------------------------------------------------------------------------
# 一、路径配置（全部基于本文件所在目录推导，保证在任意 cwd 下运行都正确）
# ---------------------------------------------------------------------------

# 工程根目录（config.py 所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据集目录：默认为工作目录下的 ./data
#   - 目录内应包含：CityFlow 路网 json（roadnet*.json）、交通需求 json（flow*.json）、
#     以及各交叉口定时信号配时 txt（signal_inter_*.txt）
#   - 允许用环境变量 RL2_DATA_DIR 覆盖（便于在别处放数据时复用本工程）
#   - **只读**：程序任何位置都不得向该目录写文件
DATA_DIR = os.environ.get("RL2_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

# 输出根目录：所有实验产物唯一出口
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# 模型权重输出目录（train_rl_aux.py 保存 .pth 到此）
MODEL_DIR = os.path.join(OUTPUT_DIR, "rl_model")

# 训练日志目录（逐 episode 指标 csv/json）
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# 运行时生成物目录（CityFlow 运行配置、数据自检报告等）
GENERATED_DIR = os.path.join(OUTPUT_DIR, "generated")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

# 数据文件发现规则（在 DATA_DIR 下按 glob 匹配；不写死文件名，适配不同数据副本）
ROADNET_PATTERNS = ("roadnet*.json", "*roadnet*.json")
FLOW_PATTERNS = ("flow*.json", "*flow*.json")
SIGNAL_PLAN_PATTERNS = ("signal_inter_*.txt", "signal_*.txt")

# 交叉口 id 与定时配时文件名的对应规则：signal_inter_<intersection_id>.txt
SIGNAL_PLAN_FILE_PREFIX = "signal_inter_"
SIGNAL_PLAN_FILE_SUFFIX = ".txt"

# ---------------------------------------------------------------------------
# 二、论文 4.2 节网络与 DQN 超参数（**直接照搬论文，禁止随意改动**）
# ---------------------------------------------------------------------------

# 折扣因子 γ = 0.8                                  【论文4.2节 DQN 超参数】
GAMMA = 0.8

# Adam 优化器初始学习率 1e-3                          【论文4.2节 DQN 超参数】
LEARNING_RATE = 1e-3

# 经验回放缓冲区容量 12000                           【论文4.2节 Replay-Buffer 容量】
REPLAY_CAPACITY = 12000

# 训练批大小 batch_size = 3000                       【论文4.2节 DQN 超参数】
BATCH_SIZE = 3000

# 奖励平衡系数 η = 0.2
#   R_target = R_self + η * (Σ R_nbr / k)            【论文4.2节 奖励函数】
ETA = 0.2

# 多头注意力头数 = 8                                  【论文4.2节 NR/NS 模块】
N_HEADS = 8

# 隐藏维度 h = 128                                    【论文4.2节 NR/NS 模块】
HIDDEN_DIM = 128

# 自注意力堆叠层数 N = 2                              【论文4.2节 NR 模块】
N_SELF_ATTN_LAYERS = 2

# 相位持续时间可选集合 {10,15,20,25,30,35,40}（单位 s）【论文4.2节 动作空间】
DURATION_OPTIONS = (10, 15, 20, 25, 30, 35, 40)

# 训练规模：总训练 episode = 100，每个 episode 内 100 轮 epoch【论文4.2节 训练设置】
TOTAL_EPISODES = 100
EPOCHS_PER_EPISODE = 100

# ---------------------------------------------------------------------------
# 三、论文 4.2 节状态空间 / 奖励函数相关的度量定义
#   注意：这些是「论文给定的度量口径」，不是数据集特征维度，因此放在 config。
# ---------------------------------------------------------------------------

# 车道状态五元组的距离分段边界（单位 m）：seg1=(0,100]，seg2=(100,200]，seg3=(200,+∞)
#   【论文4.2节 状态空间：单车道五元组 (排队车辆数, 行驶车辆, seg1, seg2, seg3)】
SEG_BAND_EDGES_M = (100.0, 200.0)

# 「排队车辆」判定阈值（m/s）：速度低于该值视为排队/等待车辆
#   选用 CityFlow 引擎内部口径：engine.cpp:641 `vehicle->getSpeed() < 0.1`
QUEUE_SPEED_THRESHOLD = 0.1

# 决策/仿真时间粒度（s）：CityFlow 推荐区间 [0.2, 1.5]（engine.cpp:169）
SIM_INTERVAL = 1.0

# 每个 episode 的仿真时长（s）
#   None → 由 data_schema 从 ./data 的 flow 数据自动推导（max(endTime)+1，即需求自然结束）
#   也可显式给数值覆盖（例如小规模冒烟测试）
EPISODE_SECONDS = None

# 需求结束后仍允许路网排空的额外时长（s），仅当 EPISODE_SECONDS 由数据推导时生效
EPISODE_DRAIN_SECONDS = 0.0

# 预热时长（s）：论文未明确给出该设置。
#   本实现默认 0，即不预热，保证 ATT/AWT/AQL 统计口径覆盖完整仿真区间（便于与论文
#   基线数值对比）。若希望路网先填充再开始学习，可设为 300 等值（此时预热期不计入
#   指标，属消融实验变量）。
WARMUP_SECONDS = 0.0

# ---------------------------------------------------------------------------
# 四、论文未明确、由本实现补充的程序级超参数（可调，已在 readme 中标注）
# ---------------------------------------------------------------------------

# ε-greedy 探索：起始值、终止值、线性衰减步数（论文未给出 ε 具体设置）
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 200000  # 按「智能体决策次数」计（16 个交叉口同时决策算 16 次）

# 目标网络同步间隔（按梯度更新次数计，论文未给出目标网络更新频率）
TARGET_UPDATE_EVERY = 200

# 梯度裁剪（防止 TD 误差偶发尖峰导致发散；论文未提及）
GRAD_CLIP_NORM = 10.0

# 严格模式：回放池样本数 < BATCH_SIZE 时不训练（论文 batch_size=3000 的忠实实现）
STRICT_BATCH_FROM_FULL_BUFFER = True

# 相位动作掩码来源（数据驱动，两条都是真实数据来源，不含任何虚构相位）：
#   "signal_plan"  : 使用 ./data/signal_inter_*.txt 定时方案里实际出现的相位索引
#                    （本数据集该集合恰为 4 个相位，与论文4.2节「4 相位」一致）
#   "roadnet_green": 使用路网 trafficLight 中所有「含直行或左转」的绿相位（本数据集为 8 个）
ACTION_MASK_SOURCE = "signal_plan"

# 训练设备：None → 自动（cuda 可用则 cuda，否则 cpu）
DEVICE = None

# 随机种子（同时写入 CityFlow config 的 seed 字段，保证仿真可复现）
SEED = 20240501

# 日志与保存频率
LOG_EVERY_EPISODE = 1
SAVE_EVERY_EPISODE = 10
PRINT_FIELDS_REPORT = True  # 启动时打印 data 全部可用字段与推导出的维度

# 数值稳定性
EPS_FOR_LOG = 1e-8


# ---------------------------------------------------------------------------
# 五、工具函数：创建输出目录（绝不创建/写入 data 目录）
# ---------------------------------------------------------------------------

def ensure_output_dirs():
    """创建全部输出目录；若 ./data 不存在则抛出异常终止（见 data_schema 的校验）。"""
    for d in (OUTPUT_DIR, MODEL_DIR, LOG_DIR, GENERATED_DIR, REPORT_DIR):
        os.makedirs(d, exist_ok=True)
    return {
        "output": OUTPUT_DIR,
        "model": MODEL_DIR,
        "log": LOG_DIR,
        "generated": GENERATED_DIR,
        "report": REPORT_DIR,
    }


def describe():
    """返回一份「论文参数 → 本实现取值」对照表，便于人工核对与论文溯源。"""
    return [
        ("仿真环境", "CityFlow", "论文4.2节"),
        ("网络架构", "DQN（在线网络 + 目标网络）", "论文4.2节"),
        ("折扣因子 γ", GAMMA, "论文4.2节"),
        ("优化器", "Adam, lr={}".format(LEARNING_RATE), "论文4.2节"),
        ("经验回放容量", REPLAY_CAPACITY, "论文4.2节"),
        ("batch_size", BATCH_SIZE, "论文4.2节"),
        ("奖励平衡系数 η", ETA, "论文4.2节"),
        ("注意力头数", N_HEADS, "论文4.2节 NR/NS 模块"),
        ("隐藏维度 h", HIDDEN_DIM, "论文4.2节 NR/NS 模块"),
        ("自注意力层数 N", N_SELF_ATTN_LAYERS, "论文4.2节 NR 模块"),
        ("持续时间动作集", DURATION_OPTIONS, "论文4.2节 动作空间"),
        ("总训练 episode", TOTAL_EPISODES, "论文4.2节 训练设置"),
        ("每 episode 训练 epoch", EPOCHS_PER_EPISODE, "论文4.2节 训练设置"),
        ("状态分段边界(m)", SEG_BAND_EDGES_M, "论文4.2节 状态空间"),
        ("排队判定速度阈值(m/s)", QUEUE_SPEED_THRESHOLD, "CityFlow engine.cpp:641"),
        ("ε 起始/终止/衰减步", (EPS_START, EPS_END, EPS_DECAY_STEPS), "论文未给出，本实现补充"),
        ("目标网络同步间隔", TARGET_UPDATE_EVERY, "论文未给出，本实现补充"),
        ("预热时长(s)", WARMUP_SECONDS, "论文未给出，本实现补充"),
    ]
