# -*- coding: utf-8 -*-
"""
deepseek_agent.py —— 把「冻结的 RL 辅助模型」作为工具接给 DeepSeek API 的智能体。

论文对应：第四章 4.2 节（RL 辅助模型）→ 第五章智能体系统（本项目只做工具接入，
不做 LLM 的 SFT/DPO 微调，也不做前端可视化）。

一、为什么是「工具调用」而不是「把模型塞进 LLM」
    RL 模型输出的是每个交叉口的离散动作（相位槽 + 持续时间档）与 Q 值，
    LLM 需要的是「现在的路况、模型做了什么、效果如何」这类结构化事实。
    因此正确的接法是：LLM 通过 function calling 调用本地工具 → 工具内部用冻结
    RL 模型推进一步 → 把 JSON 结果回灌给 LLM → LLM 负责解释/汇总/给建议。
    这样既保住了「冻结权重只做推理」的边界，也不会让 LLM 编造指标。

二、数据流（全部在本机 WSL 进程内，只有 LLM 推理走 HTTPS）
    DeepSeek API (deepseek-chat)
        │  tool_call: rl_aux_advance({steps: 5})
        ▼
    本进程 ToolRuntime ──► FrozenRLAuxPolicy（冻结权重，eval + no_grad）
        │                        │
        │                        ▼
        │                  CityFlow 仿真（./data 只读）
        ▼
    结构化 JSON（每次决策的相位/时长/Q 值/邻居权重 + ATT/AWT/AQL）
        ▲
        │  role=tool 消息回灌
        └── DeepSeek 生成最终中文结论

三、用法（工程根目录，WSL 内 conda 环境）
    export DEEPSEEK_API_KEY=sk-xxxxxxxx        # 必填，绝不要写进代码/仓库
    export DEEPSEEK_MODEL=deepseek-chat        # 可选（默认 deepseek-chat）
    python deepseek_agent.py --dry-run         # 不联网：先验证工具调用链路
    python deepseek_agent.py                   # 真实调用 DeepSeek + CityFlow
    python deepseek_agent.py --mock            # 无 CityFlow 时用 MockEngine 走链路
    python deepseek_agent.py --question "总结当前信号控制策略的表现，并指出最拥堵的两个交叉口"

四、依赖
    pip install openai>=1.0        # DeepSeek 使用 OpenAI 兼容接口
    （RL 侧依赖 torch/numpy/cityflow，见 requirements.txt）
"""

from __future__ import print_function

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional

import config

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

# ---------------------------------------------------------------------------
# 系统提示词：约束 LLM「只用工具返回的事实说话」
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是城市交通信号控制智能体。你脚下的仿真城市（杭州 4×4 路网，共 16 个受控交叉口）
由一个**已训练完成并冻结的「融合邻居信息的 RL 辅助模型」**控制，该模型对应论文第四章 4.2 节：
  - 状态：每个交叉口 12 条驶入车道的五元组（排队车辆数、行驶车辆数、seg1/seg2/seg3 分段车辆数）
          拼接当前相位 one-hot，再拼接邻居相对位置 one-hot；
  - 动作：双输出 = 相位动作（该交叉口真实可选相位内的一个）+ 相位持续时间（10~40s，7 档）；
  - 奖励：R_target = R_self + 0.2 × 邻居平均奖励（邻居信息由交叉注意力与分组 softmax 融合）。
你只能通过工具获取真实数据，规则如下：
  1) 任何指标（ATT/AWT/AQL/throughput）、动作、队列长度、Q 值，必须来自工具返回，**严禁编造**；
  2) 需要看当前时刻状态 → 调用 rl_aux_status；需要推进仿真 → 调用 rl_aux_advance；
     需要重置场景 → 调用 rl_aux_reset；
  3) 每次工具返回后，先用一两句话解读关键信息，再决定是否继续调用工具；
  4) 谈动作时必须引用具体交叉口 id、相位槽/lightphase、持续时间（秒）、以及必要的 Q 值或邻居权重；
  5) 最终用中文给出结论：当前控制效果（与固定配时基线对比更佳）、最拥堵的交叉口、
     模型动作偏好（相位/时长），以及可执行的优化建议；
  6) 不要输出与本次仿真无关的泛泛而谈；不要声称自己"训练"或"修改"了模型。"""

# ---------------------------------------------------------------------------
# 工具定义（OpenAI function calling 格式；DeepSeek 完全兼容）
# ---------------------------------------------------------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "rl_aux_status",
            "description": "查询冻结 RL 辅助模型的当前仿真状态：仿真时刻、是否结束、"
                           "待决策交叉口、累计决策数，以及当前的 ATT/AWT/AQL/throughput。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rl_aux_advance",
            "description": "让冻结的 RL 辅助模型推进 N 个决策步：内部读取 CityFlow 真实状态、"
                           "前向推理、按动作掩码选相位与持续时间并施加动作。"
                           "返回每一步每个交叉口的相位/持续时间/Q 值与邻居权重。",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {"type": "integer", "minimum": 1, "maximum": 50,
                              "description": "推进的决策步数（建议 1~10，每步约 10~40 秒仿真时间）"},
                },
                "required": ["steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rl_aux_reset",
            "description": "重置仿真 episode（回到 0 秒），清空累计指标。可选设置随机种子。",
            "parameters": {
                "type": "object",
                "properties": {
                    "seed": {"type": "integer", "description": "可选：随机种子"},
                },
                "required": [],
            },
        },
    },
]


# ===========================================================================
# 1. 工具运行时：把 FrozenRLAuxPolicy 包装成可 dispatch 的工具集合
# ===========================================================================

class ToolRuntime(object):
    """工具调度器：名称 → 冻结策略方法。所有返回值都是 JSON 可序列化的 dict。"""

    def __init__(self, policy: Any, verbose: bool = True):
        self.policy = policy
        self.verbose = verbose
        self.calls: List[Dict[str, Any]] = []

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        handler = {
            "rl_aux_status": self._status,
            "rl_aux_advance": self._advance,
            "rl_aux_reset": self._reset,
        }.get(name)
        if handler is None:
            return {"error": "未知工具：{}（可用：rl_aux_status / rl_aux_advance / rl_aux_reset）"
                             .format(name)}
        try:
            result = handler(arguments or {})
        except Exception as e:                       # noqa: BLE001 - 工具错误要回灌给 LLM
            result = {"error": "{}: {}".format(type(e).__name__, e)}
        self.calls.append({"tool": name, "arguments": arguments, "result": result})
        return result

    def handle_json(self, name: str, arguments_json: Optional[str]) -> str:
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except (ValueError, TypeError):
            return json.dumps({"error": "工具参数不是合法 JSON：{}".format(arguments_json)},
                              ensure_ascii=False)
        # allow_nan=False：工具消息必须是严格 JSON（NaN/Inf 会让模型侧解析失败）
        try:
            return json.dumps(self.dispatch(name, args), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return json.dumps({"error": "工具返回包含无法序列化的值"}, ensure_ascii=False)

    # -- 具体工具 -------------------------------------------------------
    def _status(self, _args: Dict[str, Any]) -> Dict[str, Any]:
        return self.policy.status()

    def _advance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        steps = int(args.get("steps", 1))
        steps = max(1, min(50, steps))
        return self.policy.advance(steps=steps)

    def _reset(self, args: Dict[str, Any]) -> Dict[str, Any]:
        seed = args.get("seed")
        return self.policy.reset(seed=int(seed) if seed is not None else None)


# ===========================================================================
# 2. DeepSeek 客户端封装（OpenAI 兼容协议）
# ===========================================================================

class DeepSeekAgent(object):
    def __init__(self,
                 runtime: ToolRuntime,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.2,
                 max_turns: int = 8,
                 log_path: Optional[str] = None,
                 verbose: bool = True):
        self.runtime = runtime
        self.model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        self.temperature = float(temperature)
        self.max_turns = int(max_turns)
        self.log_path = log_path
        self.verbose = verbose
        self.transcript: List[Dict[str, Any]] = []
        self._client_obj = None

    # -- 延迟导入 openai：--dry-run 不需要安装 openai -------------------
    def _client(self):
        if self._client_obj is not None:
            return self._client_obj
        if not self.api_key:
            raise RuntimeError(
                "[缺少 API Key] 请先设置环境变量 DEEPSEEK_API_KEY（或在命令行 --api-key 传入）。\n"
                "示例：export DEEPSEEK_API_KEY=sk-xxxxxxxx")
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "[缺少依赖] 未安装 openai SDK：请先 `pip install openai`（DeepSeek 用 OpenAI 兼容接口）")
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client_obj = OpenAI(**kwargs)
        return self._client_obj

    # -- 日志 -----------------------------------------------------------
    def _log(self, record: Dict[str, Any]) -> None:
        self.transcript.append(record)
        if not self.log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:                        # 日志失败不影响主流程
            pass

    # -- 主循环 ---------------------------------------------------------
    def run(self, question: str) -> str:
        client = self._client()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        self._log({"type": "user", "content": question,
                   "time": datetime.datetime.now().isoformat(timespec="seconds"),
                   "model": self.model, "base_url": self.base_url})

        last_content = ""
        for turn in range(1, self.max_turns + 1):
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=self.temperature,
            )
            msg = resp.choices[0].message
            usage = getattr(resp, "usage", None)
            tool_calls = list(getattr(msg, "tool_calls", None) or [])
            content = msg.content or ""
            last_content = content or last_content

            self._log({"type": "assistant", "turn": turn, "content": content,
                       "tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else tc
                                      for tc in tool_calls],
                       "usage": ({"prompt_tokens": getattr(usage, "prompt_tokens", None),
                                  "completion_tokens": getattr(usage, "completion_tokens", None),
                                  "total_tokens": getattr(usage, "total_tokens", None)}
                                 if usage else None)})
            if self.verbose and content:
                print("\n[DeepSeek] {}".format(content))

            if not tool_calls:
                return content

            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else {
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments},
                } for tc in tool_calls],
            })
            for tc in tool_calls:
                name = tc.function.name
                args_json = tc.function.arguments
                result_json = self.runtime.handle_json(name, args_json)
                if self.verbose:
                    preview = result_json if len(result_json) <= 600 else result_json[:600] + "...(截断)"
                    print("\n[工具] {}({}) → {}".format(name, args_json or "{}", preview))
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": result_json})

        return last_content or "[达到最大轮数 {}，模型未给出最终结论]".format(self.max_turns)


# ===========================================================================
# 3. 便捷入口：构建策略 + 运行时
# ===========================================================================

def build_runtime(args: argparse.Namespace) -> ToolRuntime:
    from rl_aux_infer import FrozenRLAuxPolicy
    policy = FrozenRLAuxPolicy(args.frozen, args.data_dir, device=args.device,
                               verbose=not args.quiet)
    engine = None
    if args.mock:
        from rl_aux_infer import _make_mock_engine
        engine = _make_mock_engine(policy.schema)
    policy.create_env(engine=engine, episode_seconds=args.episode_seconds,
                      verbose=not args.quiet)
    policy.reset()
    return ToolRuntime(policy, verbose=not args.quiet)


def dry_run(runtime: ToolRuntime, question: str, quiet: bool = False) -> str:
    """不联网演练工具调用链路：模拟 LLM 依次调用 status → advance → status。"""
    script = [
        ("rl_aux_status", {}),
        ("rl_aux_advance", {"steps": 2}),
        ("rl_aux_status", {}),
    ]
    print("\n[演练] 不调用 DeepSeek，只验证工具链路（真实运行请去掉 --dry-run 并设置 DEEPSEEK_API_KEY）")
    print("[演练] 将发送给模型的工具清单：{}".format(
        ", ".join(t["function"]["name"] for t in TOOLS)))
    for name, args in script:
        result = runtime.dispatch(name, args)
        payload = json.dumps(result, ensure_ascii=False)
        preview = payload if len(payload) <= 900 else payload[:900] + "...(截断)"
        print("[演练] {}({}) → {}".format(name, json.dumps(args, ensure_ascii=False), preview))
    print("\n[演练] 用户问题：{}".format(question))
    print("[演练] 若已配置 DEEPSEEK_API_KEY，去掉 --dry-run 即可得到模型的中文结论。")
    return "[dry-run] 工具链路 OK"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepSeek + 冻结 RL 辅助模型（工具调用）")
    p.add_argument("--frozen", default=None, help="冻结包路径（默认 output/rl_model/frozen/rl_aux_frozen.pth）")
    p.add_argument("--data-dir", default=config.DATA_DIR, help="数据目录（只读）")
    p.add_argument("--device", default="cpu", help="RL 推理设备（cpu / cuda）")
    p.add_argument("--question", default="请推进 3 个决策步，汇总当前信号控制效果，"
                                         "并指出最拥堵的交叉口与模型的动作偏好。",
                   help="向智能体提出的问题")
    p.add_argument("--model", default=None, help="DeepSeek 模型（默认 deepseek-chat）")
    p.add_argument("--api-key", default=None, help="API Key（不推荐，优先用环境变量 DEEPSEEK_API_KEY）")
    p.add_argument("--base-url", default=None, help="API 地址（默认 https://api.deepseek.com）")
    p.add_argument("--temperature", type=float, default=0.2, help="采样温度（默认 0.2）")
    p.add_argument("--max-turns", type=int, default=8, help="工具调用最大轮数（默认 8）")
    p.add_argument("--episode-seconds", type=float, default=None, help="演示用 episode 时长（可选）")
    p.add_argument("--mock", action="store_true", help="用 MockEngine 代替 CityFlow（链路验证，数值无意义）")
    p.add_argument("--dry-run", action="store_true", help="不联网，只验证工具调用链路")
    p.add_argument("--quiet", action="store_true", help="精简输出")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        runtime = build_runtime(args)
    except Exception as e:                        # noqa: BLE001 - 构建失败要给可读提示
        print("[失败] 初始化失败：{}: {}".format(type(e).__name__, e))
        print("       检查：冻结包是否已生成（python freeze_rl.py）、本机是否已装 cityflow，"
              "或加 --mock 先验证链路。")
        return 2

    try:
        if args.dry_run:
            dry_run(runtime, args.question, quiet=args.quiet)
            return 0

        dirs = config.ensure_output_dirs()
        log_path = os.path.join(dirs["log"], "deepseek_session.jsonl")
        agent = DeepSeekAgent(runtime, model=args.model, api_key=args.api_key,
                              base_url=args.base_url, temperature=args.temperature,
                              max_turns=args.max_turns, log_path=log_path,
                              verbose=not args.quiet)
        answer = agent.run(args.question)
        print("\n" + "=" * 78)
        print("[最终结论]")
        print(answer)
        print("=" * 78)
        print("[会话日志] {}".format(log_path))
        print("[工具调用] {} 次".format(len(runtime.calls)))
        return 0
    except RuntimeError as e:
        print("[失败] {}".format(e))
        return 3
    finally:
        try:
            runtime.policy.close()
        except Exception:                         # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
