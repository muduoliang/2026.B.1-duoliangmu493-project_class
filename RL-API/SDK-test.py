# -*- coding: utf-8 -*-
"""
SDK-test.py —— DeepSeek API 连通性测试（接入前先跑这一个，排除网络/Key/模型名问题）。

用法（工程根目录，先激活 conda 环境）：
    source /home/mdy493/cityflow/bin/activate
    export DEEPSEEK_API_KEY=sk-xxxxxxxx
    python SDK-test.py

检查项：
    1. 环境变量 DEEPSEEK_API_KEY 是否存在；
    2. openai SDK 是否已安装（DeepSeek 使用 OpenAI 兼容接口）；
    3. 能否用 /chat/completions 正常对话并拿到回复，打印耗时与 token 用量；
    4. 可选：--tools 测试 function calling（本工程的 DeepSeek 智能体依赖它）。

退出码：0=成功，2=缺少 Key/依赖，3=请求失败（网络、额度、模型名等）。
"""

from __future__ import print_function

import argparse
import os
import sys
import time

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="DeepSeek API 连通性测试")
    p.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    p.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--tools", action="store_true", help="额外测试 function calling（工具调用）")
    p.add_argument("--timeout", type=float, default=60.0, help="单次请求超时（秒）")
    args = p.parse_args(argv)

    print("=" * 70)
    print("DeepSeek API 连通性测试")
    print("=" * 70)
    print("  base_url : {}".format(args.base_url))
    print("  model    : {}".format(args.model))

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n[失败] 未设置环境变量 DEEPSEEK_API_KEY")
        print("  临时设置：export DEEPSEEK_API_KEY=sk-xxxxxxxx")
        print("  永久设置：把上面一行写进 ~/.bashrc 后 source ~/.bashrc")
        return 2
    print("  api_key  : {}****{}（长度 {}）".format(api_key[:6], api_key[-4:], len(api_key)))

    try:
        from openai import OpenAI
    except ImportError:
        print("\n[失败] 未安装 openai SDK：pip install 'openai>=1.0'")
        return 2

    try:
        import openai
        print("  openai   : {}（Python {}）".format(openai.__version__,
                                                  sys.version.split()[0]))
    except Exception:                                   # noqa: BLE001
        pass

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)

    # ---------- 1) 普通对话 ----------
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "只回复两个字：正常"}],
            max_tokens=16,
            temperature=0.0,
        )
    except Exception as e:                              # noqa: BLE001
        print("\n[失败] 请求异常：{}: {}".format(type(e).__name__, e))
        print("  排查顺序：1) 网络/代理  2) API Key 是否有效  3) 模型名 {} 是否可用"
              .format(args.model))
        return 3
    dt = time.time() - t0
    content = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    print("\n[成功] 对话接口正常，耗时 {:.2f}s".format(dt))
    print("  回复     : {}".format(content))
    if usage:
        print("  token    : prompt={} completion={} total={}".format(
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
            getattr(usage, "total_tokens", "?")))

    # ---------- 2) function calling（可选） ----------
    if args.tools:
        tools = [{
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "获取当前仿真时刻（测试用）",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }]
        try:
            resp2 = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": "请调用 get_time 工具"}],
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=64,
            )
        except Exception as e:                          # noqa: BLE001
            print("\n[失败] function calling 测试异常：{}: {}".format(type(e).__name__, e))
            print("  说明：本工程的 DeepSeek 智能体依赖工具调用；"
                  "请确认模型 {} 支持 tools 参数".format(args.model))
            return 3
        msg = resp2.choices[0].message
        calls = list(getattr(msg, "tool_calls", None) or [])
        if calls:
            print("\n[成功] function calling 可用：模型请求调用 {}".format(
                ", ".join(tc.function.name for tc in calls)))
        else:
            print("\n[警告] 模型没有发起工具调用（回复：{}）。"
                  "链路本身正常，但换成 deepseek-chat 再试".format(
                      (msg.content or "").strip()[:80]))

    print("\n下一步：python deepseek_agent.py --dry-run   # 验证 RL 工具链路（不联网）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
