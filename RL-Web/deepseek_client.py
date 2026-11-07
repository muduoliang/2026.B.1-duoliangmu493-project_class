# -*- coding: utf-8 -*-
"""
deepseek_client.py —— DeepSeek chat completions 代理客户端（仅标准库 urllib，py3.8 可直接跑）

为什么不用 openai SDK：本机用于打分的 py3.8 环境没有装 openai，而且只做本地调试，
用 urllib 手写 HTTP 就够；顺便能精确控制 SSE 逐块转发给前端。

安全约束：API Key 只在后端内存里，本文件不回显、不写日志、不进入任何返回给前端的字段
（redact() 用于把可能出现的密钥片段替换掉）。
"""
from __future__ import print_function

import json
import socket
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekError(RuntimeError):
    """调用 DeepSeek 失败；message 已做过密钥脱敏。"""


def redact(text, api_key=None):
    """把文本里可能出现的密钥替换掉，保证错误信息可以安全回显给前端。"""
    if not text:
        return text
    out = str(text)
    if api_key and len(api_key) >= 8:
        out = out.replace(api_key, "sk-***")
    return out


class DeepSeekClient(object):
    """OpenAI 兼容的 chat completions 客户端（只实现本项目需要的两个方法）。"""

    def __init__(self, api_key=None, base_url=None, model=None, timeout=180):
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout

    # ------------------------------------------------------------------
    # 内部：构造请求
    # ------------------------------------------------------------------
    def _endpoint(self):
        base = self.base_url
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def _payload(self, messages, temperature, max_tokens, stream, tools, json_mode):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": bool(stream),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def _request(self, payload):
        if not self.api_key:
            raise DeepSeekError("未配置 API Key（请先在页面配置栏填写并保存）")
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self._endpoint(), data=data, method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        req.add_header("Authorization", "Bearer " + self.api_key)
        req.add_header("Accept", "text/event-stream" if payload.get("stream") else "application/json")
        return req

    def _open(self, payload, attempt=0):
        req = self._request(payload)
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")
            except Exception:             # noqa: BLE001
                body = ""
            detail = _error_detail(body) or str(e.reason)
            raise DeepSeekError(redact("HTTP {} {}".format(e.code, detail), self.api_key))
        except (urllib.error.URLError, socket.timeout) as e:
            if attempt < 1:               # 网络抖动重试一次
                time.sleep(1.0)
                return self._open(payload, attempt + 1)
            reason = getattr(e, "reason", e)
            raise DeepSeekError(redact("网络错误：{}".format(reason), self.api_key))

    # ------------------------------------------------------------------
    # 非流式：一次性拿完整内容（用于必须拿到严格 JSON 的步骤）
    # ------------------------------------------------------------------
    def chat(self, messages, temperature=0.2, max_tokens=1200, tools=None, json_mode=False):
        payload = self._payload(messages, temperature, max_tokens, False, tools, json_mode)
        resp = self._open(payload)
        try:
            raw = resp.read().decode("utf-8", "replace")
        finally:
            resp.close()
        try:
            obj = json.loads(raw)
        except ValueError:
            raise DeepSeekError(redact("响应不是合法 JSON：{}".format(raw[:200]), self.api_key))
        if isinstance(obj, dict) and obj.get("error"):
            raise DeepSeekError(redact("接口返回错误：{}".format(obj["error"]), self.api_key))
        choices = obj.get("choices") or []
        if not choices:
            raise DeepSeekError(redact("响应缺少 choices：{}".format(raw[:200]), self.api_key))
        message = choices[0].get("message") or {}
        return {
            "content": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
            "finish_reason": choices[0].get("finish_reason"),
            "usage": obj.get("usage") or {},
        }

    # ------------------------------------------------------------------
    # 流式：逐块 yield {'type': 'delta'|'done', ...}，供后端 SSE 直接转发
    # ------------------------------------------------------------------
    def stream_chat(self, messages, temperature=0.2, max_tokens=1500, tools=None):
        payload = self._payload(messages, temperature, max_tokens, True, tools, False)
        resp = self._open(payload)
        collected = []
        usage = {}
        finish = None
        try:
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if isinstance(chunk, dict) and chunk.get("error"):
                    raise DeepSeekError(redact("流式返回错误：{}".format(chunk["error"]), self.api_key))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        collected.append(piece)
                        yield {"type": "delta", "content": piece}
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        finally:
            resp.close()
        yield {"type": "done", "content": "".join(collected), "usage": usage,
               "finish_reason": finish}


def _error_detail(body):
    """从错误响应体里提取 message，便于前端看到真实原因（已脱敏调用）。"""
    if not body:
        return ""
    try:
        obj = json.loads(body)
        err = obj.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
        return str(obj.get("message") or "")[:300]
    except ValueError:
        return body.strip()[:300]


# ===========================================================================
# JSON 抽取：LLM 偶尔会包一层 ```json 代码块或加前后废话
# ===========================================================================

def extract_json(text, expect="object"):
    """
    从模型输出里抠出第一个 JSON 对象/数组。
    返回 (obj, error)；失败时 obj 为 None。仅做括号配对与轻量修补，不做语义猜测。
    """
    if text is None:
        return None, "空响应"
    s = str(text).strip()
    if not s:
        return None, "空响应"
    # 去掉 ```json ... ``` 围栏
    if "```" in s:
        parts = s.split("```")
        for part in parts:
            chunk = part.strip()
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{") or chunk.startswith("["):
                s = chunk
                break
    candidates = [s]
    for i, ch in enumerate(s):
        if ch in "{[":
            candidates.append(s[i:])
            break
    for cand in candidates:
        for closing in ("}", "]"):
            idx = cand.rfind(closing)
            if idx > 0:
                piece = cand[:idx + 1]
                fixed = _repair(piece)
                try:
                    obj = json.loads(fixed)
                except ValueError:
                    continue
                if expect == "object" and not isinstance(obj, dict):
                    continue
                if expect == "array" and not isinstance(obj, list):
                    continue
                return obj, None
    return None, "未找到合法 JSON"


def _repair(text):
    """轻量修补：去掉尾随逗号、把中文全角引号换成半角。只做这两件事，避免猜测语义。"""
    out = text.replace("\r", "").replace("\n", " ")
    out = out.replace("“", '"').replace("”", '"')
    while ",}" in out:
        out = out.replace(",}", "}")
    while ",]" in out:
        out = out.replace(",]", "]")
    return out
