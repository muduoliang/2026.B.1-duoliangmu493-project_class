# -*- coding: utf-8 -*-
"""
probe_llm.py —— 开发探针：用本地假 DeepSeek 服务验证 deepseek_client（不需要真 Key）。
用法：python _dev/probe_llm.py

覆盖
    1. 非流式 chat()：请求体校验（model / messages / response_format=json_object）
    2. 流式 stream_chat()：逐帧 delta → done，[DONE] 终止符
    3. 错误路径：HTTP 401 → DeepSeekError 且带错误详情
    4. extract_json()：剥 ``` 围栏 / 尾逗号 / 前后噪声
"""
from __future__ import print_function

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))

import deepseek_client as dc  # noqa: E402

SEEN = {"requests": []}


class FakeDeepSeek(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        SEEN["requests"].append({"path": self.path,
                                 "auth": self.headers.get("Authorization"),
                                 "body": body})
        if body.get("model") == "bad-model":
            payload = json.dumps({"error": {"message": "模型不存在", "type": "invalid_request_error"}})
            data = payload.encode("utf-8")
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            frames = ["你", "好", "，", "配时", "方案"]
            for i, piece in enumerate(frames):
                chunk = {"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                         "model": body.get("model"),
                         "choices": [{"index": 0, "delta": {"content": piece},
                                      "finish_reason": None if i < len(frames) - 1 else "stop"}]}
                self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            usage = {"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                     "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 5,
                                              "total_tokens": 16}}
            self.wfile.write(("data: " + json.dumps(usage) + "\n\n").encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return

        payload = json.dumps({
            "id": "chatcmpl-fake", "object": "chat.completion", "model": body.get("model"),
            "choices": [{"index": 0, "message": {
                "role": "assistant",
                "content": '{"notes":"假服务","default_cycle":[{"phase_slot":0,"duration":30}]}'},
                "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        }, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeDeepSeek)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:{}".format(port)
    print("假 DeepSeek 服务: {}".format(base))

    client = dc.DeepSeekClient(api_key="sk-fake-key-0001", base_url=base, model="deepseek-chat")
    print("endpoint -> {}".format(client._endpoint()))

    # 1) 非流式 + json_mode
    out = client.chat([{"role": "user", "content": "给个方案"}], json_mode=True)
    print("\n[1] chat() ok: finish={} usage={}".format(out["finish_reason"], out["usage"]))
    print("    content={}".format(out["content"][:80]))
    req = SEEN["requests"][-1]
    assert req["path"].endswith("/v1/chat/completions"), req["path"]
    assert req["auth"] == "Bearer sk-fake-key-0001", req["auth"]
    assert req["body"]["response_format"] == {"type": "json_object"}, req["body"].get("response_format")
    assert req["body"]["stream"] is False
    print("    请求体校验: path/auth/response_format/stream 全部正确")

    # 2) 流式
    pieces, done = [], None
    for chunk in client.stream_chat([{"role": "user", "content": "说句话"}], max_tokens=100):
        if chunk["type"] == "delta":
            pieces.append(chunk["content"])
        else:
            done = chunk
    print("\n[2] stream_chat() 收到 {} 帧: {}".format(len(pieces), "".join(pieces)))
    print("    done: {}".format(done))
    assert "".join(pieces) == "你好，配时方案", pieces
    assert done and done["usage"], done
    assert SEEN["requests"][-1]["body"]["stream"] is True
    print("    流式帧与 [DONE] 终止符解析正确")

    # 3) 错误路径
    bad = dc.DeepSeekClient(api_key="sk-x", base_url=base, model="bad-model")
    try:
        bad.chat([{"role": "user", "content": "hi"}])
        print("\n[3] !! 没有抛错")
    except dc.DeepSeekError as e:
        msg = str(e)
        print("\n[3] DeepSeekError: {}".format(msg[:120]))
        assert "401" in msg or "模型不存在" in msg, msg
        print("    错误详情已带出，且不会泄露 Key: {}".format("sk-x" not in msg))

    # 4) JSON 抽取
    cases = [
        ('```json\n{"a":1,}\n```', "object"),
        ('前面废话 {"b": [1,2,3],} 后面废话', "object"),
        ('```json\n[{"c":1},{"c":2}]\n```', "array"),
    ]
    print("\n[4] extract_json():")
    for text, expect in cases:
        got, err = dc.extract_json(text, expect=expect)     # 返回 (对象, 错误信息)
        print("    {!r} -> {} (err={})".format(text[:26], got, err))
        assert isinstance(got, (dict, list)) and err is None, (got, err)

    # 抛不动的文本必须返回 (None, 错误) 而不是异常
    bad_obj, bad_err = dc.extract_json("这里没有 JSON", expect="object")
    print("    '这里没有 JSON' -> ({}, err={})".format(bad_obj, bad_err))
    assert bad_obj is None and bad_err, (bad_obj, bad_err)

    # 5) 脱敏
    print("\n[5] redact(): {}".format(dc.redact("key=sk-fake-key-0001 出错", "sk-fake-key-0001")))
    assert "sk-fake-key-0001" not in dc.redact("key=sk-fake-key-0001", "sk-fake-key-0001")

    httpd.shutdown()
    print("\n全部通过：非流式 / 流式 / 错误路径 / JSON 抽取 / 脱敏")


if __name__ == "__main__":
    main()
