# -*- coding: utf-8 -*-
"""
app.py —— 本地网页应用后端（只用 Python 标准库，py3.8 可直接运行）

启动： python app.py        （或用同目录的 run.bat 一行启动）
访问： http://127.0.0.1:8000

职责
    * 静态托管 web/index.html（单页前端）
    * 保存 DeepSeek API Key：只存在本进程内存里，前端只能拿到掩码，永远看不到真 Key
    * POST /api/run   ：跑五步链路，用 SSE 流式推进度/生成过程
    * POST /api/chat  ：DeepSeek 对话代理（流式）
    * GET  /api/health：环境自检（原有 RL 模块 / 冻结模型 / 数据是否就绪）
"""
from __future__ import print_function

import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pipeline
import rl_core
from deepseek_client import DeepSeekClient, DeepSeekError

HOST = os.environ.get("RL_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("RL_WEB_PORT", "8000"))
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

CHAT_SYSTEM = (
    "你是本地交通信号配时助手的对话通道。回答用中文、尽量短（不超过 6 行），"
    "只谈配时方案、RL 打分与本工具用法；数据或指标必须来自用户给的内容，不确定就说不知道。"
)

# 进程内配置：API Key 只在这里，且只在后端使用
CONFIG = {"api_key": "", "model": "deepseek-chat", "base_url": "https://api.deepseek.com"}
CONFIG_LOCK = threading.Lock()

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


def mask_key(key):
    if not key:
        return ""
    if len(key) <= 8:
        return "sk-***"
    return key[:4] + "***" + key[-4:]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"      # 简单起见：一请求一连接，SSE 结束即关闭

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def log_message(self, fmt, *args):  # noqa: A003 - 只需精简日志，且不打印任何头部
        sys.stderr.write("[web] %s %s\n" % (self.command, self.path))

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            return {}

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

    def _sse(self, event):
        """写一条 SSE 事件（data 为单行 JSON）。"""
        try:
            payload = json.dumps(event, ensure_ascii=False)
            self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, ValueError):
            return False

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if path == "/api/health":
            return self._handle_health()
        if path.startswith("/api/"):
            return self._send_json({"ok": False, "error": "未知接口"}, 404)
        return self._serve_static(path.lstrip("/"))

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/api/config":
            return self._handle_config()
        if path == "/api/run":
            return self._handle_run()
        if path == "/api/chat":
            return self._handle_chat()
        return self._send_json({"ok": False, "error": "未知接口"}, 404)

    # ------------------------------------------------------------------
    def _serve_static(self, rel):
        name = os.path.basename(rel)                 # 只取文件名，防目录穿越
        full = os.path.join(WEB_DIR, name)
        if not name or not os.path.isfile(full):
            if name == "favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            return self._send_json({"ok": False, "error": "文件不存在: " + name}, 404)
        ext = os.path.splitext(name)[1].lower()
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", STATIC_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    def _handle_health(self):
        probe = rl_core.probe()
        with CONFIG_LOCK:
            cfg = {"model": CONFIG["model"], "has_key": bool(CONFIG["api_key"]),
                   "masked": mask_key(CONFIG["api_key"])}
        return self._send_json({"ok": probe.get("ok", False), "rl": probe, "config": cfg,
                                "url": "http://{}:{}".format(HOST, PORT)})

    def _handle_config(self):
        body = self._read_json()
        with CONFIG_LOCK:
            if "api_key" in body:
                CONFIG["api_key"] = str(body.get("api_key") or "").strip()
            if body.get("model"):
                CONFIG["model"] = str(body["model"]).strip()
            if body.get("base_url"):
                CONFIG["base_url"] = str(body["base_url"]).strip()
            cfg = {"model": CONFIG["model"], "has_key": bool(CONFIG["api_key"]),
                   "masked": mask_key(CONFIG["api_key"])}
        print("[web] 配置已更新：has_key={} model={}".format(cfg["has_key"], cfg["model"]))
        return self._send_json({"ok": True, "config": cfg})

    # ------------------------------------------------------------------
    def _handle_run(self):
        body = self._read_json()
        files = body.get("files") or []
        options = body.get("options") or {}
        # 允许本次请求临时携带 Key（不落盘、不回显）；否则用后端已保存的
        transient_key = str(body.get("api_key") or "").strip()
        with CONFIG_LOCK:
            api_key = transient_key or CONFIG["api_key"]
            model = CONFIG["model"]
            base_url = CONFIG["base_url"]
        if transient_key:
            print("[web] 本次运行使用请求携带的 Key（仅内存）")

        self._start_sse()
        self._sse({"type": "log", "stage": 0, "message": "开始运行"})

        def emit(event):
            self._sse(event)

        try:
            result = pipeline.run_pipeline(
                files, api_key=api_key, model=model, base_url=base_url,
                options=options, emit=emit)
            self._sse({"type": "done",
                       "summary": {"run_id": result["run_id"], "total_score": result["score"]["total"]}})
        except Exception as e:                    # noqa: BLE001 - 必须把原因回给前端
            detail = str(e) if str(e) else type(e).__name__
            self._sse({"type": "error", "message": detail})
            if not isinstance(e, (DeepSeekError, pipeline.PipelineError)):
                sys.stderr.write("[web] 运行异常：\n" + traceback.format_exc())
        finally:
            try:
                self.wfile.flush()
            except Exception:                     # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    def _handle_chat(self):
        body = self._read_json()
        history = body.get("messages") or []
        context = (body.get("context") or "").strip()
        with CONFIG_LOCK:
            api_key = str(body.get("api_key") or "").strip() or CONFIG["api_key"]
            model = CONFIG["model"]
            base_url = CONFIG["base_url"]

        self._start_sse()
        if not api_key:
            self._sse({"type": "error", "message": "未配置 API Key：请先在配置栏保存 Key"})
            return
        messages = [{"role": "system", "content": CHAT_SYSTEM}]
        if context:
            messages.append({"role": "system", "content": "本轮结果摘要：\n" + context[:2000]})
        for item in history[-10:]:
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": str(content)[:4000]})
        client = DeepSeekClient(api_key=api_key, base_url=base_url, model=model)
        try:
            for chunk in client.stream_chat(messages, temperature=0.3, max_tokens=600):
                if chunk["type"] == "delta":
                    self._sse({"type": "delta", "content": chunk["content"]})
                else:
                    self._sse({"type": "done", "usage": chunk.get("usage") or {}})
        except DeepSeekError as e:
            self._sse({"type": "error", "message": str(e)})


def main(argv=None):
    argv = argv or sys.argv[1:]
    host, port = HOST, PORT
    for i, arg in enumerate(argv):
        if arg == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
        if arg == "--host" and i + 1 < len(argv):
            host = argv[i + 1]

    probe = rl_core.probe()
    print("=" * 68)
    print(" 交通信号配时 · LLM + RL 本地调试台")
    print("=" * 68)
    print(" RL-API 目录 : {}".format(probe.get("rlapi_home")))
    print(" 冻结模型    : {}".format(probe.get("frozen_path")))
    print(" 依赖状态    : {}".format("OK" if probe.get("ok") else "缺少依赖 -> " +
                                   ", ".join(probe.get("missing") or [])))
    print(" 引擎        : {}".format(probe.get("engine_hint")))
    print("-" * 68)
    print(" 打开浏览器访问： http://{}:{}".format(host, port))
    print(" 停止服务：Ctrl + C")
    print("=" * 68)

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] 已停止")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
