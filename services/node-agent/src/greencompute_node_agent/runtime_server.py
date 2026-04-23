"""Standalone HTTP server for inference — spawned as a subprocess by ProcessInferenceBackend."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from greencompute_protocol import ChatCompletionRequest, ChatCompletionResponse
from greencompute_node_agent.domain.model_backend import create_text_generation_backend


class RuntimeRequestHandler(BaseHTTPRequestHandler):
    artifact_payload: dict[str, Any] = {}
    deployment_id: str = ""
    hotkey: str = ""
    image: str = ""
    model_backend = None

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/healthz":
            self._send_json({"detail": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json(
            {
                "status": "ok",
                "deployment_id": self.deployment_id,
                "hotkey": self.hotkey,
                "image": self.image,
                "backend": self.model_backend.backend_name,
                "model_identifier": self.model_backend.model_identifier,
            }
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send_json({"detail": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        payload = self._read_payload()
        request_model = ChatCompletionRequest(**payload)
        response = self._build_response(request_model)
        if request_model.stream:
            self._send_stream(response)
            return
        self._send_json(response.model_dump(mode="json"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _read_payload(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.loads(body.decode() or "{}")

    def _build_response(self, payload: ChatCompletionRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            model=payload.model,
            content=self.model_backend.generate_text(payload),
            deployment_id=self.deployment_id,
            routed_hotkey=self.hotkey,
        )

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self, response: ChatCompletionResponse) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        words = response.content.split()
        for index, word in enumerate(words):
            chunk = {
                "id": response.id,
                "object": "chat.completion.chunk",
                "model": response.model,
                "deployment_id": response.deployment_id,
                "routed_hotkey": response.routed_hotkey,
                "choices": [{"index": 0, "delta": {"content": word if index == 0 else f" {word}"}}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        terminal = {
            "id": response.id,
            "object": "chat.completion.chunk",
            "model": response.model,
            "deployment_id": response.deployment_id,
            "routed_hotkey": response.routed_hotkey,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(terminal)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--hotkey", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--payload-path", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.payload_path).read_text(encoding="utf-8"))
    manifest = payload.get("runtime_manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("runtime manifest missing from payload")
    RuntimeRequestHandler.artifact_payload = payload
    RuntimeRequestHandler.deployment_id = args.deployment_id
    RuntimeRequestHandler.hotkey = args.hotkey
    RuntimeRequestHandler.image = args.image
    RuntimeRequestHandler.model_backend = create_text_generation_backend(
        manifest,
        image=args.image,
        allow_fallback=False,
    )

    server = ThreadingHTTPServer(("127.0.0.1", args.port), RuntimeRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
