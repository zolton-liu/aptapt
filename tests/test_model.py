from __future__ import annotations

import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from qfa_agent.model import HouseEndpointModel, ModelError, model_from_environment


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, limit):
        del limit
        return json.dumps(
            {
                "choices": [{"message": {"content": '{"tool":"finish","arguments":{}}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ).encode()


class HouseEndpointTests(unittest.TestCase):
    def test_normalises_official_origin_and_sends_bearer(self) -> None:
        model = HouseEndpointModel("http://model:8443", "house-pin", 7, "secret")
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _Response()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen), patch.dict(
            os.environ, {"QFA_MAX_RESPONSE_TOKENS": "9000"}, clear=False
        ):
            reply = model.complete([{"role": "user", "content": "hello"}], timeout_sec=12)

        request = captured["request"]
        self.assertEqual(request.full_url, "http://model:8443/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer secret")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "house-pin")
        self.assertEqual(payload["max_tokens"], 4000)
        self.assertEqual(reply.input_tokens, 10)

    def test_accepts_local_v1_base(self) -> None:
        model = HouseEndpointModel("http://localhost:11434/v1", "local", 0)
        self.assertEqual(model.endpoint, "http://localhost:11434/v1/chat/completions")

    def test_retries_one_transient_http_error(self) -> None:
        model = HouseEndpointModel("http://localhost:11434/v1", "local", 0)
        transient = urllib.error.HTTPError(model.endpoint, 503, "busy", {}, None)
        with patch("urllib.request.urlopen", side_effect=[transient, _Response()]) as mocked:
            reply = model.complete([{"role": "user", "content": "hello"}], timeout_sec=12)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(reply.output_tokens, 5)

    def test_retries_two_transient_connection_errors(self) -> None:
        model = HouseEndpointModel("http://localhost:11434/v1", "local", 0)
        transient = urllib.error.URLError("temporarily unavailable")
        with patch(
            "urllib.request.urlopen", side_effect=[transient, transient, _Response()]
        ) as mocked, patch("time.sleep"):
            reply = model.complete([{"role": "user", "content": "hello"}], timeout_sec=12)
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(reply.input_tokens, 10)

    def test_restricted_runtime_requires_token(self) -> None:
        values = {
            "MODEL_ENDPOINT": "http://model:8443",
            "MODEL_NAME": "house-pin",
            "QFBENCH_NETWORK": "restricted",
        }
        with patch.dict(os.environ, values, clear=True):
            with self.assertRaisesRegex(ModelError, "MODEL_TOKEN"):
                model_from_environment()


if __name__ == "__main__":
    unittest.main()
