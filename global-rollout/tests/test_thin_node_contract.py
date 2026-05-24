from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

from clients.bootstrap import BootstrapConfig, NRSIPClientBootstrap, Platform, SessionState


class _DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _DummyHttpClient:
    def __init__(self):
        self.calls: list[dict] = []

    def post(self, url: str, json: dict):
        self.calls.append({"url": url, "json": json})
        return _DummyResponse({"status": "ok", "echo": json})


def _load_edge_gateway_module():
    edge_path = Path(__file__).resolve().parents[1] / "services" / "edge-gateway" / "main.py"
    spec = importlib.util.spec_from_file_location("edge_gateway_main", edge_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestEdgeGatewayRouteContract(unittest.TestCase):
    def test_edge_exposes_unified_health_and_stream_routes(self):
        module = _load_edge_gateway_module()
        route_paths = {route.path for route in module.app.routes}
        self.assertIn("/health", route_paths)
        self.assertIn("/healthz", route_paths)
        self.assertIn("/v1/process-stream", route_paths)
        self.assertIn("/v1/stream", route_paths)
        self.assertIn("/v1/nodes", route_paths)
        self.assertIn("/v1/nodes/{node_id}", route_paths)
        self.assertIn("/v1/network/summary", route_paths)
        self.assertIn("/v1/redteam/report", route_paths)
        self.assertIn("/v1/redteam/stats", route_paths)


class TestBootstrapPacketContract(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="nrs-thin-node-")
        config = BootstrapConfig(
            edge_gateway_url="https://edge.example.test",
            platform=Platform.SERVER,
            region="us-central1",
            credential_store_kwargs={"store_dir": self.temp_dir},
        )
        self.client = NRSIPClientBootstrap(config)
        self.identity = self.client.generate_identity()
        self.client.session = SessionState(
            session_id="sess-1234",
            key_id="key-1234",
            selected_bundle="MLKEM1024_X25519_DILITHIUM3",
            expires_at_ms=9_999_999_999_999,
            signing_key="signing-key",
        )
        self.client._http = _DummyHttpClient()

    def test_query_packets_use_edge_compatible_tag_and_ttl(self):
        msg = self.client._build_message({"query": "hello", "domain": "general"})
        packet = self.client._build_packet_dict(msg)
        expected_tag = hashlib.sha256(
            f"{packet['version']}:{packet['sequence']}:{packet['nonce']}:{packet['ciphertext']}".encode("utf-8")
        ).hexdigest()
        self.assertEqual(packet["ttl_hops"], 8)
        self.assertEqual(packet["tag"], expected_tag)

    def test_media_requests_include_worker_root_fields(self):
        self.client.send_media(
            media_type="image/png",
            media_data=b"fake-image",
            domain="vision",
            mode="DETERMINISTIC",
        )
        call = self.client._http.calls[-1]
        self.assertTrue(call["url"].endswith("/v1/media"))
        self.assertEqual(call["json"]["packet_type"], "media")
        self.assertEqual(call["json"]["media_type"], "image/png")
        self.assertEqual(call["json"]["domain"], "vision")
        self.assertEqual(call["json"]["mode_override"], "DETERMINISTIC")

    def test_stream_requests_use_edge_stream_route(self):
        self.client.send_stream(
            stream_id="stream-1",
            segment_index=7,
            media_type="audio/wav",
            media_data=b"fake-audio",
            domain="audio",
            mode="HYBRID",
        )
        call = self.client._http.calls[-1]
        self.assertTrue(call["url"].endswith("/v1/process-stream"))
        self.assertEqual(call["json"]["packet_type"], "stream")
        self.assertEqual(call["json"]["stream_id"], "stream-1")
        self.assertEqual(call["json"]["segment_index"], 7)
        self.assertEqual(call["json"]["domain"], "audio")
        self.assertEqual(call["json"]["mode_override"], "HYBRID")


if __name__ == "__main__":
    unittest.main()
