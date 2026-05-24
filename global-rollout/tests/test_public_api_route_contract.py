from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


class TestPlatformApiRouteContract(unittest.TestCase):
    def test_platform_api_mounts_expected_routers(self):
        main_source = _read("services/platform-api/main.py")

        self.assertIn("app.include_router(auth_router)", main_source)
        self.assertIn("app.include_router(brain_router)", main_source)
        self.assertIn("app.include_router(devices_router)", main_source)
        self.assertIn("app.include_router(tools_router)", main_source)
        self.assertIn('app.include_router(usage_router, prefix="/usage", tags=["usage"])', main_source)
        self.assertIn('app.include_router(keys_router, prefix="/keys", tags=["keys"])', main_source)
        self.assertIn('app.include_router(public_v1_router, prefix="/v1", tags=["public"])', main_source)

    def test_platform_api_route_files_expose_expected_paths(self):
        self.assertIn('@router.post("/chat/completions")', _read("services/platform-api/routes/api.py"))
        self.assertIn('@router.get("/models",', _read("services/platform-api/routes/api.py"))
        self.assertIn('@router.post("/embeddings",', _read("services/platform-api/routes/api.py"))
        self.assertIn('@router.post("/signup",', _read("services/platform-api/routes/auth.py"))
        self.assertIn('@router.post("/login",', _read("services/platform-api/routes/auth.py"))
        self.assertIn('@router.get("/me")', _read("services/platform-api/routes/auth.py"))
        self.assertIn('@router.get("/summary",', _read("services/platform-api/routes/usage.py"))
        self.assertIn('@router.post("/personal/save")', _read("services/platform-api/routes/brain.py"))
        self.assertIn('@router.get("",', _read("services/platform-api/routes/tools.py"))
        self.assertIn('@router.post("/register"', _read("services/platform-api/routes/devices.py"))
        self.assertIn('@router.get("",', _read("services/platform-api/routes/keys.py"))


class TestEdgeGatewayWorkerAuthContract(unittest.TestCase):
    def test_worker_facing_routes_call_shared_auth_enforcer(self):
        source = _read("services/edge-gateway/main.py")

        self.assertIn("async def _enforce_capability_auth", source)
        self.assertIn('await _enforce_capability_auth(request, "image", "image:write")', source)
        self.assertIn('await _enforce_capability_auth(request, "text", "text:write")', source)
        self.assertIn('await _enforce_capability_auth(request, "code", "code:generate")', source)


if __name__ == "__main__":
    unittest.main()
