"""Tests for enterprise services -- identity, code agent, model services."""

import asyncio
import os
import sys
import time
import pytest


class TestIdentityService:
    """Test identity store, RBAC, and audit without starting the HTTP server."""

    _identity_mod = None

    @classmethod
    def _load_identity_module(cls):
        if cls._identity_mod is None:
            import importlib.util, os
            spec = importlib.util.spec_from_file_location(
                "identity_main",
                os.path.join(os.path.dirname(__file__), "..", "services", "identity", "main.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["identity_main"] = mod
            spec.loader.exec_module(mod)
            cls._identity_mod = mod
        return cls._identity_mod

    def _make_store(self):
        mod = self._load_identity_module()
        return mod.IdentityStore(), mod.Permission

    def test_create_org_and_tenant(self):
        store, _ = self._make_store()
        org = store.create_org("TestOrg", domain="test.com")
        assert org.org_id
        assert org.name == "TestOrg"
        tenant = store.create_tenant(org.org_id, "Default")
        assert tenant.tenant_id
        assert tenant.org_id == org.org_id

    def test_create_user_with_api_key(self):
        store, _ = self._make_store()
        org = store.create_org("TestOrg")
        tenant = store.create_tenant(org.org_id, "Default")
        user, api_key = store.create_user(org.org_id, tenant.tenant_id,
                                           "test@test.com", "Test User", ["admin"])
        assert user.user_id
        assert api_key.startswith("nrs_")
        authed = store.authenticate_api_key(api_key)
        assert authed is not None
        assert authed.user_id == user.user_id

    def test_invalid_api_key(self):
        store, _ = self._make_store()
        assert store.authenticate_api_key("nrs_invalid") is None

    def test_rbac_permissions(self):
        store, Permission = self._make_store()
        org = store.create_org("TestOrg")
        tenant = store.create_tenant(org.org_id, "Default")
        admin_user, _ = store.create_user(org.org_id, tenant.tenant_id,
                                           "admin@test.com", "Admin", ["admin"])
        viewer_user, _ = store.create_user(org.org_id, tenant.tenant_id,
                                            "viewer@test.com", "Viewer", ["viewer"])
        assert admin_user.has_permission(Permission.CODE_EXECUTE)
        assert admin_user.has_permission(Permission.ADMIN_WRITE)
        assert not viewer_user.has_permission(Permission.CODE_EXECUTE)
        assert viewer_user.has_permission(Permission.TEXT_READ)

    def test_session_creation_and_validation(self):
        store, _ = self._make_store()
        org = store.create_org("TestOrg")
        tenant = store.create_tenant(org.org_id, "Default")
        user, _ = store.create_user(org.org_id, tenant.tenant_id,
                                     "test@test.com", "Test", ["developer"])
        token = store.create_session(user.user_id)
        assert token
        validated = store.validate_session(token)
        assert validated is not None
        assert validated.user_id == user.user_id

    def test_invalid_session(self):
        store, _ = self._make_store()
        assert store.validate_session("invalid_token") is None

    def test_entitlement_check(self):
        store, _ = self._make_store()
        org = store.create_org("TestOrg")
        tenant = store.create_tenant(org.org_id, "Default", {"text", "image"})
        assert store.check_entitlement(tenant.tenant_id, "text")
        assert store.check_entitlement(tenant.tenant_id, "image")
        assert not store.check_entitlement(tenant.tenant_id, "video")

    def test_audit_logging(self):
        store, _ = self._make_store()
        AuditEvent = self._load_identity_module().AuditEvent
        store.log_audit(AuditEvent(
            org_id="org1", user_id="user1",
            action="test.action", resource="test", result="success",
        ))
        events = store.get_audit_log(org_id="org1")
        assert len(events) == 1
        assert events[0].action == "test.action"

    def test_policy_pack_retrieval(self):
        store, _ = self._make_store()
        org = store.create_org("StrictOrg", policy_pack="strict")
        pack = store.get_policy_pack(org.org_id)
        assert pack["compliance_mode"] == "hipaa"
        assert pack["code_execution"] is False

    def test_audit_export_filtering(self):
        store, _ = self._make_store()
        AuditEvent = self._load_identity_module().AuditEvent
        for i in range(5):
            store.log_audit(AuditEvent(
                org_id="org1", user_id=f"user{i}",
                action="query", resource="nrs",
            ))
        store.log_audit(AuditEvent(
            org_id="org2", user_id="user99",
            action="admin", resource="settings",
        ))
        org1_events = store.get_audit_log(org_id="org1")
        assert len(org1_events) == 5
        all_events = store.get_audit_log()
        assert len(all_events) == 6


_code_agent_mod = None

def _get_code_agent_module():
    global _code_agent_mod
    if _code_agent_mod is None:
        import importlib
        from prometheus_client import REGISTRY
        for name in list(REGISTRY._names_to_collectors.keys()):
            if name.startswith("code_exec"):
                try:
                    REGISTRY.unregister(REGISTRY._names_to_collectors[name])
                except Exception:
                    pass
        code_agent_path = os.path.join(os.path.dirname(__file__), "..", "services", "code-agent")
        spec = importlib.util.spec_from_file_location("code_agent_main_cached",
                                                        os.path.join(code_agent_path, "main.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["code_agent_main_cached"] = mod
        spec.loader.exec_module(mod)
        _code_agent_mod = mod
    return _code_agent_mod


class TestCodeAgentSandbox:
    """Test the code agent sandbox execution."""

    def _make_sandbox(self):
        return _get_code_agent_module().Sandbox()

    @pytest.mark.asyncio
    async def test_python_execution(self):
        sandbox = self._make_sandbox()
        result = await sandbox.execute("print('hello world')", "python", timeout=5)
        assert result["exit_code"] == 0
        assert "hello world" in result["stdout"]

    @pytest.mark.asyncio
    async def test_python_error(self):
        sandbox = self._make_sandbox()
        result = await sandbox.execute("raise ValueError('test')", "python", timeout=5)
        assert result["exit_code"] != 0
        assert "ValueError" in result["stderr"]

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self):
        sandbox = self._make_sandbox()
        result = await sandbox.execute("import time; time.sleep(10)", "python", timeout=2)
        assert result["exit_code"] == -1
        assert "timed out" in result["stderr"].lower()

    @pytest.mark.asyncio
    async def test_shell_execution(self):
        sandbox = self._make_sandbox()
        result = await sandbox.execute("echo hello", "shell", timeout=5)
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]


class TestCodeAgentAnalysis:
    def _make_indexer(self):
        return _get_code_agent_module().PythonASTIndexer()

    def test_index_python_source(self):
        indexer = self._make_indexer()
        source = '''
def hello(name):
    """Greet someone."""
    return f"Hello {name}"

class MyClass:
    """A test class."""
    def method(self):
        pass

X = 42
'''
        symbols = indexer.index_source(source)
        names = {s.name for s in symbols}
        assert "hello" in names
        assert "MyClass" in names
        assert "X" in names

    def test_function_signature(self):
        indexer = self._make_indexer()
        source = "def add(a, b): return a + b"
        symbols = indexer.index_source(source)
        func = [s for s in symbols if s.name == "add"][0]
        assert "a, b" in func.signature

    def test_class_children(self):
        indexer = self._make_indexer()
        source = '''
class Foo:
    def bar(self): pass
    def baz(self): pass
'''
        symbols = indexer.index_source(source)
        cls = [s for s in symbols if s.name == "Foo"][0]
        assert "bar" in cls.children
        assert "baz" in cls.children


class TestCodeReview:
    def _analyze(self, code: str, language: str = "python"):
        return _get_code_agent_module()._analyze_code(code, language)

    def test_basic_analysis(self):
        result = self._analyze("x = 1\ny = 2\nprint(x + y)")
        assert result["language"] == "python"
        assert result["lines"] == 3

    def test_syntax_error_detection(self):
        result = self._analyze("def foo(:\n  pass")
        assert "syntax_error" in result

    def test_counts_symbols(self):
        code = '''
def a(): pass
def b(): pass
class C:
    def d(self): pass
'''
        result = self._analyze(code)
        assert result["functions"] >= 2
        assert result["classes"] >= 1


class TestDiff:
    def _compute_diff(self, original, modified, filepath=""):
        return _get_code_agent_module()._compute_diff(original, modified, filepath)

    def test_simple_diff(self):
        diff = self._compute_diff("line1\nline2\n", "line1\nline3\n")
        assert "-line2" in diff
        assert "+line3" in diff

    def test_no_diff(self):
        diff = self._compute_diff("same\n", "same\n")
        assert diff == ""
