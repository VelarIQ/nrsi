"""Tests for NRS Code Intelligence Engine — language detection, learning, verification."""

import os
import tempfile
import shutil
import pytest

from nrsip.code_intelligence import (
    CodeLearningEngine, _SEED_EXTENSIONS, _CONTENT_FINGERPRINTS,
    _detect_naming, _detect_architecture,
)


@pytest.fixture
def engine():
    tmp = tempfile.mkdtemp(prefix="nrs-test-ci-")
    eng = CodeLearningEngine(data_dir=tmp)
    yield eng
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "main.py").write_text(
        "import os\nfrom utils import helper\n\n"
        "class AppService:\n    def start(self):\n        return True\n\n"
        "    def stop(self):\n        return False\n\n"
        "def main():\n    svc = AppService()\n    svc.start()\n"
    )
    (tmp_path / "utils.py").write_text(
        "def helper(x):\n    return x * 2\n\ndef format_output(data):\n    return str(data)\n"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text(
        "import pytest\nfrom main import AppService\n\n"
        "def test_start():\n    svc = AppService()\n    assert svc.start() is True\n"
    )
    (tmp_path / "package.json").write_text('{"name": "test", "version": "1.0.0"}')
    (tmp_path / "app.js").write_text(
        "const express = require('express');\n"
        "function handleRequest(req, res) { res.send('ok'); }\n"
        "module.exports = { handleRequest };\n"
    )
    return tmp_path


class TestLanguageDetection:
    def test_seed_extension_python(self, engine):
        lang, conf, method = engine.detect_language("app.py")
        assert lang == "python"
        assert conf == 1.0
        assert method == "seed_extension"

    def test_seed_extension_rust(self, engine):
        lang, conf, method = engine.detect_language("main.rs")
        assert lang == "rust"

    def test_seed_extension_go(self, engine):
        lang, conf, method = engine.detect_language("server.go")
        assert lang == "go"

    def test_shebang_detection(self, engine):
        lang, conf, method = engine.detect_language(
            "script.xyz", "#!/usr/bin/env python3\nprint('hello')"
        )
        assert lang == "python"
        assert method == "shebang"
        assert conf >= 0.9

    def test_content_fingerprint_rust(self, engine):
        lang, conf, method = engine.detect_language(
            "code.xxx", '#[derive(Debug)]\nfn main() { println!("hello"); }'
        )
        assert lang == "rust"
        assert method == "content_fingerprint"

    def test_content_fingerprint_php(self, engine):
        lang, conf, method = engine.detect_language(
            "page.unknown", '<?php\necho "Hello World";\n?>'
        )
        assert lang == "php"

    def test_content_fingerprint_elixir(self, engine):
        lang, conf, method = engine.detect_language(
            "mod.abc", "defmodule MyApp do\n  def hello, do: :world\nend"
        )
        assert lang == "elixir"

    def test_special_filename_dockerfile(self, engine):
        lang, conf, method = engine.detect_language("Dockerfile")
        assert lang == "dockerfile"
        assert method == "special_filename"

    def test_special_filename_makefile(self, engine):
        lang, conf, method = engine.detect_language("Makefile")
        assert lang == "make"

    def test_learns_new_extension(self, engine):
        engine.detect_language("script.xyz", "#!/usr/bin/env python3\nprint('hi')")
        assert ".xyz" in engine._learned_extensions
        assert engine._learned_extensions[".xyz"] == "python"

    def test_unknown_returns_gracefully(self, engine):
        lang, conf, method = engine.detect_language("data.bin", "")
        assert lang == "unknown"
        assert conf == 0.0

    def test_seed_extensions_count(self):
        assert len(_SEED_EXTENSIONS) > 200

    def test_content_fingerprints_count(self):
        assert len(_CONTENT_FINGERPRINTS) > 40


class TestRepoScan:
    def test_scan_finds_files(self, engine, sample_repo):
        profile = engine.scan_repo(str(sample_repo))
        assert profile.total_files >= 4
        assert profile.total_lines > 0

    def test_scan_detects_languages(self, engine, sample_repo):
        profile = engine.scan_repo(str(sample_repo))
        assert "python" in profile.languages
        assert "javascript" in profile.languages

    def test_scan_detects_architecture(self, engine, sample_repo):
        profile = engine.scan_repo(str(sample_repo))
        assert "node_project" in profile.architecture_patterns

    def test_scan_detects_naming(self, engine, sample_repo):
        profile = engine.scan_repo(str(sample_repo))
        assert profile.naming_conventions.get("functions") in ("snake_case", "camelCase", "lowercase")

    def test_scan_builds_dependency_graph(self, engine, sample_repo):
        profile = engine.scan_repo(str(sample_repo))
        assert len(profile.dependency_graph) > 0

    def test_scan_detects_test_patterns(self, engine, sample_repo):
        profile = engine.scan_repo(str(sample_repo))
        assert profile.test_patterns.get("naming") == "test_prefix"

    def test_scan_persists_conventions(self, engine, sample_repo):
        engine.scan_repo(str(sample_repo))
        assert engine._repos_scanned == 1
        assert os.path.exists(os.path.join(engine._data_dir, "global_conventions.json"))


class TestStructureExtraction:
    def test_python_functions(self, engine):
        code = "def foo():\n    pass\n\ndef bar(x):\n    return x\n"
        analysis = engine.extract_structure(code, "python")
        assert "foo" in analysis.functions
        assert "bar" in analysis.functions

    def test_python_classes(self, engine):
        code = "class MyClass:\n    pass\n\nclass OtherClass:\n    pass\n"
        analysis = engine.extract_structure(code, "python")
        assert "MyClass" in analysis.classes
        assert "OtherClass" in analysis.classes

    def test_python_imports(self, engine):
        code = "import os\nfrom pathlib import Path\n"
        analysis = engine.extract_structure(code, "python")
        assert "os" in analysis.imports
        assert "pathlib" in analysis.imports

    def test_javascript_functions(self, engine):
        code = "function handleRequest(req) { return req; }\nconst process = (x) => x;\n"
        analysis = engine.extract_structure(code, "javascript")
        assert "handleRequest" in analysis.functions

    def test_universal_extraction_unknown_lang(self, engine):
        code = "func hello() {\n  print(\"hi\")\n}\nclass Widget {\n}\n"
        analysis = engine.extract_structure(code, "some_unknown_lang")
        assert len(analysis.functions) > 0 or len(analysis.classes) > 0


class TestChangeVerification:
    def test_snapshot_and_verify_safe(self, engine):
        files = {"app.py": "def foo():\n    return 1\n"}
        sid = engine.snapshot_before_change(files)
        result = engine.verify_after_change(sid, {"app.py": "def foo():\n    return 2\n"})
        assert result["safe"] is True

    def test_snapshot_detects_removed_function(self, engine):
        files = {"app.py": "def foo():\n    pass\n\ndef bar():\n    pass\n"}
        sid = engine.snapshot_before_change(files)
        result = engine.verify_after_change(sid, {"app.py": "def foo():\n    pass\n"})
        assert result["safe"] is False
        assert any(r["type"] == "removed_functions" for r in result["regressions"])

    def test_snapshot_detects_removed_import(self, engine):
        files = {"app.py": "import os\nimport sys\ndef main(): pass\n"}
        sid = engine.snapshot_before_change(files)
        result = engine.verify_after_change(sid, {"app.py": "import os\ndef main(): pass\n"})
        assert any(r["type"] == "removed_imports" for r in result["regressions"])

    def test_snapshot_detects_syntax_error(self, engine):
        files = {"app.py": "def foo():\n    return 1\n"}
        sid = engine.snapshot_before_change(files)
        result = engine.verify_after_change(sid, {"app.py": "def foo(\n    return 1\n"})
        assert any(r["type"] == "syntax_error" for r in result["regressions"])

    def test_snapshot_not_found(self, engine):
        result = engine.verify_after_change("nonexistent", {"a.py": ""})
        assert "error" in result


class TestTestGeneration:
    def test_python_test_gen(self, engine):
        code = "def add(a, b):\n    return a + b\n\ndef greet():\n    return 'hello'\n"
        tests = engine.generate_tests(code, "python", "math_utils")
        assert "test_add" in tests
        assert "test_greet" in tests
        assert "assert True" not in tests
        assert "callable" in tests

    def test_js_test_gen(self, engine):
        code = "export function fetchData(url) { return fetch(url); }\n"
        tests = engine.generate_tests(code, "javascript", "api")
        assert "describe" in tests
        assert "fetchData" in tests

    def test_go_test_gen(self, engine):
        code = "package main\n\nfunc Hello() string { return \"hi\" }\n"
        tests = engine.generate_tests(code, "go", "main")
        assert "TestHello" in tests
        assert "testing" in tests

    def test_rust_test_gen(self, engine):
        code = "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
        tests = engine.generate_tests(code, "rust", "math")
        assert "test_add" in tests
        assert "#[test]" in tests

    def test_universal_test_gen(self, engine):
        code = "func doStuff() { }\nclass Widget { }\n"
        tests = engine.generate_tests(code, "some_lang", "module")
        assert "TEST" in tests or "test" in tests.lower()


class TestNamingDetection:
    def test_snake_case(self):
        result = _detect_naming(["get_user", "set_value", "process_data"])
        assert result["dominant_style"] == "snake_case"

    def test_camel_case(self):
        result = _detect_naming(["getUser", "setValue", "processData"])
        assert result["dominant_style"] == "camelCase"

    def test_pascal_case(self):
        result = _detect_naming(["GetUser", "SetValue", "ProcessData"])
        assert result["dominant_style"] == "PascalCase"

    def test_empty(self):
        result = _detect_naming([])
        assert result == {}


class TestPersistence:
    def test_conventions_persist_and_reload(self, engine, sample_repo):
        engine.scan_repo(str(sample_repo))
        data_dir = engine._data_dir

        engine2 = CodeLearningEngine(data_dir=data_dir)
        assert len(engine2._global_conventions) > 0

    def test_learned_languages_persist(self):
        tmp = tempfile.mkdtemp(prefix="nrs-test-persist-")
        try:
            eng1 = CodeLearningEngine(data_dir=tmp)
            eng1.detect_language("script.xyz", "#!/usr/bin/env ruby\nputs 'hi'")
            eng1._persist()

            eng2 = CodeLearningEngine(data_dir=tmp)
            assert ".xyz" in eng2._learned_extensions
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
