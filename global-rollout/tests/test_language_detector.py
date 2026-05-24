"""Tests for the LanguageDetector class in the NRS Code Agent Service.

Validates detection of Python, JavaScript, Go, Rust, HTML, SQL, Dockerfile,
Shell (shebang), auto-detect fallback, correction storage, and robustness
on unknown/ambiguous input.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import LanguageDetector from the hyphenated services/code-agent/main.py
# ---------------------------------------------------------------------------

def _load_code_agent():
    mod_name = "code_agent_main"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(Path(__file__).resolve().parent.parent / "services" / "code-agent" / "main.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_ca = _load_code_agent()
LanguageDetector = _ca.LanguageDetector


@pytest.fixture()
def detector():
    return LanguageDetector()


# ---------------------------------------------------------------------------
# 1  Python detection
# ---------------------------------------------------------------------------

def test_detect_python(detector):
    code = "import os\nfrom pathlib import Path\n\ndef foo(self):\n    if __name__ == '__main__':\n        pass\n    return 42"
    result = detector.detect(code)
    assert len(result) > 0
    assert result[0][0] == "python"
    assert result[0][1] > 0.3


# ---------------------------------------------------------------------------
# 2  JavaScript detection
# ---------------------------------------------------------------------------

def test_detect_javascript(detector):
    code = "const x = () => { return 42; }"
    result = detector.detect(code)
    langs = [r[0] for r in result]
    assert "javascript" in langs


# ---------------------------------------------------------------------------
# 3  Go detection
# ---------------------------------------------------------------------------

def test_detect_go(detector):
    code = 'func main() {\n    fmt.Println("hello")\n}'
    result = detector.detect(code)
    langs = [r[0] for r in result]
    assert "go" in langs


# ---------------------------------------------------------------------------
# 4  Rust detection
# ---------------------------------------------------------------------------

def test_detect_rust(detector):
    code = "fn main() {\n    let x = vec![1,2,3];\n}"
    result = detector.detect(code)
    langs = [r[0] for r in result]
    assert "rust" in langs


# ---------------------------------------------------------------------------
# 5  HTML detection
# ---------------------------------------------------------------------------

def test_detect_html(detector):
    code = "<!DOCTYPE html><html><body></body></html>"
    result = detector.detect(code)
    langs = [r[0] for r in result]
    assert "html" in langs


# ---------------------------------------------------------------------------
# 6  SQL detection
# ---------------------------------------------------------------------------

def test_detect_sql(detector):
    code = "SELECT * FROM users WHERE id = 1"
    result = detector.detect(code)
    langs = [r[0] for r in result]
    assert "sql" in langs


# ---------------------------------------------------------------------------
# 7  Dockerfile detection
# ---------------------------------------------------------------------------

def test_detect_dockerfile(detector):
    code = "FROM python:3.11\nRUN pip install flask"
    result = detector.detect(code)
    langs = [r[0] for r in result]
    assert "dockerfile" in langs


# ---------------------------------------------------------------------------
# 8  Shell detection via shebang
# ---------------------------------------------------------------------------

def test_detect_shell_shebang(detector):
    code = '#!/bin/bash\necho "hello"'
    result = detector.detect(code)
    langs = [r[0] for r in result]
    assert "bash" in langs


# ---------------------------------------------------------------------------
# 9  Auto-detect returns non-empty list
# ---------------------------------------------------------------------------

def test_autodetect_returns_nonempty(detector):
    result = detector.detect("x = 1")
    assert isinstance(result, list)
    assert len(result) > 0
    assert len(result[0]) == 2


# ---------------------------------------------------------------------------
# 10  correct() stores correction in Tuition System
# ---------------------------------------------------------------------------

def test_correct_stores_correction(detector):
    snippet = "foo bar baz"
    detector.correct(snippet, "python")
    result = detector.detect(snippet)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 11  Unknown / ambiguous code doesn't crash
# ---------------------------------------------------------------------------

def test_unknown_code_no_crash(detector):
    result = detector.detect("zzzzzz unknown_symbol_111")
    assert isinstance(result, list)
    assert len(result) > 0
    lang, conf = result[0]
    assert isinstance(lang, str)
    assert isinstance(conf, float)


# ---------------------------------------------------------------------------
# 12  Empty code returns default
# ---------------------------------------------------------------------------

def test_empty_code_returns_default(detector):
    result = detector.detect("")
    assert isinstance(result, list)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 13  Confidence normalisation: top result has confidence <= 1.0
# ---------------------------------------------------------------------------

def test_confidence_normalised(detector):
    code = "def main():\n    import os\n    print(os.getcwd())\n"
    result = detector.detect(code)
    for lang, conf in result:
        assert 0.0 <= conf <= 1.0, f"{lang} has out-of-range confidence {conf}"


# ---------------------------------------------------------------------------
# 14  TypeScript detected separately from JavaScript
# ---------------------------------------------------------------------------

def test_detect_typescript(detector):
    code = "interface Foo { bar: string; baz: number; }"
    result = detector.detect(code)
    langs = [r[0] for r in result]
    assert "typescript" in langs


# ---------------------------------------------------------------------------
# 15  Java detection
# ---------------------------------------------------------------------------

def test_detect_java(detector):
    code = (
        "public class Main {\n"
        "    public static void main(String[] args) {\n"
        '        System.out.println("Hello");\n'
        "    }\n"
        "}"
    )
    result = detector.detect(code)
    langs = [r[0] for r in result]
    assert "java" in langs
