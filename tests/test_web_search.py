import sys
import types
import unittest
from unittest.mock import Mock, patch

from jarvis.tools import system


def _json_response(payload):
    r = Mock()
    r.json.return_value = payload
    return r


class _FakeDDGS:
    """Stand-in for ddgs.DDGS - a context manager whose .text() returns rows."""
    rows: list = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def text(self, query, max_results=5):
        return self.rows[:max_results]


def _install_fake_ddgs(rows):
    """Put a fake `ddgs` module in sys.modules so `from ddgs import DDGS` works."""
    _FakeDDGS.rows = rows
    mod = types.ModuleType("ddgs")
    mod.DDGS = _FakeDDGS
    return patch.dict(sys.modules, {"ddgs": mod})


class WebSearchTests(unittest.TestCase):
    def test_empty_query(self):
        self.assertIn("needs a query", system.web_search(""))

    @patch("requests.get")
    def test_instant_answer_shows(self, get):
        get.return_value = _json_response({
            "AbstractText": "Paris is the capital of France.",
            "AbstractSource": "Wikipedia",
        })
        with _install_fake_ddgs([]):
            out = system.web_search("capital of france")
        self.assertIn("Answer: Paris is the capital of France.", out)
        self.assertIn("Wikipedia", out)

    @patch("requests.get")
    def test_web_results_from_ddgs(self, get):
        get.return_value = _json_response({"AbstractText": ""})   # no instant answer
        rows = [{"title": "Paris - Wikipedia", "href": "https://en.wikipedia.org/wiki/Paris",
                 "body": "Paris is the capital and most populous city of France."}]
        with _install_fake_ddgs(rows):
            out = system.web_search("paris", max_results=3)
        self.assertIn("Paris - Wikipedia", out)
        self.assertIn("https://en.wikipedia.org/wiki/Paris", out)
        self.assertIn("most populous city", out)

    @patch("requests.get")
    def test_no_results_message(self, get):
        get.return_value = _json_response({"AbstractText": ""})
        with _install_fake_ddgs([]):
            self.assertIn("no results", system.web_search("zzzznotathing"))

    @patch("requests.get")
    def test_missing_library_message(self, get):
        get.return_value = _json_response({"AbstractText": ""})
        # Hide both client libs so the import fails -> helpful install hint.
        with patch.dict(sys.modules, {"ddgs": None, "duckduckgo_search": None}):
            out = system.web_search("anything")
        self.assertIn("pip install ddgs", out)


if __name__ == "__main__":
    unittest.main()
