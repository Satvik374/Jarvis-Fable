import unittest

from jarvis.config import Config
from jarvis.perception.elements import Observation
from jarvis.tools import registry, system
from jarvis.tools.schema import ACTIONS_BY_NAME


class SystemStatusTests(unittest.TestCase):
    def test_reports_core_metrics(self):
        out = system.system_status()
        self.assertIn("os:", out)
        # At least one psutil metric should land on this machine.
        self.assertTrue(any(k in out for k in ("cpu:", "memory:", "disk:")))

    def test_action_is_wired_end_to_end(self):
        # schema knows it, registry can execute it, no screen needed.
        self.assertIn("system_status", ACTIONS_BY_NAME)
        obs = Observation(elements=[], screen_size=(1920, 1080))
        result = registry.execute("system_status", {}, obs=obs, cfg=Config())
        self.assertTrue(result.ok)
        self.assertFalse(result.needs_observe)
        self.assertIn("os:", result.message)

    def test_every_action_has_a_handler(self):
        # Guards against schema/registry drift when actions are added.
        for name in ACTIONS_BY_NAME:
            self.assertIn(name, registry._HANDLERS, f"{name} has no handler")


if __name__ == "__main__":
    unittest.main()
