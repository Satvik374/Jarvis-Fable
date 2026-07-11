import tempfile
import unittest
from unittest.mock import Mock

from jarvis.agent.loop import Agent
from jarvis.config import Config


def _agent(brain):
    cfg = Config()
    cfg.data.collect_trajectories = False          # no trajectory files in tests
    cfg.data.trajectory_dir = tempfile.mkdtemp()
    return Agent(brain, cfg)


class LazyPlanningTests(unittest.TestCase):
    def test_direct_attempt_makes_no_planning_call(self):
        brain = Mock()
        plans = _agent(brain)._generate_plans("open notepad", memory="")
        self.assertEqual(len(plans), 1)
        self.assertTrue(plans[0].get("provisional"))
        brain.complete.assert_not_called()          # the whole point: no round-trip

    def test_memory_hit_reuses_learned_plan_without_call(self):
        brain = Mock()
        memory = "- Learned Task: open notepad\n  Successful Plan: X\n"
        plans = _agent(brain)._generate_plans("open notepad", memory=memory)
        self.assertTrue(plans[0].get("from_memory"))
        brain.complete.assert_not_called()

    def test_brainstorm_is_what_calls_the_brain(self):
        brain = Mock()
        brain.complete.return_value = '[{"name": "P1", "description": "d"}]'
        plans = _agent(brain)._brainstorm_plans("open notepad")
        brain.complete.assert_called_once()
        self.assertEqual(plans[0]["name"], "P1")

    def test_brainstorm_falls_back_when_reply_is_junk(self):
        brain = Mock()
        brain.complete.return_value = "not json at all"
        plans = _agent(brain)._brainstorm_plans("open notepad")
        self.assertEqual(plans[0]["name"], "Default Action Path")


if __name__ == "__main__":
    unittest.main()
