"""Bug: Jarvis clicked a prompt box, saw the element list unchanged, decided
the click failed, and re-clicked forever. A focus-click on an editable field
must be recognised as success even though the UIA menu never changes."""

import tempfile
import unittest

from jarvis.agent.loop import Agent
from jarvis.agent.prompts import Decision
from jarvis.config import Config
from jarvis.perception.elements import Element, Observation


def _agent():
    cfg = Config()
    cfg.data.collect_trajectories = False
    cfg.data.trajectory_dir = tempfile.mkdtemp()
    return Agent(object(), cfg)


def _obs():
    return Observation(
        elements=[
            Element(0, "Edit", "Message", (10, 10, 200, 40), (105, 25)),
            Element(1, "Button", "Send", (210, 10, 260, 40), (235, 25)),
        ],
        screen_size=(1920, 1080),
    )


def _click(**args):
    return Decision(thought="", action="click", args=args)


class ClickedEditableTests(unittest.TestCase):
    def setUp(self):
        self.agent, self.obs = _agent(), _obs()

    def test_click_on_edit_is_editable(self):
        self.assertTrue(self.agent._clicked_editable(_click(element=0), self.obs))

    def test_click_on_button_is_not(self):
        self.assertFalse(self.agent._clicked_editable(_click(element=1), self.obs))

    def test_raw_coordinate_click_is_not(self):
        self.assertFalse(self.agent._clicked_editable(_click(x=105, y=25), self.obs))

    def test_missing_element_id_is_not(self):
        self.assertFalse(self.agent._clicked_editable(_click(element=99), self.obs))

    def test_non_click_action_is_not(self):
        d = Decision(thought="", action="type", args={"text": "hi"})
        self.assertFalse(self.agent._clicked_editable(d, self.obs))


if __name__ == "__main__":
    unittest.main()
