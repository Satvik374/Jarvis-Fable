import unittest

from jarvis.utils.voice import _EnergyVAD


def _run(levels, silence_after=0.6, chunk=0.1, cap_seconds=12.0):
    """Feed levels through the VAD; return (started?, seconds recorded, hit_cap)."""
    det = _EnergyVAD(silence_after, chunk)
    recorded = 0.0
    for lv in levels:
        state = det.feed(lv)
        if state == "listening":
            continue
        recorded += chunk
        if state == "stop":
            return det.started, recorded, False
        if recorded >= cap_seconds:
            return det.started, recorded, True
    return det.started, recorded, False


class EnergyVADTests(unittest.TestCase):
    def test_steady_fan_noise_never_starts(self):
        # Loud, slightly fluctuating fan noise and no speech: must stay idle.
        levels = [500, 560, 480, 620, 540, 500, 590, 470, 610, 530] * 4
        started, recorded, _ = _run(levels)
        self.assertFalse(started)
        self.assertEqual(recorded, 0.0)

    def test_brief_fan_surge_is_ignored(self):
        # A 2-chunk surge above the floor is shorter than START_HOLD (3) -> no.
        levels = [500] * 8 + [2000, 2000] + [500] * 10
        started, _, _ = _run(levels)
        self.assertFalse(started)

    def test_speech_over_fan_starts_and_stops(self):
        # Same fan floor (~500), then real speech, then back to fan noise.
        levels = [500] * 8 + [3000] * 10 + [500] * 20
        started, recorded, hit_cap = _run(levels)
        self.assertTrue(started)
        self.assertFalse(hit_cap)
        self.assertLess(recorded, 2.5)

    def test_stops_soon_after_speech_ends(self):
        levels = [80] * 8 + [3000] * 10 + [80] * 20
        started, recorded, hit_cap = _run(levels)
        self.assertTrue(started)
        self.assertFalse(hit_cap)
        self.assertLess(recorded, 2.0)

    def test_background_rises_after_warmup_still_stops(self):
        # Warmup on a quiet room, but background after speaking is higher.
        # Speech-relative stop still fires (floor-based stop used to wedge here).
        levels = [80] * 8 + [4000] * 8 + [300] * 40
        started, recorded, hit_cap = _run(levels)
        self.assertTrue(started)
        self.assertFalse(hit_cap)
        self.assertLess(recorded, 2.5)

    def test_silence_only_never_starts(self):
        started, recorded, _ = _run([90] * 50)
        self.assertFalse(started)
        self.assertEqual(recorded, 0.0)


if __name__ == "__main__":
    unittest.main()
