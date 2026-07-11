import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from jarvis.scheduler import (
    Scheduler, ScheduleError, parse_schedule, next_run, is_recurring)


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class ParseTests(unittest.TestCase):
    def test_interval(self):
        self.assertEqual(parse_schedule("every 30 minutes"), ("interval", 1800, None, None))
        self.assertEqual(parse_schedule("every 2 h")[1], 7200)

    def test_once_and_daily(self):
        self.assertEqual(parse_schedule("in 10 min")[0], "once_in")
        self.assertEqual(parse_schedule("daily at 08:00")[0], "daily")
        self.assertEqual(parse_schedule("daily 8:00"), ("daily", None, 8, 0))
        self.assertEqual(parse_schedule("at 14:30")[0], "once_at")
        self.assertEqual(parse_schedule("14:30")[0], "once_at")

    def test_rejects_garbage(self):
        for bad in ("banana", "every 0 minutes", "at 25:00", "every 5 lightyears"):
            with self.assertRaises(ScheduleError):
                parse_schedule(bad)

    def test_is_recurring(self):
        self.assertTrue(is_recurring("every 5 minutes"))
        self.assertTrue(is_recurring("daily at 08:00"))
        self.assertFalse(is_recurring("in 5 minutes"))
        self.assertFalse(is_recurring("at 08:00"))


class NextRunTests(unittest.TestCase):
    def test_daily_picks_future_time(self):
        now = datetime(2026, 7, 9, 7, 0, 0).timestamp()
        nr = next_run(parse_schedule("daily at 08:00"), now)
        dt = datetime.fromtimestamp(nr)
        self.assertEqual((dt.hour, dt.minute), (8, 0))
        self.assertGreater(nr, now)

    def test_time_already_passed_rolls_to_tomorrow(self):
        now = datetime(2026, 7, 9, 7, 0, 0).timestamp()
        nr = next_run(parse_schedule("at 06:00"), now)
        self.assertEqual(datetime.fromtimestamp(nr).day, 10)

    def test_interval_is_now_plus_interval(self):
        self.assertEqual(next_run(parse_schedule("every 1 minutes"), 1000.0), 1060.0)


class RunDueTests(unittest.TestCase):
    def _sched(self, clock):
        path = Path(tempfile.mkdtemp()) / "cron.json"
        calls = []
        s = Scheduler(path, runner=calls.append, clock=clock)
        return s, calls

    def test_fires_due_reschedules_recurring_disables_oneshot(self):
        clock = _Clock(1000.0)
        s, calls = self._sched(clock)
        once = s.add("in 1 minutes", "do A")       # once_in -> next 1060
        repeat = s.add("every 1 minutes", "do B")  # interval -> next 1060

        clock.t = 1059
        self.assertEqual(s.run_due(), [])          # nothing due yet
        self.assertEqual(calls, [])

        clock.t = 1060
        fired = s.run_due()
        self.assertEqual(len(fired), 2)
        self.assertEqual(calls, ["do A", "do B"])

        jobs = {j.id: j for j in s.jobs()}
        self.assertFalse(jobs[once.id].enabled)    # one-shot done
        self.assertTrue(jobs[repeat.id].enabled)   # recurring stays
        self.assertEqual(jobs[repeat.id].next_run, 1120.0)

    def test_bad_job_does_not_stop_the_others(self):
        clock = _Clock(1000.0)
        path = Path(tempfile.mkdtemp()) / "cron.json"

        def boom(_cmd):
            raise RuntimeError("job blew up")

        s = Scheduler(path, runner=boom, clock=clock)
        j = s.add("every 1 minutes", "explode")
        clock.t = 1060
        fired = s.run_due()                        # must not raise
        self.assertEqual(len(fired), 1)
        self.assertTrue({x.id: x for x in s.jobs()}[j.id].enabled)

    def test_persistence_round_trip(self):
        clock = _Clock(datetime(2026, 7, 9, 7, 0, 0).timestamp())
        path = Path(tempfile.mkdtemp()) / "cron.json"
        s1 = Scheduler(path, runner=lambda c: None, clock=clock)
        j = s1.add("daily at 08:00", "morning brief")
        s2 = Scheduler(path, runner=lambda c: None, clock=clock)
        self.assertIn(j.id, {x.id for x in s2.jobs()})
        self.assertEqual({x.id: x for x in s2.jobs()}[j.id].command, "morning brief")


if __name__ == "__main__":
    unittest.main()
