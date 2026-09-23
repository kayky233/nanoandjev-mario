"""Supervisor lifecycle checks; no viewer, model server or tunnel is launched."""
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import supervise


class ProcessLivenessTests(unittest.TestCase):
    def test_posix_uses_signal_zero(self):
        with patch.object(supervise.sys, "platform", "darwin"), patch.object(
            supervise.os, "kill"
        ) as kill, patch.object(supervise.subprocess, "run") as run:
            self.assertTrue(supervise._alive(123))
            kill.assert_called_once_with(123, 0)
            run.assert_not_called()

    def test_missing_and_inaccessible_posix_processes(self):
        with patch.object(supervise.sys, "platform", "linux"), patch.object(
            supervise.os, "kill", side_effect=ProcessLookupError
        ):
            self.assertFalse(supervise._alive(123))
        with patch.object(supervise.sys, "platform", "linux"), patch.object(
            supervise.os, "kill", side_effect=PermissionError
        ):
            self.assertTrue(supervise._alive(123))
        self.assertFalse(supervise._alive(0))
        self.assertFalse(supervise._alive(-1))

    def test_windows_matches_pid_column_exactly(self):
        result = subprocess.CompletedProcess([], 0, '"python123.exe","1234","Console"\n')
        with patch.object(supervise.sys, "platform", "win32"), patch.object(
            supervise.subprocess, "run", return_value=result
        ):
            self.assertFalse(supervise._alive(123))
            self.assertTrue(supervise._alive(1234))

    def test_failed_windows_query_does_not_steal_lock(self):
        with patch.object(supervise.sys, "platform", "win32"), patch.object(
            supervise.subprocess, "run", side_effect=FileNotFoundError
        ):
            self.assertTrue(supervise._alive(123))


class LockTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "supervise.lock"
        self.path_patch = patch.object(supervise, "LOCK_FILE", self.path)
        self.path_patch.start()
        supervise._lock_identity = None
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self.path_patch.stop)
        self.addCleanup(supervise.release_lock)

    def test_acquire_and_release(self):
        self.assertTrue(supervise.take_singleton_lock(8123))
        self.assertEqual(self.path.read_text(), str(os.getpid()))
        supervise.release_lock()
        self.assertFalse(self.path.exists())

    def test_live_owner_is_preserved(self):
        self.path.write_text("123")
        with patch.object(supervise, "_alive", return_value=True):
            self.assertFalse(supervise.take_singleton_lock(8123))
        supervise.release_lock()
        self.assertEqual(self.path.read_text(), "123")

    def test_dead_owner_is_reclaimed_even_when_viewer_survives(self):
        self.path.write_text("123")
        with patch.object(supervise, "_alive", return_value=False), patch.object(
            supervise, "port_busy", return_value=True
        ):
            self.assertTrue(supervise.take_singleton_lock(8123))
        self.assertEqual(self.path.read_text(), str(os.getpid()))

    def test_recent_empty_lock_is_not_stolen(self):
        self.path.touch()
        self.assertFalse(supervise.take_singleton_lock(8123))
        self.assertEqual(self.path.read_text(), "")

    def test_old_invalid_lock_is_reclaimed(self):
        self.path.write_text("interrupted write")
        os.utime(self.path, (1, 1))
        with patch.object(supervise, "port_busy", return_value=False):
            self.assertTrue(supervise.take_singleton_lock(8123))

    def test_release_preserves_replacement_lock(self):
        self.assertTrue(supervise.take_singleton_lock(8123))
        replacement = self.path.with_suffix(".replacement")
        replacement.write_text("456")
        replacement.replace(self.path)
        supervise.release_lock()
        self.assertEqual(self.path.read_text(), "456")


class ChildLifecycleTests(unittest.TestCase):
    def setUp(self):
        supervise.stop.clear()
        self.addCleanup(supervise.stop.clear)

    def test_shutdown_reaps_child(self):
        child = Mock()
        child.poll.return_value = None
        supervise.terminate_child(child)
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=5)
        child.kill.assert_not_called()

    def test_shutdown_kills_child_that_ignores_termination(self):
        child = Mock()
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("viewer", 5), 0]
        supervise.terminate_child(child)
        child.kill.assert_called_once_with()
        self.assertEqual(child.wait.call_count, 2)

    def test_supervisor_shutdown_closes_output_and_reaps_child(self):
        child = Mock()
        child.poll.return_value = None
        child.stdout = io.StringIO("")
        with patch.object(supervise.subprocess, "Popen", return_value=child), patch.object(
            supervise.stop, "wait", side_effect=lambda *_: supervise.stop.set()
        ):
            supervise.supervise("viewer", ["python", "viewer.py"])
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=5)
        self.assertTrue(child.stdout.closed)

    def test_three_immediate_failures_stop_restarting(self):
        children = []

        def exited_child(*args, **kwargs):
            child = Mock()
            child.poll.return_value = child.returncode = 1
            child.stdout = io.StringIO("")
            children.append(child)
            return child

        with patch.object(supervise.subprocess, "Popen", side_effect=exited_child), patch.object(
            supervise.time, "perf_counter", side_effect=[0, 0.1, 1, 1.1, 2, 2.1]
        ), patch.object(supervise.stop, "wait"):
            supervise.supervise("viewer", ["python", "viewer.py"])
        self.assertEqual(len(children), 3)
        self.assertTrue(all(child.stdout.closed for child in children))


class MainTests(unittest.TestCase):
    def setUp(self):
        supervise.stop.clear()
        self.addCleanup(supervise.stop.clear)

    def test_replay_is_forwarded_without_starting_tunnel(self):
        launched = []

        def capture(name, argv, *args):
            launched.append((name, argv))
            supervise.stop.set()

        with patch.object(supervise.sys, "argv", ["supervise.py", "--replay-only"]), patch.object(
            supervise.signal, "signal"
        ), patch.object(supervise, "take_singleton_lock", return_value=True), patch.object(
            supervise, "release_lock"
        ) as release, patch.object(supervise, "port_busy", return_value=False), patch.object(
            supervise, "supervise", side_effect=capture
        ):
            supervise.main()
        self.assertEqual(len(launched), 1)
        self.assertEqual(launched[0][0], "viewer")
        self.assertIn("--replay-only", launched[0][1])
        release.assert_called_once_with()

    def test_model_only_is_forwarded_to_dual_viewer(self):
        launched = []

        def capture(name, argv, *args):
            launched.append((name, argv))
            supervise.stop.set()

        with patch.object(supervise.sys, "argv", [
            "supervise.py", "--model-only", "--tunnel", "none"
        ]), patch.object(supervise.signal, "signal"), patch.object(
            supervise, "take_singleton_lock", return_value=True
        ), patch.object(supervise, "release_lock"), patch.object(
            supervise, "port_busy", return_value=False
        ), patch.object(supervise, "supervise", side_effect=capture):
            supervise.main()
        self.assertEqual(len(launched), 1)
        name, argv = launched[0]
        self.assertEqual(name, "viewer")
        self.assertIn("--model-only", argv)
        self.assertEqual(argv[argv.index("--model") + 1], "dual")

    def test_model_only_rejects_route_and_coach_options(self):
        for option in ("--replay-only", "--verified-route", "--web-search"):
            with self.subTest(option=option), patch.object(supervise.sys, "argv", [
                "supervise.py", "--model-only", option
            ]), patch("sys.stderr", new_callable=io.StringIO), patch.object(
                supervise, "take_singleton_lock"
            ) as take_lock, self.assertRaises(SystemExit) as error:
                supervise.main()
            self.assertEqual(error.exception.code, 2)
            take_lock.assert_not_called()

    def test_replay_rejects_explicit_public_tunnel(self):
        with patch.object(supervise.sys, "argv", [
            "supervise.py", "--replay-only", "--tunnel", "ngrok"
        ]), patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
            supervise.main()
        self.assertEqual(error.exception.code, 2)

    def test_all_failed_workers_exit_with_error_and_release_lock(self):
        with patch.object(supervise.sys, "argv", ["supervise.py", "--tunnel", "none"]), patch.object(
            supervise.signal, "signal"
        ), patch.object(supervise, "take_singleton_lock", return_value=True), patch.object(
            supervise, "release_lock"
        ) as release, patch.object(supervise, "port_busy", return_value=False), patch.object(
            supervise, "supervise"
        ), self.assertRaises(SystemExit) as error:
            supervise.main()
        self.assertEqual(error.exception.code, 1)
        release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
