"""Runtime protocol configuration must reach the viewer without stale labels."""

import contextlib
import io
import threading
import unittest
from unittest.mock import Mock, patch

import supervise
import watch_local as viewer


class LayaViewerConfigTests(unittest.TestCase):
    def run_viewer_config(self, mode=None):
        latest = {c: {"meta": {}} for c in viewer.CHANNELS}
        stats = {c: {} for c in viewer.CHANNELS}
        stop = threading.Event()
        stop.set()
        server = Mock()

        class StartupOnlyThread:
            def __init__(self, target, **kwargs):
                self.target = target

            def start(self):
                if self.target.__name__ == "serve":
                    self.target()

        def load_laya_env():
            # Simulate .env being read after play_local's constants were imported.
            viewer.os.environ.update(LOCAL_POLICY_MODEL="english", LOCAL_POLICY_MODE="laya")

        argv = ["watch_local.py", "--model-only", "--model", "dual"]
        if mode is not None:
            argv.extend(["--mode", mode])
        with patch("sys.argv", argv), patch.dict(viewer.os.environ, {}, clear=False), \
             patch.object(viewer.P, "LOCAL_POLICY_MODEL", "old-qwen-model"), \
             patch.object(viewer.P, "load_env", side_effect=load_laya_env), \
             patch.object(viewer, "LATEST", latest), patch.object(viewer, "STATS", stats), \
             patch.object(viewer, "STOP", stop), \
             patch.object(viewer, "ThreadingHTTPServer", return_value=server), \
             patch.object(viewer.threading, "Thread", StartupOnlyThread), \
             patch.object(viewer, "play_episode", side_effect=AssertionError("no game may start")), \
             contextlib.redirect_stdout(io.StringIO()):
            viewer.main()
        server.shutdown.assert_called_once_with()
        return latest

    def test_loaded_laya_config_labels_local_lane_and_preserves_jev(self):
        latest = self.run_viewer_config()
        self.assertEqual(latest["local"]["meta"]["mode"], "laya")
        self.assertEqual(latest["local"]["meta"]["model_name"], "Laya (english)")
        self.assertEqual(latest["jev"]["meta"]["model_name"], "Jev (jev-latest)")
        for channel in viewer.CHANNELS:
            self.assertTrue(latest[channel]["meta"]["model_only"])
            self.assertTrue(latest[channel]["meta"]["enabled"])

    def test_explicit_legacy_protocol_is_not_mislabeled_laya(self):
        for mode in ("chat", "score"):
            with self.subTest(mode=mode):
                latest = self.run_viewer_config(mode)
                self.assertEqual(latest["local"]["meta"]["mode"], mode)
                self.assertEqual(latest["local"]["meta"]["model_name"], "english")
                self.assertEqual(latest["jev"]["meta"]["model_name"], "Jev (jev-latest)")

    def test_frame_publish_keeps_laya_identity(self):
        from PIL import Image
        import numpy as np

        latest = {"local": {"seq": 0, "meta": {
            "model_name": "Laya (english)", "model_only": True, "enabled": True,
        }}}
        with patch.object(viewer, "LATEST", latest), patch.object(viewer, "FRAME_TIMES", []):
            viewer.publish(np.asarray(Image.new("RGB", (2, 2))), {"mode": "laya"})
        self.assertEqual(latest["local"]["meta"]["model_name"], "Laya (english)")
        self.assertTrue(latest["local"]["meta"]["model_only"])


class SupervisorProtocolTests(unittest.TestCase):
    def test_only_explicit_mode_is_forwarded(self):
        for mode in (None, "laya", "chat", "score"):
            with self.subTest(mode=mode):
                launched = []

                def capture(name, argv, *args):
                    launched.append((name, argv))
                    supervise.stop.set()

                argv = ["supervise.py", "--model-only", "--tunnel", "none"]
                if mode is not None:
                    argv.extend(["--mode", mode])
                with patch.object(supervise.sys, "argv", argv), \
                     patch.dict(supervise.os.environ, {"LOCAL_POLICY_MODE": "laya"}), \
                     patch.object(supervise.signal, "signal"), \
                     patch.object(supervise, "take_singleton_lock", return_value=True), \
                     patch.object(supervise, "release_lock"), \
                     patch.object(supervise, "port_busy", return_value=False), \
                     patch.object(supervise, "supervise", side_effect=capture), \
                     contextlib.redirect_stdout(io.StringIO()):
                    try:
                        supervise.main()
                    finally:
                        supervise.stop.clear()
                self.assertEqual(len(launched), 1)
                name, child_argv = launched[0]
                self.assertEqual(name, "viewer")
                self.assertIn("--model-only", child_argv)
                if mode is None:
                    self.assertNotIn("--mode", child_argv)
                else:
                    self.assertEqual(child_argv[child_argv.index("--mode") + 1], mode)


if __name__ == "__main__":
    unittest.main()
