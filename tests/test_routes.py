"""Behavioral checks against the real emulator, with all network calls forbidden."""

import contextlib
import io
import unittest
from unittest.mock import patch

import watch_local as viewer


class VerifiedRouteTests(unittest.TestCase):
    def test_routes_reach_flag_without_network_and_count_actual_frames(self):
        for level in viewer.VERIFIED_ROUTES:
            with self.subTest(level=level):
                observed = []
                def capture(_obs, meta, _channel="local"):
                    if meta.get("status") == "playing":
                        observed.append(meta["frame"])

                with patch.object(viewer.httpx.Client, "send", side_effect=AssertionError("network forbidden")) as send, \
                     patch.object(viewer, "publish", capture), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = viewer.play_episode(
                        level, 1_000_000, "replay", 1,
                        model="replay", verified_route=True,
                    )
                send.assert_not_called()
                self.assertTrue(result["flag"])
                self.assertTrue(result["route_complete"])
                self.assertEqual(result["calls"], 0)
                self.assertEqual(observed, list(range(1, result["frames"] + 1)))
                for step in result["verified_route_steps"]:
                    self.assertIsNone(step["model_choice"])
                    self.assertEqual(step["model_source"], "offline_replay")

    def test_unsupported_route_is_rejected_before_starting_emulator(self):
        with patch.object(viewer, "JoypadSpace") as env:
            with self.assertRaisesRegex(ValueError, "No verified route"):
                viewer.play_episode("8-4", 60, "replay", 1, model="replay", verified_route=True)
            env.assert_not_called()

    def test_replay_requires_route_and_positive_fps(self):
        with self.assertRaisesRegex(ValueError, "requires a verified route"):
            viewer.play_episode("1-2", 60, "replay", 1, model="replay")
        with self.assertRaisesRegex(ValueError, "fps must be positive"):
            viewer.play_episode("1-2", 0, "replay", 1, model="replay", verified_route=True)

    def test_exhausted_route_stops_without_falling_back_to_another_controller(self):
        with patch.dict(viewer.VERIFIED_ROUTES, {"1-2": (("stand", None),)}), \
             patch.object(viewer, "publish"), \
             patch.object(viewer.httpx.Client, "send", side_effect=AssertionError("network forbidden")) as send, \
             contextlib.redirect_stdout(io.StringIO()):
            result = viewer.play_episode("1-2", 1_000_000, "replay", 1, model="replay", verified_route=True)
        self.assertFalse(result["flag"])
        self.assertFalse(result["route_complete"])
        self.assertEqual(len(result["verified_route_steps"]), 1)
        self.assertLess(result["frames"], 200)
        send.assert_not_called()

    def test_failed_model_request_is_not_reported_as_a_model_answer(self):
        with patch.dict(viewer.VERIFIED_ROUTES, {"1-2": (("stand", None),)}), \
             patch.object(viewer, "publish"), \
             patch.object(viewer.P, "ask_local", side_effect=RuntimeError("unavailable")), \
             contextlib.redirect_stdout(io.StringIO()):
            result = viewer.play_episode("1-2", 1_000_000, "chat", 1, sync=True, verified_route=True)
        self.assertEqual(result["calls"], 1)  # attempt count, not successful answers
        self.assertEqual(result["model_decision_sources"], {"fallback": 1})
        self.assertIsNone(result["verified_route_steps"][0]["model_choice"])
        self.assertEqual(result["verified_route_steps"][0]["model_source"], "fallback")


if __name__ == "__main__":
    unittest.main()
