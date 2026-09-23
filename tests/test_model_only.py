"""Model provenance checks with a real emulator and no network requests."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import play_local as player


class ModelOnlyTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.runs = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.steps = 0
        real_joypad = player.JoypadSpace

        def make_counted_env(*args):
            env = real_joypad(*args)
            real_step = env.step

            def step(action):
                self.steps += 1
                return real_step(action)

            env.step = step
            return env

        self.stack.enter_context(patch.object(player, "JoypadSpace", side_effect=make_counted_env))
        self.stack.enter_context(patch.object(player, "RUNS", self.runs))
        self.stack.enter_context(patch.object(player, "MAX_FRAMES", 70))
        self.real_mimsave = player.imageio.mimsave
        self.gif_writer = self.stack.enter_context(patch.object(player.imageio, "mimsave"))
        self.stack.enter_context(patch.object(player.httpx.Client, "send", side_effect=AssertionError("network forbidden")))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def read_log(self):
        return [json.loads(line) for line in next(self.runs.glob("*.log.jsonl")).read_text().splitlines()]

    def test_model_only_executes_model_choices_without_guard_or_unstick(self):
        for bot in ("local", "jev"):
            with self.subTest(bot=bot):
                self.steps = 0
                answer = ("run and jump right", None, 0, 0.01)
                if bot == "local":
                    answer += ("model",)
                with patch.object(player, f"ask_{bot}", return_value=answer) as model, \
                     patch.object(player, "hazard_guard", side_effect=AssertionError("guard forbidden")), \
                     patch.object(player, "unstick", side_effect=AssertionError("unstick forbidden")):
                    result = player.run(bot, model_only=True)
                self.assertTrue(result["model_only"])
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["valid_model_answers"], model.call_count)
                self.assertGreater(model.call_count, 0)
                self.assertEqual(result["guard_rewrites"], 0)
                self.assertEqual(result["unstick_rewrites"], 0)
                self.assertEqual(result["frames"], self.steps)
                self.assertGreater(result["decision_clock"], result["frames"])
                for row in self.read_log():
                    self.assertEqual(row["model_source"], "model")
                    self.assertEqual(row["model_choice"], row["executed_choice"])
                    self.assertEqual(row["choice"], row["model_choice"])
                for path in self.runs.glob("*.log.jsonl"):
                    path.unlink()

    def test_model_only_rejects_fallback_without_executing_it_or_recording_success(self):
        with patch.object(player, "ask_local", return_value=("run right", None, 0, 0.01, "rules_fallback")), \
             patch.object(player, "hazard_guard") as guard, patch.object(player, "unstick") as unstick:
            with self.assertRaisesRegex(RuntimeError, "rejected.*rules_fallback"):
                player.run("local", model_only=True)
        self.assertEqual(self.steps, 0)
        result = json.loads((self.runs / "results.jsonl").read_text())
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertIn("rules_fallback", result["error"])
        self.assertFalse(result["flag"])
        self.assertEqual(result["valid_model_answers"], 0)
        self.assertIsNone(result["gif"])
        guard.assert_not_called()
        unstick.assert_not_called()

    def test_default_mode_keeps_rewrites_and_original_model_choice(self):
        with patch.object(player, "ask_local", return_value=("run right", None, 0, 0.01, "model")) as model, \
             patch.object(player, "hazard_guard", return_value="jump right") as guard, \
             patch.object(player, "unstick", return_value="stand") as unstick:
            result = player.run("local")
        self.assertFalse(result["model_only"])
        self.assertEqual(result["guard_rewrites"], model.call_count)
        self.assertEqual(result["unstick_rewrites"], model.call_count)
        self.assertEqual(guard.call_count, model.call_count)
        self.assertEqual(unstick.call_count, model.call_count)
        for row in self.read_log():
            self.assertEqual(row["model_choice"], "run right")
            self.assertEqual(row["model_source"], "model")
            self.assertEqual(row["choice"], "stand")
            self.assertEqual(row["local_source"], "unstick")

    def test_model_only_rejects_non_model_runs_before_creating_emulator(self):
        with patch.object(player, "JoypadSpace") as env:
            for bot, dump in (("rules", False), ("local", True), ("run right", False)):
                with self.subTest(bot=bot, dump=dump), self.assertRaisesRegex(ValueError, "requires"):
                    player.run(bot, dump=dump, model_only=True)
            env.assert_not_called()

    def test_gif_duration_matches_emulated_time_including_partial_failed_runs(self):
        with patch.object(player, "ask_local", side_effect=[
            ("run right", None, 0, 0.01, "model"), ValueError("invalid model reply"),
        ]), patch.object(player.imageio, "mimsave", wraps=self.real_mimsave):
            with self.assertRaisesRegex(ValueError, "invalid model reply"):
                player.run("local", model_only=True)
        result = json.loads((self.runs / "results.jsonl").read_text())
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["flag"])
        self.assertEqual(result["frames"], self.steps)
        self.assertEqual(result["valid_model_answers"], 1)
        self.assertEqual(len(self.read_log()), 1)
        with Image.open(self.runs / result["gif"]) as gif:
            duration_ms = 0
            for i in range(gif.n_frames):
                gif.seek(i)
                duration_ms += gif.info.get("duration", 0)
        self.assertAlmostEqual(duration_ms, result["frames"] * 1000 / 60, delta=5)


class StrictResponseTests(unittest.TestCase):
    def test_chat_accepts_only_one_candidate_letter_without_rules_fallback(self):
        state = {"summary": "Clear ground ahead.", "grid": ""}
        for text, valid in (("A", True), (" a\n", True), ("A or B", False), ("I cannot choose", False)):
            with self.subTest(text=text), \
                 patch.dict(player.os.environ, {"LOCAL_POLICY_MODE": "chat"}), \
                 patch.object(player, "policy", side_effect=AssertionError("rules forbidden")) as policy, \
                 player.httpx.Client(transport=player.httpx.MockTransport(
                     lambda request: player.httpx.Response(200, json={"choices": [{"message": {"content": text}}]}),
                 )) as client:
                if valid:
                    answer = player.ask_local(client, state, strict=True)
                    self.assertEqual(answer[0], "stand")
                    self.assertEqual(answer[-1], "model")
                else:
                    with self.assertRaises(ValueError) as caught:
                        player.ask_local(client, state, strict=True)
                    self.assertIn(text, str(caught.exception))
                policy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
