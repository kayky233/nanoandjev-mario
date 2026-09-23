"""Regressions from failed model runs; observations and prompts require no network."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import play_local as player


EVIDENCE = Path(__file__).resolve().parents[1] / "artifacts/model-eval-20260923"


def recorded_state(relative_path, line):
    return json.loads((EVIDENCE / relative_path).read_text().splitlines()[line - 1])


def flat_grid():
    rows = [list("." * 20) for _ in range(13)]
    rows[6][4] = "M"
    rows[7] = list("#" * 20)
    return rows


def grid_text(rows):
    return "\n".join("".join(row) for row in rows)


class ObservationTests(unittest.TestCase):
    def test_recorded_1_3_gap_is_detected_beyond_old_eight_tile_limit(self):
        recorded = recorded_state("jev/1-3/1-3-jev-20260923-103227.log.jsonl", 8)
        self.assertIn("Solid ground ahead", recorded["summary"])
        state = player.features(recorded["grid"], 48, airborne=False, visible=10)
        self.assertEqual(state["gap_ahead"], {"tiles": 9, "width": 2, "width_known": False})
        self.assertEqual(state["unknown_from_tile"], 11)
        self.assertNotIn("Solid ground ahead", state["summary"])
        self.assertIn("at least 2", state["summary"])

    def test_fifteenth_tile_is_observed_and_gap_width_stops_at_boundary(self):
        rows = flat_grid()
        rows[7][19] = "."
        state = player.features(grid_text(rows), visible=15)
        self.assertEqual(state["gap_ahead"], {"tiles": 15, "width": 1, "width_known": False})
        self.assertEqual(state["unknown_from_tile"], 16)

    def test_unseen_wall_gap_and_enemy_are_not_claimed_as_observations(self):
        rows = flat_grid()
        rows[6][13] = "#"  # nine tiles ahead, beyond the chosen view
        rows[7][14] = "."
        rows[6][15] = "G"
        state = player.features(grid_text(rows), visible=8)
        self.assertIsNone(state["wall_ahead"])
        self.assertIsNone(state["gap_ahead"])
        self.assertEqual(state["enemies"], [])
        self.assertTrue(all(segment["to"] <= 8 for segment in state["terrain"]))
        self.assertEqual(state["unknown_from_tile"], 9)

    def test_gap_width_is_known_only_when_far_support_is_observed(self):
        rows = flat_grid()
        for dx in (3, 4, 5):
            rows[7][4 + dx] = "."
        state = player.features(grid_text(rows), visible=6)
        self.assertEqual(state["gap_ahead"], {"tiles": 3, "width": 3, "width_known": True})
        unseen_end = player.features(grid_text(rows), visible=5)
        self.assertFalse(unseen_end["gap_ahead"]["width_known"])

    def test_zero_visible_tiles_does_not_assert_clear_ground_ahead(self):
        state = player.features(grid_text(flat_grid()), visible=0)
        self.assertEqual(state["unknown_from_tile"], 1)
        self.assertIsNone(state["gap_ahead"])
        self.assertEqual(state["enemies"], [])
        self.assertIn("next 0 tiles", state["summary"])

    def test_recorded_front_wall_is_not_a_ceiling(self):
        recorded = recorded_state("jev-repeat-2/1-1/1-1-jev-20260923-103545.log.jsonl", 14)
        self.assertIn("Blocks overhead 1 tiles up", recorded["summary"])
        state = player.features(recorded["grid"], 48, airborne=False, visible=8)
        self.assertEqual(state["wall_ahead"], {"tiles": 6, "height": 3})
        self.assertNotIn("headroom", state)
        self.assertEqual(state["jump_lands_tiles_ahead"], 9)

    def test_real_low_ceiling_still_reduces_estimated_reach(self):
        rows = flat_grid()
        rows[3][4:12] = "#" * 8
        state = player.features(grid_text(rows), 48, airborne=False)
        self.assertIsNone(state["wall_ahead"])
        self.assertEqual(state["headroom"], 2)
        self.assertEqual(state["jump_lands_tiles_ahead"], 4)
        self.assertIn("estimated landing", state["summary"])

    def test_separate_overhead_block_does_not_inflate_wall_height(self):
        rows = flat_grid()
        rows[6][6] = "#"
        rows[3][6] = "#"
        state = player.features(grid_text(rows))
        self.assertEqual(state["wall_ahead"], {"tiles": 2, "height": 1})

    def test_terrain_preserves_all_visible_solids_and_elevated_platforms(self):
        rows = flat_grid()
        for dx in range(6, 12):
            rows[7][4 + dx] = "."
        rows[4][12:15] = "###"  # elevated landing platform above a gap
        rows[2][4:7] = "###"  # actual ceiling
        state = player.features(grid_text(rows), visible=12)
        reconstructed = set()
        for segment in state["terrain"]:
            for dx in range(segment["from"], segment["to"] + 1):
                for low, high in segment["solid_up"]:
                    for up in range(low, high + 1):
                        reconstructed.add((6 - up, 4 + dx))
        expected = {(r, c) for r in range(13) for c in range(17) if rows[r][c] == "#"}
        self.assertEqual(reconstructed, expected)
        self.assertIn((4, 12), reconstructed)
        self.assertEqual(state["gap_ahead"]["tiles"], 6)

    def test_recorded_collision_risk_is_an_estimate_not_conflicting_orders(self):
        recorded = recorded_state("jev-repeat-2/1-2/1-2-jev-20260923-103556.log.jsonl", 24)
        self.assertIn("Do not take a full jump", recorded["summary"])
        self.assertIn("full jump forward is the only way", recorded["summary"])
        state = player.features(recorded["grid"], 24, airborne=False, visible=8)
        self.assertTrue(state["enemies_near_landing_spot"])
        self.assertTrue(state["risks"])
        self.assertNotIn("Do not take a full jump", state["summary"])
        self.assertNotIn("only way", state["summary"])
        self.assertIn("future positions are uncertain", state["summary"])


class PromptTests(unittest.TestCase):
    def test_compact_state_keeps_geometry_enemies_speed_and_jump_estimates(self):
        rows = flat_grid()
        rows[6][6] = "G"
        state = player.features(grid_text(rows), 28, airborne=False, visible=10)
        state.update(grid=grid_text(rows), action_before=3)
        compact = json.loads(player._state_block(state))
        self.assertEqual(compact["speed"], 28)
        self.assertEqual(compact["unknown_from"], 11)
        self.assertEqual(compact["enemies"], [[2, 0, "goomba"]])
        self.assertEqual(compact["estimated_jump"]["landing"], state["jump_lands_tiles_ahead"])
        self.assertEqual(compact["terrain"], [[s["from"], s["to"], s["solid_up"]] for s in state["terrain"]])
        self.assertNotIn("summary", compact)
        self.assertNotIn("grid", compact)

    def test_both_models_get_shared_rules_and_local_keeps_stuck_history(self):
        captured = []

        def respond(request):
            captured.append(json.loads(request.content))
            if request.url.path.endswith("chat/completions"):
                return httpx.Response(200, json={"choices": [{"message": {"content": "D"}}]})
            return httpx.Response(200, json={"answers": {"action": {
                "choice": "run right", "probabilities": {"run right": 1.0},
            }}, "usage": {"input_tokens": 1}})

        state = player.features(grid_text(flat_grid()), 0, airborne=False)
        state["grid"] = grid_text(flat_grid())
        history = [("hop right", 722)] * 3
        with patch.dict(player.os.environ, {"TYPESAFE_API_KEY": "test-only", "LOCAL_POLICY_MODE": "chat"}), \
                httpx.Client(transport=httpx.MockTransport(respond)) as client:
            player.ask_local(client, state, history, strict=True)
            player.ask_jev(client, state, history)
        local, jev = captured
        self.assertIn(player.RULES, local["messages"][0]["content"])
        self.assertIn(player.RULES, jev["questions"]["action"]["instructions"])
        self.assertIn("STUCK:", local["messages"][1]["content"])
        self.assertIn("STUCK:", jev["state"]["recent_decisions"])
        self.assertNotIn("Example:", local["messages"][0]["content"])
        self.assertNotIn("decision_context", jev["state"])
        for description in player.ACTION_HELP.values():
            self.assertIn(description, local["messages"][0]["content"])
            self.assertNotIn(description, local["messages"][1]["content"])

    def test_legacy_summary_only_callers_keep_their_observation(self):
        self.assertIn("Clear ground", player._state_block({"summary": "Clear ground", "grid": ""}))


if __name__ == "__main__":
    unittest.main()
