"""Laya typed choices must reach the executor without parsing or rule substitution."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import play_local as p


def answer():
    probs = {name: 0.0 for name in p.ACTIONS}
    probs['run right'] = 1.0
    return {'answers': {'action': {'choice': 'run right', 'probabilities': probs}},
            'usage': {'input_tokens': 302},
            'runtime': {'checkpoint': 'english', 'token_budget': {'action': {'truncated': False}}}}


class LayaPolicyTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(p.os.environ, {
            'LOCAL_POLICY_MODE': 'laya', 'LOCAL_POLICY_MODEL': 'english',
            'LOCAL_POLICY_BASE_URL': 'http://127.0.0.1:11505/v1',
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.state = {'summary': 'On ground. Visible floor ahead.', 'grid': ''}

    def call(self, body, status=200):
        def respond(request):
            self.assertEqual(request.url.path, '/v1/systemone')
            payload = json.loads(request.content)
            self.assertNotIn('messages', payload)
            self.assertEqual(payload['model'], 'english')
            self.assertEqual(set(payload['questions']['action']['criteria']), set(p.ACTIONS))
            self.assertEqual(payload['questions']['action']['type'], 'choice')
            self.assertNotIn('Answer with one letter', payload['state'])
            return httpx.Response(status, json=body)
        with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
             patch.object(p, 'policy', side_effect=AssertionError('no rules fallback')):
            return p.ask_local(client, self.state, strict=True)

    def test_typed_choice_preserves_action_probabilities_and_real_usage(self):
        result = self.call(answer())
        self.assertEqual(result[0], 'run right')
        self.assertEqual(result[1], answer()['answers']['action']['probabilities'])
        self.assertEqual(result[2], 302)
        self.assertEqual(result[-1], 'model')

    def test_invalid_choice_distribution_or_usage_fails_closed(self):
        cases = []
        for value in ['Z', 'run right or jump right', None, ['run right']]:
            value_case = answer(); value_case['answers']['action']['choice'] = value; cases.append(value_case)
        for value in [float('nan'), float('inf'), -0.1, True, '1']:
            value_case = answer(); value_case['answers']['action']['probabilities']['run right'] = value; cases.append(value_case)
        value_case = answer(); del value_case['answers']['action']['probabilities']['stand']; cases.append(value_case)
        value_case = answer(); value_case['usage']['input_tokens'] = -1; cases.append(value_case)
        value_case = answer(); value_case['runtime']['token_budget']['action']['truncated'] = True; cases.append(value_case)
        value_case = answer(); value_case['runtime']['checkpoint'] = 'typed-decisions'; cases.append(value_case)
        for value_case in cases:
            with self.subTest(value=value_case), self.assertRaises(ValueError):
                self.call(value_case)

    def test_rounded_probabilities_and_failures(self):
        body = answer(); body['answers']['action']['probabilities'] = {key: 0.1111 for key in p.ACTIONS}
        self.assertEqual(self.call(body)[0], 'run right')
        with self.assertRaises(httpx.HTTPStatusError):
            self.call({'detail': 'token budget exceeded'}, 422)
        with self.assertRaises(KeyError):
            self.call({'answers': {}})

    def test_observation_keeps_geometry_and_latest_three_decisions(self):
        rows = [list('.' * 20) for _ in range(13)]
        rows[6][4] = 'M'; rows[7] = list('#' * 20)
        grid = '\n'.join(''.join(row) for row in rows)
        state = p.features(grid, 48, airborne=False, visible=9)
        state.update(grid=grid, action_before=3)
        history = [('run right', n * 10) for n in range(6)]
        payload = p.laya_request(state, history)
        packet = json.loads(payload['state']); observation = packet['observation']
        self.assertEqual(packet['history_action_x'], [list(row) for row in history[-3:]])
        self.assertEqual(observation['terrain'], [[s['from'], s['to'], s['solid_up']] for s in state['terrain']])
        self.assertEqual(observation['enemies'], [])
        self.assertEqual(observation['visible_ahead'], 9)
        self.assertEqual(observation['speed'], 48)
        self.assertNotIn('grid', observation)

    def test_local_game_counts_usage_without_applying_jev_pricing(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(p, 'RUNS', Path(directory)), patch.object(p, 'MAX_FRAMES', 7), \
             patch.object(p, 'ask_local', return_value=('run right', None, 302, 0.01, 'model')) as model, \
             patch.object(p.imageio, 'mimsave'), \
             patch.object(p, 'hazard_guard', side_effect=AssertionError('no guard')), \
             patch.object(p, 'unstick', side_effect=AssertionError('no unstick')):
            result = p.run('local', model_only=True)
        self.assertEqual(result['input_tokens'], 302 * model.call_count)
        self.assertEqual(result['model'], 'english')
        self.assertEqual(result['local_mode'], 'laya')
        self.assertIsNone(result['cost_usd'])
        self.assertEqual(result['guard_rewrites'], 0)
        self.assertEqual(result['unstick_rewrites'], 0)


if __name__ == '__main__':
    unittest.main()
