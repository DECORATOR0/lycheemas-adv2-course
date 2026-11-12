import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lychee_mas.course.io import ChatClient, judge_schema_complete


def response(value):
    return {'choices': [{'message': {'content': json.dumps(value)}, 'finish_reason': 'stop'}]}


class JudgeSchemaTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_verdict_retries_without_defaulting_to_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            client = ChatClient('http://127.0.0.1:1/v1', 'unused', 1, Path(directory) / 'calls.jsonl')
            missing = response({'feedback': 'The matrix argument is unfinished.'})
            valid = response({'violations': ['reasoning_validity'], 'feedback': 'Recompute the determinant.'})
            with mock.patch.object(client, '_request', side_effect=[missing, valid]):
                verdict = json.loads(await client.judge('Audit the unchanged original candidate.'))
            self.assertEqual(verdict['violations'], ['reasoning_validity'])
            self.assertFalse(client.calls[0]['output_complete'])
            self.assertEqual(client.calls[1]['retry_reason'], 'invalid_judge_schema')
            self.assertIn('exactly two keys', client.calls[1]['prompt'])
            with mock.patch.object(client, '_request', return_value=missing) as request:
                with self.assertRaisesRegex(ValueError, 'invalid after 2 format retries'):
                    await client.judge('Audit another unchanged candidate.')
            self.assertEqual(request.call_count, 3)

    def test_wrong_types_and_missing_feedback_fail_shape_validation(self):
        self.assertTrue(judge_schema_complete({'violations': [], 'feedback': 'Checked.'}))
        for invalid in [
            {'violations': []}, {'violations': [], 'feedback': 1},
            {'violations': 'reasoning_validity', 'feedback': 'Check.'},
            {'violations': ['reasoning_validity'], 'feedback': ' '},
            {'violations': ['x', 'x'], 'feedback': 'Check.'},
            {'violations': [], 'feedback': 'Checked.', 'extra': True},
        ]:
            self.assertFalse(judge_schema_complete(invalid))
