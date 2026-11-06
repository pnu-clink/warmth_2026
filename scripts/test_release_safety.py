"""Input-free tests for the upload guard and preserved DPO pair builder."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
from verify_submission import audit, inspect_text, ROOT
TEST_TMP = os.environ.get('RELEASE_TEST_TMPDIR')

spec = importlib.util.spec_from_file_location('archived_pair_builder', ROOT / 'server_code/build_dpo_pairs.py')
pairs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pairs)


class ReleaseSafety(unittest.TestCase):
    def test_provider_pattern(self):
        value = 'ghp_' + 'x' * 36
        self.assertTrue(inspect_text(value))
        self.assertNotIn(value, repr(inspect_text(value)))

    def test_literal_assignment(self):
        line = 'api' + '_key = ' + repr('abc123examplecredential987654')
        self.assertTrue(inspect_text(line))

    def test_placeholder(self):
        self.assertFalse(inspect_text('OPENAI_API_KEY="YOUR_API_KEY"'))

    def test_env_lookup(self):
        self.assertFalse(inspect_text('client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])'))

    def test_extra_file_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as folder:
            root = Path(folder)
            (root / 'MANIFEST.json').write_text(json.dumps({'files': []}), encoding='utf-8')
            (root / 'extra.txt').write_text('unexpected', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unlisted file'):
                audit(root)

    def test_env_not_allowed(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as folder:
            root = Path(folder)
            (root / 'MANIFEST.json').write_text(json.dumps({'files': []}), encoding='utf-8')
            (root / '.env').write_text('not-a-real-key', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'must not be uploaded'):
                audit(root)

    def test_pair_default_final_turn(self):
        original = [{'from': 'human', 'value': 'First?'}, {'from': 'gpt', 'value': 'First original'},
                    {'from': 'human', 'value': 'Second?'}, {'from': 'gpt', 'value': 'Second original'}]
        warm = [dict(x) for x in original]
        warm[1]['value'] = 'First warm'
        warm[3]['value'] = 'Second warm'
        result = pairs.extract_pairs(pairs.normalize(warm), pairs.normalize(original))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['chosen'], 'Second warm')
        self.assertEqual(result[0]['rejected'], 'Second original')
        self.assertEqual(result[0]['prompt'][1]['content'], 'First warm')

    def test_pair_all_turns(self):
        original = [{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': 'Original'}]
        warm = [dict(x) for x in original]
        warm[1]['content'] = 'Warm'
        self.assertEqual(len(pairs.extract_pairs(warm, original, all_turns=True)), 1)

    def test_pair_identical_rejected(self):
        seq = [{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': 'Same'}]
        self.assertEqual(pairs.extract_pairs(seq, seq), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
