"""Offline credential tests. Uses synthetic .env files, never a real API key."""
import ast
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import api_credentials as auth
import run_with_dotenv as runner


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ.get('RELEASE_TEST_TMPDIR'))
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.root_patch = patch.object(auth, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def env(self, text):
        target = self.root/'.env'
        target.write_text(text, encoding='utf-8-sig')
        return target

    def test_dotenv_overrides_shell_without_modifying_shell(self):
        self.env('OPENAI_API_KEY="test_file_key"\n')
        os.environ['OPENAI_API_KEY'] = 'test_shell_key'
        self.assertEqual(auth.load_openai_key(), 'test_file_key')
        self.assertEqual(os.environ['OPENAI_API_KEY'], 'test_shell_key')

    def test_only_key_read_no_interpolation_or_endpoint(self):
        self.env('OPENAI_API_KEY=test_${IGNORED}\nOPENAI_BASE_URL=https://example.invalid\n')
        self.assertEqual(auth.load_openai_key(), 'test_${IGNORED}')
        self.assertNotIn('OPENAI_BASE_URL', os.environ)

    def test_shell_fallback_only_when_default_file_absent(self):
        os.environ['OPENAI_API_KEY'] = 'test_shell_key'
        self.assertEqual(auth.load_openai_key(), 'test_shell_key')
        self.env('UNRELATED=value\n')
        with self.assertRaises(SystemExit):
            auth.load_openai_key()

    def test_explicit_missing_file_no_fallback(self):
        os.environ['OPENAI_API_KEY'] = 'test_shell_key'
        with self.assertRaises(SystemExit):
            auth.load_openai_key(self.root/'missing.env')

    def test_invalid_and_missing_never_echo_key(self):
        for content in ('', 'OPENAI_API_KEY=replace_with_your_api_key\n', 'OPENAI_API_KEY="test bad key"\n'):
            self.env(content)
            with self.assertRaises(SystemExit) as error:
                auth.load_openai_key()
            self.assertNotIn('test bad key', str(error.exception))

    def test_no_logs_on_success(self):
        self.env('OPENAI_API_KEY=test_private_fixture\n')
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(auth.load_openai_key(), 'test_private_fixture')
        self.assertEqual(out.getvalue()+err.getvalue(), '')

    def test_missing_dependency_clear_error(self):
        self.env('OPENAI_API_KEY=test_private_fixture\n')
        with patch.dict(sys.modules, {'dotenv': None}):
            with self.assertRaisesRegex(SystemExit, 'python-dotenv'):
                auth.load_openai_key()

    def test_wrapper_help_does_not_read_key(self):
        script = self.root/'fixture.py'
        script.write_text('pass\n', encoding='utf-8')
        with patch.object(runner, 'load_openai_key') as load, patch.object(runner.runpy, 'run_path') as execute:
            runner.main([str(script), '--help'])
            load.assert_not_called()
            execute.assert_called_once()

    def test_wrapper_explicit_run_restores_state(self):
        script = self.root/'fixture.py'
        script.write_text('pass\n', encoding='utf-8')
        original_argv = sys.argv[:]
        os.environ['OPENAI_API_KEY'] = 'test_original'
        def capture(*args, **kwargs):
            self.assertEqual(os.environ['OPENAI_API_KEY'], 'test_replacement')
            self.assertEqual(sys.argv, [str(script), '--run'])
            raise RuntimeError('synthetic failure')
        with patch.object(runner, 'load_openai_key', return_value='test_replacement'), patch.object(runner.runpy, 'run_path', side_effect=capture):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                runner.main([str(script), '--run'])
        self.assertEqual(os.environ['OPENAI_API_KEY'], 'test_original')
        self.assertEqual(sys.argv, original_argv)

    def test_legacy_csv_judge_has_no_key_command_line_option(self):
        tree = ast.parse((ROOT/'label_errors_run.py').read_text(encoding='utf-8-sig'))
        options = [arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
                   for arg in node.args if isinstance(arg, ast.Constant)]
        self.assertIn('--env-file', options)
        self.assertNotIn('--api_key', options)


if __name__ == '__main__':
    unittest.main(verbosity=2)
