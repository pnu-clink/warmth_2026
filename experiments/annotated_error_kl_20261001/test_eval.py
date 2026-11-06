"""CPU unit/integration tests. Toy-model checks do not certify real GPU execution."""
from contextlib import contextmanager, nullcontext
import copy
import math
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import kl_core as core
import kl_runtime as runtime

HERE = Path(__file__).resolve().parent


class ToyTokenizer:
    is_fast = True
    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        ids = [ord(char) % 7 for char in text]
        if add_special_tokens:
            ids = [0] + ids
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "\n".join(m["role"] + ":" + m["content"] for m in messages) + "assistant:"
    def convert_ids_to_tokens(self, token_id):
        return str(token_id)
    def decode(self, ids, **kwargs):
        return "/".join(str(i) for i in ids)


class ToyTensor:
    def __init__(self, data):
        self.data = np.asarray(data)
    @property
    def shape(self):
        return self.data.shape
    @property
    def ndim(self):
        return self.data.ndim
    def __getitem__(self, key):
        return ToyTensor(self.data[key])
    def detach(self):
        return self
    def to(self, **kwargs):
        return self
    def contiguous(self):
        return self
    def numpy(self):
        return self.data


class ToyTorch:
    long, float32 = "long", "float32"
    @staticmethod
    def tensor(data, **kwargs):
        return ToyTensor(data)
    @staticmethod
    def ones_like(tensor):
        return ToyTensor(np.ones_like(tensor.data))
    inference_mode = staticmethod(nullcontext)


class ToyModel:
    def __init__(self):
        self.active = True
        self.calls = []
    def eval(self):
        return self
    @contextmanager
    def disable_adapter(self):
        previous = self.active
        self.active = False
        try:
            yield
        finally:
            self.active = previous
    def __call__(self, **kwargs):
        assert kwargs["use_cache"] is False
        self.calls.append(self.active)
        ids = kwargs["input_ids"].data[0]
        # Causal toy logits depend ONLY on tokens up to each prediction position.
        x = np.cumsum(ids)[:, None] / (np.arange(7)[None, :] + 1) / 100
        if self.active:
            x = x + np.arange(7)[None, :] / 10
        keep = kwargs.get("logits_to_keep", 0)
        if keep:
            x = x[-keep:]
        return types.SimpleNamespace(logits=ToyTensor(x[None, :, :]))


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.input_manifest = core.load_inputs(HERE / "prepared/inputs.jsonl", HERE / "prepared/manifest.json")

    def test_fixed_input_counts(self):
        self.assertEqual(len(self.rows), 140)
        self.assertEqual({m: sum(r["model"] == m for r in self.rows) for m in core.MODELS},
                         {"qwen": 37, "phi4": 39, "llama": 32, "gemma": 32})

    def test_formatter_matches_archived_source(self):
        import ast
        path = HERE.parent / "early_checkpoint_eval/reference/mmlu_original.py"
        # On the server, external archive is not required by this self-contained bundle.
        if not path.is_file():
            self.skipTest("External archive absent; execution does not depend on it.")
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "format_mmlu_pro_prompt")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        for row in self.rows:
            self.assertEqual(core.format_prompt(row["question"], row["options"]),
                             namespace[function.name](row["question"], row["options"]))

    def test_shift_for_all_and_suffix_logits(self):
        for p in range(1, 9):
            for t in range(1, 9):
                ids = list(range(p + t - 1))
                result = ids[core.response_logit_slice(len(ids), p, t)]
                self.assertEqual(result, list(range(p - 1, p + t - 1)))
                self.assertEqual(list(range(t))[core.response_logit_slice(t, p, t)], list(range(t)))
        with self.assertRaises(ValueError):
            core.response_logit_slice(99, 2, 3)

    def test_kl_against_independent_scalar_formula(self):
        b = np.array([[1., 2., 3.], [-2., 4., 0.]])
        w = np.array([[3., 2., 1.], [-1., 2., 0.]])
        results = core.distribution_metrics(b, w, [0, 1])
        for i in range(2):
            pb = [math.exp(x) / sum(math.exp(z) for z in b[i]) for x in b[i]]
            pw = [math.exp(x) / sum(math.exp(z) for z in w[i]) for x in w[i]]
            expected_bw = sum(p * math.log(p / q) for p, q in zip(pb, pw))
            expected_wb = sum(q * math.log(q / p) for p, q in zip(pb, pw))
            self.assertAlmostEqual(results["kl_base_to_warm"][i], expected_bw, places=12)
            self.assertAlmostEqual(results["kl_warm_to_base"][i], expected_wb, places=12)
        identical = core.distribution_metrics(b, b, [0, 1])
        self.assertEqual(identical["kl_base_to_warm"], [0., 0.])

    def test_kl_extreme_logits_and_validation(self):
        result = core.distribution_metrics([[10000, -10000]], [[-10000, 10000]], [0])
        self.assertAlmostEqual(result["kl_base_to_warm"][0], 20000)
        self.assertAlmostEqual(result["kl_warm_to_base"][0], 20000)
        for b, w, ids in [([[0, 1]], [[0]], [0]), ([[float("nan"), 1]], [[0, 1]], [0]), ([[0, 1]], [[0, 1]], [2])]:
            with self.assertRaises(ValueError):
                core.distribution_metrics(b, w, ids)

    def test_gemma_system_fallback_and_tokenization(self):
        class RejectSystem(ToyTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                if messages[0]["role"] == "system":
                    raise ValueError("System role not supported")
                return super().apply_chat_template(messages, **kwargs)
        for key in core.MODELS:
            row = next(r for r in self.rows if r["model"] == key)
            owner = RejectSystem() if key == "gemma" else ToyTokenizer()
            result = core.tokenize_reference(row, owner, owner, key)
            self.assertEqual(result["input_ids"], result["prompt_ids"] + result["response_ids"][:-1])
            self.assertEqual(result["prompt_add_special_tokens"], key != "gemma")
            self.assertEqual(result["rendering"], "system_folded_into_user" if key == "gemma" else "native_system_user")
            self.assertTrue(result["rendered_prompt"].endswith("assistant:"))
            self.assertEqual(len(result["offsets"]), len(result["response_ids"]))

    def make_toy_result(self, logits_mode="auto"):
        row = copy.deepcopy(self.rows[0])
        row["answer_text"] = "short answer"
        row["response_sha256"] = core.text_hash(row["answer_text"])
        tokenizer = ToyTokenizer()
        tokenized = core.tokenize_reference(row, tokenizer, tokenizer, "qwen")
        model = ToyModel()
        args = types.SimpleNamespace(metric_chunk=3, logits_mode=logits_mode)
        bundle = {"torch": ToyTorch(), "tokenizer": tokenizer}
        result = runtime.evaluate(row, tokenized, args, bundle, (model, "cpu", "logits_to_keep"))
        self.assertEqual(model.calls, [False, True])
        self.assertTrue(model.active)
        self.assertEqual(len(result["tokens"]), len(row["answer_text"]))
        return row, result

    def test_toy_forward_suffix_and_all_are_identical(self):
        _, suffix = self.make_toy_result("auto")
        _, full = self.make_toy_result("all")
        self.assertEqual(suffix["tokens"], full["tokens"])
        self.assertGreater(suffix["summary"]["mean_kl_base_to_warm"], 0)

    def test_regions_for_short_answers(self):
        self.assertEqual([core.region(i, 1) for i in range(1)], ["late"])
        self.assertEqual([core.region(i, 6) for i in range(6)], ["early", "early", "middle", "middle", "late", "late"])

    def test_atomic_store_resume_and_export(self):
        row, result = self.make_toy_result()
        plan = {"item_ids": [row["item_id"]], "model": "qwen",
                "selected_groups": [{k: row[k] for k in ("item_id", "condition", "error_group", "human_unanimous")}]}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "run"
            store = core.RunStore(folder, plan, False)
            store.save(result)
            with self.assertRaises(ValueError):
                store.save(result)
            store.export()
            store.close()
            with self.assertRaises(FileExistsError):
                core.RunStore(folder, plan, False)
            resumed = core.RunStore(folder, plan, True)
            self.assertEqual(len(resumed.rows), 1)
            resumed.close()
            with self.assertRaises(ValueError):
                core.RunStore(folder, {**plan, "model": "phi4"}, True)
            saved = core.read_json(folder / "items" / (row["item_id"] + ".json"))
            saved["tokens"][0]["kl_base_to_warm"] += 1
            core.atomic_json(folder / "items" / (row["item_id"] + ".json"), saved)
            with self.assertRaises(ValueError):
                core.RunStore(folder, plan, True)
            self.assertTrue((folder / "token_metrics.csv").is_file())
            self.assertTrue((folder / "summary_by_group.csv").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
