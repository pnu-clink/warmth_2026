"""CPU-only helper tests. They do not import HF/PyTorch or load any model."""
import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import train_lora_dpo_gemma as train


def pair():
    return {"id": "example_0", "prompt": [{"role": "user", "content": "Explain this."}],
            "chosen": "A warmer answer here.", "rejected": "An original answer here."}


class TrainingTests(unittest.TestCase):
    def test_valid_pairs_unchanged(self):
        data = [pair()]
        before = copy.deepcopy(data)
        train.validate_pairs(data, 1)
        self.assertEqual(data, before)

    def test_pair_count(self):
        with self.assertRaises(ValueError):
            train.validate_pairs([pair()], 1489)

    def test_duplicate(self):
        with self.assertRaises(ValueError):
            train.validate_pairs([pair(), pair()], 2)

    def test_invalid_responses(self):
        for bad in (None, "", "x", "a" * 6001):
            row = pair()
            row["chosen"] = bad
            with self.assertRaises(ValueError):
                train.validate_pairs([row], 1)
        row = pair()
        row["chosen"] = row["rejected"]
        with self.assertRaises(ValueError):
            train.validate_pairs([row], 1)

    def test_non_text_prompt_rejected(self):
        row = pair()
        row["prompt"][0]["content"] = [{"type": "image"}]
        with self.assertRaises(ValueError):
            train.validate_pairs([row], 1)

    def test_blank_historical_turn_kept_like_original_builder(self):
        row = pair()
        row["prompt"] = [{"role": "user", "content": "First question"},
                         {"role": "assistant", "content": "Prior answer"},
                         {"role": "user", "content": ""}]
        before = copy.deepcopy(row)
        report = train.validate_pairs([row], 1)
        self.assertEqual(report["blank_prompt_messages"], 1)
        self.assertEqual(report["pairs_with_blank_prompt_messages"], 1)
        self.assertEqual(row, before)

    def test_prompt_must_end_with_user(self):
        row = pair()
        row["prompt"][0]["role"] = "assistant"
        with self.assertRaises(ValueError):
            train.validate_pairs([row], 1)

    def test_safe_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            adapter = root / "sft"
            adapter.mkdir()
            train.ensure_fresh_output(root / "new_dpo", [adapter])
            for bad in (root, adapter, adapter / "child"):
                with self.assertRaises(ValueError):
                    train.ensure_fresh_output(bad, [adapter])
            existing = root / "existing_dpo"
            existing.mkdir()
            (existing / "keep.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                train.ensure_fresh_output(existing, [adapter])
            self.assertEqual((existing / "keep.txt").read_text(), "keep")

    def test_safetensors_names(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = Path(td)
            mod = "model.language_model.layers.0.self_attn.q_proj"
            header = {f"base_model.model.{mod}.lora_{x}.weight": {} for x in ("A", "B")}
            raw = json.dumps(header).encode()
            (adapter / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
            self.assertEqual(train.adapter_modules(adapter), [mod])
            del header[f"base_model.model.{mod}.lora_B.weight"]
            raw = json.dumps(header).encode()
            (adapter / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
            with self.assertRaises(ValueError):
                train.adapter_modules(adapter)

    def test_targets_match_saved_sft_without_silent_filtering(self):
        names = ["model.language_model.q_proj", "model.vision_tower.q_proj"]
        self.assertEqual(train.resolve_targets(names[:1], names), names[:1])
        self.assertEqual(train.resolve_targets(names, names), names)
        self.assertEqual(train.non_language_targets(names), names[1:])
        with self.assertRaises(ValueError):
            train.resolve_targets(["q_proj"], names)

    def test_legacy_text_route(self):
        class Legacy:
            is_vision_model = True

            def _prepare_dataset(self, value):
                return self.is_vision_model, value

        instance = train.make_trainer_class(Legacy)()
        self.assertEqual(instance._prepare_dataset("data"), (False, "data"))

    def test_modern_route_unchanged(self):
        class Modern:
            _is_vlm = False

            def _prepare_dataset(self, value):
                return self._is_vlm, value

        instance = train.make_trainer_class(Modern)()
        self.assertEqual(instance._prepare_dataset("data"), (False, "data"))
        self.assertFalse(hasattr(instance, "is_vision_model"))

    def test_archived_dpo_defaults(self):
        args = train.parse_args([])
        cfg = train.training_kwargs(args, Path("out"))
        self.assertEqual(cfg["beta"], 0.1)
        self.assertEqual(cfg["num_train_epochs"], 3)
        self.assertEqual(cfg["gradient_accumulation_steps"], 16)
        self.assertEqual(cfg["learning_rate"], 5e-5)
        self.assertEqual(cfg["max_length"], 1024)
        self.assertNotIn("max_prompt_length", cfg)
        self.assertNotIn("max_steps", cfg)

    def test_smoke_uses_same_length_and_accumulation(self):
        args = train.parse_args(["--smoke_test"])
        cfg = train.training_kwargs(args, Path("out_SMOKE"))
        self.assertEqual(cfg["max_steps"], 2)
        self.assertEqual(cfg["max_length"], 1024)
        self.assertEqual(cfg["gradient_accumulation_steps"], 16)
        self.assertEqual(cfg["save_strategy"], "no")

    def test_explicit_prompt_override(self):
        args = train.parse_args(["--max_prompt_length", "400"])
        self.assertEqual(train.training_kwargs(args, Path("out"))["max_prompt_length"], 400)

    def test_empty_completion_guard(self):
        args = SimpleNamespace(max_length=4, truncation_mode="keep_start")
        row = {"prompt_input_ids": [1, 2, 3, 4], "chosen_input_ids": [5, 6], "rejected_input_ids": [7, 8]}
        trainer = SimpleNamespace(args=args, train_dataset=[row])
        with self.assertRaises(ValueError):
            train.audit_completion_tokens(trainer, strict=True)
        row["prompt_input_ids"] = [1, 2]
        report = train.audit_completion_tokens(trainer)
        self.assertEqual(report["minimum_scored_completion_tokens"], 2)

    def test_default_empty_completion_audit_does_not_block_or_filter(self):
        row = {"prompt_input_ids": [1, 2, 3, 4], "chosen_input_ids": [5, 6], "rejected_input_ids": [7, 8]}
        before = copy.deepcopy(row)
        trainer = SimpleNamespace(args=SimpleNamespace(max_length=4, truncation_mode="keep_start"),
                                  train_dataset=[row])
        report = train.audit_completion_tokens(trainer)
        self.assertEqual(report["pairs_with_empty_completion"], 1)
        self.assertEqual(report["first_empty_pair_indices"], [0])
        self.assertFalse(report["strict"])
        self.assertEqual(trainer.train_dataset, [before])
        self.assertFalse(train.parse_args([]).strict_token_audit)

    def test_keep_end_causal_first_token(self):
        self.assertEqual(train.retained_completion_counts(2, [5, 2], 4, "keep_end", "per_sequence"), [3, 2])

    def test_trl_flush_right_is_not_old_shared_slice(self):
        old = "input_ids = input_ids[:, -self.max_length :]"
        new = "attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)\n" + old
        self.assertEqual(train.legacy_end_alignment(old), "shared_padded")
        self.assertEqual(train.legacy_end_alignment(new), "per_sequence")
        # Previously the newer implementation was mistaken for the older one,
        # which incorrectly reported no short-answer tokens for this valid pair.
        self.assertEqual(train.retained_completion_counts(10, [30, 2], 16, "keep_end", "per_sequence"), [15, 2])
        self.assertEqual(train.retained_completion_counts(10, [30, 2], 16, "keep_end", "shared_padded"), [15, 0])

    def test_unknown_end_slice_does_not_guess(self):
        with self.assertRaises(RuntimeError):
            train.legacy_end_alignment("return some_unknown_truncation()")

    def test_relative_arguments_follow_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            self.assertEqual(train.under_root(root, Path("data/pairs.json"), "unused"), root / "data/pairs.json")
            self.assertEqual(train.under_root(root, None, "hf_cache/models"), root / "hf_cache/models")
            self.assertEqual(train.under_root(root, root / "absolute.json", "unused"), root / "absolute.json")

    def test_explicit_original_cache_and_revision(self):
        self.assertEqual(train.hf_load_kwargs(Path("cache"), "abc123"),
                         {"trust_remote_code": True, "cache_dir": "cache", "revision": "abc123"})
        self.assertNotIn("revision", train.hf_load_kwargs(Path("cache")))

    def test_empty_prompt_does_not_count_first_token(self):
        self.assertEqual(train.retained_completion_counts(0, [1, 3], 4, "keep_start", "not_needed"), [0, 2])

    def test_eos_only_is_reported_without_filtering(self):
        row = {"prompt_input_ids": [1, 2], "chosen_input_ids": [99], "rejected_input_ids": [7, 99]}
        trainer = SimpleNamespace(args=SimpleNamespace(max_length=4, truncation_mode="keep_start"),
                                  processing_class=SimpleNamespace(eos_token_id=99), train_dataset=[row])
        self.assertEqual(train.audit_completion_tokens(trainer)["pairs_with_eos_only_completion"], 1)
        self.assertEqual(len(trainer.train_dataset), 1)


if __name__ == "__main__":
    unittest.main()
