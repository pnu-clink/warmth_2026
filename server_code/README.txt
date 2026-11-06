Historical server sources

These 14 files were supplied from the research server. The release changes only
absolute filesystem paths and identifying path examples. Model IDs, prompts,
training hyperparameters, decoding, answer extraction and analysis algorithms
are preserved. See ../docs/SERVER_SOURCE_PROVENANCE.json for SHA-256 hashes.

RUN LOCATION
Use a private working directory containing data/, adapters and heldout_prompts.csv.
Set WARM_MODEL_ROOT to its absolute path and run commands from that directory.
Without that variable, the current working directory is used. Relative --data,
CSV inputs and output names still resolve from the current directory.
Do not put private inputs, credentials or generated model weights in this repo.

Example in the existing Linux/Docker environment (paths are illustrative):
  export WARM_MODEL_ROOT=/workspace/warm_models
  cd /workspace/warm_models
  python /path/to/repository/server_code/build_dpo_pairs.py --warm data/dataset_warm_final.json --neutral data/dataset_w_sampled_ready.json --out data/dpo_pairs.json
  python /path/to/repository/server_code/train_lora_dpo.py --model phi4 --data data/dpo_pairs.json --output_dir new_runs/phi4_dpo

These commands are real training/data-generation commands, not offline checks.
Choose new output paths/directories. Some historical scripts overwrite fixed CSVs
or adapter folders. Do not run them in a directory containing irreplaceable runs.

SOURCE MAP
Original-SFT (called neutral in historical names):
  train_lora_neutral.py                 --model qwen|llama|phi4, --data
  train_lora_neutral_gemma.py           Gemma; fixed constants, no CLI

DPO:
  build_dpo_pairs.py                    --warm, --neutral, --out
  train_lora_dpo.py                     --model qwen|llama|phi4, --data, --output_dir
  Gemma DPO source is ../experiments/gemma_dpo/train_lora_dpo_gemma.py

MMLU-Pro:
  eval_mmlu_neutral.py                  --model qwen|llama|phi4
  evaluate_mmlu_pro_neutral_gemma.py    Gemma Original-SFT; fixed constants
  gemma_evaluate_mmlu.py                Gemma Base/Warm-SFT; fixed constants
  eval_mmlu_dpo.py                      --model qwen|llama|phi4|gemma

SocioT:
  generate_society_dpo.py               --model, --prompts, --out
  analyze_sociot.py                     scored CSV files; --no_plot available
  Scoring itself is in ../peft_handoff/scripts/score_sociot.py and reference/.

Diagnostics:
  analyze_lora_space.py                 --root, --subspace, --top_k
  analyze_lora_baseline.py              --root, --seed
  compare_two_adapters.py               --a, --b, --out
  measure_kl.py                        --model, --responses, --text_from, --out

PRESERVED BEHAVIOR AND CAVEATS
- Gemma Original-SFT trains at top level. Do not import it or use --help as a
  harmless check. It requires the compatible multimodal Transformers class.
- Some evaluation scripts without argparse also ignore --help; do not use a
  blanket command that executes every file with --help.
- DPO starts from Base, not a pre-trained Warm-SFT adapter.
- In train_lora_dpo.py, --max_prompt_length is currently parsed but unused.
  The effective prompt/completion truncation follows installed TRL behavior.
- build_dpo_pairs.py defaults to the last aligned assistant turn, merges
  consecutive same-role messages, and applies its recorded role/length filters.
  Prompts use the warm conversation prefix. --all_turns changes the design.
  An assistant-role alias not listed by normalize() is not silently remapped.
  Compare any regenerated data with the historical pair count/hash.
- Historical evaluation scripts may re-sample upstream datasets. Match IDs,
  references and source revisions before treating a rerun as item-paired.
- Historical print messages/comments are retained as source records; they are
  not a replacement for the manuscript's final claims or inferential analysis.
- trust_remote_code=True is retained where present. Use approved model sources.
- No original Docker image/digest was supplied with these scripts. The observed
  six package versions are not a complete, tested install lockfile.

AUTHENTICATION
Use HF_TOKEN (when required for model access) through the container environment
or a local Hugging Face login outside the repository. Judge/rewrite API entry
points use OPENAI_API_KEY through environment/local .env. No real key is shipped.

SAFE OFFLINE CHECKS (from repository root, not the private working directory)
  python scripts/verify_submission.py
  python -B scripts/test_release_safety.py
  python -B scripts/test_numeric_core.py

No GPU training, model download or paid API request is part of these checks.
