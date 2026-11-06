Warmer Language Models, Different Errors: Adaptation Choices and Error Reduction
Anonymous Software and Data release with privacy-redacted text

Extract warmth_software.zip and warmth_data.zip into the SAME directory. Both
archives contain a warmth_review/ root, with no overlapping archive paths.
Use the two updated archives together; the older Software manifest will not
match the expanded Data archive.

Quick start from warmth_review/ (Python 3.11 or newer):
  python -B scripts/verify_submission.py
  python -m pip install -r requirements-analysis.txt
  python -B scripts/reproduce_available.py --output ../reproduced_results

Choose a NEW output directory outside warmth_review/. Numeric reproduction is
CPU-only and makes no model downloads, API requests, training runs or uploads.

Software
- Training, evaluation, judging and analysis entry points remain unchanged.
  Only the packaging credential validator receives a token-prefix boundary
  fix, preventing ordinary URL words from being mistaken for provider keys.
- Available training, evaluation, SocioT, judging, KL, LoRA and statistical code,
  requirements, source notices and offline tests are retained.

Data
- Existing paired correctness/LLM-judge labels, all 12 DPO+Steering cells, DoRA
  aggregate counts, 32 method summaries, SycoBench trials and numerical diagnostics.
- Privacy-redacted Original and Warm training conversations, 1,526 IDs each.
- Full-context generated responses for the historical Base/Original-SFT/Warm-SFT/
  Warm-DPO conditions and Warm-DPO+Steering on the three knowledge benchmarks.
- Saved LLM-judge reasons and step-by-step assessments where archived.
- Available held-out SocioT prompts/responses and SycoBench pressure responses.
- A text schema, field-level redaction log and privacy-check scope.

The JSONL text files link to numerical inputs by benchmark, model, condition and
question ID; historical knowledge records also retain their recorded record ID.
No text is executed. Correct responses have label 0 and no fabricated rationale.
Some archived Other labels originate from operational judge failures, which are
retained and flagged. No fresh judgments were generated for this release.

Privacy and provenance
Credential-shaped strings, contact details, user home paths, self-identifying
names/profile links and known research-author markers are replaced with typed
REDACTED placeholders in the released text. Original local research files and
all frozen numerical results remain unchanged. No records or turns are removed
by release preparation. Redacted copies need not reproduce the historical text
byte-for-byte; historical scores and lengths are not recomputed from those copies.
Heuristic checks and targeted review cannot guarantee that public text cannot
be re-identified by search. See data/text/privacy_review.json and
data/text/redaction_log.json for the performed checks and their limits.

Individual human annotator votes and confidence scores remain withheld.
Human agreement cannot be independently recomputed from this release.

Limits
Complete collaborator Steering/DoRA implementations, all collaborator text
traces, the original Docker image digest and model weights are not included.
This is not a complete end-to-end replay of every evaluated method.
The preserved reproduce_summaries.py still requires withheld human labels;
reproduce_available.py is the current numeric entry point.

Licensing
The data retain their applicable upstream licenses and attribution notices.
In particular, inherited FinEval material remains subject to CC-BY-NC-4.0.
No new blanket license for all repository code is assigned by this package.
Public release of the available code on GitHub and the redacted warmth-rewritten
training conversations on Hugging Face is planned, with applicable license,
privacy and GPT-4o-mini modification notices retained. No public upload is
performed by preparing these review archives.

Read README_DATA.txt and docs/DATA_ACCESS.txt, docs/MISSING.txt,
docs/REPRODUCTION.txt and docs/PAPER_DATA_VERSION.json for the precise scope.
