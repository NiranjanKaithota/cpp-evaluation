# HPE Support Bundle Evaluation Pipeline

This repository contains the HPE Support Bundle Evaluation Pipeline: an end-to-end framework that compresses large network diagnostic bundles and evaluates whether LLMs can still perform accurate root-cause diagnosis on the compressed payload.

Contents in this README
- Quickstart (install, env)
- High-level features and key upgrades
- Data formats supported
- Two-phase evaluation harness (orchestrator / run_eval)
- Usage examples and flags
- Directory layout and ground-truth format

---

## Quickstart

1) Install dependencies

```powershell
pip install -r requirements.txt
# or (if you prefer explicit):
pip install tiktoken cohere google-generativeai
```

2) Set API keys (example `.env` or shell vars)

```env
COHERE_API_KEY="your_cohere_key_here"
COHERE_MODEL="command-r-08-2024"
GEMINI_API_KEY="your_gemini_key_here"
```

3) Run an evaluation (examples below).

---

## Features & Key Upgrades

- Domain-agnostic: supports HPE network bundles and generic logs (.log, .csv, Kubernetes audit logs) with domain-aware question routing.
- Multi-format: evaluates `text`, `graph` (causal summaries), `json`, and `toon` compression outputs.
- Exact tokenization & anti-crash guards: uses `tiktoken` / Cohere tokenizers and bounds payloads (~80k tokens) to avoid API size errors.
- 3-pass self-consistency LLM judge: majority-vote grading across 3 passes with an "Absent Entity Rule" to handle trap questions.
- Resilient JSON parsing: aggressive JSON hunter + a 3-attempt retry loop to mitigate API truncation.
- Adversarial cross-model testing: primary inference on Cohere, adversarial question generation with Gemini.

---

## Data Generation & Supported Structures

Two primary paradigms are supported:

1) Strict Support Bundles (networking)
- `messages.log` — raw syslog streams
- `routeinfo.txt` — RIB/FIB routing tables (OSPF, BGP)
- `showtech.txt` — hardware diagnostics
- `metadata.json` — ground truth (failure entity, type, remediation)

2) Agnostic Logs
- Single `.log` or `.csv` inputs (e.g., Kubernetes audit CSVs). When metadata is missing, the orchestrator can generate a synthetic bundle and route to domain-specific question banks. Use `--skip-raw-inference` to bypass strict bundle checks.

---

## Evaluation Harness (high level)

The pipeline provides two entrypoints:
- `orchestrator.py` — higher-level workflow that supports domain-agnostic flows and orchestrates both phases.
- `run_eval.py` — Phase-1 focused entrypoint used in older workflows (kept for compatibility).

Two phases:
- Phase 1 (RCM): run inference on raw vs compressed payloads and compute compression ratio. Exact matches to `metadata.json` are scored as PASS.
- Phase 2 (Generic / Domain-aware Q&A): runs ~30 (configurable) qualitative questions, graded by the 3-pass LLM judge, and produces JSON reports in `inference/` and human-readable contexts in `compressed_logs/`.

---

## Usage examples

Orchestrator — standard network bundle

```powershell
python orchestrator.py --raw-path bundle_001_port_flap_fiber --compressed-name bundle_001_port_flap_fiber --pipeline-type toon
```

Orchestrator — agnostic logs (skip Phase 1 raw inference)

```powershell
python orchestrator.py --raw-path k8s_test_bundle --compressed-name k8s_test_bundle --pipeline-type json --skip-raw-inference
```

Legacy/run-eval example

```powershell
python run_eval.py --dataset evaluation_dataset --provider cohere --model command-r-08-2024 --bundle HPE-test-bundle --pipeline-type toon --use-existing-compressed
```

---

## Command flags (summary)

- `--raw-path` : raw bundle folder name (e.g., `bundle_001_port_flap_fiber`)
- `--compressed-name` : target compressed name/file to evaluate
- `--pipeline-type` : `text` | `graph` | `json` | `toon`
- `--skip-raw-inference` : bypass Phase 1 raw processing (for agnostic logs)
- `--use-existing-compressed` : use pre-compressed artifacts under `dev_compressed_logs/`
- `--provider` / `--model` : LLM provider and model used for inference
- `--mock-llm` : dry-run checks without calling external APIs

---

## Supported pipeline types

- `text`  — stitches together `.txt` summaries
- `graph` — markdown/JSON causal summaries
- `json`  — unified `incident_context.json`
- `toon`  — Token Oriented Object Notation (`.toon`) files

---

## Evaluation metrics (Phase 2)

Grades are determined by majority vote across 3 passes. Typical weighting:
- Root Cause Attribution — 35%
- Temporal Ordering — 20%
- Causal Reasoning — 20%
- Semantic Equivalence — 10%
- Negative Evidence — 10% (subject to Absent Entity Rule)
- Context Retrieval — 5%

---

## Dynamic / Adversarial checks

When `metadata.json` is present, Phase 2 appends:
- Actionability checks: exact CLI remediation extraction for the final report
- Gemini adversarial traps: negative distractors, deep causality probes, and multi-hop trace questions

---

## Directory layout (representative)

```
cpp-evaluation/
├── batch_generator.py
├── bundle_generator.py
├── orchestrator.py
├── run_eval.py
├── eval_generic_pipeline.py
├── compression_pipeline/
│   ├── __init__.py
│   ├── main.py
│   └── layer*/
├── evaluation/
│   ├── llm_wrapper.py
│   ├── loader.py
│   └── metrics.py
├── evaluation_dataset/
├── dev_compressed_logs/
├── compressed_logs/
├── inference/
├── question_response/
└── README.md
```

---

## Ground truth format (`metadata.json`) — example

```json
{
  "bundle_version": "1.0",
  "generated_at": "2026-04-20T23:03:49.564441",
  "hostname": "PSCSCTLEAFB03",
  "ground_truth": {
    "scenario_name": "vlan_mismatch",
    "description": "VLAN mismatch on 1/1/12: expected 10, got 20",
    "severity": "medium",
    "failure_timestamp": "2026-04-20T23:19:21.878000",
    "root_cause_entity": "1/1/12",
    "root_cause_type": "configuration_error",
    "expected_symptoms": [
      "frame_drops",
      "vlan_violation",
      "connectivity_partial"
    ],
    "recommended_action": "Verify VLAN configuration on 1/1/12 and connected device"
  }
}
```

---

If you'd like, I can also:
- add a short troubleshooting section for common environment issues,
- or run a quick grep to ensure all referenced entrypoints (`orchestrator.py`, `run_eval.py`) exist and wire examples to actual scripts.