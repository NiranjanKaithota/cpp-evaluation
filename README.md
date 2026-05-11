# HPE Support Bundle Evaluation Pipeline

This repository hosts the **HPE Support Bundle Evaluation Pipeline**, a sophisticated end-to-end framework designed to drastically reduce the sheer volume of network telemetry data via semantic compression, while proving mathematically that Large Language Models (LLMs) can still successfully diagnose network faults on the compressed payload with 100% fidelity.

---

## 1. Data Generation

Network telemetry is notoriously vast and noisy. To simulate real-world environments, this project utilizes a custom data generator tool.

### Structure of a Support Bundle
When executing data generation (e.g., `python batch_generator.py suite --name regression_test`), the framework produces deeply realistic support bundles mimicking actual enterprise router and switch arrays.
Each support bundle contains:
- `messages.log`: Raw syslog streams detailing all systemic and networking events chronologically.
- `routeinfo.txt`: Rib/Fib routing tables containing protocol states (OSPF, BGP, connected routes).
- `showtech.txt`: Deep hardware and software diagnostic metrics (interface states, transceiver power levels).
- `metadata.json`: The **Ground Truth** failure state containing the exact failure mechanism, interface, and chronological symptoms.

<!-- ---

## 2. Evaluation Harness (Phase 1 & Generic Pipeline)

The `run_eval.py` framework tests how an LLM inherently performs on the massive raw log versus the distilled compressed log. 

### Phase 1: Core Inference & Compression Ratio
- **Raw RCM Pass vs Comp RCM Pass**: The pipeline sends both payloads strictly to **Cohere** via the API wrapper. If the LLM predicts the exact entity (e.g. `1/1/3`) and fault type as defined in `metadata.json`, it scores a **PASS**.
- **Compression Ratio (CR)**: Calculates precisely how much data was mitigated (e.g., *167.3x reduction / 99.4%*).

### Generic Questions Pipeline
After the primary bundle Phase 1 assessment completes, the test framework will automatically trigger the `eval_generic_pipeline.py` script. This secondary pipeline natively prompts the Cohere model with 10 generic, ground-truth-verified qualitative questions regarding the network's extracted causal chains (e.g., identifying the most dominant signal categories or IP addresses) to stress test context comprehension over isolated events.

### Results Output
The runtime gracefully exports all diagnostic metrics per bundle inside a local `./inference/` folder as structured JSON, while simultaneously dumping the exact finalized markdown context into the `./compressed_logs/` directory for manual inspection. -->

---

## Quickstart Guide

### 1. Requirements
Install the required libraries. Ensure you have network access for HuggingFace model pulling.
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` in the root directory to authorize queries. The framework runs exclusively on Cohere.
```env
COHERE_API_KEY="your_cohere_key_here"
```

### 3. Execution
Launch the evaluation harness against a specific bundle or a full batch. The workflow will run Phase 1 evaluations first, then automatically transition to the Generic Pipeline.

```bash
# Run on a single bundle using Cohere's Developer API natively
python run_eval.py --dataset ./evaluation_dataset/bundle_001_port_flap_fiber
```

---

# Network Log Compression Evaluator

An automated, LLM-driven evaluation pipeline designed to benchmark different compression and summarization strategies for massive network support bundles.

When dealing with 5MB+ diagnostic logs, LLMs hit context limits and token density issues. This pipeline tests whether compressed representations (Text, Graph, JSON, or TOON formats) preserve the critical causal chains, temporal ordering, and actionable configuration states required for automated network troubleshooting.

---

## 🚀 Features

- **Multi-Format Support**: Natively loads and evaluates `text`, `graph`, `json`, and `toon` compression outputs.
- **Smart Context Truncation**: Uses `tiktoken` for exact, offline token counting to safely maximize the 128k context window without API crashes.
- **Two-Phase Evaluation**:
  - **Phase 1 (RCM)**: Tests immediate root cause identification and compression ratio/size reductions.
  - **Phase 2 (Generic Q&A)**: A rigorous 30+ question LLM-as-a-Judge pipeline grading 6 distinct diagnostic categories.
- **Adversarial Cross-Model Testing**: Uses Cohere (`command-r`) as the primary diagnostic engine, while utilizing Google Gemini (`gemini-1.5-flash`) to dynamically generate hyper-specific, adversarial trap questions based on ground truth data.

---

## 📁 Directory Structure

The repo's actual layout used by the pipeline (trimmed for brevity) is shown below:

```plaintext
cpp-evaluation/
├── batch_generator.py
├── bundle_generator.py
├── bundle_generator.py
├── run_eval.py
├── eval_generic_pipeline.py
├── QUICKSTART.md
├── README.md
├── requirements.txt
├── compression_pipeline/                # Dummy Compressor Pipeline
│   ├── __init__.py
│   ├── main.py
│   ├── layer1_ingestion/
│   ├── layer2_compression/
│   ├── layer3_semantics/
│   ├── layer4_causality/
│   └── layer5_encoding/
├── evaluation/
│   ├── __init__.py
│   ├── compression_wrapper.py
│   ├── llm_wrapper.py
│   ├── loader.py
│   ├── metrics.py
│   └── report.py
├── evaluation_dataset/                  # Raw, uncompressed bundles
│   ├── bundle_001_port_flap_fiber/
│   ├── bundle_002_port_flap_copper/
│   └── HPE-test-bundle/
├── dev_compressed_logs/
│   ├── compressed_text_logs/
│   ├── compressed_graph_logs/
│   ├── compressed_json_logs/
│   └── compressed_toon_logs/
├── compressed_logs/                     # Generated compressed markdown/text
├── inference/                           # Evaluation output JSON reports
├── inference-copy/
├── question_response/
└── __pycache__/
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/NiranjanKaithota/cpp-evaluation.git
cd cpp-evaluation
```

### 2. Install Dependencies

```bash
pip install tiktoken cohere google-generativeai
```

### 3. Set Your API Keys

Export your keys as environment variables.

The pipeline requires:
- **Cohere** → Diagnostic engine
- **Gemini** → Dynamic question generation

#### Windows (CMD)

```cmd
set COHERE_API_KEY=your_cohere_key
set GEMINI_API_KEY=your_gemini_key
```

#### You can also use .env file to store the API keys

---

## 💻 Usage

Run the evaluation pipeline using `run_eval.py`.

The script will automatically:
- Locate compressed files
- Stitch them together
- Calculate compression ratios
- Trigger the Phase 2 Q&A evaluation

### Basic Command

```bash
python run_eval.py \
  --dataset evaluation_dataset \
  --provider cohere \
  --model command-r-08-2024 \
  --bundle <BUNDLE_NAME> \
  --pipeline-type <PIPELINE_TYPE> \
  --use-existing-compressed
```

## ⚙️ Command Flags Explained

| Flag | Description |
|---|---|
| `--dataset` | The directory containing your raw, uncompressed support bundles (e.g., `evaluation_dataset`). |
| `--provider` | The LLM backend to use for the primary evaluation (e.g., `cohere`). |
| `--model` | The specific model version to use for inference and grading (e.g., `command-r-08-2024`). |
| `--bundle` | The exact folder name of the specific log bundle you want to evaluate (e.g., `HPE-test-bundle`). |
| `--pipeline-type` | Specifies the format of the compressed logs being evaluated (`text`, `graph`, `json`, or `toon`). |
| `--use-existing-compressed` | A critical flag that tells the script to look for pre-compressed files inside the `dev_compressed_logs` directory rather than trying to compress the raw logs on the fly. |
| `--mock-llm` | A flag used to consistency of all the dependencies and paths that are initialed in the scripts. Does only the checks and doesn't run the evaluation pipeline. |


---

## 📌 Supported Pipeline Types (`--pipeline-type`)

| Pipeline Type | Description |
|---|---|
| `text` | Stitches together multiple `.txt` summaries |
| `graph` | Looks for markdown/JSON graph representations |
| `json` | Parses a unified `incident_context.json` file |
| `toon` | Loads Token Oriented Object Notation (`.toon`) files |

---

## ▶️ Example (Evaluating a TOON Compression)

```bash
python run_eval.py \
  --dataset evaluation_dataset \
  --provider cohere \
  --model command-r-08-2024 \
  --bundle HPE-test-bundle \
  --pipeline-type toon \
  --use-existing-compressed
```

---

## 📊 Evaluation Metrics (Phase 2)

The Phase 2 LLM Judge evaluates the compressed logs across 6 static categories:

### 1. Root Cause Attribution
Did the specific failing entity survive compression?

### 2. Causal Reasoning
Is the chain of events logically sound?

### 3. Temporal Ordering
Are timestamps and sequences preserved?

### 4. Semantic Equivalence
Does the high-level summary match the technical reality?

### 5. Negative Evidence
Did the compressor avoid hallucinating fake errors (e.g., DNS failures, Spanning Tree loops)?

### 6. Context Retrieval
Are exact IP addresses, MAC addresses, and VLAN IDs still present?

---

## 🧠 Dynamic / Adversarial Questions

If `metadata.json` contains valid Ground Truth data, the pipeline dynamically appends:

- **Actionability Checks**
  - Asking the LLM to generate exact CLI remediation commands.

- **Gemini Adversarial Traps**
  - Generating 2(to be expanded further) highly specific, contextual trap questions to ensure the diagnostic model isn't just guessing.

---

## 📝 Ground Truth Format (`metadata.json`)

To enable dynamic questions, ensure your raw bundle directory contains a `metadata.json` file formatted like this (same used in bundle_006):

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