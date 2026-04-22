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

---

## 2. Semantic Compression Pipeline

The core engine responsible for shrinking 250,000+ byte support bundles down into ~1,500 byte payloads. This pipeline strips out the noise while perfectly preserving the causal root event.

### The 5 Stages of Compression:

1. **Ingestion & Parsing (Layer 1)**: The system pulls in unstructured `messages.log`, parses metric tables from `routeinfo.txt`, and absorbs diagnostic telemetry from `showtech.txt`.
2. **Mining & Deduplication (Layer 2)**: Identifies verbose, repeating templates (like routine spanning-tree updates) and deduplicates them. It calculates the strict "state deltas" — recording only when a system drastically changed states.
3. **Semantic Embeddings (Layer 3)**: Utilizing HuggingFace's `sentence-transformers/all-MiniLM-L6-v2`, the pipeline mathematically maps the raw syslog text meaning into N-dimensional vectors.
4. **Dependency Graph & Noise Neutralization (Layer 4)**: It builds a chronological dependency graph linking events causally. The built-in **Noise Filter** algorithms detach and drop isolated graph nodes that have zero correlation to the primary fault cluster, achieving >99% volume reduction.
5. **Prompt Formatting (Layer 5)**: Reassembles the concentrated fault nodes into an optimized Markdown string explicitly built to maximize LLM context interpretation.

---

## 3. Evaluation Harness (Phase 1 & Generic Pipeline)

The `run_eval.py` framework tests how an LLM inherently performs on the massive raw log versus the distilled compressed log. 

### Phase 1: Core Inference & Compression Ratio
- **Raw RCM Pass vs Comp RCM Pass**: The pipeline sends both payloads strictly to **Cohere** via the API wrapper. If the LLM predicts the exact entity (e.g. `1/1/3`) and fault type as defined in `metadata.json`, it scores a **PASS**.
- **Compression Ratio (CR)**: Calculates precisely how much data was mitigated (e.g., *167.3x reduction / 99.4%*).

### Generic Questions Pipeline
After the primary bundle Phase 1 assessment completes, the test framework will automatically trigger the `eval_generic_pipeline.py` script. This secondary pipeline natively prompts the Cohere model with 10 generic, ground-truth-verified qualitative questions regarding the network's extracted causal chains (e.g., identifying the most dominant signal categories or IP addresses) to stress test context comprehension over isolated events.

### Results Output
The runtime gracefully exports all diagnostic metrics per bundle inside a local `./inference/` folder as structured JSON, while simultaneously dumping the exact finalized markdown context into the `./compressed_logs/` directory for manual inspection.

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
