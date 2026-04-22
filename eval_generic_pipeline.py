#!/usr/bin/env python3
"""
Generic Pipeline — Multi-LLM Evaluation
========================================
Tests multiple LLMs on BOTH generic pipeline outputs:
  - data/processed/messages_generic/       (messages.log)
  - data/processed/messages_gen_generic/   (messages_gen.log)

Each log gets 10 ground-truth questions verified against actual chain data.

Backends:
  cohere  : COHERE_API_KEY=...

Usage:
    # Cohere
    COHERE_API_KEY=... COHERE_MODEL=command-r-08-2024 .venv/bin/python eval_generic_pipeline.py
"""

import os, json, time, textwrap
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BASE     = Path(__file__).parent
EVAL_DIR = BASE / "data/processed/generic_eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT_TOKENS = 6000

# qwen3-32b has a 6k TPM limit — auto-reduce context for small-window models
MODEL_TOKEN_LIMITS = {
    "qwen/qwen3-32b"                          : 2000,
    "llama-3.1-8b-instant"                    : 2000,
    "allam-2-7b"                              : 2000,
    "moonshotai/kimi-k2-instruct"             : 2000,
    "moonshotai/kimi-k2-instruct-0905"        : 2000,
}

# ── Backend ────────────────────────────────────────────────────────────────────
BACKEND = "cohere"
import cohere
MODEL   = os.environ.get("COHERE_MODEL", "command-r-08-2024")
_cohere = cohere.ClientV2(os.environ.get("COHERE_API_KEY", ""), timeout=360.0)
print(f"Backend : Cohere  |  Model : {MODEL}")


# ── Ground-truth Q&A — verified against actual chain data ─────────────────────
# Each entry: id, question, keywords_correct, keywords_partial, ground_truth

QUESTIONS_MESSAGES = [
    {
        "id": "M-Q1",
        "question": "What is the most dominant signal category in these causal chains?",
        "keywords_correct": ["port_link", "port", "link"],
        "keywords_partial":  ["port"],
        "ground_truth": "PORT_LINK (969 chains) — port/link state changes dominate",
    },
    {
        "id": "M-Q2",
        "question": "Which IP address appears most frequently across all chain steps?",
        "keywords_correct": ["10.233.131.22"],
        "keywords_partial":  ["10.233"],
        "ground_truth": "10.233.131.22 (5,680 step occurrences)",
    },
    {
        "id": "M-Q3",
        "question": "Which PID appears most frequently in BGP-related chain steps?",
        "keywords_correct": ["4714"],
        "keywords_partial":  ["471"],
        "ground_truth": "pid:4714 (235 occurrences in BGP steps)",
    },
    {
        "id": "M-Q4",
        "question": "How many causal chains have BGP as their terminal signal?",
        "keywords_correct": ["216"],
        "keywords_partial":  ["200", "21"],
        "ground_truth": "216 chains end with BGP signal",
    },
    {
        "id": "M-Q5",
        "question": "What category most commonly precedes PORT_LINK signal events in the chains?",
        "keywords_correct": ["kernel", "api", "infra", "bgp"],
        "keywords_partial":  ["kernel", "api"],
        "ground_truth": "KERNEL and API categories most commonly precede PORT_LINK signals",
    },
    {
        "id": "M-Q6",
        "question": "What is the approximate time span covered by these chains?",
        "keywords_correct": ["11:00", "11:16", "16 min", "16 minutes"],
        "keywords_partial":  ["11:0", "16"],
        "ground_truth": "11:00:23 to 11:16:24 — approximately 16 minutes",
    },
    {
        "id": "M-Q7",
        "question": "Is there any evidence of OSPF routing protocol events in these chains?",
        "keywords_correct": ["yes", "ospf"],
        "keywords_partial":  ["ospf"],
        "ground_truth": "Yes — OSPF appears as a signal category with 51 chains",
    },
    {
        "id": "M-Q8",
        "question": "Does IP address 192.168.0.1 appear in any chain step?",
        "keywords_correct": ["no", "not", "does not", "absent"],
        "keywords_partial":  ["no"],
        "ground_truth": "No — 192.168.0.1 is not present. Main IP is 10.233.131.22",
    },
    {
        "id": "M-Q9",
        "question": "Which daemon PID is associated with INFRA category events?",
        "keywords_correct": ["1210"],
        "keywords_partial":  ["121"],
        "ground_truth": "pid:1210 (417 occurrences in INFRA steps)",
    },
    {
        "id": "M-Q10",
        "question": "Summarise the top 3 failure types in this log based on the chains.",
        "keywords_correct": ["port", "api", "bgp"],
        "keywords_partial":  ["port", "bgp"],
        "ground_truth": "1. PORT_LINK (969 chains), 2. API (523 chains), 3. BGP (216 chains)",
    },
]

QUESTIONS_MESSAGES_GEN = [
    {
        "id": "G-Q1",
        "question": "What is the most dominant signal category in these causal chains?",
        "keywords_correct": ["port_link", "port", "link"],
        "keywords_partial":  ["port"],
        "ground_truth": "PORT_LINK (860 chains) — port/link state changes dominate",
    },
    {
        "id": "G-Q2",
        "question": "Which IP address appears most frequently across all chain steps?",
        "keywords_correct": ["10.0.0.2"],
        "keywords_partial":  ["10.0.0"],
        "ground_truth": "10.0.0.2 (1,518 step occurrences)",
    },
    {
        "id": "G-Q3",
        "question": "How many causal chains have BGP as their terminal signal?",
        "keywords_correct": ["387"],
        "keywords_partial":  ["38", "380"],
        "ground_truth": "387 chains end with BGP signal",
    },
    {
        "id": "G-Q4",
        "question": "How many causal chains have OSPF as their terminal signal?",
        "keywords_correct": ["177"],
        "keywords_partial":  ["17", "180"],
        "ground_truth": "177 chains end with OSPF signal",
    },
    {
        "id": "G-Q5",
        "question": "What is the most common two-step causal sequence in the chains?",
        "keywords_correct": ["bgp", "bgp→bgp", "bgp -> bgp", "bgp to bgp"],
        "keywords_partial":  ["bgp"],
        "ground_truth": "BGP → BGP (1,858 occurrences) — BGP events chain into themselves",
    },
    {
        "id": "G-Q6",
        "question": "What is the approximate time span covered by these chains?",
        "keywords_correct": ["14:43", "15:02", "18 min", "18 minutes"],
        "keywords_partial":  ["14:4", "15:0", "18"],
        "ground_truth": "14:43:55 to 15:02:28 — approximately 18 minutes",
    },
    {
        "id": "G-Q7",
        "question": "How many steps does the longest causal chain have, and what is its signal?",
        "keywords_correct": ["8", "vlan"],
        "keywords_partial":  ["8", "vlan"],
        "ground_truth": "8 steps, signal: VLAN",
    },
    {
        "id": "G-Q8",
        "question": "Is there any evidence of NTP (time synchronisation) events in these chains?",
        "keywords_correct": ["yes", "ntp"],
        "keywords_partial":  ["ntp"],
        "ground_truth": "Yes — NTP appears as a signal category with 91 chains",
    },
    {
        "id": "G-Q9",
        "question": "Does IP address 10.0.0.2 appear in the chains, and in which categories?",
        "keywords_correct": ["10.0.0.2", "bgp"],
        "keywords_partial":  ["10.0.0.2"],
        "ground_truth": "Yes — 10.0.0.2 appears 1,518 times, primarily in BGP steps",
    },
    {
        "id": "G-Q10",
        "question": "Summarise the top 3 failure types in this log based on the chains.",
        "keywords_correct": ["port", "bgp", "ospf"],
        "keywords_partial":  ["port", "bgp"],
        "ground_truth": "1. PORT_LINK (860 chains), 2. BGP (387 chains), 3. OSPF (177 chains)",
    },
]

# ── Context builder ────────────────────────────────────────────────────────────
def build_context(jsonl_path: Path, token_budget: int = MAX_CONTEXT_TOKENS) -> list:
    chains = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    selected, used = [], 0
    by_signal = defaultdict(list)
    for c in chains:
        by_signal[c["signal"]].append(c)
    buckets = [sorted(v, key=lambda x: -len(x["steps"])) for v in by_signal.values()]
    i = 0
    while any(buckets):
        bucket = buckets[i % len(buckets)]
        if not bucket:
            buckets.pop(i % len(buckets))
            continue
        c = bucket.pop(0)
        cost = len(json.dumps(c)) // 4
        if used + cost > token_budget:
            break
        selected.append(c)
        used += cost
        i += 1
    return selected, used

SYSTEM = textwrap.dedent("""\
    You are an expert network engineer analysing switch/router logs.
    Answer each question concisely and precisely based ONLY on the causal chain data provided.
    If the data does not contain enough information, say so explicitly.
    Do not guess — only state what is directly supported by the data.

    CAUSAL CHAINS (JSON, one per line):
    {chains}
""")

# ── LLM call ───────────────────────────────────────────────────────────────────
def ask_llm(chains: list, question: str, retries: int = 3) -> str:
    chain_text = "\n".join(json.dumps(c) for c in chains)
    prompt = SYSTEM.format(chains=chain_text) + f"\nQUESTION: {question}\nANSWER:"
    for attempt in range(retries):
        try:
            resp = _cohere.chat(
                model=MODEL,
                message=prompt,
                conversation_id="",  # needed for some models
                use_cache=False,
            )
            return resp.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"    ⚠️  Retry {attempt+1} ({wait}s): {str(e)[:80]}")
                time.sleep(wait)
            else:
                return f"ERROR: {str(e)[:200]}"

# ── Scoring ────────────────────────────────────────────────────────────────────
def score(answer: str, q: dict) -> str:
    a = answer.lower()
    if any(kw.lower() in a for kw in q["keywords_correct"]):
        return "correct"
    if any(kw.lower() in a for kw in q["keywords_partial"]):
        return "partial"
    return "wrong"

# ── Run one log ────────────────────────────────────────────────────────────────
def run_log(log_label: str, jsonl_path: Path, questions: list) -> dict:
    print(f"\n{'='*65}")
    print(f"  LOG: {log_label}")
    print(f"{'='*65}")

    ctx_budget = MODEL_TOKEN_LIMITS.get(MODEL, MAX_CONTEXT_TOKENS)
    ctx, ctx_tokens = build_context(jsonl_path, ctx_budget)
    print(f"  Context: {len(ctx)} chains, ~{ctx_tokens:,} tokens\n")

    results = []
    counts  = defaultdict(int)

    for q in questions:
        print(f"  [{q['id']}] {q['question'][:60]}...")
        answer  = ask_llm(ctx, q["question"])
        verdict = score(answer, q)
        icon    = "✅" if verdict == "correct" else ("⚠️ " if verdict == "partial" else "❌")
        print(f"    {icon} {verdict.upper()}  |  {answer[:100]}")
        counts[verdict] += 1
        results.append({
            "id": q["id"], "question": q["question"],
            "ground_truth": q["ground_truth"],
            "answer": answer, "verdict": verdict,
        })
        time.sleep(1)

    total = len(questions)
    score_pct = (counts["correct"] + 0.5 * counts["partial"]) / total * 100
    print(f"\n  Score: {counts['correct']} correct, {counts['partial']} partial, "
          f"{counts['wrong']} wrong  →  {score_pct:.1f}%")

    return {
        "log": log_label, "model": MODEL, "backend": BACKEND,
        "context_chains": len(ctx), "context_tokens": ctx_tokens,
        "correct": counts["correct"], "partial": counts["partial"],
        "wrong": counts["wrong"], "total": total,
        "score": round(score_pct, 1),
        "results": results,
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_tag = f"{BACKEND}_{MODEL.split('/')[-1].replace(':','_')}"

    logs = [
        (
            "messages.log (generic)",
            BASE / "data/processed/messages_generic/causal_graph/causal_chains_llm.jsonl",
            QUESTIONS_MESSAGES,
        ),
        (
            "messages_gen.log (generic)",
            BASE / "data/processed/messages_gen_generic/causal_graph/causal_chains_llm.jsonl",
            QUESTIONS_MESSAGES_GEN,
        ),
    ]

    all_results = []
    for log_label, jsonl_path, questions in logs:
        # Note: if the log file doesn't exist, we skip it gracefully
        if not os.path.exists(jsonl_path):
            print(f"Skipping {log_label} - file not found at {jsonl_path}")
            continue
        result = run_log(log_label, jsonl_path, questions)
        all_results.append(result)
        
    if not all_results:
        print("No logs were found or processed.")
        return

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS  |  {BACKEND.upper()} {MODEL}")
    print(f"{'='*65}")
    print(f"  {'Log':<35} {'Correct':>8} {'Partial':>8} {'Wrong':>8} {'Score':>8}")
    print(f"  {'-'*63}")
    for r in all_results:
        print(f"  {r['log']:<35} {r['correct']:>8} {r['partial']:>8} {r['wrong']:>8} {r['score']:>7.1f}%")
    overall = sum(r['score'] for r in all_results) / len(all_results)
    print(f"  {'-'*63}")
    print(f"  {'AVERAGE':<35} {'':>8} {'':>8} {'':>8} {overall:>7.1f}%")
    print(f"{'='*65}")

    # ── Save results ───────────────────────────────────────────────────────────
    out_path = EVAL_DIR / f"results_{backend_tag}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL, "backend": BACKEND,
            "timestamp": datetime.now().isoformat(),
            "overall_avg": round(overall, 1),
            "logs": all_results,
        }, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    # ── Save human-readable text report ───────────────────────────────────────
    txt_path = EVAL_DIR / f"results_{backend_tag}_{timestamp}.txt"
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write(f"GENERIC PIPELINE EVALUATION\n")
        f.write(f"Model  : {MODEL}\n")
        f.write(f"Backend: {BACKEND}\n")
        f.write(f"Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*65 + "\n\n")
        for r in all_results:
            f.write(f"LOG: {r['log']}\n")
            f.write(f"Score: {r['score']}%  ({r['correct']} correct, {r['partial']} partial, {r['wrong']} wrong)\n")
            f.write("-"*65 + "\n")
            for res in r["results"]:
                icon = "✅" if res["verdict"]=="correct" else ("⚠️ " if res["verdict"]=="partial" else "❌")
                f.write(f"\n[{res['id']}] {icon} {res['verdict'].upper()}\n")
                f.write(f"Q  : {res['question']}\n")
                f.write(f"GT : {res['ground_truth']}\n")
                f.write(f"LLM: {res['answer'][:300]}\n")
            f.write("\n")
        f.write(f"\nOVERALL AVERAGE: {overall:.1f}%\n")
    print(f"  Text report  → {txt_path}")

if __name__ == "__main__":
    main()
