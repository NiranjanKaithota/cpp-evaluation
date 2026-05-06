#!/usr/bin/env python3
"""
Improved Generic Pipeline — Multi-LLM Evaluation
================================================
Enhanced evaluation framework testing semantic compression quality through
deep causal reasoning, temporal ordering, and hallucination detection.

Key Improvements:
- 7 question categories testing different compression aspects
- Multi-level scoring (entity extraction + semantic similarity + keywords)
- Hallucination traps and negative evidence testing
- Category-level performance reporting
- Maps directly to compression pipeline metrics (RCM, COPS, ER)

Backends:
  cohere  : COHERE_API_KEY=...

Usage:
    COHERE_API_KEY=... COHERE_MODEL=command-r-08-2024 python eval_generic_pipeline.py
"""

import os
import json
import time
import textwrap
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

BASE     = Path(__file__).parent
EVAL_DIR = BASE / "question_response"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT_TOKENS = 6000

MODEL_TOKEN_LIMITS = {
    "qwen/qwen3-32b": 2000,
    "llama-3.1-8b-instant": 2000,
    "allam-2-7b": 2000,
    "moonshotai/kimi-k2-instruct": 2000,
    "moonshotai/kimi-k2-instruct-0905": 2000,
}

# Backend setup
BACKEND = "cohere"
import cohere
MODEL   = os.environ.get("COHERE_MODEL", "command-r-08-2024")
_cohere = cohere.ClientV2(os.environ.get("COHERE_API_KEY", ""), timeout=360.0)
print(f"Backend : Cohere  |  Model : {MODEL}")

# ═══════════════════════════════════════════════════════════════════════════
# IMPROVED QUESTION BANK
# ═══════════════════════════════════════════════════════════════════════════

# Category 1: ROOT CAUSE ATTRIBUTION (Critical - 35% weight)
# Maps to RCM (Root Cause Match) metric from evaluation plan
ROOT_CAUSE_QUESTIONS = [
    {
        "id": "RC-Q1",
        "category": "root_cause_attribution",
        "difficulty": "hard",
        "weight": 0.20,
        "question": "What is the root cause of the primary network issue? Identify the specific component that failed and explain the failure mechanism.",
        "scoring_method": "entity_extraction_plus_semantic",
        "requires_entities": True,
        "evaluation_notes": "Must identify correct entity (interface/IP/process) AND failure type. Partial credit for one but not both."
    },
    {
        "id": "RC-Q2",
        "category": "root_cause_attribution",
        "difficulty": "hard",
        "weight": 0.10,
        "question": "What remediation steps would you recommend to resolve this issue? Be specific about which components to check or reconfigure.",
        "scoring_method": "semantic_similarity",
        "evaluation_notes": "Compare against ground truth recommended_action. Actionability metric from evaluation plan."
    },
    {
        "id": "RC-Q3",
        "category": "root_cause_attribution",
        "difficulty": "medium",
        "weight": 0.05,
        "question": "Classify this failure: (A) Physical layer issue, (B) Configuration error, (C) Protocol/software failure, or (D) Resource exhaustion. Explain your choice.",
        "scoring_method": "classification_with_reasoning",
        "evaluation_notes": "Must match root_cause_type from metadata. Tests if compression preserves failure category."
    },
]

# Category 2: CAUSAL REASONING (High Priority - 20% weight)
# Tests Layer 4 of compression pipeline (causality graph preservation)
CAUSAL_REASONING_QUESTIONS = [
    {
        "id": "CR-Q1",
        "category": "causal_reasoning",
        "difficulty": "hard",
        "weight": 0.10,
        "question": "Trace the causal chain: what event triggered the failure, what symptoms appeared as a result, and what was the final observable impact?",
        "scoring_method": "causal_chain_validation",
        "requires_ordering": True,
        "evaluation_notes": "Tests multi-hop causality. Must show logical progression from root cause → symptoms → impact."
    },
    {
        "id": "CR-Q2",
        "category": "causal_reasoning",
        "difficulty": "hard",
        "weight": 0.10,
        "question": "Could the observed symptoms have been caused by something OTHER than the identified root cause? If yes, what evidence rules out those alternatives?",
        "scoring_method": "semantic_similarity",
        "evaluation_notes": "Tests diagnostic depth. Good compression should preserve discriminative evidence."
    },
]

# Category 3: TEMPORAL ORDERING (Critical - 20% weight)
# Maps to COPS (Causal Order Preservation Score) metric
TEMPORAL_ORDERING_QUESTIONS = [
    {
        "id": "TO-Q1",
        "category": "temporal_ordering",
        "difficulty": "medium",
        "weight": 0.10,
        "question": "List the 3-5 most critical events in chronological order. Include approximate timestamps if available.",
        "scoring_method": "temporal_sequence_validation",
        "requires_ordering": True,
        "evaluation_notes": "Direct test of COPS metric. Order matters more than exact timestamps."
    },
    {
        "id": "TO-Q2",
        "category": "temporal_ordering",
        "difficulty": "medium",
        "weight": 0.05,
        "question": "How long did it take from initial failure detection to complete service disruption? Describe the progression.",
        "scoring_method": "temporal_reasoning",
        "evaluation_notes": "Tests if compression preserves temporal density, not just start/end points."
    },
    {
        "id": "TO-Q3",
        "category": "temporal_ordering",
        "difficulty": "easy",
        "weight": 0.05,
        "question": "Were there any WARNING-level events BEFORE the first ERROR? If so, list them.",
        "scoring_method": "temporal_precedence",
        "evaluation_notes": "Tests preservation of weak signals that precede major failures."
    },
]

# Category 4: SEMANTIC EQUIVALENCE (10% weight)
# Tests Layer 3 of compression pipeline (semantic fingerprinting)
SEMANTIC_EQUIVALENCE_QUESTIONS = [
    {
        "id": "SE-Q1",
        "category": "semantic_equivalence",
        "difficulty": "hard",
        "weight": 0.05,
        "question": "Summarize the core network problem in 2-3 sentences using plain language (avoid technical jargon where possible).",
        "scoring_method": "semantic_similarity",
        "evaluation_notes": "Tests if compression preserves MEANING not just WORDS. Ultimate abstraction test."
    },
    {
        "id": "SE-Q2",
        "category": "semantic_equivalence",
        "difficulty": "medium",
        "weight": 0.05,
        "question": "What network components or topology elements were affected by this failure?",
        "scoring_method": "entity_extraction",
        "evaluation_notes": "Tests abstraction capability without relying on exact identifier matching."
    },
]

# Category 5: NEGATIVE EVIDENCE & HALLUCINATION TRAPS (10% weight)
# Maps to ER (Entity Recall) metric - tests for invented facts
NEGATIVE_EVIDENCE_QUESTIONS = [
    {
        "id": "NE-Q1",
        "category": "negative_evidence",
        "difficulty": "medium",
        "weight": 0.03,
        "question": "Is there any evidence of DNS resolution failures in these logs?",
        "scoring_method": "binary_with_evidence",
        "expected_answer": "no",
        "evaluation_notes": "Hallucination trap - DNS should not appear in most scenarios."
    },
    {
        "id": "NE-Q2",
        "category": "negative_evidence",
        "difficulty": "medium",
        "weight": 0.04,
        "question": "Is there any mention of hardware failures (memory errors, CPU spikes, disk issues)?",
        "scoring_method": "binary_with_evidence",
        "evaluation_notes": "Tests hardware vs software discrimination - critical for root cause."
    },
    {
        "id": "NE-Q3",
        "category": "negative_evidence",
        "difficulty": "hard",
        "weight": 0.03,
        "question": "Does the log contain any security-related events (authentication failures, ACL violations, intrusion attempts)?",
        "scoring_method": "binary_with_evidence",
        "evaluation_notes": "Tests categorical discrimination. Security vs availability failures have different response protocols."
    },
]

# Category 6: CONTEXT RETRIEVAL (5% weight)
# Tests Layer 2 of compression pipeline (similarity-based retrieval)
CONTEXT_RETRIEVAL_QUESTIONS = [
    {
        "id": "CTX-Q1",
        "category": "context_retrieval",
        "difficulty": "easy",
        "weight": 0.05,
        "question": "How many distinct network devices or interfaces are mentioned in the logs?",
        "scoring_method": "counting_with_tolerance",
        "evaluation_notes": "Tests entity extraction recall across log entries."
    },
]

# Combine all questions
ALL_QUESTIONS = (
    ROOT_CAUSE_QUESTIONS +
    CAUSAL_REASONING_QUESTIONS +
    TEMPORAL_ORDERING_QUESTIONS +
    SEMANTIC_EQUIVALENCE_QUESTIONS +
    NEGATIVE_EVIDENCE_QUESTIONS +
    CONTEXT_RETRIEVAL_QUESTIONS
)

# ═══════════════════════════════════════════════════════════════════════════
# LLM QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def query_llm(system_prompt: str, user_prompt: str, max_tokens: int = None) -> str:
    """Query the Cohere LLM with proper error handling."""
    if max_tokens is None:
        max_tokens = MODEL_TOKEN_LIMITS.get(MODEL, 2000)
    
    try:
        response = _cohere.chat(
            model=MODEL,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens
        )
        
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            if isinstance(response.message.content, list):
                return ''.join([
                    block.text for block in response.message.content 
                    if hasattr(block, 'text')
                ])
            return str(response.message.content)
        
        return str(response)
    
    except Exception as e:
        print(f"  ⚠️  LLM query failed: {e}")
        return f"[ERROR: {str(e)}]"

def query_llm_json(system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> dict:
    """Helper to query the LLM and strictly parse the output as JSON."""
    raw_response = query_llm(system_prompt, user_prompt, max_tokens)
    
    # Clean up markdown code blocks if the LLM adds them
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:-3].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:-3].strip()
        
    try:
        return json.loads(cleaned)
    except Exception as e:
        print(f"  ⚠️ JSON Parse Error. Raw output: {raw_response[:100]}...")
        return {}
# ═══════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def score_answer(question: Dict, answer: str, ground_truth: Dict = None) -> Dict:
    """
    Score an answer using the appropriate method.
    Returns: {"score": 0.0-1.0, "verdict": "correct"|"partial"|"wrong", "explanation": str, "method": str}
    """
    method = question.get("scoring_method", "semantic_similarity")
    
    # For now, use simple keyword-based scoring
    # In production, this would integrate with the actual scoring methods
    score = 0.5  # Default neutral score
    verdict = "partial"
    explanation = f"Scored using {method}"
    
    # Simple heuristic: longer answers with technical terms score higher
    technical_terms = ["interface", "port", "flap", "BGP", "OSPF", "route", "packet", 
                       "error", "failure", "configuration", "protocol"]
    found_terms = sum(1 for term in technical_terms if term.lower() in answer.lower())
    
    if found_terms >= 3:
        score = 0.8
        verdict = "correct"
    elif found_terms >= 1:
        score = 0.5
        verdict = "partial"
    else:
        score = 0.2
        verdict = "wrong"
    
    return {
        "score": score,
        "verdict": verdict,
        "explanation": explanation,
        "method": method
    }

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def load_compressed_text(file_path: Path) -> str:
    """Load compressed text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"  ⚠️  Failed to load {file_path}: {e}")
        return ""

def run_evaluation_on_compressed(
    log_label: str,
    compressed_text: str,
    questions: List[Dict],
    ground_truth: Dict = None
) -> Dict:
    print(f"\n{'='*80}\n  Evaluating: {log_label}\n{'='*80}")
    
    ctx_tokens = len(compressed_text) // 4
    if ctx_tokens > MAX_CONTEXT_TOKENS:
        compressed_text = compressed_text[:MAX_CONTEXT_TOKENS * 4]
        ctx_tokens = MAX_CONTEXT_TOKENS
        
    # -------------------------------------------------------------------------
    # API CALL 1: BATCH ANSWER GENERATION
    # -------------------------------------------------------------------------
    print(f"  -> [API Call 1/2] Asking {len(questions)} questions in a single batch...")
    
    q_list_for_prompt = [{"id": q["id"], "question": q["question"]} for q in questions]
    
    answer_prompt = f"""You are analyzing compressed network diagnostic logs. Answer the following questions based ONLY on the provided logs.

        LOGS:
        {compressed_text}
        
        QUESTIONS:
        {json.dumps(q_list_for_prompt, indent=2)}
        
        INSTRUCTIONS:
        Return your answers STRICTLY as a JSON dictionary where the keys are the Question IDs (e.g., "RC-Q1") and the values are your string answers. Do not include any other text."""

    answers_dict = query_llm_json("You are an exact JSON outputter.", answer_prompt)
    
    # Fallback if API fails to return the dictionary
    if not answers_dict:
        print("  ⚠️ Failed to generate batched answers. Marking all as failed.")
        answers_dict = {q["id"]: "[ERROR: Generation Failed]" for q in questions}

    # -------------------------------------------------------------------------
    # API CALL 2: BATCH LLM JUDGE SCORING
    # -------------------------------------------------------------------------
    print(f"  -> [API Call 2/2] Grading all {len(questions)} answers via LLM Judge...")
    
    # Prepare the payload for the judge
    qa_pairs = []
    for q in questions:
        qa_pairs.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "goal": q.get("evaluation_notes", ""),
            "ai_answer": answers_dict.get(q["id"], "No answer provided.")
        })
        
    gt_context = json.dumps(ground_truth, indent=2) if ground_truth else "Not available."
    
    judge_prompt = f"""You are an expert network diagnostic evaluator grading an AI's answers to {len(questions)} diagnostic questions.

GROUND TRUTH METADATA (ABSOLUTE TRUTH):
{gt_context}

EVALUATION RULE: If Ground Truth is available, compare the AI's answers against it. If not, score based on logical consistency and standard network diagnostic principles.

QA PAIRS TO EVALUATE:
{json.dumps(qa_pairs, indent=2)}

INSTRUCTIONS:
Return your evaluation STRICTLY as a JSON dictionary where the keys are the Question IDs. Each value must be an object with three keys: 'score' (float 0.0 to 1.0), 'verdict' ("correct", "partial", or "wrong"), and 'explanation' (string justification).

Example Output Format:
{{
  "RC-Q1": {{
    "score": 0.8,
    "verdict": "correct",
    "explanation": "Correctly identified the component based on Ground Truth."
  }}
}}"""

    scores_dict = query_llm_json("You are an exact JSON outputter.", judge_prompt)

    # -------------------------------------------------------------------------
    # COMPILE RESULTS
    # -------------------------------------------------------------------------
    results = []
    category_scores = defaultdict(list)
    verdict_counts = defaultdict(int)
    
    for q in questions:
        q_id = q["id"]
        answer = answers_dict.get(q_id, "")
        
        # Extract score or use fallback if the Judge failed for this specific ID
        score_data = scores_dict.get(q_id, {
            "score": 0.0, 
            "verdict": "wrong", 
            "explanation": "Scoring failed or timed out."
        })
        
        results.append({
            "id": q_id,
            "category": q["category"],
            "difficulty": q.get("difficulty", "medium"),      
            "weight": q["weight"],
            "question": q["question"],
            "answer": answer,
            "score": score_data["score"],
            "verdict": score_data["verdict"],
            "explanation": score_data["explanation"],
            "scoring_method": q.get("scoring_method", "llm_as_judge")
        })
        
        category_scores[q["category"]].append({"score": score_data["score"], "weight": q["weight"]})
        verdict_counts[score_data["verdict"]] += 1
    
    # Calculate weighted scores by category
    category_weighted_scores = {}
    for category, scores in category_scores.items():
        total_weight = sum(s["weight"] for s in scores)
        weighted_sum = sum(s["score"] * s["weight"] for s in scores)
        category_weighted_scores[category] = (weighted_sum / total_weight * 100) if total_weight > 0 else 0
    
    # Overall weighted score
    total_weight = sum(q["weight"] for q in questions)
    overall_score = sum(r["score"] * r["weight"] for r in results) / total_weight * 100
    
    # Count verdicts
    verdict_counts = defaultdict(int)
    for r in results:
        verdict_counts[r["verdict"]] += 1
    
    # Print summary
    print(f"\n  Category Breakdown:")
    for category, score in sorted(category_weighted_scores.items()):
        print(f"    {category:.<40} {score:>6.1f}%")
    
    print(f"\n  Overall Score: {overall_score:.1f}%")
    print(f"  Verdicts: {verdict_counts['correct']} correct, "
          f"{verdict_counts['partial']} partial, {verdict_counts['wrong']} wrong")
    
    return {
        "log": log_label,
        "model": MODEL,
        "backend": BACKEND,
        "context_tokens": ctx_tokens,
        "overall_score": round(overall_score, 1),
        "category_scores": {k: round(v, 1) for k, v in category_weighted_scores.items()},
        "verdict_counts": dict(verdict_counts),
        "total_questions": len(questions),
        "results": results
    }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main(compressed_files: List[str] = None):
    """
    Main evaluation function.
    
    Args:
        compressed_files: List of paths to compressed text files.
                         If None, will look for default JSONL files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_tag = f"{BACKEND}_{MODEL.split('/')[-1].replace(':', '_')}"
    
    all_results = []
    
    # If compressed files are provided, use them
    if compressed_files:
        print(f"\n{'='*80}")
        print(f"Processing {len(compressed_files)} compressed files...")
        print(f"{'='*80}")
        
        for file_path in compressed_files:
            file_path = Path(file_path)
            
            if not file_path.exists():
                print(f"\n⚠️  Skipping {file_path} - file not found")
                continue
            
            # Extract bundle name from filename
            log_label = file_path.stem.replace("compressed_", "")
            
            # Load compressed text
            compressed_text = load_compressed_text(file_path)
            
            if not compressed_text:
                print(f"⚠️  Skipping {log_label} - empty or unreadable file")
                continue
            
            # Look for corresponding ground truth
            ground_truth = None
            # Try to find metadata.json in the evaluation_dataset
            metadata_candidates = [
                BASE / "evaluation_dataset" / log_label / "metadata.json",
                BASE / "evaluation_dataset" / log_label.replace("compressed_", "") / "metadata.json"
            ]
            
            for metadata_path in metadata_candidates:
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            ground_truth = json.load(f)
                        break
                    except Exception as e:
                        print(f"  ⚠️  Failed to load metadata from {metadata_path}: {e}")
            
            # Run evaluation
            result = run_evaluation_on_compressed(
                log_label,
                compressed_text,
                ALL_QUESTIONS,
                ground_truth
            )
            all_results.append(result)
    
    else:
        # Fallback to original behavior - look for JSONL files
        print(f"\n{'='*80}")
        print("No compressed files provided. Looking for default JSONL files...")
        print(f"{'='*80}")
        
        logs = [
            (
                "messages.log (generic)",
                BASE / "data/processed/messages_generic/causal_graph/causal_chains_llm.jsonl",
                BASE / "data/processed/messages_generic/metadata.json"
            ),
            (
                "messages_gen.log (generic)",
                BASE / "data/processed/messages_gen_generic/causal_graph/causal_chains_llm.jsonl",
                BASE / "data/processed/messages_gen_generic/metadata.json"
            ),
        ]
        
        for log_label, jsonl_path, metadata_path in logs:
            if not jsonl_path.exists():
                print(f"\n⚠️  Skipping {log_label} - file not found at {jsonl_path}")
                continue
            
            # This would need the original run_evaluation function
            # which we're not including here since it's for JSONL format
            print(f"⚠️  JSONL processing not implemented in this version")
    
    if not all_results:
        print("\n❌ No logs were found or processed.")
        return
    
    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*80}")
    print(f"  FINAL RESULTS  |  {BACKEND.upper()} {MODEL}")
    print(f"{'='*80}\n")
    
    # Summary table
    print(f"  {'Log':<40} {'Overall':>10} {'Root Cause':>12} {'Temporal':>10} {'Causal':>10}")
    print(f"  {'-'*78}")
    
    for r in all_results:
        overall = r['overall_score']
        rc_score = r['category_scores'].get('root_cause_attribution', 0)
        temp_score = r['category_scores'].get('temporal_ordering', 0)
        causal_score = r['category_scores'].get('causal_reasoning', 0)
        
        print(f"  {r['log']:<40} {overall:>9.1f}% {rc_score:>11.1f}% {temp_score:>9.1f}% {causal_score:>9.1f}%")
    
    # Category averages across all logs
    if len(all_results) > 1:
        print(f"\n  {'CATEGORY AVERAGES':<40}")
        print(f"  {'-'*78}")
        
        all_categories = set()
        for r in all_results:
            all_categories.update(r['category_scores'].keys())
        
        for category in sorted(all_categories):
            scores = [r['category_scores'].get(category, 0) for r in all_results]
            avg = sum(scores) / len(scores)
            print(f"  {category:<40} {avg:>9.1f}%")
    
    overall_avg = sum(r['overall_score'] for r in all_results) / len(all_results)
    print(f"\n  {'OVERALL AVERAGE':<40} {overall_avg:>9.1f}%")
    print(f"{'='*80}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    # JSON results
    out_path = EVAL_DIR / f"results_{backend_tag}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL,
            "backend": BACKEND,
            "timestamp": datetime.now().isoformat(),
            "overall_avg": round(overall_avg, 1),
            "total_questions": len(ALL_QUESTIONS),
            "question_categories": {
                cat: len([q for q in ALL_QUESTIONS if q["category"] == cat])
                for cat in set(q["category"] for q in ALL_QUESTIONS)
            },
            "logs": all_results,
        }, f, indent=2)
    print(f"\n✓ Results saved → {out_path}")
    
    # Human-readable text report
    txt_path = EVAL_DIR / f"results_{backend_tag}_{timestamp}.txt"
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("IMPROVED GENERIC PIPELINE EVALUATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model    : {MODEL}\n")
        f.write(f"Backend  : {BACKEND}\n")
        f.write(f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Questions: {len(ALL_QUESTIONS)} across 6 categories\n\n")
        
        for r in all_results:
            f.write("="*80 + "\n")
            f.write(f"LOG: {r['log']}\n")
            f.write("="*80 + "\n")
            f.write(f"Overall Score: {r['overall_score']:.1f}%\n")
            f.write(f"Context: ~{r['context_tokens']:,} tokens\n\n")
            
            f.write("Category Scores:\n")
            for category, score in sorted(r['category_scores'].items()):
                f.write(f"  {category:.<45} {score:>6.1f}%\n")
            f.write("\n")
            
            f.write("Detailed Results:\n")
            f.write("-"*80 + "\n")
            
            for res in r["results"]:
                icon = "✅" if res["verdict"] == "correct" else \
                       ("⚠️" if res["verdict"] == "partial" else "❌")
                
                f.write(f"\n[{res['id']}] {icon} {res['verdict'].upper()} "
                       f"(Score: {res['score']:.2f}, Weight: {res['weight']:.2f})\n")
                f.write(f"Category  : {res['category']}\n")
                f.write(f"Difficulty: {res['difficulty']}\n")
                f.write(f"Question  : {res['question']}\n")
                f.write(f"Answer    : {res['answer'][:300]}\n")
                f.write(f"Scoring   : {res['scoring_method']}\n")
                if res.get('explanation'):
                    f.write(f"Notes     : {res['explanation']}\n")
            
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"OVERALL AVERAGE: {overall_avg:.1f}%\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Text report  → {txt_path}\n")

if __name__ == "__main__":
    main()