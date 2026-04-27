"""
Improved Generic Pipeline — Multi-LLM Evaluation
================================================
Enhanced evaluation framework testing semantic compression quality through
deep causal reasoning, temporal ordering, and hallucination detection.

Key additions:
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
        "question": "The monitoring team reported high packet loss on VLAN 500. Is this supported by the log data?",
        "scoring_method": "adversarial_verification",
        "trap_entity": "VLAN 500",
        "evaluation_notes": "Adversarial question - VLAN 500 likely doesn't exist. Tests if LLM invents evidence."
    },
]

# Category 6: QUANTITATIVE PRECISION (5% weight)
# Tests if compression preserves numerical accuracy (Layer 2 deduplication counts)
QUANTITATIVE_QUESTIONS = [
    {
        "id": "QP-Q1",
        "category": "quantitative_precision",
        "difficulty": "medium",
        "weight": 0.03,
        "question": "Approximately how many error-level events or failures are documented in these logs?",
        "scoring_method": "magnitude_estimation",
        "acceptable_variance": 0.25,
        "evaluation_notes": "Tests frequency preservation. 'Approximately' allows for compression loss."
    },
    {
        "id": "QP-Q2",
        "category": "quantitative_precision",
        "difficulty": "easy",
        "weight": 0.02,
        "question": "What is the approximate time span covered by these log events?",
        "scoring_method": "temporal_span",
        "acceptable_variance": 0.15,
        "evaluation_notes": "Basic sanity check for temporal coverage."
    },
]

# Compile all questions
ALL_QUESTIONS = (
    ROOT_CAUSE_QUESTIONS +
    CAUSAL_REASONING_QUESTIONS +
    TEMPORAL_ORDERING_QUESTIONS +
    SEMANTIC_EQUIVALENCE_QUESTIONS +
    NEGATIVE_EVIDENCE_QUESTIONS +
    QUANTITATIVE_QUESTIONS
)

# Verify weights sum to ~1.0
total_weight = sum(q["weight"] for q in ALL_QUESTIONS)
print(f"Total question weight: {total_weight:.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM = textwrap.dedent("""\
    You are an expert network engineer analyzing network failure logs.
    
    Your task is to answer diagnostic questions based ONLY on the provided log data below.
    
    IMPORTANT GUIDELINES:
    - Be precise and specific - cite evidence from the data
    - If the data doesn't contain enough information, explicitly state this
    - Do NOT guess or fabricate details
    - Focus on causal relationships, not just isolated events
    - Distinguish between root causes and symptoms
    
    COMPRESSED LOG SUMMARY (Markdown format):
    {context}
""")

# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDING
# ═══════════════════════════════════════════════════════════════════════════

def build_context(file_path: Path, token_budget: int = MAX_CONTEXT_TOKENS) -> Tuple[str, int]:
    """
    Read the Layer 5 compressed Markdown string.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    cost = len(content) // 4  # Rough token estimate
    
    # Simple truncation if it exceeds budget
    if cost > token_budget:
        content = content[:token_budget * 4] + "\n...[TRUNCATED]"
        cost = token_budget
        
    return content, cost

# ═══════════════════════════════════════════════════════════════════════════
# LLM INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def ask_llm(context_text: str, question: str, retries: int = 3) -> str:
    """Call LLM with context and question, with retry logic."""
    prompt = SYSTEM.format(context=context_text) + f"\n\nQUESTION: {question}\n\nANSWER:"
    
    for attempt in range(retries):
        try:
            resp = _cohere.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.message.content[0].text.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"    ⚠️  Retry {attempt+1} ({wait}s): {str(e)[:80]}")
                time.sleep(wait)
            else:
                return f"ERROR: {str(e)[:200]}"

# ═══════════════════════════════════════════════════════════════════════════
# IMPROVED SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def extract_entities(text: str) -> set:
    """Extract network entities (IPs, interfaces, processes) from text."""
    entities = set()
    
    # IP addresses
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    entities.update(re.findall(ip_pattern, text))
    
    # Interface names (e.g., 1/1/3, eth0, pvnet0)
    iface_pattern = r'\b(?:\d+/\d+/\d+|eth\d+|pvnet\d+|1/1/\d+)\b'
    entities.update(re.findall(iface_pattern, text))
    
    # Process IDs (e.g., pid:4714)
    pid_pattern = r'(?:pid:?|PID:?)\s*(\d+)'
    entities.update(re.findall(pid_pattern, text))
    
    # Protocol names
    protocols = ['bgp', 'ospf', 'vlan', 'arp', 'dns', 'ntp']
    text_lower = text.lower()
    entities.update(p for p in protocols if p in text_lower)
    
    return entities

def simple_semantic_similarity(text1: str, text2: str) -> float:
    """
    Simple semantic similarity based on word overlap.
    For production, replace with sentence-transformers embeddings.
    """
    # Normalize and tokenize
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    # Remove stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'was', 'are'}
    words1 -= stop_words
    words2 -= stop_words
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

def score_answer(answer: str, question: Dict, context_text: str) -> Dict[str, Any]:
    """
    Multi-level scoring based on question type.
    Returns score (0-1), verdict (correct/partial/wrong), and explanation.
    """
    method = question.get("scoring_method", "keyword_matching")
    
    # Extract metadata from context for comparison
    chain_text = context_text
    
    # Default scoring
    result = {
        "score": 0.0,
        "verdict": "wrong",
        "explanation": "",
        "method": method
    }
    
    # ERROR responses get zero score
    if answer.startswith("ERROR:"):
        result["explanation"] = "LLM error occurred"
        return result
    
    # Method-specific scoring
    if method == "entity_extraction_plus_semantic":
        # Extract entities from answer
        answer_entities = extract_entities(answer)
        
        # Check for key failure indicators in chains
        chain_entities = extract_entities(chain_text)
        
        # Entity overlap
        if answer_entities and chain_entities:
            overlap = len(answer_entities & chain_entities) / len(chain_entities)
            entity_score = min(overlap * 1.5, 1.0)  # Boost for good entity recall
        else:
            entity_score = 0.0
        
        # Semantic check for failure type
        failure_keywords = {
            'physical': ['cable', 'fiber', 'link down', 'transceiver', 'optical'],
            'configuration': ['config', 'mismatch', 'vlan', 'misconfigured'],
            'protocol': ['bgp', 'ospf', 'routing', 'protocol', 'session'],
            'performance': ['bandwidth', 'congestion', 'packet loss', 'latency']
        }
        
        semantic_score = 0.0
        for category, keywords in failure_keywords.items():
            if any(kw in answer.lower() for kw in keywords):
                semantic_score = max(semantic_score, 0.5)
        
        # Combined score
        result["score"] = 0.6 * entity_score + 0.4 * semantic_score
        
    elif method == "semantic_similarity":
        # Placeholder - in production, use actual embeddings
        # For now, use simple word overlap
        result["score"] = simple_semantic_similarity(answer, chain_text[:500])
        
    elif method == "temporal_sequence_validation":
        # Check if answer contains temporal ordering words
        temporal_indicators = ['first', 'then', 'next', 'after', 'before', 'finally', 'initially']
        has_ordering = any(word in answer.lower() for word in temporal_indicators)
        
        # Check for timestamps or time references
        has_timestamps = bool(re.search(r'\d{1,2}:\d{2}', answer))
        
        result["score"] = (0.5 if has_ordering else 0.0) + (0.5 if has_timestamps else 0.0)
        
    elif method == "binary_with_evidence":
        # Check for clear yes/no answer
        answer_lower = answer.lower()
        if question.get("expected_answer") == "no":
            # Should say "no" or "no evidence"
            if any(phrase in answer_lower for phrase in ['no', 'not found', 'no evidence', 'does not appear', 'absent']):
                result["score"] = 1.0
            elif any(phrase in answer_lower for phrase in ['yes', 'evidence of', 'shows']):
                result["score"] = 0.0  # Hallucination
        else:
            # Generic binary scoring
            result["score"] = 0.5  # Neutral if unclear
            
    elif method == "adversarial_verification":
        # Check if LLM resists the trap
        trap_entity = question.get("trap_entity", "")
        answer_lower = answer.lower()
        trap_lower = trap_entity.lower()
        
        if trap_lower in answer_lower:
            # LLM mentioned the trap entity - likely hallucinated
            if any(phrase in answer_lower for phrase in ['no evidence', 'not mentioned', 'does not appear']):
                result["score"] = 1.0  # Correctly rejected
            else:
                result["score"] = 0.0  # Hallucination!
                result["explanation"] = f"Hallucinated evidence about {trap_entity}"
        else:
            # Didn't mention trap entity - good
            result["score"] = 1.0
            
    elif method == "magnitude_estimation":
        # Extract numbers from answer
        numbers = re.findall(r'\b\d+\b', answer)
        if numbers:
            result["score"] = 0.5  # Partial for attempting quantification
        # Full scoring would require ground truth comparison
        
    elif method == "classification_with_reasoning":
        # Check if answer contains classification choice
        classifications = {
            'A': ['physical', 'cable', 'fiber', 'link'],
            'B': ['configuration', 'config', 'mismatch'],
            'C': ['protocol', 'software', 'bgp', 'ospf'],
            'D': ['resource', 'capacity', 'bandwidth', 'memory']
        }
        
        for choice, keywords in classifications.items():
            if any(kw in answer.lower() for kw in keywords) or choice in answer.upper():
                result["score"] = 0.8  # High score for categorization
                break
    
    else:
        # Default keyword matching
        result["score"] = 0.3  # Minimal baseline
    
    # Convert score to verdict
    if result["score"] >= 0.75:
        result["verdict"] = "correct"
    elif result["score"] >= 0.4:
        result["verdict"] = "partial"
    else:
        result["verdict"] = "wrong"
    
    # Generate explanation if not set
    if not result["explanation"]:
        result["explanation"] = f"Score: {result['score']:.2f} via {method}"
    
    return result

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(log_label: str, file_path: Path, questions: List[Dict]) -> Dict[str, Any]:
    """Run complete evaluation on one log file."""
    print(f"\n{'='*80}")
    print(f"  LOG: {log_label}")
    print(f"{'='*80}")
    
    # Build context
    ctx_budget = MODEL_TOKEN_LIMITS.get(MODEL, MAX_CONTEXT_TOKENS)
    ctx, ctx_tokens = build_context(file_path, ctx_budget)
    print(f"  Context: {len(ctx)} chars, ~{ctx_tokens:,} tokens\n")
    
    # Run questions
    results = []
    category_scores = defaultdict(list)
    
    for q in questions:
        print(f"  [{q['id']}] {q['question'][:70]}...")
        
        # Get LLM answer
        answer = ask_llm(ctx, q["question"])
        
        # Score answer
        score_result = score_answer(answer, q, ctx)
        
        # Icon for display
        icon = "✅" if score_result["verdict"] == "correct" else \
               ("⚠️ " if score_result["verdict"] == "partial" else "❌")
        
        print(f"    {icon} {score_result['verdict'].upper()} ({score_result['score']:.2f})")
        print(f"       {answer[:120]}...")
        
        # Track by category
        category_scores[q["category"]].append({
            "score": score_result["score"],
            "weight": q["weight"]
        })
        
        # Store result
        results.append({
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "weight": q["weight"],
            "question": q["question"],
            "answer": answer,
            "score": score_result["score"],
            "verdict": score_result["verdict"],
            "explanation": score_result["explanation"],
            "scoring_method": score_result["method"]
        })
        
        time.sleep(1)  # Rate limiting
    
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
        "context_chars": len(ctx),
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

def main(log_files=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_tag = f"{BACKEND}_{MODEL.split('/')[-1].replace(':', '_')}"
    
    # Define log files to process
    logs = []
    if log_files is not None:
        for file_path in log_files:
            if isinstance(file_path, str):
                file_path = Path(file_path)
            logs.append((file_path.name, file_path))
    else:
        compressed_dir = BASE / "compressed_logs"
        if compressed_dir.exists():
            for txt_file in compressed_dir.glob("*.txt"):
                logs.append((txt_file.name, txt_file))
            
    all_results = []
    
    for log_label, file_path in logs:
        if not file_path.exists():
            print(f"\n⚠️  Skipping {log_label} - file not found at {file_path}")
            continue
        
        result = run_evaluation(log_label, file_path, ALL_QUESTIONS)
        all_results.append(result)
    
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
            f.write(f"Context: {r.get('context_chars', 0)} chars, ~{r['context_tokens']:,} tokens\n\n")
            
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