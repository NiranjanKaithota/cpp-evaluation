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

import google.generativeai as genai

# Gemini Setup for Adversarial Question Generation
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    print("  [WARNING] GEMINI_API_KEY not found. Dynamic adversarial questions will be skipped.")
    gemini_model = None

# ═══════════════════════════════════════════════════════════════════════════
# IMPROVED QUESTION BANK
# ═══════════════════════════════════════════════════════════════════════════

# Category 1: ROOT CAUSE ATTRIBUTION (Critical - 35% weight)
# Maps to RCM (Root Cause Match) metric from evaluation plan
ROOT_CAUSE_QUESTIONS = [
    {
        "id": "RC-Q1", "category": "root_cause_attribution", "difficulty": "hard", "weight": 0.20,
        "question": "What is the root cause of the primary network issue? Identify the specific component that failed and explain the failure mechanism.",
        "scoring_method": "entity_extraction_plus_semantic"
    },
    {
        "id": "RC-Q2", "category": "root_cause_attribution", "difficulty": "hard", "weight": 0.10,
        "question": "What remediation steps would you recommend to resolve this issue? Be specific about which components to check or reconfigure.",
        "scoring_method": "semantic_similarity"
    },
    {
        "id": "RC-Q3", "category": "root_cause_attribution", "difficulty": "medium", "weight": 0.05,
        "question": "Classify this failure: (A) Physical layer issue, (B) Configuration error, (C) Protocol/software failure, or (D) Resource exhaustion. Explain your choice.",
        "scoring_method": "classification_with_reasoning"
    },
    {
        "id": "RC-Q4", "category": "root_cause_attribution", "difficulty": "hard", "weight": 0.10,
        "question": "Was this a single point of failure, or a cascading failure where one component's crash caused others to fail? Provide evidence.",
        "scoring_method": "causal_chain_validation"
    },
    {
        "id": "RC-Q5", "category": "root_cause_attribution", "difficulty": "easy", "weight": 0.05,
        "question": "Based on standard networking practices, what is the severity level of this issue (e.g., Critical outage, Degraded performance, Informational warning)? Justify the severity.",
        "scoring_method": "semantic_similarity"
    }
]

# Category 2: CAUSAL REASONING (High Priority - 20% weight)
# Tests Layer 4 of compression pipeline (causality graph preservation)
CAUSAL_REASONING_QUESTIONS = [
    {
        "id": "CR-Q1", "category": "causal_reasoning", "difficulty": "hard", "weight": 0.10,
        "question": "Trace the causal chain: what event triggered the failure, what symptoms appeared as a result, and what was the final observable impact?",
        "scoring_method": "causal_chain_validation"
    },
    {
        "id": "CR-Q2", "category": "causal_reasoning", "difficulty": "hard", "weight": 0.10,
        "question": "Could the observed symptoms have been caused by something OTHER than the identified root cause? If yes, what evidence rules out those alternatives?",
        "scoring_method": "semantic_similarity"
    },
    {
        "id": "CR-Q3", "category": "causal_reasoning", "difficulty": "medium", "weight": 0.10,
        "question": "Did any automated network recovery mechanisms (e.g., STP reconvergence, BGP route recalculation, Link Aggregation failover) attempt to mitigate this failure? Detail their actions.",
        "scoring_method": "entity_extraction_plus_semantic"
    },
    {
        "id": "CR-Q4", "category": "causal_reasoning", "difficulty": "medium", "weight": 0.05,
        "question": "How did this specific failure impact the overall network topology (e.g., routing tables, forwarding paths, neighbor adjacencies)?",
        "scoring_method": "semantic_similarity"
    }
]

# Category 3: TEMPORAL ORDERING (Critical - 20% weight)
# Maps to COPS (Causal Order Preservation Score) metric
TEMPORAL_ORDERING_QUESTIONS = [
    {
        "id": "TO-Q1", "category": "temporal_ordering", "difficulty": "medium", "weight": 0.10,
        "question": "List the 3-5 most critical events in chronological order. Include approximate timestamps if available.",
        "scoring_method": "temporal_sequence_validation"
    },
    {
        "id": "TO-Q2", "category": "temporal_ordering", "difficulty": "medium", "weight": 0.05,
        "question": "How long did it take from initial failure detection to complete service disruption? Describe the progression.",
        "scoring_method": "temporal_reasoning"
    },
    {
        "id": "TO-Q3", "category": "temporal_ordering", "difficulty": "easy", "weight": 0.05,
        "question": "Were there any WARNING-level events BEFORE the first ERROR? If so, list them.",
        "scoring_method": "temporal_precedence"
    },
    {
        "id": "TO-Q4", "category": "temporal_ordering", "difficulty": "medium", "weight": 0.05,
        "question": "Are there any cyclic or repeating error patterns indicating a continuous retry/fail loop (e.g., continuous interface flapping or repeated protocol handshakes)?",
        "scoring_method": "binary_with_evidence"
    },
    {
        "id": "TO-Q5", "category": "temporal_ordering", "difficulty": "easy", "weight": 0.05,
        "question": "Identify the timestamp or approximate time when the system first returned to a stable or fully disconnected state after the chaos.",
        "scoring_method": "entity_extraction"
    }
]

# Category 4: SEMANTIC EQUIVALENCE (10% weight)
# Tests Layer 3 of compression pipeline (semantic fingerprinting)
SEMANTIC_EQUIVALENCE_QUESTIONS = [
    {
        "id": "SE-Q1", "category": "semantic_equivalence", "difficulty": "hard", "weight": 0.05,
        "question": "Summarize the core network problem in 2-3 sentences using plain language (avoid technical jargon where possible).",
        "scoring_method": "semantic_similarity"
    },
    {
        "id": "SE-Q2", "category": "semantic_equivalence", "difficulty": "medium", "weight": 0.05,
        "question": "What network components or topology elements were affected by this failure?",
        "scoring_method": "entity_extraction"
    },
    {
        "id": "SE-Q3", "category": "semantic_equivalence", "difficulty": "hard", "weight": 0.10,
        "question": "If you were escalating this ticket to a Tier 3 Network Engineer, write the 1-sentence technical TL;DR they need to read first.",
        "scoring_method": "semantic_similarity"
    },
    {
        "id": "SE-Q4", "category": "semantic_equivalence", "difficulty": "easy", "weight": 0.05,
        "question": "Translate this technical failure into its likely business or user-facing impact (e.g., loss of redundancy vs. complete user outage).",
        "scoring_method": "semantic_similarity"
    }
]

# Category 5: NEGATIVE EVIDENCE & HALLUCINATION TRAPS (10% weight)
# Maps to ER (Entity Recall) metric - tests for invented facts
NEGATIVE_EVIDENCE_QUESTIONS = [
    {
        "id": "NE-Q1", "category": "negative_evidence", "difficulty": "medium", "weight": 0.05,
        "question": "Is there any evidence of DNS resolution failures in these logs?",
        "scoring_method": "binary_with_evidence"
    },
    {
        "id": "NE-Q2", "category": "negative_evidence", "difficulty": "medium", "weight": 0.05,
        "question": "Is there any mention of hardware failures (memory errors, CPU spikes, ASIC faults, disk issues)?",
        "scoring_method": "binary_with_evidence"
    },
    {
        "id": "NE-Q3", "category": "negative_evidence", "difficulty": "hard", "weight": 0.05,
        "question": "Does the log contain any security-related events (authentication failures, ACL drops, MAC spoofing, intrusion attempts)?",
        "scoring_method": "binary_with_evidence"
    },
    {
        "id": "NE-Q4", "category": "negative_evidence", "difficulty": "medium", "weight": 0.05,
        "question": "Is there any evidence of a power failure, brownout, or unscheduled system reboot?",
        "scoring_method": "binary_with_evidence"
    },
    {
        "id": "NE-Q5", "category": "negative_evidence", "difficulty": "hard", "weight": 0.05,
        "question": "Are there any Spanning Tree Protocol (STP) loop detections, topology changes (TCs), or broadcast storms mentioned?",
        "scoring_method": "binary_with_evidence"
    },
    {
        "id": "NE-Q6", "category": "negative_evidence", "difficulty": "medium", "weight": 0.05,
        "question": "Does the context explicitly mention MTU mismatches or packet fragmentation issues?",
        "scoring_method": "binary_with_evidence"
    },
    {
        "id": "NE-Q7", "category": "negative_evidence", "difficulty": "medium", "weight": 0.05,
        "question": "Is there evidence of an OSPF or BGP routing loop (e.g., TTL expired in transit)?",
        "scoring_method": "binary_with_evidence"
    }
]

# Category 6: CONTEXT RETRIEVAL (5% weight)
# Tests Layer 2 of compression pipeline (similarity-based retrieval)
CONTEXT_RETRIEVAL_QUESTIONS = [
    {
        "id": "CTX-Q1", "category": "context_retrieval", "difficulty": "easy", "weight": 0.05,
        "question": "How many distinct network devices or physical interfaces are explicitly mentioned in the logs?",
        "scoring_method": "counting_with_tolerance"
    },
    {
        "id": "CTX-Q2", "category": "context_retrieval", "difficulty": "medium", "weight": 0.05,
        "question": "What specific OS versions, firmware versions, or software builds are explicitly mentioned in the context?",
        "scoring_method": "entity_extraction"
    },
    {
        "id": "CTX-Q3", "category": "context_retrieval", "difficulty": "hard", "weight": 0.05,
        "question": "List all exact IP addresses and MAC addresses that can be extracted directly from the logs.",
        "scoring_method": "entity_extraction"
    },
    {
        "id": "CTX-Q4", "category": "context_retrieval", "difficulty": "medium", "weight": 0.05,
        "question": "Identify any specific VLAN IDs or Virtual Routing and Forwarding (VRF) instances referenced.",
        "scoring_method": "entity_extraction"
    },
    {
        "id": "CTX-Q5", "category": "context_retrieval", "difficulty": "easy", "weight": 0.05,
        "question": "What diagnostic commands (e.g., 'show tech', 'ping', 'traceroute') were executed to generate these logs, if visible?",
        "scoring_method": "entity_extraction"
    }
]

# Define the base static questions that run on every bundle
STATIC_QUESTIONS = (
    ROOT_CAUSE_QUESTIONS +
    CAUSAL_REASONING_QUESTIONS +
    TEMPORAL_ORDERING_QUESTIONS +
    SEMANTIC_EQUIVALENCE_QUESTIONS +
    NEGATIVE_EVIDENCE_QUESTIONS +
    CONTEXT_RETRIEVAL_QUESTIONS
)

def generate_adversarial_questions(ground_truth: Dict) -> List[Dict]:
    """Uses Gemini to generate dynamic, adversarial questions based on the ground truth."""
    if not gemini_model or not ground_truth:
        return []
        
    entity = ground_truth.get('root_cause_entity', 'unknown')
    fault_type = ground_truth.get('root_cause_type', 'unknown')
    action = ground_truth.get('recommended_action', 'unknown')
    
    if entity == "unknown":
        return []
        
    prompt = f"""You are a Master Network Architect writing an exam to test an AI diagnostic tool. 
    The AI has just read a compressed network log.
    
    The ACTUAL Ground Truth of the failure is:
    - Failing Entity: {entity}
    - Failure Type: {fault_type}
    - Remediation: {action}
    
    Generate EXACTLY TWO highly specific, difficult questions to test if the AI truly understood the nuances of this failure.
    - Question 1 should test a "Negative Distractor" (e.g., asking if a related but incorrect component also failed).
    - Question 2 should test "Deep Causality" specific to the {fault_type} of {entity}.
    
    Output STRICTLY as a JSON array of two objects with this schema, and no other text or markdown blocks:
    [
      {{
        "id": "ADV-Q1",
        "category": "adversarial_testing",
        "difficulty": "hard",
        "weight": 0.10,
        "question": "<your question here>",
        "scoring_method": "llm_as_judge"
      }}
    ]
    """
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.7)
        )
        
        # Clean up markdown if present
        cleaned = response.text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:-3].strip()
            
        generated_questions = json.loads(cleaned)
        return generated_questions
    except Exception as e:
        print(f"  [WARNING] Gemini failed to generate adversarial questions: {e}")
        return []

def get_dynamic_questions(ground_truth: Dict = None) -> List[Dict]:
    """
    Generates a tailored list of questions. 
    Injects dynamic Entity and CLI questions if ground truth is available.
    """
    # Start with the baseline static questions
    questions = list(STATIC_QUESTIONS)
    
    if ground_truth and isinstance(ground_truth, dict):
        entity = ground_truth.get("root_cause_entity", "unknown")
        action = ground_truth.get("recommended_action", "unknown")
        
        # 1. Add your standard dynamic template questions
        if entity and entity.lower() != "unknown":
            
            # ---------------------------------------------------------
            # IDEA 3: Dynamic Contextual Questioning
            # ---------------------------------------------------------
            questions.append({
                "id": "DYN-Q1",
                "category": "dynamic_entity_tracing",
                "difficulty": "hard",
                "weight": 0.15,
                "question": f"Analyze the logs specifically for the entity '{entity}'. Detail the exact sequence of events, errors, or state changes that occurred on this component before the failure.",
                "scoring_method": "dynamic_trace_validation",
                "evaluation_notes": "Tests if compression preserved the granular causal chain for the exact failing component."
            })
            
        if action and action.lower() != "unknown":
            
            # ---------------------------------------------------------
            # IDEA 4: Actionability and Remediation Testing (CLI)
            # ---------------------------------------------------------
            questions.append({
                "id": "REM-Q1",
                "category": "remediation_actionability",
                "difficulty": "hard",
                "weight": 0.15,
                "question": f"The recommended action is roughly: '{action}'. Provide the exact network OS CLI commands (e.g., HPE/Aruba OS syntax or standard Linux network utilities) a network engineer would type to execute this remediation and verify the fix.",
                "scoring_method": "cli_syntax_validation",
                "evaluation_notes": "Tests if the compressed logs preserved enough configuration state to allow the LLM to generate valid, specific CLI commands."
            })
            
        # 2. Ask Gemini to generate custom adversarial questions
        print("  -> Generating adversarial questions via Gemini...")
        adversarial_qs = generate_adversarial_questions(ground_truth)
        if adversarial_qs:
            questions.extend(adversarial_qs)
            print(f"  -> Added {len(adversarial_qs)} Gemini-authored questions.")
    
    return questions

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
            
            # Extract bundle name from filename - FIX: strip 'combined_' as well
            log_label = file_path.stem.replace("compressed_", "").replace("combined_", "")
            
            # Load compressed text
            compressed_text = load_compressed_text(file_path)
            
            if not compressed_text:
                print(f"⚠️  Skipping {log_label} - empty or unreadable file")
                continue
            
            # Look for corresponding ground truth
            ground_truth = None
            # Try to find metadata.json in the evaluation_dataset
            metadata_candidates = [
                BASE / "evaluation_dataset" / log_label / "metadata.json"
            ]
            
            for metadata_path in metadata_candidates:
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            raw_meta = json.load(f)
                            # FIX: Unwrap the 'ground_truth' key if it exists
                            ground_truth = raw_meta.get("ground_truth", raw_meta)
                            print(f"  -> Successfully loaded ground truth for {log_label}")
                        break
                    except Exception as e:
                        print(f"  ⚠️  Failed to load metadata from {metadata_path}: {e}")
            
            if not ground_truth:
                print(f"  ⚠️  No valid ground truth found. Proceeding with static questions only.")
            
            # Run evaluation
            # 1. Generate the tailored questions for this specific bundle
            dynamic_question_list = get_dynamic_questions(ground_truth)
            
            # 2. Run evaluation using the dynamically generated list
            result = run_evaluation_on_compressed(
                log_label,
                compressed_text,
                dynamic_question_list,
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
    out_path = EVAL_DIR / f"results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL,
            "backend": BACKEND,
            "timestamp": datetime.now().isoformat(),
            "overall_avg": round(overall_avg, 1),
            # Count the maximum questions used in any bundle during this run
            "max_questions_used": max((len(r["results"]) for r in all_results), default=0),
            "logs": all_results,
        }, f, indent=2)
    print(f"\n✓ Results saved → {out_path}")
    
    # Human-readable text report
    txt_path = EVAL_DIR / f"results_{timestamp}.txt"
    max_q = max((len(r["results"]) for r in all_results), default=0)
    
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("IMPROVED GENERIC PIPELINE EVALUATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model    : {MODEL}\n")
        f.write(f"Backend  : {BACKEND}\n")
        f.write(f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Questions: Up to {max_q} per bundle (Dynamic/Adversarial mode)\n\n") # <--- FIXED
        
        for r in all_results:
            f.write("="*80 + "\n")
            f.write(f"LOG: {r['log']}\n")
            f.write("="*80 + "\n")
            f.write(f"Overall Score: {r['overall_score']:.1f}%\n")
            f.write(f"Total Questions Evaluated: {len(r['results'])}\n") # <--- ADDED
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