#!/usr/bin/env python3
"""
Improved Generic Pipeline — Multi-LLM Evaluation
================================================
Enhanced evaluation framework testing semantic compression quality through
deep causal reasoning, temporal ordering, and hallucination detection.
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

# Upgraded to 100k to fully leverage Command-R 08-2024
MAX_CONTEXT_TOKENS = 100000

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
        "question": "Were there any WARNING-level events BEFORE the first ERROR? If so, summarize them or list up to 3 examples.",
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

SEMANTIC_EQUIVALENCE_QUESTIONS = [
    {
        "id": "SE-Q1", "category": "semantic_equivalence", "difficulty": "hard", "weight": 0.05,
        "question": "Summarize the core network problem in 2-3 sentences using plain language (avoid technical jargon where possible).",
        "scoring_method": "semantic_similarity"
    },
    {
        "id": "SE-Q2", "category": "semantic_equivalence", "difficulty": "medium", "weight": 0.05,
        "question": "What primary network components or topology elements were affected by this failure? (List the most critical or up to 5 components).",
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
        "question": "Extract a sample of 5 to 10 exact IP addresses and 5 to 10 MAC addresses from the logs.",
        "scoring_method": "entity_extraction"
    },
    {
        "id": "CTX-Q4", "category": "context_retrieval", "difficulty": "medium", "weight": 0.05,
        "question": "Identify up to 5 specific VLAN IDs or Virtual Routing and Forwarding (VRF) instances referenced in the logs.",
        "scoring_method": "entity_extraction"
    },
    {
        "id": "CTX-Q5", "category": "context_retrieval", "difficulty": "easy", "weight": 0.05,
        "question": "What diagnostic commands (e.g., 'show tech', 'ping', 'traceroute') were executed to generate these logs, if visible?",
        "scoring_method": "entity_extraction"
    }
]

STATIC_QUESTIONS = (
    ROOT_CAUSE_QUESTIONS +
    CAUSAL_REASONING_QUESTIONS +
    TEMPORAL_ORDERING_QUESTIONS +
    SEMANTIC_EQUIVALENCE_QUESTIONS +
    NEGATIVE_EVIDENCE_QUESTIONS +
    CONTEXT_RETRIEVAL_QUESTIONS
)

K8S_STATIC_QUESTIONS = [
    {
        "id": "K8S-RC-Q1", "category": "root_cause_attribution", "difficulty": "hard", "weight": 0.25,
        "question": "Identify any persistent error patterns (e.g., HTTP 403 Forbidden, 500 Internal Server Error) and the Kubernetes components or users involved.",
        "scoring_method": "entity_extraction_plus_semantic"
    },
    {
        "id": "K8S-CR-Q1", "category": "causal_reasoning", "difficulty": "hard", "weight": 0.20,
        "question": "Analyze the API requests made by the nodes (e.g., system:node). Were their requests for configurations like 'kube-flannel-cfg' and 'kube-root-ca.crt' successful, or were they denied? What HTTP status code proves this?",
        "scoring_method": "causal_chain_validation"
    },
    {
        "id": "K8S-TO-Q1", "category": "temporal_ordering", "difficulty": "medium", "weight": 0.15,
        "question": "List the first 3 API requests logged in chronological order, including the verbs (e.g., watch, list) and the target resources.",
        "scoring_method": "temporal_sequence_validation"
    },
    {
        "id": "K8S-SE-Q1", "category": "semantic_equivalence", "difficulty": "medium", "weight": 0.15,
        "question": "Summarize the core behavior seen in these logs. Are these standard routine checks, or is there an indication of permission/RBAC misconfigurations?",
        "scoring_method": "semantic_similarity"
    },
    {
        "id": "K8S-NE-Q1", "category": "negative_evidence", "difficulty": "hard", "weight": 0.15,
        "question": "Is there any evidence of Pod evictions, Node NotReady states, or CrashLoopBackOff errors in these logs?",
        "scoring_method": "binary_with_evidence"
    },
    {
        "id": "K8S-CTX-Q1", "category": "context_retrieval", "difficulty": "easy", "weight": 0.10,
        "question": "Extract up to 5 distinct Kubernetes namespaces or system users (e.g., system:kube-scheduler) mentioned in the context.",
        "scoring_method": "entity_extraction"
    }
]

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

    Generate exactly THREE highly specific, difficult questions to test if the AI truly understood the nuances of this failure. Follow these strict guidelines:
    - Question 1 (Negative Distractor): Test if the AI hallucinates by asking if a related but incorrect component also failed.
    - Question 2 (Deep Causality): Test specific technical nuances of the {fault_type} on {entity}.
    - Question 3 (Multi-Hop Trace): Ask a question that CANNOT be answered by looking at a single log line. It MUST require the AI to chronologically trace a chain of at least 3 distinct events, state changes, or network hops to arrive at the correct answer.

    Output STRICTLY as a JSON array of three objects with this schema, and no other text or markdown blocks:
    [
      {{
        "id": "ADV-Q1",
        "category": "negative_evidence",
        "difficulty": "hard",
        "weight": 0.10,
        "question": "<your negative distractor question here>",
        "scoring_method": "llm_as_judge"
      }},
      {{
        "id": "ADV-Q2",
        "category": "causal_reasoning",
        "difficulty": "hard",
        "weight": 0.10,
        "question": "<your deep causality question here>",
        "scoring_method": "llm_as_judge"
      }},
      {{
        "id": "ADV-Q3",
        "category": "temporal_ordering",
        "difficulty": "hard",
        "weight": 0.15,
        "question": "<your multi-hop trace question here>",
        "scoring_method": "llm_as_judge"
      }}
    ]
    """
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.7)
        )
        
        cleaned = response.text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:-3].strip()
            
        return json.loads(cleaned)
    except Exception as e:
        print(f"  [WARNING] Gemini failed to generate adversarial questions: {e}")
        return []

def get_dynamic_questions(log_label: str, ground_truth: Dict = None) -> List[Dict]:
    # Auto-detect Kubernetes logs based on the bundle name
    if "k8s" in log_label.lower() or "kubernetes" in log_label.lower():
        questions = list(K8S_STATIC_QUESTIONS)
        print("  -> ☸️  Kubernetes domain detected. Using K8s-specific evaluation questions.")
    else:
        questions = list(STATIC_QUESTIONS)
    
    if ground_truth and isinstance(ground_truth, dict):
        entity = ground_truth.get("root_cause_entity", "unknown")
        action = ground_truth.get("recommended_action", "unknown")
        
        if entity and entity.lower() != "unknown":
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
            questions.append({
                "id": "REM-Q1",
                "category": "remediation_actionability",
                "difficulty": "hard",
                "weight": 0.15,
                "question": f"The recommended action is roughly: '{action}'. Provide the exact network OS CLI commands (e.g., HPE/Aruba OS syntax or standard Linux network utilities) a network engineer would type to execute this remediation and verify the fix.",
                "scoring_method": "cli_syntax_validation",
                "evaluation_notes": "Tests if the compressed logs preserved enough configuration state to allow the LLM to generate valid, specific CLI commands."
            })
            
        print("  -> Generating adversarial questions via Gemini...")
        adversarial_qs = generate_adversarial_questions(ground_truth)
        if adversarial_qs:
            questions.extend(adversarial_qs)
            print(f"  -> Added {len(adversarial_qs)} Gemini-authored questions.")
    
    return questions

# ═══════════════════════════════════════════════════════════════════════════
# HARDENED LLM QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def query_llm(system_prompt: str, user_prompt: str, max_tokens: int = None) -> str:
    if max_tokens is None:
        max_tokens = MODEL_TOKEN_LIMITS.get(MODEL, 4000)
    
    try:
        response = _cohere.chat(
            model=MODEL,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=max_tokens
        )
        
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            if isinstance(response.message.content, list):
                return ''.join([block.text for block in response.message.content if hasattr(block, 'text')])
            return str(response.message.content)
        return str(response)
    
    except Exception as e:
        print(f"  ⚠️  LLM query failed: {e}")
        return f"[ERROR: {str(e)}]"

def query_llm_json(system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> dict:
    """Bulletproof JSON parser that aggressively hunts for dicts and handles unexpected lists."""
    raw_response = query_llm(system_prompt, user_prompt, max_tokens)
    cleaned = raw_response.strip()
    
    # 1. Clean markdown code blocks if the LLM added them
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # 2. Aggressively hunt for the outermost JSON container ([ or {)
    start_dict, start_list = cleaned.find('{'), cleaned.find('[')
    start_idx = min(i for i in [start_dict, start_list] if i != -1) if (start_dict != -1 or start_list != -1) else -1
    
    end_dict, end_list = cleaned.rfind('}'), cleaned.rfind(']')
    end_idx = max(end_dict, end_list)
    
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        cleaned = cleaned[start_idx:end_idx+1]
    
    # 3. Parse and auto-correct structures
    try:
        parsed = json.loads(cleaned)
        
        # If it returned a dictionary as expected, return it
        if isinstance(parsed, dict):
            return parsed
            
        # If the LLM wrapped the dictionary in a list (e.g., [{"RC-Q1": {...}}])
        elif isinstance(parsed, list):
            merged_dict = {}
            for item in parsed:
                if isinstance(item, dict):
                    merged_dict.update(item)
            return merged_dict
            
        else:
            print(f"  ⚠️ JSON Parse Error. Expected Dict/List, got {type(parsed).__name__}.")
            return {}
            
    except Exception as e:
        print(f"  ⚠️ JSON Parse Error. Raw output was garbled: {raw_response[:100]}...")
        return {}

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def load_compressed_text(file_path: Path) -> str:
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
    
    actual_tokens = len(compressed_text) // 4
    is_truncated = False
    if actual_tokens > MAX_CONTEXT_TOKENS:
        compressed_text = compressed_text[:MAX_CONTEXT_TOKENS * 4]
        is_truncated = True
        print(f"  ⚠️ Context truncated from ~{actual_tokens} to {MAX_CONTEXT_TOKENS} tokens to fit limits.")
        
    # -------------------------------------------------------------------------
    # API CALL 1: BATCH ANSWER GENERATION (CHUNKED)
    # -------------------------------------------------------------------------
    print(f"  -> [API Call 1/2] Asking {len(questions)} questions (Chunked to prevent token limits)...")
    
    answers_dict = {}
    chunk_size = 5  # Small chunk size to guarantee we never hit 4k limits
    
    for i in range(0, len(questions), chunk_size):
        chunk = questions[i:i + chunk_size]
        print(f"     Processing questions {i+1} to {min(i+chunk_size, len(questions))}...")
        
        q_list_for_prompt = [{"id": q["id"], "question": q["question"]} for q in chunk]
        
        answer_prompt = f"""You are analyzing compressed network diagnostic logs. Answer the following questions based ONLY on the provided logs.

        LOGS:
        {compressed_text}
        
        QUESTIONS:
        {json.dumps(q_list_for_prompt, indent=2)}
        
        INSTRUCTIONS:
        Return your answers STRICTLY as a JSON dictionary where the keys are the Question IDs (e.g., "RC-Q1") and the values are your string answers. Do not include any other text."""

        chunk_answers = query_llm_json("You are an exact JSON outputter.", answer_prompt)
        
        if chunk_answers:
            answers_dict.update(chunk_answers)
        else:
            print(f"     ⚠️ Chunk failed to parse. Marking {len(chunk)} questions as failed.")
            for q in chunk:
                answers_dict[q["id"]] = "[ERROR: Generation Failed]"

    # -------------------------------------------------------------------------
    # API CALL 2: BATCH LLM JUDGE SCORING (3 PASSES, CHUNKED, SELF-CONSISTENCY)
    # -------------------------------------------------------------------------
    print(f"  -> [API Call 2/2] Grading answers via LLM Judge (3 Passes for Self-Consistency)...")
    
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
    
    all_passes_scores = []
    chunk_size_judge = 10  # Judge output is shorter, so 10 is perfectly safe
    
    for pass_num in range(3):
        print(f"     Pass {pass_num+1}/3...")
        pass_scores_dict = {}
        
        for j in range(0, len(qa_pairs), chunk_size_judge):
            chunk_qa = qa_pairs[j:j + chunk_size_judge]
            
            judge_prompt = f"""You are an expert network diagnostic evaluator grading an AI's answers to {len(chunk_qa)} diagnostic questions.

            GROUND TRUTH METADATA (ABSOLUTE TRUTH):
            {gt_context}
            
            EVALUATION RULES:
            1. BASELINE: If Ground Truth is available, compare the AI's answers against it. If not, score based on logical consistency.
            2. ABSENT ENTITY RULE (CRITICAL): If a question asks for details, sequences, or root causes regarding a specific component, but the AI correctly identifies that this component does NOT exist or is absent from the logs, you MUST score it as 'correct' (1.0). Do NOT penalize the AI for missing details about an entity that wasn't in the logs.
            
            QA PAIRS TO EVALUATE:
            {json.dumps(chunk_qa, indent=2)}
            
            INSTRUCTIONS:
            Return your evaluation STRICTLY as a JSON dictionary. Each value must be an object with four keys: 'score' (float 0.0 to 1.0), 'verdict' ("correct", "partial", or "wrong"), 'explanation' (string justification), and 'extracted_solution' (string). 
            
            For the 'extracted_solution' key, if the AI's answer contains specific CLI commands, remediation steps, or a definitive root cause, extract that specific part. If not, return "None provided."
            
            Example Output Format:
            {{
              "RC-Q1": {{
                "score": 1.0,
                "verdict": "correct",
                "explanation": "Correctly identified the component based on Ground Truth.",
                "extracted_solution": "switch# configure terminal \\n switch(config)# interface vlan 100"
              }}
            }}"""
            
            chunk_scores = query_llm_json("You are an exact JSON outputter.", judge_prompt)
            
            if chunk_scores:
                pass_scores_dict.update(chunk_scores)
            else:
                print(f"       ⚠️ Judge chunk failed to parse. Marking {len(chunk_qa)} questions as wrong for this pass.")
                for q in chunk_qa:
                    pass_scores_dict[q["id"]] = {
                        "score": 0.0,
                        "verdict": "wrong",
                        "explanation": "Judge JSON parsing failed for this chunk.",
                        "extracted_solution": "None"
                    }
        
        all_passes_scores.append(pass_scores_dict)

    # -------------------------------------------------------------------------
    # AGGREGATE VIA MAJORITY VOTE
    # -------------------------------------------------------------------------
    scores_dict = {}
    for q in questions:
        q_id = q["id"]
        # Extract the results for this specific question across all 3 passes
        q_results = [
            s.get(q_id, {"score": 0.0, "verdict": "wrong", "explanation": "Pass failed.", "extracted_solution": "None"}) 
            for s in all_passes_scores
        ]
        
        # 1. Find the majority verdict
        verdicts = [res.get("verdict", "wrong") for res in q_results]
        majority_verdict = max(set(verdicts), key=verdicts.count)
        
        # 2. Average the score of the passes that agreed with the majority
        majority_items = [res for res in q_results if res.get("verdict", "wrong") == majority_verdict]
        
        if majority_items:
            avg_score = sum(res.get("score", 0.0) for res in majority_items) / len(majority_items)
            best_explanation = majority_items[0].get("explanation", "No explanation.")
            best_solution = majority_items[0].get("extracted_solution", "None")
        else:
            avg_score = 0.0
            best_explanation = "Failed to determine consensus."
            best_solution = "None"
        
        # 3. Store the finalized consensus
        scores_dict[q_id] = {
            "score": avg_score,
            "verdict": majority_verdict,
            "explanation": f"[Consensus: {verdicts.count(majority_verdict)}/3 passes] {best_explanation}",
            "extracted_solution": best_solution
        }

    # -------------------------------------------------------------------------
    # COMPILE RESULTS
    # -------------------------------------------------------------------------
    results = []
    category_scores = defaultdict(list)
    verdict_counts = defaultdict(int)
    
    for q in questions:
        q_id = q["id"]
        answer = answers_dict.get(q_id, "")
        
        score_data = scores_dict.get(q_id, {
            "score": 0.0, 
            "verdict": "wrong", 
            "explanation": "Scoring failed or timed out.",
            "extracted_solution": "None"
        })
        
        results.append({
            "id": q_id,
            "category": q["category"],
            "difficulty": q.get("difficulty", "medium"),      
            "weight": q["weight"],
            "question": q["question"],
            "answer": answer,
            "score": score_data.get("score", 0.0),
            "verdict": score_data.get("verdict", "wrong"),
            "explanation": score_data.get("explanation", ""),
            "extracted_solution": score_data.get("extracted_solution", "None"),
            "scoring_method": q.get("scoring_method", "llm_as_judge")
        })
        
        category_scores[q["category"]].append({"score": score_data.get("score", 0.0), "weight": q["weight"]})
        verdict_counts[score_data.get("verdict", "wrong")] += 1
    
    category_weighted_scores = {}
    category_weight_percentages = {}
    total_eval_weight = sum(q["weight"] for q in questions)
    
    for category, scores in category_scores.items():
        cat_weight = sum(s["weight"] for s in scores)
        weighted_sum = sum(s["score"] * s["weight"] for s in scores)
        
        category_weighted_scores[category] = (weighted_sum / cat_weight * 100) if cat_weight > 0 else 0
        category_weight_percentages[category] = (cat_weight / total_eval_weight * 100) if total_eval_weight > 0 else 0
    
    overall_score = sum(r["score"] * r["weight"] for r in results) / total_eval_weight * 100 if total_eval_weight > 0 else 0
    
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
        "context_tokens": actual_tokens,          
        "is_truncated": is_truncated,             
        "overall_score": round(overall_score, 1),
        "category_scores": {k: round(v, 1) for k, v in category_weighted_scores.items()},
        "category_weights": {k: round(v, 1) for k, v in category_weight_percentages.items()},
        "verdict_counts": dict(verdict_counts),
        "total_questions": len(questions),
        "results": results
    }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main(compressed_files: List[str] = None, raw_bundle_name: str = None):
    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
    target_bundle_name = "batch" 
    
    if compressed_files and len(compressed_files) == 1:
        target_bundle_name = Path(compressed_files[0]).stem.replace("compressed_", "").replace("combined_", "")
        
    all_results = []
    
    if compressed_files:
        print(f"\n{'='*80}\nProcessing {len(compressed_files)} compressed files...\n{'='*80}")
        
        for file_path in compressed_files:
            file_path = Path(file_path)
            
            if not file_path.exists():
                print(f"\n⚠️  Skipping {file_path} - file not found")
                continue
            
            log_label = file_path.stem.replace("compressed_", "").replace("combined_", "")
            compressed_text = load_compressed_text(file_path)
            
            if not compressed_text:
                print(f"⚠️  Skipping {log_label} - empty or unreadable file")
                continue
            
            ground_truth = None
            metadata_folder = raw_bundle_name if raw_bundle_name else log_label
            
            metadata_candidates = [
                BASE / "evaluation_dataset" / metadata_folder / "metadata.json",
                Path(metadata_folder) / "metadata.json" if raw_bundle_name else None 
            ]
            
            metadata_candidates = [m for m in metadata_candidates if m is not None]
            
            for metadata_path in metadata_candidates:
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            raw_meta = json.load(f)
                            ground_truth = raw_meta.get("ground_truth", raw_meta)
                            print(f"  -> Successfully loaded ground truth for {log_label}")
                        break
                    except Exception as e:
                        print(f"  ⚠️  Failed to load metadata from {metadata_path}: {e}")
            
            if not ground_truth:
                print(f"  ⚠️  No valid ground truth found. Proceeding with static questions only.")
            
            dynamic_question_list = get_dynamic_questions(log_label, ground_truth)
            
            result = run_evaluation_on_compressed(
                log_label,
                compressed_text,
                dynamic_question_list,
                ground_truth
            )
            all_results.append(result)
    else:
        print("No compressed files provided.")
        return
    
    if not all_results:
        print("\n❌ No logs were found or processed.")
        return
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    overall_avg = sum(r['overall_score'] for r in all_results) / len(all_results)
    
    out_path = EVAL_DIR / f"results_{target_bundle_name}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL,
            "backend": BACKEND,
            "timestamp": datetime.now().isoformat(),
            "overall_avg": round(overall_avg, 1),
            "max_questions_used": max((len(r["results"]) for r in all_results), default=0),
            "logs": all_results,
        }, f, indent=2)
    print(f"\n✓ Results saved → {out_path}")
    
    txt_path = EVAL_DIR / f"results_{target_bundle_name}_{timestamp}.txt"
    max_q = max((len(r["results"]) for r in all_results), default=0)
    
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("IMPROVED GENERIC PIPELINE EVALUATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model    : {MODEL}\n")
        f.write(f"Backend  : {BACKEND}\n")
        f.write(f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Questions: Up to {max_q} per bundle (Dynamic/Adversarial mode)\n\n") 
        
        for r in all_results:
            f.write("="*80 + "\n")
            f.write(f"LOG: {r['log']}\n")
            f.write("="*80 + "\n")
            f.write(f"Overall Score: {r['overall_score']:.1f}%\n")
            f.write(f"Total Questions Evaluated: {len(r['results'])}\n")
            
            token_str = f"~{r['context_tokens']:,} tokens"
            if r.get("is_truncated"):
                token_str += f" (⚠️ Truncated to {MAX_CONTEXT_TOKENS:,} to fit LLM limits)"
            f.write(f"Context: {token_str}\n\n")
            
            f.write("Weighted Category Scores (0-100%):\n")
            for category, score in sorted(r['category_scores'].items()):
                cat_weight = r.get('category_weights', {}).get(category, 0)
                f.write(f"  {category:.<45} {score:>6.1f}%  ({cat_weight:>4.1f}% weight)\n")
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
                ans_str = str(res['answer'])
                f.write(f"Answer    : {ans_str[:300]}...\n")
                f.write(f"Scoring   : {res['scoring_method']}\n")
                if res.get('explanation'):
                    f.write(f"Judge Note: {res['explanation']}\n")
                
                # Fetch directly from the res object securely
                extracted = res.get('extracted_solution', 'None')
                if extracted and extracted != "None" and extracted != "None provided.":
                    f.write("\n  ┌── EXECUTABLE SOLUTION / CLI COMMANDS ───────────\n")
                    for line in extracted.split('\n'):
                        f.write(f"  │ {line}\n")
                    f.write("  └─────────────────────────────────────────────────\n")
            
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"OVERALL AVERAGE: {overall_avg:.1f}%\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Text report  → {txt_path}\n")

if __name__ == "__main__":
    main()