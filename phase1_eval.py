import os
from time import time
from datetime import datetime
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from evaluation.loader import TestCaseLoader
from evaluation.llm_wrapper import LLMEngine
from evaluation.metrics import calculate_compression_ratio, evaluate_rcm, get_file_size
from evaluation.report import ReportGenerator

# ---------------------------------------------------------
# Streamlined Loaders (No recursive searching)
# ---------------------------------------------------------

def load_text_context(compressed_name):
    target_dir = Path("dev_compressed_logs") / "compressed_text_logs" / compressed_name
    compressed_text = ""
    if target_dir.exists() and target_dir.is_dir():
        for cf in target_dir.glob("*.txt"):
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    compressed_text += f"\n--- Source: {cf.name} ---\n{f.read()}\n"
            except Exception as e:
                print(f"  [ERROR] Failed to read {cf}: {e}")
    else:
        print(f"  [ERROR] Directory not found: {target_dir}")
    return compressed_text

def load_graph_context(compressed_name):
    target_dir = Path("dev_compressed_logs") / "compressed_graph_logs" / compressed_name
    context_parts = []
    
    files_to_check = [
        (f"{compressed_name}_summary.md", "# 1. Causal Chain Summary\n"),
        # ("compressed_graph.json", "# 2. Compressed Causal Graph\n"),
        # ("cat_temp_mapping.csv", "# 3. Categorical Template Mapping\n")
    ]
    
    if target_dir.exists() and target_dir.is_dir():
        for filename, header in files_to_check:
            filepath = target_dir / filename
            if filepath.exists():
                print(f"  [INFO] Loading {filename}...")
                with open(filepath, 'r', encoding='utf-8') as f:
                    context_parts.append(header + f.read())
    else:
        print(f"  [ERROR] Directory not found: {target_dir}")
        
    return "\n\n".join(context_parts)

def load_json_context(compressed_name):
    target_dir = Path("dev_compressed_logs") / "compressed_json_logs" / compressed_name
    json_file = target_dir / "incident_context.json"
    
    if json_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return f"--- Source: incident_context.json ---\n{json.dumps(data, indent=2)}\n"
        except Exception as e:
            print(f"  [ERROR] Failed to read {json_file}: {e}")
    else:
        print(f"  [ERROR] File not found: {json_file}")
    return ""

def load_toon_context(compressed_name):
    target_dir = Path("dev_compressed_logs") / "compressed_toon_logs" / compressed_name
    compressed_text = ""
    
    if target_dir.exists() and target_dir.is_dir():
        toon_files = list(target_dir.glob("*.toon"))
        for tf in toon_files:
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    compressed_text += f"\n--- Source: {tf.name} ---\n{f.read()}\n"
                    print("    - Loaded:", tf.name)
            except Exception as e:
                print(f"  [ERROR] Failed to read {tf}: {e}")
    else:
         print(f"  [ERROR] Directory not found: {target_dir}")
    return compressed_text

# ---------------------------------------------------------
# Main Execution Logic
# ---------------------------------------------------------

# def run_phase1(raw_path: str, compressed_name: str, pipeline_type: str, mock_llm: bool, skip_raw: bool):
#     loader = TestCaseLoader("evaluation_dataset/" + raw_path)
#     bundles = loader.load_bundles()
    
#     # if not bundles:
#     #     print(f"[ERROR] No bundles loaded from {raw_path}")
#     #     return []
    
#     if not bundles:
#         if skip_raw:
#             print(f"  [INFO] No standard network bundles found in {raw_path}.")
#             print(f"  [INFO] Since --skip-raw-inference is active, generating a synthetic bundle to evaluate compressed logs directly...")
#             bundles = [{"name": os.path.basename(os.path.normpath(raw_path)), "metadata": {}}]
#         else:
#             print(f"[ERROR] No bundles loaded from {raw_path}. If testing agnostic CSV logs, use --skip-raw-inference.")
#             return []
        
#     print(f"Loaded {len(bundles)} test bundles from Raw Path.")
    
#     llm = LLMEngine(mock_mode=mock_llm, provider="cohere", model_name="command-r-08-2024")
#     reporter = ReportGenerator("evaluation_results.json") 
    
#     generated_compressed_files = []
    
#     for bundle in bundles:
#         print(f"\nProcessing Raw Bundle: {bundle['name']} | Targeting Compressed: {compressed_name}...")
        
#         raw_text = loader.get_raw_text(bundle)
#         raw_size = sum(get_file_size(bundle[k]) for k in ["messages_path", "showtech_path", "routeinfo_path"])
#         gt = bundle["metadata"].get("ground_truth", {})
        
#         # 1. Evaluate on Raw Logs
#         raw_eval = {"passed": False, "note": "Skipped"}
#         raw_pred = {}
        
#         if not skip_raw:
#             print(f"  -> Running raw inference ({raw_size} bytes)...")
#             t0 = time()
#             raw_pred = llm.run_inference(raw_text)
#             raw_latency = round(time() - t0, 2)
#             raw_eval = evaluate_rcm(raw_pred, gt)
#             print(f"     [+] Raw Inference took {raw_latency}s")
#         else:
#             raw_latency = 0.0
#             print("  -> ⏭️ Skipping raw inference.")
        
#         # 2. Load Compressed Logs
#         print(f"  -> Loading {pipeline_type} compressed context for '{compressed_name}'...")
#         if pipeline_type == "text":
#             compressed_text = load_text_context(compressed_name)
#         elif pipeline_type == "graph":
#             compressed_text = load_graph_context(compressed_name)
#         elif pipeline_type == "json":                 
#             compressed_text = load_json_context(compressed_name)
#         elif pipeline_type == "toon":
#             compressed_text = load_toon_context(compressed_name)
#         else:
#             compressed_text = ""
            
#         comp_size = len(compressed_text.encode('utf-8')) if compressed_text else 0
#         comp_ratio = calculate_compression_ratio(raw_size, comp_size) if raw_size > 0 else 0
        
#         comp_eval = {"passed": False}
#         comp_pred = {}
        
#         if compressed_text:
#             os.makedirs("compressed_logs", exist_ok=True)
#             # Save using the explicitly passed compressed_name
#             comp_path = os.path.join("compressed_logs", f"combined_{compressed_name}.txt")
            
#             try:
#                 with open(comp_path, "w", encoding="utf-8") as f:
#                     f.write(compressed_text)
#                 generated_compressed_files.append(comp_path)
#             except Exception as e:
#                 print(f"  [ERROR] Failed to save unified context file: {e}")
                
#             print(f"  -> Running compressed inference ({comp_size} bytes)...")
#             t0 = time()
#             comp_pred = llm.run_inference(compressed_text)
#             comp_latency = round(time() - t0, 2)
#             comp_eval = evaluate_rcm(comp_pred, gt) if gt else {"passed": False, "note": "No GT evaluated"}
#             print(f"     [+] Compressed Inference took {comp_latency}s")
#         else:
#             print("  [WARNING] Compressed text was empty. Inference skipped.")
            
#         bundle_result = {
#             "raw_rcm_passed": raw_eval.get("passed", False),
#             "compressed_rcm_passed": comp_eval.get("passed", False),
#             "compression_ratio": comp_ratio,
#             "raw_size": raw_size,
#             "comp_size": comp_size,
#             "raw_details": raw_pred,
#             "comp_details": comp_pred,
#             "ground_truth": gt
#         }
            
#         reporter.add_result(bundle["name"], bundle_result)
        
#         # Save individual bundle report
        
#         os.makedirs("inference", exist_ok=True)
#         timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
#         individual_report_path = os.path.join("inference", f"evaluation_report_{compressed_name}_{timestamp}.json")
#         try:
#             with open(individual_report_path, "w", encoding="utf-8") as f:
#                 json.dump({"raw_bundle": bundle["name"], "compressed_name": compressed_name, **bundle_result}, f, indent=2)
#         except Exception as e:
#             print(f"  [ERROR] Failed to save individual report: {e}")
        
#     reporter.generate()
#     print("\n" + "="*80)
#     print("PHASE 1 EVALUATION COMPLETE.")
#     print("="*80)
    
#     return generated_compressed_files

def run_phase1(raw_path: str, compressed_name: str, pipeline_type: str, mock_llm: bool, skip_raw: bool):
    loader = TestCaseLoader("evaluation_dataset/" + raw_path)
    bundles = loader.load_bundles()
    
    # ---------------------------------------------------------
    # AGILE BUNDLE HANDLING (Fix for CSV/K8s Agnostic Logs)
    # ---------------------------------------------------------
    if not bundles:
        if skip_raw:
            print(f"  [INFO] No standard network bundles found in {raw_path}.")
            print(f"  [INFO] Since --skip-raw-inference is active, generating a synthetic bundle to evaluate compressed logs directly...")
            bundles = [{"name": os.path.basename(os.path.normpath(raw_path)), "metadata": {}}]
        else:
            print(f"[ERROR] No bundles loaded from {raw_path}. If testing agnostic CSV logs, use --skip-raw-inference.")
            return []
            
    print(f"Loaded {len(bundles)} test bundles from Raw Path.")
    
    llm = LLMEngine(mock_mode=mock_llm, provider="cohere", model_name="command-r-08-2024")
    reporter = ReportGenerator("evaluation_results.json") 
    
    generated_compressed_files = []
    
    for bundle in bundles:
        print(f"\nProcessing Raw Bundle: {bundle['name']} | Targeting Compressed: {compressed_name}...")
        
        # SAFELY initializing these variables. 
        # (This removes the rogue loader.get_raw_text call that caused the KeyError)
        raw_text = ""
        raw_size = 0
        gt = bundle.get("metadata", {}).get("ground_truth", {})
        
        # 1. Evaluate on Raw Logs
        raw_eval = {"passed": False, "note": "Skipped"}
        raw_pred = {}
        
        if not skip_raw:
            try:
                raw_text = loader.get_raw_text(bundle)
                raw_size = sum(get_file_size(bundle.get(k, "")) for k in ["messages_path", "showtech_path", "routeinfo_path"] if k in bundle)
            except Exception as e:
                print(f"  [WARNING] Failed to load raw network logs: {e}")
                
            print(f"  -> Running raw inference ({raw_size} bytes)...")
            raw_pred = llm.run_inference(raw_text)
            raw_eval = evaluate_rcm(raw_pred, gt)
        else:
            print("  -> ⏭️ Skipping raw inference.")
        
        # 2. Load Compressed Logs
        print(f"  -> Loading {pipeline_type} compressed context for '{compressed_name}'...")
        if pipeline_type == "text":
            compressed_text = load_text_context(compressed_name)
        elif pipeline_type == "graph":
            compressed_text = load_graph_context(compressed_name)
        elif pipeline_type == "json":                 
            compressed_text = load_json_context(compressed_name)
        elif pipeline_type == "toon":
            compressed_text = load_toon_context(compressed_name)
        else:
            compressed_text = ""
            
        comp_size = len(compressed_text.encode('utf-8')) if compressed_text else 0
        comp_ratio = calculate_compression_ratio(raw_size, comp_size) if raw_size > 0 else 0
        
        comp_eval = {"passed": False}
        comp_pred = {}
        
        if compressed_text:
            os.makedirs("compressed_logs", exist_ok=True)
            # Save using the explicitly passed compressed_name
            comp_path = os.path.join("compressed_logs", f"combined_{compressed_name}.txt")
            
            try:
                with open(comp_path, "w", encoding="utf-8") as f:
                    f.write(compressed_text)
                generated_compressed_files.append(comp_path)
            except Exception as e:
                print(f"  [ERROR] Failed to save unified context file: {e}")
                
            print(f"  -> Running compressed inference ({comp_size} bytes)...")
            comp_pred = llm.run_inference(compressed_text)
            comp_eval = evaluate_rcm(comp_pred, gt) if gt else {"passed": False, "note": "No GT evaluated"}
        else:
            print("  [WARNING] Compressed text was empty. Inference skipped.")
            
        bundle_result = {
            "timestamp": datetime.now().strftime("%d%m%y_%H%M%S"),
            "raw_rcm_passed": raw_eval.get("passed", False),
            "compressed_rcm_passed": comp_eval.get("passed", False),
            "compression_ratio": comp_ratio,
            "raw_size": raw_size,
            "comp_size": comp_size,
            "raw_details": raw_pred,
            "comp_details": comp_pred,
            "ground_truth": gt
        }
            
        reporter.add_result(bundle["name"], bundle_result)
        
        # Save individual bundle report
        os.makedirs("inference", exist_ok=True)
        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        individual_report_path = os.path.join("inference", f"evaluation_report_{compressed_name}_{timestamp}.json")
        try:
            with open(individual_report_path, "w", encoding="utf-8") as f:
                json.dump({"raw_bundle": bundle["name"], "compressed_name": compressed_name, **bundle_result}, f, indent=2)
        except Exception as e:
            print(f"  [ERROR] Failed to save individual report: {e}")
        
    reporter.generate()
    print("\n" + "="*80)
    print("PHASE 1 EVALUATION COMPLETE.")
    print("="*80)
    
    return generated_compressed_files