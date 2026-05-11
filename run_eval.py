import argparse
import os
from time import time
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from evaluation.loader import TestCaseLoader
from evaluation.compression_wrapper import CompressionWrapper
from evaluation.llm_wrapper import LLMEngine
from evaluation.metrics import calculate_compression_ratio, evaluate_rcm, get_file_size
from evaluation.report import ReportGenerator

def load_json_context(bundle_name):
    """Bulletproof loader for the single incident_context.json file."""
    base_dir = Path("dev_compressed_logs")
    json_file = None
    
    # Recursively search for the bundle folder, then check for the JSON file inside it
    for path in base_dir.rglob(bundle_name):
        if path.is_dir():
            potential_file = path / "incident_context.json"
            if potential_file.exists():
                json_file = potential_file
                break
                
    compressed_text = ""
    
    if json_file:
        print(f"  -> [SUCCESS] Found JSON context at: {json_file}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                # Load and re-dump the JSON to ensure it is valid and formatted cleanly for the LLM
                data = json.load(f)
                compressed_text = f"--- Source: incident_context.json ---\n{json.dumps(data, indent=2)}\n"
        except Exception as e:
            print(f"  [ERROR] Failed to read or parse {json_file}: {e}")
    else:
        print(f"  [ERROR] Could not find 'incident_context.json' inside any folder named '{bundle_name}'")
        
    return compressed_text

def load_graph_context(bundle_name):
    """Loads and concatenates the 3 graph pipeline files into a single context string."""
    graph_dir = os.path.join("dev_compressed_logs", "compressed_graph_logs", bundle_name)
    
    graph_path = os.path.join(graph_dir, "compressed_graph.json")
    summary_path = os.path.join(graph_dir, "casual_chain_summary.md")
    mapping_path = os.path.join(graph_dir, "cat_temp_mapping.csv")
    
    context_parts = []
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            context_parts.append("# 1. Causal Chain Summary\n" + f.read())
            
    if os.path.exists(graph_path):
        with open(graph_path, 'r', encoding='utf-8') as f:
            context_parts.append("# 2. Compressed Causal Graph\n" + f.read())
            
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            context_parts.append("# 3. Categorical Template Mapping\n" + f.read())
            
    if not context_parts:
        return ""
        
    return "\n\n".join(context_parts)

def load_existing_text_context(bundle_name):
    """Loads all existing text files from a specific bundle's compressed directory."""
    # FIX: We now append the bundle_name to the end of the path
    bundle_dir = Path("dev_compressed_logs") / "compressed_text_logs" / bundle_name
    compressed_text = ""
    
    if bundle_dir.exists() and bundle_dir.is_dir():
        comp_files = list(bundle_dir.glob("*.txt"))
        
        if not comp_files:
             print(f"  [WARNING] Folder found at {bundle_dir}, but no .txt files inside.")
             
        for cf in comp_files:
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    compressed_text += f"\n--- Source: {cf.name} ---\n"
                    compressed_text += f.read() + "\n"
            except Exception as e:
                print(f"  [ERROR] Failed to read {cf}: {e}")
    else:
        # Added a debug print to show exactly what path it failed to find
        print(f"  [ERROR] Directory does not exist: {bundle_dir}")
        
    return compressed_text

def load_toon_context(bundle_name):
    """Bulletproof loader for the .toon format files."""
    base_dir = Path("dev_compressed_logs")
    toon_files = []
    
    # Recursively search for the bundle folder, then grab any .toon files inside
    for path in base_dir.rglob(bundle_name):
        if path.is_dir():
            toon_files = list(path.glob("*.toon"))
            if toon_files:
                break
                
    compressed_text = ""
    
    if toon_files:
        print(f"  -> [SUCCESS] Found TOON context at: {toon_files[0].parent}")
        for tf in toon_files:
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    compressed_text += f"\n--- Source: {tf.name} ---\n"
                    compressed_text += f.read() + "\n"
            except Exception as e:
                print(f"  [ERROR] Failed to read {tf}: {e}")
    else:
        print(f"  [ERROR] Could not find any '.toon' files inside any folder named '{bundle_name}'")
        
    return compressed_text

def main():
    parser = argparse.ArgumentParser(description="HPE Support Bundle Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output metrics file")
    parser.add_argument("--provider", type=str, default="cohere", choices=["cohere"], help="LLM Provider")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g., command-r-08-2024)")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM responses")
    
    # Execution Toggles
    parser.add_argument("--skip-generic-eval", action="store_true", help="Skip the generic questions evaluation")
    parser.add_argument("--skip-raw-inference", action="store_true", help="Skip sending raw logs to the LLM (saves API quota)")
    parser.add_argument("--use-existing-compressed", action="store_true", help="Bypass CompressionWrapper, load existing compressed files directly")
    
    # Target Toggles
    # Update choices to include "json"
    parser.add_argument("--pipeline-type", type=str, default="text", choices=["text", "graph", "json", "toon"], help="Type of compression output")
    parser.add_argument("--bundle", type=str, default=None, help="Name of a specific bundle to run")
    
    args = parser.parse_args()
    
    loader = TestCaseLoader(args.dataset)
    bundles = loader.load_bundles()
    
    if not bundles:
        print(f"No bundles found in {args.dataset}")
        return
        
    if args.bundle:
        bundles = [b for b in bundles if b['name'] == args.bundle]
        if not bundles:
            print(f"[ERROR] Bundle '{args.bundle}' not found in {args.dataset}.")
            return
        
    print(f"Loaded {len(bundles)} test bundles.")
    
    compressor = CompressionWrapper()
    llm = LLMEngine(mock_mode=args.mock_llm, provider=args.provider, model_name=args.model)
    reporter = ReportGenerator(args.output)
    
    generated_compressed_files = []
    
    for bundle in bundles:
        print(f"\nProcessing {bundle['name']}...")
        
        raw_text = loader.get_raw_text(bundle)
        raw_size = sum(get_file_size(bundle[k]) for k in ["messages_path", "showtech_path", "routeinfo_path"])
        gt = bundle["metadata"].get("ground_truth", {})
        
        # 1. Evaluate on Raw Logs (Bypassable)
        raw_eval = {"passed": False, "note": "Skipped"}
        raw_pred = {}
        
        if not args.skip_raw_inference:
            print(f"  -> Running raw inference ({raw_size} bytes)...")
            raw_pred = llm.run_inference(raw_text)
            raw_eval = evaluate_rcm(raw_pred, gt)
        else:
            print("  -> ⏭️ Skipping raw inference (--skip-raw-inference flag active).")
        
        # 2. Evaluate on Compressed Logs
        print(f"  -> Gathering {args.pipeline_type} compressed context...")
        compressed_text = ""
        try:
            if args.pipeline_type == "text":
                if args.use_existing_compressed:
                    print("  -> Loading existing text files from dev_compressed_logs...")
                    compressed_text = load_existing_text_context(bundle["name"])
                else:
                    print("  -> Executing CompressionWrapper pipeline...")
                    compressed_text = compressor.run_compression(
                        bundle["messages_path"],
                        bundle["showtech_path"],
                        bundle["routeinfo_path"]
                    )
            elif args.pipeline_type == "graph":
                compressed_text = load_graph_context(bundle["name"])
            elif args.pipeline_type == "json":                 
                compressed_text = load_json_context(bundle["name"])
            elif args.pipeline_type == "toon":
                compressed_text = load_toon_context(bundle["name"])
                
            if not compressed_text:
                print(f"  [WARNING] Compressed context is empty for {bundle['name']}. Files may be missing.")
        except Exception as e:
            print(f"  [ERROR] Pipeline gathering failed: {e}")
            
        comp_size = len(compressed_text.encode('utf-8')) if compressed_text else 0
        comp_ratio = calculate_compression_ratio(raw_size, comp_size) if raw_size > 0 else 0
        
        comp_eval = {"passed": False}
        comp_pred = {}
        
        if compressed_text:
            # We save a combined file so Phase 2 has a single path to target
            os.makedirs("compressed_logs", exist_ok=True)
            comp_path = os.path.join("compressed_logs", f"combined_{bundle['name']}.txt")
            
            try:
                with open(comp_path, "w", encoding="utf-8") as f:
                    f.write(compressed_text)
                generated_compressed_files.append(comp_path)
            except Exception as e:
                print(f"  [ERROR] Failed to save unified context file: {e}")
                
            print(f"  -> Running compressed inference ({comp_size} bytes)...")
            comp_pred = llm.run_inference(compressed_text)
            comp_eval = evaluate_rcm(comp_pred, gt) if gt else {"passed": False, "note": "No GT evaluated"}
            
        bundle_result = {
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
        import json
        timestamp = int(time())
        individual_report_path = os.path.join("inference", f"evaluation_report_{bundle['name']}_{timestamp}.json")
        try:
            with open(individual_report_path, "w", encoding="utf-8") as f:
                json.dump({"bundle": bundle["name"], **bundle_result}, f, indent=2)
        except Exception as e:
            print(f"  [ERROR] Failed to save individual report: {e}")
        
    reporter.generate()
    
    print("\n" + "="*80)
    print("PHASE 1 EVALUATION COMPLETE.")
    print("="*80)
    
    # 3. Generic Questions Pipeline
    if generated_compressed_files and not args.skip_generic_eval and not args.mock_llm:
        print("\n" + "="*80)
        print("STARTING GENERIC QUESTIONS PIPELINE")
        print("="*80)
        
        try:
            import sys
            import importlib.util
            
            fixed_path = os.path.join(os.path.dirname(__file__), "eval_generic_pipeline.py")
            if os.path.exists(fixed_path):
                spec = importlib.util.spec_from_file_location("eval_generic_pipeline", fixed_path)
                eval_generic_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(eval_generic_module)
                
                # Pass the unified files to Phase 2
                eval_generic_module.main(compressed_files=generated_compressed_files)
            else:
                print(f"[ERROR] Could not find {fixed_path}")
                
        except Exception as e:
            print(f"\n[ERROR] Generic pipeline failed: {e}")
    elif args.mock_llm:
        print("\n⏭️  Skipping generic questions evaluation (--mock-llm flag set)")
    elif args.skip_generic_eval:
        print("\n⏭️  Skipping generic questions evaluation (--skip-generic-eval flag set)")

if __name__ == "__main__":
    main()