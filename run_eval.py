import argparse
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from evaluation.loader import TestCaseLoader
from evaluation.compression_wrapper import CompressionWrapper
from evaluation.llm_wrapper import LLMEngine
from evaluation.metrics import calculate_compression_ratio, evaluate_rcm, get_file_size
from evaluation.report import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="HPE Support Bundle Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output metrics file")
    parser.add_argument("--provider", type=str, default="cohere", choices=["cohere"], help="LLM Provider")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g., command-r-08-2024, gemini-2.5-pro)")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM responses")
    args = parser.parse_args()
    
    loader = TestCaseLoader(args.dataset)
    bundles = loader.load_bundles()
    
    if not bundles:
        print(f"No bundles found in {args.dataset}")
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
        
        # 1. Evaluate on Raw Logs
        print(f"  -> Running raw inference ({raw_size} bytes)...")
        raw_pred = llm.run_inference(raw_text)
        gt = bundle["metadata"].get("ground_truth", {})
        raw_eval = evaluate_rcm(raw_pred, gt)
        
        # 2. Evaluate on Compressed Logs
        print("  -> Executing compression pipeline...")
        compressed_text = ""
        try:
            compressed_text = compressor.run_compression(
                bundle["messages_path"],
                bundle["showtech_path"],
                bundle["routeinfo_path"]
            )
        except Exception as e:
            print(f"  [ERROR] Compression failed: {e}")
            
        comp_size = len(compressed_text.encode('utf-8')) if compressed_text else 0
        comp_ratio = calculate_compression_ratio(raw_size, comp_size)
        from evaluation.metrics import calculate_compression_percentage
        comp_percent = calculate_compression_percentage(raw_size, comp_size)
        
        comp_eval = {"passed": False}
        
        if compressed_text:
            os.makedirs("compressed_logs", exist_ok=True)
            comp_path = os.path.join("compressed_logs", f"compressed_{bundle['name']}.txt")
            with open(comp_path, "w", encoding="utf-8") as f:
                f.write(compressed_text)
            generated_compressed_files.append(comp_path)
                
            print(f"  -> Running compressed inference ({comp_size} bytes - {comp_ratio:.2f}x reduction / {comp_percent:.2f}%)...")
            comp_pred = llm.run_inference(compressed_text)
            comp_eval = evaluate_rcm(comp_pred, gt)
            
        bundle_result = {
            "raw_rcm_passed": raw_eval.get("passed", False),
            "compressed_rcm_passed": comp_eval.get("passed", False),
            
            "compression_ratio": comp_ratio,
            "compression_percentage": comp_percent,
            "raw_size": raw_size,
            "comp_size": comp_size,
            "raw_details": raw_pred,
            "comp_details": comp_pred if compressed_text else {},
            "ground_truth": gt
        }
            
        reporter.add_result(bundle["name"], bundle_result)
        
        # Save individual bundle report to /inference directory
        os.makedirs("inference", exist_ok=True)
        import json
        individual_report_path = os.path.join("inference", f"evaluation_report_{bundle['name']}.json")
        with open(individual_report_path, "w", encoding="utf-8") as f:
            json.dump({"bundle": bundle["name"], **bundle_result}, f, indent=2)
        
        
    reporter.generate()
    
    print("\n" + "="*80)
    print("PHASE 1 EVALUATION COMPLETE.")
    print("="*80)
    
    print("\n" + "="*80)
    print("STARTING GENERIC QUESTIONS PIPELINE")
    print("="*80)
    import eval_generic_pipeline
    eval_generic_pipeline.main(generated_compressed_files)

if __name__ == "__main__":
    main()
