#!/usr/bin/env python3
import argparse
import os

from phase1_eval import run_phase1
from eval_generic_pipeline import main as run_phase2

def main():
    parser = argparse.ArgumentParser(description="Master Evaluation Pipeline Orchestrator (Bypass Mode)")
    
    # Updated Arguments for explicit targeting
    parser.add_argument("--raw-path", type=str, required=True, help="Path to the evaluation dataset or specific raw bundle directory.")
    parser.add_argument("--compressed-name", type=str, required=True, help="Exact name of the compressed bundle folder to search for.")

    # Pipeline control
    parser.add_argument("--pipeline-type", type=str, default="text", choices=["text", "graph", "json", "toon"], help="Determines which subdirectory in dev_compressed_logs to read from.")
    
    # Execution Toggles
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM responses")
    parser.add_argument("--skip-generic-eval", action="store_true", help="Skip the generic questions evaluation (Phase 2)")
    parser.add_argument("--skip-raw-inference", action="store_true", help="Skip sending raw logs to the LLM (saves API quota)")

    args = parser.parse_args()

    print("="*80)
    print(f"MASTER ORCHESTRATOR STARTED (Bypass Mode)")
    print(f"Raw Path        : {args.raw_path}")
    print(f"Compressed Name : {args.compressed_name}")
    print(f"Pipeline Type   : {args.pipeline_type}")
    print(f"Mock LLM        : {args.mock_llm}")
    print("="*80)

    # --- Execute Phase 1 (Direct Load & RCM Eval) ---
    print("\n>>> INITIATING PHASE 1: Loading & RCM Inference")
    generated_compressed_files = run_phase1(
        raw_path=args.raw_path,
        compressed_name=args.compressed_name,
        pipeline_type=args.pipeline_type,
        mock_llm=args.mock_llm,
        skip_raw=args.skip_raw_inference
    )

    if not generated_compressed_files:
        print("\n[WARNING] Phase 1 yielded no compressed files. Halting pipeline.")
        return

    # --- Execute Phase 2 (Generic Eval) ---
    if not args.skip_generic_eval and not args.mock_llm:
        print("\n>>> INITIATING PHASE 2: Generic Questions Evaluation")
        try:
            # Extract just the folder name from the raw path
            raw_bundle = os.path.basename(os.path.normpath(args.raw_path))
            
            # Pass it into Phase 2
            run_phase2(compressed_files=generated_compressed_files, raw_bundle_name=raw_bundle)
        except Exception as e:
            print(f"\n[ERROR] Phase 2 (Generic Pipeline) failed: {e}")
    else:
        reason = "(--skip-generic-eval is active)" if args.skip_generic_eval else "(--mock-llm is active)"
        print(f"\n>>> SKIPPING PHASE 2 {reason}")

    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE.")
    print("="*80)

if __name__ == "__main__":
    main()