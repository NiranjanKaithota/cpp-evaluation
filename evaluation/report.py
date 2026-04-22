# evaluation/report.py
import json

class ReportGenerator:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.results = []
        
    def add_result(self, bundle_name: str, metrics: dict):
        self.results.append({"bundle": bundle_name, **metrics})
        
    def generate(self):
        print("\n" + "="*160)
        print("EVALUATION PIPELINE REPORT - PHASE 1")
        print("="*160)
        print(f"{'Bundle':<30} | {'RAW: pass':<25} | {'COMP: pass':<25} | {'CR Ratio':<10} | {'CR %':<8} | {'Raw Size':<10} | {'Comp Size':<10}")
        print("-" * 160)
        
        for r in self.results:
            r_pass = "P" if r.get("raw_rcm_passed") else "F"
            raw_str = f"{r_pass}"
            
            c_pass = "P" if r.get("compressed_rcm_passed") else "F"
            comp_str = f"{c_pass}"
            
            cr_ratio = r.get("compression_ratio", 0.0)
            cr_pct = r.get("compression_percentage", 0.0)
            cr_str = f"{cr_ratio:.2f}x"
            pct_str = f"{cr_pct:.1f}%"
            raw_sz = f"{r.get('raw_size', 0):,}B"
            cmp_sz = f"{r.get('comp_size', 0):,}B"
            
            print(f"{r['bundle']:<30} | {raw_str:<25} | {comp_str:<25} | {cr_str:<10} | {pct_str:<8} | {raw_sz:<10} | {cmp_sz:<10}")
            
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
            
        print("="*80)
        print(f"Report saved to {self.output_path}")
