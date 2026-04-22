import os

# --- LAYER 1 IMPORTS ---
from layer1_ingestion.stream_parser import StreamParser
from layer1_ingestion.table_extractor import TableExtractor
from layer1_ingestion.block_parser import BlockParser

# --- LAYER 2 IMPORTS ---
from layer2_compression.template_miner import TemplateMiner
from layer2_compression.config_dedup import ConfigDedupEngine
from layer2_compression.delta_engine import StateDeltaEngine

# --- LAYER 3 IMPORTS ---
from layer3_semantics.embedder import SemanticGrouper

# --- LAYER 4 IMPORTS ---
from layer4_causality.graph_builder import CausalityGraphBuilder
from layer4_causality.noise_filter import NoiseFilter

# --- LAYER 5 IMPORTS ---
from layer5_encoding.formatter import MarkdownFormatter
from layer5_encoding.token_manager import TokenManager

def main():
    print("==================================================")
    print("  HPE Support Bundle Reducer - Pipeline Started   ")
    print("==================================================\n")

    # Standardized Enterprise Data Paths
    syslog_path = r"D:\Projects\HPE_CPP3\cpp3-61\data\raw\sim_logs\messages.log"
    route_log_path = r"D:\Projects\HPE_CPP3\cpp3-61\data\raw\sim_logs\routeinfo.txt"
    showtech_log_path = r"D:\Projects\HPE_CPP3\cpp3-61\data\raw\sim_logs\showtech.txt"

    # ==========================================
    # --- LAYER 1: INGESTION & ROUTING ---
    # ==========================================
    print("🔽 LAYER 1: Modality Routing & Parsing")
    print("-" * 50)
    
    syslog_df = None
    if os.path.exists(syslog_path):
        syslog_df = StreamParser(syslog_path).parse_syslog()
        print(f" [✓] Stream Parser : Ingested {syslog_df.height} event logs from messages.log")
    else:
        print(f" [!] WARNING: Could not find {syslog_path}")

    route_blocks = None
    if os.path.exists(route_log_path):
        route_blocks = TableExtractor(route_log_path).extract_blocks()
        print(f" [✓] Table Extractor: Parsed {len(route_blocks)} routing state tables")
    else:
        print(f" [!] WARNING: Could not find {route_log_path}")

    showtech_blocks = None
    if os.path.exists(showtech_log_path):
        showtech_blocks = BlockParser(showtech_log_path).parse_showtech()
        print(f" [✓] Block Parser  : Extracted {len(showtech_blocks)} telemetry blocks")
    else:
        print(f" [!] WARNING: Could not find {showtech_log_path}")

    print("\n")

    # ==========================================
    # --- LAYER 2: STRUCTURAL COMPRESSION ---
    # ==========================================
    print("🔽 LAYER 2: Structural Compression & Deduplication")
    print("-" * 50)
    
    compressed_syslog = None
    original_lines = 0

    # 1. Compress Log Streams using Drain3
    if syslog_df is not None:
        print(" [*] Operation: Template Extraction & Burst Deduplication")
        print("     -> Engine: Drain3 (Online Log Parsing)")
        miner = TemplateMiner()
        compressed_syslog = miner.compress_stream(syslog_df, "messages.log")
        
        original_lines = syslog_df.height
        compressed_lines = compressed_syslog.height
        
        if original_lines > 0:
            reduction = (1 - (compressed_lines / original_lines)) * 100
            print(f"     -> Result: Collapsed {original_lines} raw lines into {compressed_lines} unique templates.")
            print(f"     -> Metric: {reduction:.2f}% Volume Reduction Achieved\n")

    # 2. Clean Telemetry
    if showtech_blocks is not None:
        print(" [*] Operation: Static Boilerplate Removal")
        print("     -> Engine: Config Dedup Engine")
        dedup_engine = ConfigDedupEngine()
        clean_showtech = dedup_engine.clean_telemetry(showtech_blocks)
        
    # 3. Compute Routing State Deltas
    state_deltas = None
    if route_blocks is not None:
        print(" [*] Operation: State Delta Computation")
        print("     -> Engine: State Delta Engine")
        delta_engine = StateDeltaEngine()
        
        # --- SIMULATION FOR POC ---
        # In production, HPE will provide the 'baseline' from a previous passing health check.
        # For this PoC, we grab the current route table...
        current_routes = route_blocks.get('ip -4 route show table all', '')
        
        # ...and we simulate a "Past State" where a critical database subnet existed.
        mock_past_routes = current_routes + "\n10.99.0.0/16 dev br-database proto kernel scope link src 10.99.0.1"
        
        # Now we actually compute the delta!
        state_deltas = delta_engine.compute_delta(mock_past_routes, current_routes)
        
        print(f"     -> Result: Computed deltas against baseline state.")
        if state_deltas["removed"]:
            print(f"     -> [!] CAUSALITY TRIGGER: Detected {len(state_deltas['removed'])} missing route(s)!")
            for route in state_deltas["removed"]:
                print(f"        - Dropped: {route}")

    print("\n")

    # ==========================================
    # --- LAYER 3: SEMANTIC FINGERPRINTING ---
    # ==========================================
    print("🔽 LAYER 3: Semantic Fingerprinting (The Context Engine)")
    print("-" * 50)
    
    semantic_syslog = None
    
    if compressed_syslog is not None and original_lines > 0:
        print(" [*] Operation: NLP Vector Embedding & Cosine Clustering")
        print("     -> Model : all-MiniLM-L6-v2 (Local Inference)")
        
        grouper = SemanticGrouper(threshold=0.15)
        semantic_syslog = grouper.group_templates(compressed_syslog)
        
        drain_count = compressed_syslog.height
        semantic_count = semantic_syslog.height
        
        print(f"     -> Result: Merged {drain_count} syntactic templates into {semantic_count} semantic meaning clusters.")
        
        # Check if any templates were successfully merged and display an example
        import polars as pl
        merged_clusters = semantic_syslog.filter(pl.col("drain_templates_merged") > 1)
        if merged_clusters.height > 0:
            top_merge = merged_clusters.row(0, named=True)
            print(f"     -> Insight: Merged {top_merge['drain_templates_merged']} different log structures into -> '{top_merge['representative_template']}'")

    print("\n[+] Layer 3 Complete. Data is semantically grouped.\n")

    # ==========================================
    # --- LAYER 4: CAUSALITY PRESERVATION ---
    # ==========================================
    print("🔽 LAYER 4: Causality Graph (SEAL-Inspired)")
    print("-" * 50)
    
    critical_path = []
    
    if semantic_syslog is not None and state_deltas is not None:
        print(" [*] Operation: Dependency Graph Construction")
        print("     -> Engine: NetworkX Directed Graph")
        
        # 1. Build the raw graph
        graph_builder = CausalityGraphBuilder()
        causal_graph, edge_count = graph_builder.build_graph(semantic_syslog, state_deltas)
        print(f"     -> Result: Graph built with {causal_graph.number_of_nodes()} nodes and {edge_count} causal edges.\n")
        
        # 2. Filter the noise
        noise_filter = NoiseFilter()
        critical_path = noise_filter.extract_critical_paths(causal_graph)
        
        if critical_path:
            print("\n     [!] CRITICAL CAUSAL CHAIN DETECTED:")
            for item in critical_path:
                prefix = "🔴 ROOT CAUSE:" if item['type'] == 'state_change' else "🟡 SYMPTOM  :"
                print(f"         {prefix} {item['description'][:80]}...")

    print("\n[+] Pipeline Execution Paused at Layer 4.")
    print("    Causality extracted. Ready for Layer 5 Token Encoding.")
    
    # ==========================================
    # --- LAYER 5: TOKEN ENCODING & EXPORT ---
    # ==========================================
    print("🔽 LAYER 5: Formatting & Token Validation")
    print("-" * 50)
    
    # 1. Format the data into Markdown
    formatter = MarkdownFormatter()
    final_prompt = formatter.generate_payload(
        telemetry=clean_showtech,
        deltas=state_deltas,
        causal_chain=critical_path,
        semantic_logs=semantic_syslog
    )
    
    # 2. Validate the token budget
    token_manager = TokenManager()
    metrics = token_manager.validate_budget(final_prompt)
    
    print("\n==================================================")
    print(" 🚀 PIPELINE COMPLETE. READY FOR LLM INFERENCE. ")
    print("==================================================\n")
    
    # Save the final prompt to disk
    output_path = r"D:\Projects\HPE_CPP3\cpp3-61\data\output\llm_final_prompt.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_prompt)
        
    print(f"[+] Saved final LLM prompt ({metrics['tokens']} tokens) to: {output_path}")

if __name__ == "__main__":
    main()