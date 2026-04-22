# evaluation/compression_wrapper.py
import os
import sys

# Add compression_pipeline to sys.path
pipeline_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compression_pipeline")
if pipeline_path not in sys.path:
    sys.path.append(pipeline_path)

# Monkey-patch functools.update_wrapper for Python 3.13 datasets compatibility
import functools
_original_update_wrapper = functools.update_wrapper
def _patched_update_wrapper(wrapper, wrapped, *args, **kwargs):
    try:
        return _original_update_wrapper(wrapper, wrapped, *args, **kwargs)
    except AttributeError:
        return wrapper
functools.update_wrapper = _patched_update_wrapper

class CompressionWrapper:
    def __init__(self):
        pass

    def run_compression(self, messages_path: str, showtech_path: str, routeinfo_path: str) -> str:
        from layer1_ingestion.stream_parser import StreamParser
        from layer1_ingestion.table_extractor import TableExtractor
        from layer1_ingestion.block_parser import BlockParser
        from layer2_compression.template_miner import TemplateMiner
        from layer2_compression.config_dedup import ConfigDedupEngine
        from layer2_compression.delta_engine import StateDeltaEngine
        from layer3_semantics.embedder import SemanticGrouper
        from layer4_causality.graph_builder import CausalityGraphBuilder
        from layer4_causality.noise_filter import NoiseFilter
        from layer5_encoding.formatter import MarkdownFormatter

        # 1. Parsing
        syslog_df = None
        if os.path.exists(messages_path):
            syslog_df = StreamParser(messages_path).parse_syslog()
            
        route_blocks = None
        if os.path.exists(routeinfo_path):
            route_blocks = TableExtractor(routeinfo_path).extract_blocks()
            
        showtech_blocks = None
        if os.path.exists(showtech_path):
            showtech_blocks = BlockParser(showtech_path).parse_showtech()
            
        # 2. Structural Compression
        compressed_syslog = None
        original_lines = 0
        if syslog_df is not None:
            miner = TemplateMiner()
            compressed_syslog = miner.compress_stream(syslog_df, "messages.log")
            original_lines = syslog_df.height
            
        clean_showtech = None
        if showtech_blocks is not None:
            dedup_engine = ConfigDedupEngine()
            clean_showtech = dedup_engine.clean_telemetry(showtech_blocks)
            
        state_deltas = None
        if route_blocks is not None:
            delta_engine = StateDeltaEngine()
            current_routes = route_blocks.get('ip -4 route show table all', '')
            state_deltas = delta_engine.compute_delta(current_routes, current_routes)
            
        # 3. Semantics
        semantic_syslog = None
        if compressed_syslog is not None and original_lines > 0:
            grouper = SemanticGrouper(threshold=0.15)
            semantic_syslog = grouper.group_templates(compressed_syslog)
            
        # 4. Causality
        critical_path = []
        if semantic_syslog is not None and state_deltas is not None:
            graph_builder = CausalityGraphBuilder()
            causal_graph, _ = graph_builder.build_graph(semantic_syslog, state_deltas)
            noise_filter = NoiseFilter()
            critical_path = noise_filter.extract_critical_paths(causal_graph)
            
        # 5. Export Output
        formatter = MarkdownFormatter()
        final_prompt = formatter.generate_payload(
            telemetry=clean_showtech,
            deltas=state_deltas,
            causal_chain=critical_path,
            semantic_logs=semantic_syslog
        )
        return final_prompt
