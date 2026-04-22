import polars as pl

class MarkdownFormatter:
    def __init__(self):
        pass

    def generate_payload(self, telemetry: dict, deltas: dict, causal_chain: list, semantic_logs: pl.DataFrame) -> str:
        """
        Formats the compressed data into a clean, LLM-readable Markdown prompt.
        """
        print(" [*] Operation: Prompt Generation & Formatting")
        print("     -> Engine: Markdown Builder")
        
        payload = ["# HPE Support Bundle Diagnostics\n"]
        
        # 1. System Environment
        payload.append("## 1. System Environment State")
        if telemetry:
            sys_info = telemetry.get('show system', 'System info unavailable')
            version_info = telemetry.get('show version', 'Version info unavailable')
            payload.append(f"**OS/Version:**\n```text\n{version_info}\n```")
            payload.append(f"**System State:**\n```text\n{sys_info}\n```\n")

        # 2. State Deltas 
        payload.append("## 2. Configuration Deltas")
        if deltas and deltas.get("removed"):
            payload.append("**Critical Missing Routes:**")
            for route in deltas["removed"]:
                payload.append(f"- {route}")
            payload.append("\n")
        else:
            payload.append("*No critical state drops detected.*\n")

        # 3. Causal Narrative
        payload.append("## 3. Causal Event Narrative")
        if causal_chain:
            payload.append("The following events are mathematically correlated to the state drops:")
            for item in causal_chain:
                node_type = "🔴 CAUSE" if item['type'] == 'state_change' else "🟡 SYMPTOM"
                payload.append(f"- **[{node_type}]** {item['description']}")
            payload.append("\n")
        else:
            payload.append("*Graph analysis complete. No direct causal links found between logs and state drops.*\n")

        # 4. Top Semantic Anomalies
        payload.append("## 4. Top Semantic Log Clusters")
        payload.append("Highest frequency system events during the window:")
        if semantic_logs is not None and not semantic_logs.is_empty():
            for row in semantic_logs.head(10).iter_rows(named=True):
                payload.append(f"- [Freq: {row['total_frequency']}] {row['representative_template']}")

        return "\n".join(payload)