import networkx as nx
import polars as pl
import re

class CausalityGraphBuilder:
    def __init__(self):
        # We use a Directed Graph because causality implies a sequence (A caused B)
        self.graph = nx.DiGraph()

    def _extract_entities(self, text: str) -> set:
        """
        Extracts system entities (IPs, MACs, Interface names like 'eth0' or 'br-database')
        to find hidden links between different log files.
        """
        # Find alphanumeric strings with dashes or colons
        words = set(re.findall(r'[a-zA-Z0-9:-]+', str(text)))
        
        # Blacklist common networking stopwords so we don't draw false edges
        stopwords = {'dev', 'proto', 'kernel', 'scope', 'link', 'src', 'to', 'from', 
                     'the', 'is', 'and', 'INFO', 'WARN', 'DBG', 'ERR', 'AUDIT'}
        
        return {w for w in words if w not in stopwords and len(w) > 2}

    def build_graph(self, semantic_df: pl.DataFrame, deltas: dict) -> tuple:
        """
        Constructs the causal graph from state changes and log events.
        """
        print("[*] Layer 4: Constructing Dependency Graph...")
        
        delta_nodes = []
        log_nodes = []

        # 1. Inject State Deltas as "Root Cause" Nodes
        if deltas and "removed" in deltas:
            for idx, route in enumerate(deltas["removed"]):
                node_id = f"DELTA_DROP_{idx}"
                # High weight because configuration drops are usually root causes
                self.graph.add_node(node_id, type="state_change", desc=f"Dropped Route: {route}", weight=10)
                delta_nodes.append((node_id, route))

        # 2. Inject Semantic Clusters as "Symptom" Nodes
        if semantic_df is not None and not semantic_df.is_empty():
            for row in semantic_df.iter_rows(named=True):
                cluster_id = f"LOG_CLUSTER_{row['semantic_cluster_id']}"
                template = row['representative_template']
                
                # Dynamic weighting: Errors/Drops get higher graph priority than normal polling
                weight = 5 if any(x in template.lower() for x in ['down', 'fail', 'error', 'drop', 'timeout']) else 1
                
                self.graph.add_node(cluster_id, type="log_event", desc=template, weight=weight)
                log_nodes.append((cluster_id, template))

        # 3. Draw Causal Edges (The SEAL Approach)
        edges_created = 0
        for delta_id, delta_text in delta_nodes:
            delta_entities = self._extract_entities(delta_text)
            
            for log_id, log_text in log_nodes:
                log_entities = self._extract_entities(log_text)
                
                # If the missing route and the log error mention the same interface/IP
                shared_entities = delta_entities.intersection(log_entities)
                if shared_entities:
                    self.graph.add_edge(delta_id, log_id, relation="shared_entity", entities=list(shared_entities))
                    edges_created += 1
                    
        return self.graph, edges_created

    def extract_critical_paths(self) -> list:
        """
        Graph Compression: Drops floating nodes (noise) and keeps only connected 
        components (the actual causal chain of the failure).
        """
        # Find nodes that actually have connections
        connected_nodes = [n for n in self.graph.nodes() if self.graph.degree(n) > 0]
        
        # Extract the exact payload for the LLM
        critical_story = []
        for node in connected_nodes:
            node_data = self.graph.nodes[node]
            critical_story.append({
                "id": node,
                "type": node_data["type"],
                "description": node_data["desc"]
            })
            
        return critical_story