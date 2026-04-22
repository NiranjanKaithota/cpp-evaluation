import networkx as nx

class NoiseFilter:
    def __init__(self):
        pass

    def extract_critical_paths(self, causal_graph: nx.DiGraph) -> list:
        """
        Graph Compression Engine (SEAL-Inspired).
        Takes the raw dependency graph and mathematically isolates the 
        connected components that represent the actual failure chain.
        """
        print(" [*] Operation: Graph Compression & Noise Neutralization")
        print("     -> Engine: Noise Filter (Component Isolation)")
        
        # 1. Identify nodes that are actually part of a causal chain (degree > 0)
        connected_nodes = [n for n in causal_graph.nodes() if causal_graph.degree(n) > 0]
        
        # 2. Extract the exact payload for the LLM
        critical_story = []
        for node in connected_nodes:
            node_data = causal_graph.nodes[node]
            critical_story.append({
                "id": node,
                "type": node_data["type"],
                "description": node_data["desc"]
            })
            
        # Calculate reduction metrics
        total_nodes = causal_graph.number_of_nodes()
        noise_dropped = total_nodes - len(critical_story)
        
        print(f"     -> Result: Dropped {noise_dropped} isolated noise nodes.")
        if total_nodes > 0:
            reduction = (noise_dropped / total_nodes) * 100
            print(f"     -> Metric: {reduction:.2f}% graph volume reduction.")
            
        return critical_story