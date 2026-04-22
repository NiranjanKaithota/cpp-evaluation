import polars as pl
from drain3 import TemplateMiner as Drain3Miner
from drain3.template_miner_config import TemplateMinerConfig

class TemplateMiner:
    def __init__(self):
        # Initialize Drain3 with default configurations
        config = TemplateMinerConfig()
        config.profiling_enabled = False
        self.miner = Drain3Miner(persistence_handler=None, config=config)

    def compress_stream(self, df: pl.DataFrame, stream_name: str) -> pl.DataFrame:
        """
        Feeds the 'message' column of a Polars DataFrame into Drain3.
        Returns a compressed DataFrame containing only the unique templates and counts.
        """
        print(f"[*] Layer 2: Mining Templates for {stream_name}...")
        
        # Ensure we actually have a message column
        if "message" not in df.columns:
            print(f"    [!] Error: No 'message' column found in {stream_name}")
            return df

        # Feed each message into the Drain algorithm
        # (Drain builds a parsing tree dynamically in memory)
        messages = df["message"].to_list()
        for msg in messages:
            if msg:  # Skip empty messages
                self.miner.add_log_message(msg)
                
        # Extract the compressed clusters
        clusters = self.miner.drain.clusters
        
        # Format the output into a new, heavily reduced Polars DataFrame
        compressed_data = [
            {
                "cluster_id": cluster.cluster_id,
                "template": cluster.get_template(),
                "frequency": cluster.size,
                "modality": stream_name
            }
            for cluster in clusters
        ]
        
        compressed_df = pl.DataFrame(compressed_data)
        
        # Sort by frequency (highest first) to see the worst offenders
        compressed_df = compressed_df.sort("frequency", descending=True)
        
        return compressed_df