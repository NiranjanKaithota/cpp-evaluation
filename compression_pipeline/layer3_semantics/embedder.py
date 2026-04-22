import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import warnings

# Suppress sklearn warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=FutureWarning)

class SemanticGrouper:
    def __init__(self, threshold=0.15):
        """
        threshold: The cosine distance threshold for grouping. 
        0.15 means templates must be 85% semantically similar to be grouped together.
        """
        # all-MiniLM-L6-v2 is an 80MB model, perfect for fast local inference
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.clustering_engine = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=threshold, 
            metric='cosine', 
            linkage='average'
        )

    def group_templates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Takes the Drain3 templates and groups them by semantic meaning.
        """
        if df.is_empty():
            return df

        templates = df["template"].to_list()
        
        # 1. Convert text to vector embeddings
        embeddings = self.model.encode(templates, show_progress_bar=False)
        
        # 2. Cluster the embeddings
        # If there's only 1 template, clustering will throw an error, so we catch that
        if len(templates) > 1:
            cluster_labels = self.clustering_engine.fit_predict(embeddings)
        else:
            cluster_labels = [0]
            
        # 3. Add the semantic cluster IDs back to our Polars DataFrame
        df = df.with_columns(pl.Series("semantic_cluster_id", cluster_labels))
        
        # 4. Roll up the data: Group by the new semantic ID, sum the frequencies, 
        # and keep the first template as the "Representative Template" for the LLM
        semantic_df = (
            df.group_by("semantic_cluster_id")
            .agg([
                pl.col("template").first().alias("representative_template"),
                pl.col("frequency").sum().alias("total_frequency"),
                pl.col("template").count().alias("drain_templates_merged")
            ])
            .sort("total_frequency", descending=True)
        )
        
        return semantic_df