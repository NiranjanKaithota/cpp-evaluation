import polars as pl
import re
from pathlib import Path

class StreamParser:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.file_path}")

    def parse_ovs_runtime(self) -> pl.DataFrame:
        """
        Parses OVS runtime logs.
        Format: 2026-03-12T13:56:49.872Z|00001|vlog|INFO|opened log file...
        """
        print(f"[*] Layer 1: Ingesting OVS Stream -> {self.file_path.name}")
        
        # Read the raw text file into a single-column Polars DataFrame
        df = pl.read_csv(self.file_path, has_header=False, new_columns=["raw_line"], separator="\x1e") # using a rare separator to read whole lines
        
        # Extract fields using regex
        # OVS pattern: timestamp | thread | module | severity | message
        regex_pattern = r"^(?P<timestamp>[^|]+)\|(?P<thread>[^|]+)\|(?P<module>[^|]+)\|(?P<severity>[^|]+)\|(?P<message>.*)$"
        
        parsed_df = df.select(
            pl.col("raw_line").str.extract_groups(regex_pattern)
        ).unnest("raw_line")
        
        # Drop empty rows (if any regex failed) and add a modality tag
        parsed_df = parsed_df.drop_nulls().with_columns(
            pl.lit("STREAM_OVS").alias("modality")
        )
        
        return parsed_df

    def parse_syslog(self) -> pl.DataFrame:
        """
        Parses standard system messages (like messages_sim.log).
        Format: 2026-03-12T22:39:52.019083+05:30 kali kernel: IPv6: ADDRCONF...
        """
        print(f"[*] Layer 1: Ingesting Syslog Stream -> {self.file_path.name}")
        
        df = pl.read_csv(self.file_path, has_header=False, new_columns=["raw_line"], separator="\x1e")
        
        # Syslog pattern: timestamp hostname service[pid]: message
        regex_pattern = r"^(?P<timestamp>\S+)\s+(?P<hostname>\S+)\s+(?P<service>[^:]+):\s+(?P<message>.*)$"
        
        parsed_df = df.select(
            pl.col("raw_line").str.extract_groups(regex_pattern)
        ).unnest("raw_line")
        
        # We assign 'INFO' as default severity if not explicitly stated in syslog
        parsed_df = parsed_df.drop_nulls().with_columns(
            pl.lit("INFO").alias("severity"),
            pl.lit("STREAM_SYSLOG").alias("modality")
        )
        
        return parsed_df