class ConfigDedupEngine:
    def __init__(self):
        # Add commands here that provide zero debugging value to the LLM
        self.blacklist = [
            "show clock",
            # We keep "show version" and "show system" as they contain 
            # vital OS and memory limits needed for context.
        ]

    def clean_telemetry(self, showtech_blocks: dict) -> dict:
        """
        Strips out blacklisted telemetry blocks to save token space.
        """
        print(f"[*] Layer 2: Deduplicating Telemetry Blocks...")
        
        cleaned_blocks = {}
        original_count = len(showtech_blocks)
        
        for cmd, data in showtech_blocks.items():
            if cmd not in self.blacklist:
                cleaned_blocks[cmd] = data
                
        new_count = len(cleaned_blocks)
        print(f"    -> Dropped {original_count - new_count} redundant blocks.")
        
        return cleaned_blocks