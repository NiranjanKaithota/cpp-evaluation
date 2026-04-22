class TableExtractor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_blocks(self) -> dict:
        """
        Reads files like routeinfo.txt and splits them into distinct dictionary blocks
        based on the command executed (e.g., 'ip -4 route show').
        """
        print(f"[*] Layer 1: Extracting Tables -> {self.file_path}")
        blocks = {}
        current_cmd = None
        current_data = []
        
        with open(self.file_path, 'r') as f:
            for line in f:
                # Basic logic to detect command headers based on your sim file structure
                if line.startswith("ip ") or line.startswith("netstat "):
                    if current_cmd:
                        blocks[current_cmd] = "\n".join(current_data)
                    current_cmd = line.strip()
                    current_data = []
                elif not line.startswith("---") and current_cmd:
                    current_data.append(line.strip())
                    
            if current_cmd:
                 blocks[current_cmd] = "\n".join(current_data)
                 
        return blocks