import os

class BlockParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Log file not found: {self.file_path}")

    def parse_showtech(self) -> dict:
        """
        Parses block-formatted telemetry data from showtech logs.
        Extracts the output for each 'Command : <cmd>' section into a dictionary.
        """
        print(f"[*] Layer 1: Extracting Telemetry Blocks -> {os.path.basename(self.file_path)}")
        
        parsed_blocks = {}
        current_command = None
        current_block_data = []
        
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Detect the start of a new command block
            if line.startswith("Command : "):
                # Save the previous block before starting a new one
                if current_command:
                    # Join the collected lines and strip trailing whitespace
                    parsed_blocks[current_command] = "\n".join(current_block_data).strip()
                    
                # Update the state to the new command (e.g., "show system")
                current_command = line.replace("Command : ", "").strip()
                current_block_data = []
                
                # Skip the immediate next line if it's the asterisk border
                if i + 1 < len(lines) and lines[i+1].startswith("****"):
                    i += 1
            
            # Accumulate data if we are actively inside a command block
            elif current_command and not line.startswith("****"):
                # We use rstrip() to preserve leading spaces (which are important for 
                # tabular text formatting) but remove trailing whitespace/newlines
                current_block_data.append(lines[i].rstrip())
                
            i += 1
            
        # Catch the final block when the loop ends
        if current_command and current_block_data:
            parsed_blocks[current_command] = "\n".join(current_block_data).strip()
            
        return parsed_blocks