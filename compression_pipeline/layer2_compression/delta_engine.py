import hashlib

class StateDeltaEngine:
    def _hash_row(self, row_str: str) -> str:
        return hashlib.sha256(row_str.encode('utf-8')).hexdigest()

    def _parse_table(self, raw_table_text: str) -> dict:
        parsed_state = {}
        lines = raw_table_text.strip().split('\n')
        for line in lines:
            if not line.strip() or line.startswith('Iface') or line.startswith('Destination'):
                continue
            row_hash = self._hash_row(line)
            parsed_state[row_hash] = line.strip()
        return parsed_state

    def compute_delta(self, state_t1_text: str, state_t2_text: str) -> dict:
        """Computes added/removed rows between two state snapshots."""
        print("[*] Layer 2: Computing State Deltas...")
        state1 = self._parse_table(state_t1_text)
        state2 = self._parse_table(state_t2_text)
        
        s1_keys = set(state1.keys())
        s2_keys = set(state2.keys())
        
        return {
            "added": [state2[k] for k in (s2_keys - s1_keys)],
            "removed": [state1[k] for k in (s1_keys - s2_keys)]
        }