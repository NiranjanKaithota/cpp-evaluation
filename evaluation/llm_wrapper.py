import os
import json
import tiktoken
try:
    import cohere
except ImportError:
    cohere = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class LLMEngine:
    def __init__(self, api_key: str = None, model_name: str = None, provider: str = "cohere", mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.provider = provider
        
        self.model_name = model_name if model_name else "command-r-08-2024"
            
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.environ.get("COHERE_API_KEY")
                
        if not self.mock_mode and self.api_key and cohere:
            self.co_client = cohere.ClientV2(self.api_key, timeout=360.0)
            
        # Initialize the local tokenizer once (cl100k_base is an excellent, fast proxy for modern LLMs)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"  [WARNING] Could not load tokenizer: {e}")
            self.tokenizer = None
            
    def run_inference(self, context: str) -> dict:

        # We set the limit to 100,000 to leave an 8k token buffer for your prompt instructions and the LLM's output
        MAX_TOKENS = 100000

        if self.tokenizer:
            # 1. Encode the massive text string into an exact array of token integers
            # disallowed_special=() prevents errors if logs contain token-like strings
            tokens = self.tokenizer.encode(context, disallowed_special=())
            
            token_count = len(tokens)
            
            # 2. Check and Truncate
            if token_count > MAX_TOKENS:
                print(f"  [LLMEngine] Truncating locally from {token_count:,} tokens down to {MAX_TOKENS:,} tokens.")
                
                # Slice the array to the exact limit
                truncated_tokens = tokens[:MAX_TOKENS]
                
                # Decode the tokens back into a string to send to Cohere
                context = self.tokenizer.decode(truncated_tokens)
            else:
                print(f"  [LLMEngine] Context size is {token_count:,} tokens (Well within 128k limit).")
                
        else:
            # Fallback to the character heuristic ONLY if tiktoken fails to load
            MAX_CHARS = 250000
            if len(context) > MAX_CHARS:
                print(f"  [LLMEngine] Fallback: Truncating from {len(context)} chars to {MAX_CHARS} chars.")
                context = context[:MAX_CHARS]

        if self.mock_mode or not self.api_key:
            return {
                "root_cause_entity": "mock_entity",
                "root_cause_type": "mock_failure",
                "recommended_action": "mock_action"
            }
            
        system_instruction = """
        You are an expert network operations engineer. Analyze the following logs and determine the root cause of the failure.
        You must return ONLY a valid JSON response in the following schema without any markdown blocks or explanation.
        {
            "root_cause_entity": "The specific component that failed (e.g., 1/1/3, 10.233.255.1)",
            "root_cause_type": "The category of failure (e.g., physical_layer_failure, missing_route, configuration_error)",
            "observed_symptoms": ["List up to exactly 3 of the most critical events chronologically"],
            "recommended_action": "The recommended remediation step"
        }
        """
        
        prompt = f"{system_instruction}\n\nLogs:\n{context}"
        
        import time
        for attempt in range(3):
            try:
                response = self.co_client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                return json.loads(response.message.content[0].text)
            except Exception as e:
                error_msg = str(e).lower()
                if '429' in error_msg or 'quota' in error_msg or 'too many requests' in error_msg:
                    print(f"  [LLM] Rate limit hit. Waiting 35 seconds to retry... (Attempt {attempt+1}/3)")
                    time.sleep(35)
                else:
                    return {
                        "root_cause_entity": "error",
                        "root_cause_type": "error",
                        "recommended_action": str(e)
                    }
        return {
            "root_cause_entity": "error",
            "root_cause_type": "error",
            "recommended_action": "Max retries exceeded for rate limit"
        }

