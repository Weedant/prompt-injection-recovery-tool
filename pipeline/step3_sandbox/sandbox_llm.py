# ============================================================================
# pipeline/step3_sandbox/sandbox_llm.py
# ============================================================================
"""
Sandbox wrapper for Groq API (Llama 3.3 70B).
Handles rate limiting, error handling, and structured output queries.
"""
import os
import time
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIError

# Load environment variables
load_dotenv()

logger = logging.getLogger("recovery.sandbox")

class SandboxLLM:
    """
    Wrapper for Groq API to sandbox suspicious prompts.
    Model: llama-3.3-70b-versatile
    Limits: 30 requests/minute (enforced via sleep)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found. Sandbox will fail if queried.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        self.last_request_time = 0
        self.min_interval = 2.0  # Safe buffer (30 RPM = 1 req/2s)

    def _enforce_rate_limit(self):
        """Sleep if requests are too frequent."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def query(self, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Send a prompt to the sandbox LLM.
        
        Args:
            prompt: The suspicious user prompt to test.
            temperature: Creativity parameter.
            
        Returns:
            Dict containing output text, model used, and token usage.
        """
        if not self.api_key:
            return {"error": "Missing API Key", "output": ""}

        self._enforce_rate_limit()

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant." # Standard system prompt to test jailbreaks against
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                temperature=temperature,
                max_tokens=1024,
            )
            
            response = chat_completion.choices[0].message.content
            usage = chat_completion.usage
            
            return {
                "output": response,
                "model": self.model,
                "tokens": {
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "total": usage.total_tokens
                }
            }

        except RateLimitError as e:
            logger.error(f"Rate limit hit: {e}. Retrying not implemented.")
            return {"error": "Rate Limit Exceeded", "output": ""}
        except APIError as e:
            logger.error(f"Groq API Error: {e}")
            return {"error": "Groq API error", "output": ""}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": "Unexpected internal sandbox error", "output": ""}
