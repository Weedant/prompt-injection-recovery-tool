# ============================================================================
# pipeline/step3_sandbox/pipeline.py
# ============================================================================
"""
Step 3 Pipeline: Orchestrates Prefilter -> Sandbox -> Behavior Analysis.
"""
import logging
from typing import Dict, Any

from pipeline.step2_prefilter.service import is_suspicious
from pipeline.step3_sandbox.sandbox_llm import SandboxLLM
from pipeline.step3_sandbox.behavior_detectors import BehaviorAnalyzer

logger = logging.getLogger("recovery.pipeline.step3")

class Step3Pipeline:
    """
    Integrates Step 2 (Prefilter) and Step 3 (Sandbox).
    """
    def __init__(self):
        self.sandbox = SandboxLLM()
        self.analyzer = BehaviorAnalyzer()

    def process(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user input through the defense pipeline.
        
        Args:
            user_input: The raw prompt string.
            
        Returns:
            Dict containing processing details, behavior analysis, and routing decision.
        """
        # Step 2: Prefilter
        prefilter_result = is_suspicious(user_input)
        
        if not prefilter_result["suspicious"]:
            return {
                "input": user_input,
                "stage": "prefilter",
                "route": "production",
                "prefilter": prefilter_result,
                "sandbox": None,
                "behavior": None
            }
            
        # Step 3: Sandbox
        logger.info(f"Suspicious input detected (score={prefilter_result['score']:.2f}). Routing to Sandbox.")
        
        sandbox_result = self.sandbox.query(user_input)
        llm_output = sandbox_result.get("output", "")
        
        if "error" in sandbox_result:
            logger.error(f"Sandbox error: {sandbox_result['error']}")
            # Fail-safe: if sandbox errors, likely block or escalate (here we route to reject for safety)
            return {
                "input": user_input,
                "stage": "sandbox_error",
                "route": "reject",
                "prefilter": prefilter_result,
                "sandbox": sandbox_result,
                "behavior": None
            }
            
        # Behavior Analysis
        behavior_result = self.analyzer.analyze(user_input, llm_output)
        
        if behavior_result["compromised"]:
            route = "repair"  # Route to Step 4
        else:
            route = "production" # False positive from prefilter
            
        return {
            "input": user_input,
            "stage": "behavior_analysis",
            "route": route,
            "prefilter": prefilter_result,
            "sandbox": sandbox_result,
            "behavior": behavior_result
        }
