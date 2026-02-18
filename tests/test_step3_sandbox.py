# ============================================================================
# tests/test_step3_sandbox.py  (placeholder — populated when Step 3 is built)
# ============================================================================
"""
Unit tests for Step 3 — Sandbox & Behavior Analysis.

Will test:
    - SandboxLLM.query() response format
    - InstructionFollowingDetector.detect()
    - RoleSwitchDetector.detect()
    - DataExfiltrationDetector.detect()
    - BehaviorAnalyzer.analyze() confidence / severity levels
    - Step3Pipeline.process() routing decisions

Usage:
    python -m pytest tests/test_step3_sandbox.py -v
    python scripts/run_tests.py --step 3
"""
import unittest


class TestStep3Placeholder(unittest.TestCase):
    """Placeholder — will be replaced when Step 3 is implemented."""

    def test_placeholder(self):
        """Remove this test once Step 3 is implemented."""
        self.skipTest("Step 3 not yet implemented.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
