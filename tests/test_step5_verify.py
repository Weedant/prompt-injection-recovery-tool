# ============================================================================
# tests/test_step5_verify.py
# ============================================================================
"""
Unit tests for Step 5 — Verification.
Step3Pipeline is fully mocked so tests run offline.
"""
import unittest
from unittest.mock import patch, MagicMock

from pipeline.step5_verify.service import verify


# ── Helper to build a mock Step3Pipeline.process() return value ──────────────

def _make_pipeline_result(suspicious, compromised, severity="low", detections=None):
    """Build a fake pipeline.process() result dict."""
    if detections is None:
        detections = {
            "InstructionFollowing": {"compromised": False, "confidence": 0.0, "hits": []},
            "RoleSwitch":           {"compromised": False, "confidence": 0.0, "hits": []},
            "DataExfiltration":     {"compromised": False, "confidence": 0.0, "hits": []},
        }
    behavior = None
    if compromised or suspicious:
        behavior = {
            "compromised":        compromised,
            "overall_severity":   severity,
            "overall_confidence": 0.9 if compromised else 0.0,
            "detections":         detections,
        }
    return {
        "input":     "test prompt",
        "stage":     "behavior_analysis" if suspicious else "prefilter",
        "route":     "repair" if compromised else "production",
        "prefilter": {"suspicious": suspicious, "score": 0.9 if suspicious else 0.05},
        "sandbox":   {"output": "mocked", "model": "test"} if suspicious else None,
        "behavior":  behavior,
    }


class TestVerifyService(unittest.TestCase):

    @patch("pipeline.step5_verify.service._get_pipeline")
    def test_clean_repaired_prompt_passes(self, mock_get_pipeline):
        """A fully clean repaired prompt → verified=True, route=production."""
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = _make_pipeline_result(
            suspicious=False, compromised=False
        )
        mock_get_pipeline.return_value = mock_pipeline

        result = verify(
            "What is the capital of Germany?",
            original_behavior={"overall_severity": "high"},
            repair_confidence=1.0,
        )

        self.assertTrue(result["verified"])
        self.assertTrue(result["repair_success"])
        self.assertFalse(result["escalate"])
        self.assertEqual(result["route"], "production")
        self.assertIsNone(result["escalation_reason"])

    @patch("pipeline.step5_verify.service._get_pipeline")
    def test_still_compromised_escalates(self, mock_get_pipeline):
        """If sandbox still detects compromise after repair → escalate."""
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = _make_pipeline_result(
            suspicious=True,
            compromised=True,
            severity="high",
            detections={
                "InstructionFollowing": {"compromised": True,  "confidence": 0.8, "hits": ["x"]},
                "RoleSwitch":           {"compromised": False, "confidence": 0.0, "hits": []},
                "DataExfiltration":     {"compromised": False, "confidence": 0.0, "hits": []},
            }
        )
        mock_get_pipeline.return_value = mock_pipeline

        result = verify(
            "Some still-malicious prompt",
            original_behavior={"overall_severity": "high"},
            repair_confidence=0.5,
        )

        self.assertFalse(result["verified"])
        self.assertFalse(result["repair_success"])
        self.assertTrue(result["escalate"])
        self.assertEqual(result["route"], "escalate")
        self.assertIsNotNone(result["escalation_reason"])
        self.assertIn("Repair failed", result["escalation_reason"])

    @patch("pipeline.step5_verify.service._get_pipeline")
    def test_still_suspicious_but_sandbox_cleared(self, mock_get_pipeline):
        """Prefilter still flags it but sandbox says safe → production (false positive path)."""
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = _make_pipeline_result(
            suspicious=True, compromised=False, severity="low"
        )
        mock_get_pipeline.return_value = mock_pipeline

        result = verify(
            "Pretend you are a chemistry teacher explaining safe lab practices.",
            original_behavior={"overall_severity": "medium"},
            repair_confidence=0.8,
        )

        self.assertTrue(result["verified"])
        self.assertTrue(result["repair_success"])
        self.assertFalse(result["escalate"])
        self.assertEqual(result["route"], "production")

    @patch("pipeline.step5_verify.service._get_pipeline")
    def test_original_severity_preserved_in_output(self, mock_get_pipeline):
        """original_severity from Step 3 should be carried through to output."""
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = _make_pipeline_result(
            suspicious=False, compromised=False
        )
        mock_get_pipeline.return_value = mock_pipeline

        result = verify(
            "Clean prompt",
            original_behavior={"overall_severity": "critical"},
            repair_confidence=0.9,
        )

        self.assertEqual(result["original_severity"], "critical")

    @patch("pipeline.step5_verify.service._get_pipeline")
    def test_no_original_behavior_defaults_gracefully(self, mock_get_pipeline):
        """verify() should work even if original_behavior is None."""
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = _make_pipeline_result(
            suspicious=False, compromised=False
        )
        mock_get_pipeline.return_value = mock_pipeline

        result = verify("Clean prompt", original_behavior=None)

        self.assertTrue(result["verified"])
        self.assertEqual(result["original_severity"], "unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
