# ============================================================================
# scripts/load_renellm_json.py
# ============================================================================
"""
Parser for the ReNeLLM-Jailbreak JSON dataset.
Converts it into a structured format (CSV/DataFrame) for analysis.
"""
import sys
import json
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger("recovery.dataset")

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/renellm/renellm_jailbreak.json"

def load_renellm(path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load ReNeLLM dataset from JSON to DataFrame.
    """
    if not path.exists():
        logger.error(f"ReNeLLM dataset not found at: {path}")
        return pd.DataFrame()
        
    try:
        # The file might be large, read line by line or json load depending on format
        # Standard JSON array vs. JSON Lines
        # Prompt says: "Format: JSON array"
        data = json.loads(path.read_text(encoding="utf-8"))
        
        # Flatten structure
        records = []
        for entry in data:
            records.append({
                "prompt": entry.get("nested_prompt", ""),
                "response": entry.get("claude2_output", ""),
                "original_harm": entry.get("original_harm_behavior", ""),
                "label": int(entry.get("model_label", "1")) # 1 = successful jailbreak
            })
            
        return pd.DataFrame(records)
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = load_renellm()
    print(f"Loaded {len(df)} records.")
    print(df.head())
