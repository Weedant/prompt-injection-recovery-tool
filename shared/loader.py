# ============================================================================
# shared/loader.py
# ============================================================================
"""
Generic dataset loader shared across pipeline steps.

Usage:
    from shared.loader import load_data
    train_df, test_df = load_data()
"""
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT      = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "filtered_data.csv"


def load_data(path: Path = DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load filtered_data.csv and split into train/test (80/20 stratified).

    Args:
        path: Path to the processed CSV file.

    Returns:
        Tuple of (train_df, test_df), both reset-indexed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if not {"id", "text", "label"}.issubset(df.columns):
        raise ValueError("Missing required columns: id, text, label")

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
