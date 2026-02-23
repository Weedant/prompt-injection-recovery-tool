import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def main():
    root_dir = Path(r"c:\Users\VEDANT\Desktop\recovery_prefilter")
    test_dir = root_dir / "Test Data"
    out_dir = root_dir / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = test_dir.glob("*.parquet")
    
    dataframes = []
    
    for f in files:
        df = pd.read_parquet(f)
        file_name = f.name
        
        # Normalize columns based on origin
        if "geeky" in file_name:
            # Already has 'prompt' and 'label'
            df = df[['prompt', 'label']]
        elif "sinaw" in file_name:
            # Has 'text' and 'label'
            df = df.rename(columns={'text': 'prompt'})
            df = df[['prompt', 'label']]
        elif "mindguard" in file_name:
            # Has 'original_sample' and 'modified_sample'. Assuming these are attacks (label=1)
            df = df[['modified_sample']].rename(columns={'modified_sample': 'prompt'})
            df['label'] = 1
        else:
            print(f"Unknown format for file {file_name}")
            continue
            
        print(f"Loaded {file_name}: {df.shape[0]} rows")
        dataframes.append(df)
        
    # Combine everything
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Clean up empty strings and exact duplicates
    combined_df['prompt'] = combined_df['prompt'].astype(str).str.strip()
    combined_df = combined_df[combined_df['prompt'] != ""]
    combined_df = combined_df.drop_duplicates(subset=['prompt']).reset_index(drop=True)
    
    print(f"\nTotal Combined Test Size: {combined_df.shape[0]:,} rows")
    print(f"Label Distribution:\n{combined_df['label'].value_counts()}")
    
    # Save the full combined dataset
    full_out_path = test_dir / "full_combined_test.csv"
    combined_df.to_csv(full_out_path, index=False)
    print(f"Saved full test dataset to: {full_out_path}")
    
    # --- SOLUTION TO API RATE LIMITS ---
    # Create an extremely small stratified subsample for Groq Sandboxing! 
    # (50 benign, 50 malicious = 100 total) to protect the FREE TIER for tomorrow's presentation.
    print("\nGenerating a presentation-safe 100 row stratified sample to protect Groq API free tier...")
    
    SAMPLE_SIZE_PER_CLASS = 50
    
    if len(combined_df[combined_df['label'] == 1]) >= SAMPLE_SIZE_PER_CLASS:
        malicious = combined_df[combined_df['label'] == 1].sample(SAMPLE_SIZE_PER_CLASS, random_state=42)
    else:
        malicious = combined_df[combined_df['label'] == 1]
        
    if len(combined_df[combined_df['label'] == 0]) >= SAMPLE_SIZE_PER_CLASS:
        benign = combined_df[combined_df['label'] == 0].sample(SAMPLE_SIZE_PER_CLASS, random_state=42)
    else:
        benign = combined_df[combined_df['label'] == 0]
        
    sampled_df = pd.concat([benign, malicious]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    sampled_out_path = test_dir / "sampled_test_for_api.csv"
    sampled_df.to_csv(sampled_out_path, index=False)
    print(f"Saved Groq-safe sample to: {sampled_out_path}")

if __name__ == "__main__":
    main()
