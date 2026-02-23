
# =============================================================================
# scripts/prepare_all_datasets.py
# =============================================================================
"""
Master dataset preparation script.

Reads all 6 DATASETS/ subfolders, identifies each dataset, normalises it
into the canonical schema and writes merged outputs to data/raw/:

  data/raw/prefilter_merged.csv        → Step 2 prefilter training
  data/raw/harmful_intent_merged.csv   → Harmful-intent classifier
  data/raw/sandbox_behavior.csv        → Step 3 sandbox behavior classifier

Dataset mapping
───────────────
DS1  DATASETS/1/  (parquet)  question + label(1=harmful)
     → harmful_intent  (question=text, label as-is)
     → sandbox (has completion column too — behaviour labels)

DS2  DATASETS/2/  (parquet)  text + label(0=benign,1=injection)
     → prefilter  (direct use)

DS3  DATASETS/3/  (JSON)     prompt + is_jailbreak (bool)
     → prefilter  (is_jailbreak → 1/0)
     → harmful_intent  (jailbreak prompts = harmful)

DS4  DATASETS/4/  (CSV)      prompt + jailbreak(bool)  [verazuo/jailbreak_llms]
     → prefilter  (jailbreak=True → 1, regular CSV → 0)

DS5  DATASETS/5/  (TSV 530MB) vanilla + adversarial + completion + data_type
     → sandbox behavior  (adversarial_harmful=1, others=0)
     → prefilter          (adversarial prompts = injection, vanilla_benign = benign)

DS6  DATASETS/6/  (CSV downloaded mini + full)  jailbreak_query + redteam_query + policy
     → prefilter  (jailbreak_query = 1, redteam_query = 1)
     → harmful_intent (all jailbreak queries)

Usage:
    python scripts/prepare_all_datasets.py
    python scripts/prepare_all_datasets.py --datasets 1 2 3   # specific datasets only
    python scripts/prepare_all_datasets.py --no-sandbox       # skip heavy DS5
"""
import argparse
import json
import sys
import os

# Force UTF-8 output on Windows (avoids cp1252 errors with box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS_ROOT = ROOT / "DATASETS"
RAW_OUT = ROOT / "data" / "raw"
RAW_OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace("nan", pd.NA).dropna()


def make_prefilter_row(text: str, label: int, source: str) -> dict:
    return {"text": str(text).strip(), "label": int(label), "source": source}


def make_harmful_row(text: str, label: int, source: str, category: str = "") -> dict:
    return {"text": str(text).strip(), "label": int(label), "source": source, "category": category}


def make_sandbox_row(prompt: str, response: str, label: int, source: str, data_type: str = "") -> dict:
    return {
        "prompt": str(prompt).strip(),
        "response": str(response).strip(),
        "label": int(label),          # 1 = sandbox compromised/complied, 0 = refused/safe
        "source": source,
        "data_type": data_type,
    }


def save_csv(rows: list[dict], path: Path, name: str) -> None:
    if not rows:
        log(f"  ⚠  No rows for {name}, skipping.")
        return
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["text"] if "text" in df.columns else ["prompt"])
    df = df.drop_duplicates(
        subset=["text"] if "text" in df.columns else ["prompt"]
    ).reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log(f"  ✓ Saved {len(df):,} rows → {path.relative_to(ROOT)}")
    if "label" in df.columns:
        log(f"    Label dist: {df['label'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_ds1(prefilter: list, harmful: list, sandbox: list) -> None:
    """
    DS1: parquet with columns [question, text(=LLM response), label, q_plus_a]
    label=1 → harmful question, label=0 → benign question
    """
    log("── Dataset 1 (parquet, harmful intent + sandbox) ──")
    base = DS_ROOT / "1"
    for split in ["train-00000-of-00001.parquet", "test-00000-of-00001 (1).parquet"]:
        p = base / split
        if not p.exists():
            log(f"  Missing: {p.name}")
            continue
        df = pd.read_parquet(p)
        log(f"  {split}: {len(df):,} rows, cols={list(df.columns)}")

        for _, row in df.iterrows():
            q = str(row.get("question", "")).strip()
            resp = str(row.get("text", "")).strip()
            lbl = int(row.get("label", 0))

            if not q:
                continue

            # Harmful intent — use the question
            harmful.append(make_harmful_row(q, lbl, "ds1_harmbench", "harmbench_behavior"))

            # Prefilter — question as prompt
            prefilter.append(make_prefilter_row(q, lbl, "ds1_harmbench"))

            # Sandbox — we have question + LLM response
            # label=1 means harmful request; response is the LLM completion.
            # We treat: if label=1 AND response doesn't contain a refusal → compromised (label=1)
            # Simple heuristic: response starts with "I cannot" → refused (label=0)
            if resp and lbl == 1:
                refused_keywords = ["i cannot", "i can't", "i am unable", "i'm sorry", "i apologize"]
                is_refusal = any(kw in resp.lower()[:120] for kw in refused_keywords)
                sandbox_label = 0 if is_refusal else 1
                sandbox.append(make_sandbox_row(q, resp, sandbox_label, "ds1_harmbench", "harmful_behavior_response"))


def load_ds2(prefilter: list) -> None:
    """
    DS2: parquet with [text, label]. label=0=benign, label=1=injection.
    Direct use for prefilter.
    """
    log("── Dataset 2 (parquet, prompt injection) ──")
    base = DS_ROOT / "2"
    for split in ["train-00000-of-00001.parquet", "test-00000-of-00001 (1).parquet"]:
        p = base / split
        if not p.exists():
            log(f"  Missing: {p.name}")
            continue
        df = pd.read_parquet(p)
        log(f"  {split}: {len(df):,} rows")
        for _, row in df.iterrows():
            t = str(row.get("text", "")).strip()
            lbl = int(row.get("label", 0))
            if t:
                prefilter.append(make_prefilter_row(t, lbl, "ds2_injection"))


def load_ds3(prefilter: list, harmful: list) -> None:
    """
    DS3: JSON with [prompt, is_jailbreak(bool), source, metadata]
    is_jailbreak=True → label=1
    Sources: verazuo/jailbreak_llms + cais/wmdp
    """
    log("── Dataset 3 (JSON, jailbreak prompts) ──")
    p = DS_ROOT / "3" / "dataset.json"
    if not p.exists():
        log(f"  Missing: {p}")
        return
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    log(f"  Loaded {len(data):,} records")

    for item in data:
        prompt = str(item.get("prompt", "")).strip()
        is_jb = bool(item.get("is_jailbreak", False))
        lbl = 1 if is_jb else 0
        src_raw = str(item.get("source", "ds3"))
        # Simplify source name
        if "verazuo" in src_raw or "jailbreak_llms" in src_raw:
            src = "ds3_verazuo"
        elif "wmdp" in src_raw:
            src = "ds3_wmdp"
        else:
            src = "ds3_guardrails"

        if prompt:
            prefilter.append(make_prefilter_row(prompt, lbl, src))
            if is_jb:
                harmful.append(make_harmful_row(prompt, 1, src, "jailbreak"))


def load_ds4(prefilter: list, harmful: list) -> None:
    """
    DS4: verazuo/jailbreak_llms CSVs (downloaded)
    jailbreak_prompts CSV: prompt + jailbreak(bool)
    regular_prompts CSV: prompt  (all benign)
    """
    log("── Dataset 4 (verazuo/jailbreak_llms CSVs) ──")
    base = DS_ROOT / "4"

    # Jailbreak prompts
    jb_path = base / "jailbreak_prompts_2023_12_25.csv"
    if jb_path.exists():
        try:
            df = pd.read_csv(jb_path, encoding="utf-8", on_bad_lines="skip")
            log(f"  jailbreak CSV: {len(df):,} rows, cols={list(df.columns)}")
            for _, row in df.iterrows():
                prompt = str(row.get("prompt", "")).strip()
                is_jb = str(row.get("jailbreak", "True")).lower() in ("true", "1", "yes")
                lbl = 1 if is_jb else 0
                if prompt:
                    prefilter.append(make_prefilter_row(prompt, lbl, "ds4_verazuo"))
                    if is_jb:
                        harmful.append(make_harmful_row(prompt, 1, "ds4_verazuo", "jailbreak"))
        except Exception as e:
            log(f"  ⚠ Error reading jailbreak CSV: {e}")
    else:
        log("  jailbreak CSV not downloaded yet — run again after download")

    # Regular prompts → benign
    reg_path = base / "regular_prompts_2023_12_25.csv"
    if reg_path.exists():
        try:
            df = pd.read_csv(reg_path, encoding="utf-8", on_bad_lines="skip", usecols=["prompt"])
            log(f"  regular CSV: {len(df):,} rows (benign)")
            for prompt in df["prompt"].dropna():
                prompt = str(prompt).strip()
                if prompt:
                    prefilter.append(make_prefilter_row(prompt, 0, "ds4_verazuo_regular"))
        except Exception as e:
            log(f"  ⚠ Error reading regular CSV: {e}")
    else:
        log("  regular CSV not downloaded yet")


def load_ds5(prefilter: list, sandbox: list, no_sandbox: bool = False) -> None:
    """
    DS5: TSV with [vanilla, adversarial, completion, data_type]
    data_type: adversarial_harmful | adversarial_benign | vanilla_harmful | vanilla_benign

    → Prefilter: adversarial/vanilla prompt with label from data_type
    → Sandbox: (prompt, completion) pairs — all rows, no cap

    NOTE: train.tsv is 530MB — read in chunks to avoid RAM issues.
    """
    log("── Dataset 5 (TSV, adversarial + sandbox behavior) ──")
    base = DS_ROOT / "5"

    for fname in ["train.tsv", "eval.tsv"]:
        p = base / fname
        if not p.exists():
            log(f"  Missing: {p.name}")
            continue

        log(f"  Reading {fname} (chunked) ...")
        chunk_prefilter: list[dict] = []
        chunk_sandbox:   list[dict] = []

        try:
            for chunk in pd.read_csv(p, sep="\t", chunksize=10_000, on_bad_lines="skip"):
                for _, row in chunk.iterrows():
                    data_type = str(row.get("data_type", "")).strip()
                    completion = str(row.get("completion", "")).strip()
                    adv = str(row.get("adversarial", "")).strip()
                    van = str(row.get("vanilla", "")).strip()

                    # ── Prefilter rows ────────────────────────────────────────
                    if adv and adv != "nan":
                        lbl = 1 if "harmful" in data_type else 0
                        chunk_prefilter.append(make_prefilter_row(adv, lbl, f"ds5_{fname}"))
                    elif van and van != "nan":
                        lbl = 1 if "harmful" in data_type else 0
                        chunk_prefilter.append(make_prefilter_row(van, lbl, f"ds5_{fname}"))

                    # ── Sandbox rows ──────────────────────────────────────────
                    if not no_sandbox and completion and completion != "nan":
                        prompt_for_sandbox = adv if (adv and adv != "nan") else van
                        if prompt_for_sandbox:
                            sandbox_lbl = 0
                            if data_type == "adversarial_harmful":
                                refused_kws = ["i cannot", "i can't", "i am unable", "i'm sorry", "i apologize", "i'm unable"]
                                is_refusal = any(kw in completion.lower()[:150] for kw in refused_kws)
                                sandbox_lbl = 0 if is_refusal else 1
                            chunk_sandbox.append(make_sandbox_row(
                                prompt_for_sandbox, completion, sandbox_lbl,
                                f"ds5_{fname}", data_type
                            ))

        except Exception as e:
            log(f"  ⚠ Error reading {fname}: {e}")
            continue

        prefilter.extend(chunk_prefilter)
        sandbox.extend(chunk_sandbox)
        log(f"  {fname}: {len(chunk_prefilter):,} prefilter rows, {len(chunk_sandbox):,} sandbox rows")


def load_ds6(prefilter: list, harmful: list) -> None:
    """
    DS6: JailBreakV-28K
    Files: mini_JailBreakV_28K.csv (280 rows, downloaded)
           JailBreakV_28K.csv (28K, download if missing)
    Cols: jailbreak_query, redteam_query, policy, format, ...
    → prefilter: jailbreak_query=1, redteam_query (original query) used as the text
    → harmful: all queries with policy category
    """
    log("── Dataset 6 (JailBreakV-28K) ──")
    base = DS_ROOT / "6"

    def _policy_to_category(policy: str) -> str:
        mapping = {
            "Malware": "malware", "Illegal Activity": "hacking",
            "Fraud": "phishing", "Economic Harm": "pii_exfiltration",
            "Hate Speech": "hate_speech", "Physical Harm": "weapons",
            "Violence": "weapons", "Child Abuse Content": "csam",
            "Privacy Violation": "pii_exfiltration",
        }
        return mapping.get(str(policy), "jailbreak")

    # Try mini first, then full
    candidates = ["JailBreakV_28K.csv", "mini_JailBreakV_28K.csv"]
    loaded = False
    for fname in candidates:
        p = base / fname
        if not p.exists():
            # Try to auto-download mini if nothing is there
            if fname == "mini_JailBreakV_28K.csv":
                try:
                    import urllib.request
                    url = "https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/resolve/main/JailBreakV_28K/mini_JailBreakV_28K.csv"
                    log(f"  Downloading {fname} ...")
                    urllib.request.urlretrieve(url, p)
                    log(f"  Downloaded {fname}")
                except Exception as e:
                    log(f"  ⚠ Download failed: {e}")
                    continue
            else:
                continue

        try:
            df = pd.read_csv(p, encoding="utf-8")
            log(f"  {fname}: {len(df):,} rows")
            for _, row in df.iterrows():
                jb_q = str(row.get("jailbreak_query", "")).strip()
                rt_q = str(row.get("redteam_query", "")).strip()
                policy = str(row.get("policy", "jailbreak"))
                cat = _policy_to_category(policy)

                if jb_q and jb_q != "nan":
                    prefilter.append(make_prefilter_row(jb_q, 1, "ds6_jailbreakv"))
                    harmful.append(make_harmful_row(jb_q, 1, "ds6_jailbreakv", cat))
                if rt_q and rt_q != "nan" and rt_q != jb_q:
                    prefilter.append(make_prefilter_row(rt_q, 1, "ds6_jailbreakv_redteam"))
                    harmful.append(make_harmful_row(rt_q, 1, "ds6_jailbreakv_redteam", cat))
            loaded = True
            break
        except Exception as e:
            log(f"  ⚠ Error: {e}")

    if not loaded:
        log("  ⚠ No DS6 file available. Skipping.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all datasets for training.")
    parser.add_argument("--datasets", nargs="*", type=int,
                        help="Which dataset numbers to process (1-6). Default: all.")
    parser.add_argument("--no-sandbox", action="store_true",
                        help="Skip sandbox behavior extraction (faster, skips DS5 sandbox).")
    args = parser.parse_args()

    enabled = set(args.datasets) if args.datasets else {1, 2, 3, 4, 5, 6}
    log(f"Processing datasets: {sorted(enabled)}")

    prefilter_rows: list[dict] = []
    harmful_rows:   list[dict] = []
    sandbox_rows:   list[dict] = []

    if 1 in enabled:
        load_ds1(prefilter_rows, harmful_rows, sandbox_rows)
    if 2 in enabled:
        load_ds2(prefilter_rows)
    if 3 in enabled:
        load_ds3(prefilter_rows, harmful_rows)
    if 4 in enabled:
        load_ds4(prefilter_rows, harmful_rows)
    if 5 in enabled:
        load_ds5(prefilter_rows, sandbox_rows, no_sandbox=args.no_sandbox)
    if 6 in enabled:
        load_ds6(prefilter_rows, harmful_rows)

    log("")
    log("── Saving outputs ──")

    # --- Prefilter ---
    save_csv(prefilter_rows, RAW_OUT / "prefilter_merged.csv", "prefilter")

    # --- Harmful intent ---
    save_csv(harmful_rows, RAW_OUT / "harmful_intent_merged.csv", "harmful_intent")

    # --- Sandbox ---
    if not args.no_sandbox:
        save_csv(sandbox_rows, RAW_OUT / "sandbox_behavior.csv", "sandbox")

    log("")
    log("All done. Next steps:")
    log("  python scripts/train_prefilter.py")
    log("  python scripts/train_harmful_intent.py")
    if not args.no_sandbox:
        log("  python scripts/train_sandbox_classifier.py")


if __name__ == "__main__":
    main()
