"""
preprocess.py
─────────────────────────────────────────────────────────────
Preprocessing utilities for the Sindhi POS Tagger project.
Handles loading, cleaning, and expanding the AMBILE WordNet corpus.

Authors : Mohsin Ali (721983), Zeynaddin Papakhov (721981)
Course  : Data Mining and Machine Learning — M.Sc. AIDE, UniPi
"""

import pandas as pd
import numpy as np

# The 8 valid POS tags defined in the project proposal
VALID_TAGS = {'noun', 'verb', 'adj', 'adv', 'pro', 'pp', 'int', 'con'}


def load_corpus(filepath: str) -> pd.DataFrame:
    """
    Load the raw AMBILE Sindhi WordNet CSV.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns.
    """
    return pd.read_csv(filepath, encoding='utf-8')


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and expand the raw corpus into a flat supervised dataset.

    Steps:
      1. Fix tag inconsistency: replace isolated 'v' with 'verb' in the tags.
      2. Drop rows with no tag (missing, dash, or empty values).
      3. Drop rows with empty/whitespace-only words.
      4. Add 'is_ambiguous' flag for originally multi-tagged words (those containing commas).
      5. Explode multi-label rows (split comma-separated tags) into separate rows.
      6. Filter to VALID_TAGS only.
      7. Drop exact (word, label) duplicates to avoid data leakage and redundancy.
      8. Reset index and return cleaned columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_corpus().

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with columns: ['word', 'label', 'is_ambiguous', 'word_len']
    """
    df = df.copy()

    # Step 1: Fix 'v' abbreviation to 'verb'
    if 'tags' in df.columns:
        df['tags'] = df['tags'].astype(str).str.replace(r'\bv\b', 'verb', regex=True)

        # Step 2: Drop rows with missing tags
        df = df[~df['tags'].isin(['-', '', 'nan']) & df['tags'].notna()]

    # Step 3: Drop empty/missing words
    if 'word' in df.columns:
        df['word'] = df['word'].astype(str).str.strip()
        df = df[df['word'] != '']

    # Step 4: Mark ambiguous words
    df['is_ambiguous'] = df['tags'].str.contains(',', na=False)

    # Step 5: Explode multi-labels -> one row per label
    df['label'] = df['tags'].str.split(',')
    df = df.explode('label')
    df['label'] = df['label'].str.strip()

    # Step 6: Keep only valid tags
    df = df[df['label'].isin(VALID_TAGS)]

    # Step 7: Drop (word, label) duplicates
    df['word_len'] = df['word'].str.len()
    df = df.drop_duplicates(subset=['word', 'label'])

    # Keep only target columns and reset index
    return df[['word', 'label', 'is_ambiguous', 'word_len']].reset_index(drop=True)

