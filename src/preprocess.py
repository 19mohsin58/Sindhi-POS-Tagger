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
    # TODO: implement
    pass


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and expand the raw corpus into a flat supervised dataset.

    Steps:
      1. Keep only 'word' and 'tags' columns.
      2. Drop rows with missing / dash-placeholder values.
      3. Expand multi-tag rows (e.g. 'noun,adv') into separate rows.
      4. Add an 'is_ambiguous' flag for originally multi-tagged words.
      5. Filter to VALID_TAGS only.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_corpus().

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with columns: ['word', 'label', 'is_ambiguous']
    """
    # TODO: implement
    pass
