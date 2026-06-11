"""
features.py
─────────────────────────────────────────────────────────────
Feature engineering for the Sindhi POS Tagger project.
Builds TF-IDF character n-gram feature matrices.

Authors : Mohsin Ali (721983), Zeynaddin Papakhov (721981)
Course  : Data Mining and Machine Learning — M.Sc. AIDE, UniPi
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    ngram_range: tuple = (2, 4),
    max_features: int = 50_000,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer configured for character-level n-grams.

    Why character n-grams?
        Sindhi is morphologically rich — prefixes and suffixes strongly
        signal the POS category. Character n-grams capture these patterns
        and generalise to unseen words at inference time.

    Parameters
    ----------
    ngram_range  : tuple
        Range of n-gram sizes. Default (2, 4) → bigrams through 4-grams.
    max_features : int
        Maximum vocabulary size. Default 50,000 as per proposal.
    sublinear_tf : bool
        Apply log(1+tf) scaling to reduce the dominance of high-frequency
        n-grams. Default True.

    Returns
    -------
    TfidfVectorizer
        Unfitted vectorizer ready to call .fit_transform() on training data.
    """
    return TfidfVectorizer(
        analyzer='char',
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf
    )

