# 🔤 Sindhi POS Tagger

<p align="center">
  <img src="https://img.shields.io/badge/Language-Sindhi%20(Perso--Arabic)-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-POS%20Tagging-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Models-3%20Classifiers-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge" />
</p>

A supervised **Part-of-Speech (POS) Tagger** for the morphologically rich, low-resource **Sindhi language**, built on the AMBILE WordNet Corpus (163,337 words in Perso-Arabic script).

This project conducts a **systematic comparison of 3 ML classifiers** — Logistic Regression, LinearSVC, and Random Forest — trained on character-level TF-IDF n-gram features to identify the best-performing model for Sindhi POS prediction.

> **Course:** Data Mining and Machine Learning — M.Sc. Artificial Intelligence and Data Engineering, University of Pisa  
> **Authors:** Mohsin Ali (721983) · Zeynaddin Papakhov (721981)

---

## 🎯 Objective

Given a single Sindhi word as input, predict its grammatical category from **8 POS classes**:

| Label | Category | Example |
|-------|----------|---------|
| `noun` | Noun | — |
| `verb` | Verb | — |
| `adj` | Adjective | — |
| `adv` | Adverb | — |
| `pro` | Pronoun | — |
| `pp` | Postposition | — |
| `int` | Interjection | — |
| `con` | Conjunction | — |

**Primary metric:** Macro-F1 (preferred over accuracy due to severe class imbalance)  
**Target performance:** 80–90% Macro-F1

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | AMBILE Sindhi WordNet Tagging Corpus |
| **Size** | 163,337 unique Sindhi words |
| **Script** | Perso-Arabic (right-to-left) |
| **Format** | CSV — columns: `word_id, word, category, gender, invariants, tags, tenses, hyp, antonyms, synonyms` |
| **Target column** | `tags` — comma-separated POS labels |
| **Training instances** | ~187,000 after multi-label expansion |
| **Class imbalance** | noun = 65.9% vs. con = 0.07% (1,000:1 ratio) |
| **Source** | [Kaggle](https://kaggle.com/datasets/ambile/sindhi-wordnet-tagged-corpus) · [IEEE DataPort](https://ieee-dataport.org) (DOI: 10.21227/fy2b-6211) |

---

## 🧪 Methods

### Feature Engineering
- **Character n-grams** (bigrams, trigrams, 4-grams) — capture Sindhi morphological prefixes/suffixes
- **TF-IDF weighting** with `sublinear_tf=True` (log scaling)
- **Max vocabulary:** 50,000 features

### Models Compared
| Model | Key Setting | Rationale |
|-------|------------|-----------|
| Logistic Regression | `class_weight='balanced'` | Interpretable linear baseline |
| LinearSVC (SVM) | `class_weight='balanced'` | Mirrors Mahar & Memon (2010), strong on sparse high-dim features |
| Random Forest | 200 trees, `class_weight='balanced'` | Ensemble baseline |

### Evaluation Protocol
- **Stratified 80/20 train/test split**
- **Stratified 5-Fold Cross-Validation** on training set
- **Metrics:** Macro-F1 (primary), Accuracy, Per-class F1, Training Time

---

## 📁 Repository Structure

```
sindhi-pos-tagger/
│
├── 📄 README.md                          ← You are here
├── 📄 .gitignore
├── 📄 requirements.txt
│
├── 📂 data/
│   └── Wordnet-Corpus 10-30-25.csv       ← AMBILE WordNet dataset
│
├── 📂 notebooks/
│   └── sindhi_pos_tagger.ipynb           ← Main Colab notebook (all experiments)
│
├── 📂 src/
│   ├── __init__.py
│   ├── preprocess.py                     ← Data loading & cleaning utilities
│   └── features.py                       ← TF-IDF feature extraction
│
└── 📂 results/
    ├── metrics.csv                        ← Model comparison table (auto-generated)
    └── plots/
        ├── class_distribution.png
        ├── word_length.png
        ├── confusion_matrices.png
        └── per_class_f1.png
```

---

## 🚀 How to Run

### Option A — Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com)
2. Click **File → Open Notebook → GitHub**
3. Paste this repository URL and open `notebooks/sindhi_pos_tagger.ipynb`
4. Run cells from top to bottom (`Shift+Enter`)

### Option B — Local Setup
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/sindhi-pos-tagger.git
cd sindhi-pos-tagger

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook notebooks/sindhi_pos_tagger.ipynb
```

---

## 📈 Results

> ⏳ Results will be updated after experiments are complete.

| Model | CV Macro-F1 | Test Macro-F1 | Test Accuracy | Train Time |
|-------|-------------|--------------|--------------|------------|
| LinearSVC | — | — | — | — |
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |

---

## 📚 References

1. **Ali et al. (2021)** — CRF-based Sindhi POS tagger; 90.34% accuracy on sentence-level news text.
2. **Mahar & Memon (2010)** — SVM-based Sindhi POS tagger; 97% on a 5-class WordNet subset.
3. **AMBILE Sindhi WordNet Corpus** — DOI: [10.21227/fy2b-6211](https://doi.org/10.21227/fy2b-6211)

---

## 📝 License

This project is submitted as academic coursework for the M.Sc. AIDE programme at the University of Pisa.
