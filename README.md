# 🔤 Sindhi POS Tagger

<p align="center">
  <img src="https://img.shields.io/badge/Language-Sindhi%20(Perso--Arabic)-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-POS%20Tagging-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Models-3%20Classifiers-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge" />
</p>

A supervised **Part-of-Speech (POS) Tagger** for the morphologically rich, low-resource **Sindhi language**, built on the AMBILE WordNet Corpus (~163K words in Perso-Arabic script).

This project conducts a **systematic comparison of 3 ML classifiers** — Logistic Regression, LinearSVC, and Random Forest — trained on character-level TF-IDF n-gram features to identify the best-performing model for Sindhi POS prediction.

> **Course:** Data Mining and Machine Learning — M.Sc. Artificial Intelligence and Data Engineering, University of Pisa  
> **Authors:** Mohsin Ali (721983) · Zeynaddin Papakhov (721981)

---

## 🌐 Domain & Core Idea

This project is in the domain of **Natural Language Processing (NLP)** and **Computational Linguistics**, focusing on lexical morphological classification for the low-resource Sindhi language.

The core idea is to transform the **AMBILE Sindhi WordNet** — a lexical database of 163,337 annotated words in Perso-Arabic script — into a supervised classification dataset and systematically compare three machine learning algorithms to identify which best predicts the POS category of a Sindhi word from its character-level structure alone.

> [!IMPORTANT]
> **Lexical Morphological Classification:**  
> Since models operate on single isolated words without sentence context, this is a **lexical morphological classifier** rather than a full contextual POS tagger. The inability to disambiguate by sentence context is acknowledged as a core limitation of the lexicon-level approach.

---

## 💡 Motivation & Prior Work

Sindhi remains digitally under-resourced despite having over 30 million speakers worldwide. To date, no classifier has been trained on the **AMBILE WordNet** — the largest open Sindhi lexical resource.

Prior work has several gaps:
- **Ali et al. (2021)** and **Mahar & Memon (2010)** used different corpora, relied on a single classifier, and evaluated their models using accuracy only.
- In morphologically rich languages with severe class imbalances, accuracy is an unreliable metric (e.g., a naive classifier predicting `noun` achieves ~66% accuracy but is practically useless).

We address these gaps by:
1. Comparing three distinct machine learning classifiers on identical feature sets.
2. Using **Stratified 5-Fold Cross-Validation** to ensure stable estimation.
3. Adopting **Macro-F1** as the primary evaluation metric.
4. Measuring **unseen-word generalisation** via per-class F1 on a held-out test set.

---

## 🎯 Objective

Given a single Sindhi word in isolation, predict its grammatical category from **8 POS classes**:

| Label | Category | Example / Notes |
|-------|----------|-----------------|
| `noun` | Noun | — |
| `verb` | Verb | — |
| `adj` | Adjective | — |
| `adv` | Adverb | — |
| `pro` | Pronoun | — |
| `pp` | Postposition | — |
| `int` | Interjection | — |
| `con` | Conjunction | — |

- **Primary metric:** Macro-F1 (preferred over accuracy due to severe class imbalance)  
- **Target performance:** 80–90% Macro-F1

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | AMBILE Sindhi WordNet Tagging Corpus |
| **Size** | 163,337 unique Sindhi words |
| **Script** | Perso-Arabic (right-to-left) |
| **Format** | CSV — columns: `word_id, word, category, gender, invariants, tags, tenses, hyp, antonyms, synonyms` |
| **Target column** | `tags` — comma-separated POS labels |
| **Multi-Label Resolution** | Multi-label words are resolved by **row expansion** (one row per label). |
| **Ambiguity Handling** | The `is_ambiguous` flag is retained for dataset description purposes only. |
| **Class imbalance** | noun = 65.9% vs. con = 0.07% (1,000:1 ratio) |
| **Source** | [Kaggle](https://kaggle.com/datasets/ambile/sindhi-wordnet-tagged-corpus) · [IEEE DataPort](https://ieee-dataport.org) (DOI: 10.21227/fy2b-6211) |

---

## ⚠️ Core Limitations

- **Context-Free Disambiguation:** Models predict grammatical categories based entirely on isolated words. Disambiguating homographs (e.g., words that are nouns in one context and verbs in another) is impossible without syntactic sentence context.
- **Lexicon-Level Approach:** The system operates as a morphological analyzer on lexical databases, not on running text.

---

## 🧪 Methods

### Feature Engineering
- **Character n-grams:** Bigrams, trigrams, and 4-grams (n=2–4) are extracted.
- **Morphological Encoding:** Character n-grams capture Sindhi morphological prefixes and suffixes, which encode patterns that distinguish POS classes and generalise to unseen words.
- **TF-IDF Weighting:** Applied with sublinear term frequency scaling (`sublinear_tf=True`) to diminish the dominance of high-frequency character patterns.
- **Max vocabulary:** 50,000 features.

### Models Compared
| Model | Key Setting | Rationale |
|-------|------------|-----------|
| **Logistic Regression** | `class_weight='balanced'` | Interpretable linear baseline; coefficients reveal the importance of specific morphological character-level patterns per class. |
| **LinearSVC (SVM)** | `class_weight='balanced'` | Extends the SVM approach of Mahar & Memon (2010). Strong performance on high-dimensional sparse TF-IDF features. |
| **Random Forest** | 200 trees, `class_weight='balanced'` | Ensemble baseline; tests whether non-linear decision boundaries yield performance gains over the linear SVM. |

> [!NOTE]
> All models use `class_weight='balanced'` to offset the severe class imbalance (noun count is 1,000× larger than conjunction count).

### Evaluation Protocol
- **Data Splitting:** Stratified 80/20 train/test split.
- **Cross-Validation:** Stratified 5-Fold Cross-Validation on the training set.
- **Unseen-Word Generalisation:** Evaluated using per-class F1 on the held-out test set, which naturally contains words absent from training.
- **Metrics Reported:** Macro-F1 (primary), Accuracy, Per-class F1, and Training Time.

---

## 📁 Repository Structure

```
sindhi-pos-tagger/
│
├── 📄 [README.md](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/README.md)                          ← Project overview & documentation
├── 📄 [requirements.txt](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/requirements.txt)                    ← Python dependencies
├── 📄 [Sindhi_POS_Proposal_Revised.docx](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/Sindhi_POS_Proposal_Revised.docx)    ← Revised Project Proposal
│
├── 📂 [data](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/data)/
│   └── 📄 [Wordnet-Corpus 10-30-25.csv](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/data/Wordnet-Corpus%2010-30-25.csv)       ← Raw AMBILE WordNet dataset
│
├── 📂 [notebooks](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/notebooks)/
│   └── 📓 [sindhi_pos_tagger.ipynb](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/notebooks/sindhi_pos_tagger.ipynb)           ← Main Jupyter/Colab notebook (all experiments)
│
├── 📂 [src](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/src)/
│   ├── 📄 [__init__.py](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/src/__init__.py)
│   ├── 📄 [preprocess.py](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/src/preprocess.py)                     ← Preprocessing pipeline & utilities
│   └── 📄 [features.py](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/src/features.py)                         ← TF-IDF feature extraction
│
└── 📂 [results](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/results)/
    ├── 📄 [metrics.csv](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/results/metrics.csv)                        ← Model comparison table (auto-generated)
    └── 📂 plots/                                  ← Visualization artifacts (EDA & metrics)
```

---

## 🚀 How to Run

### Option A — Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Click **File → Open Notebook → GitHub**.
3. Paste this repository URL and open [sindhi_pos_tagger.ipynb](file:///d:/MS%20AIDE%20Unipi/Data%20Mining%20and%20Machine%20Learning/Project/notebooks/sindhi_pos_tagger.ipynb).
4. Run cells sequentially (`Shift+Enter`).

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

> ⏳ Results will be updated after running the full suite of experiments.

| Model | CV Macro-F1 | Test Macro-F1 | Test Accuracy | Train Time |
|-------|-------------|--------------|--------------|------------|
| **LinearSVC** | — | — | — | — |
| **Logistic Regression** | — | — | — | — |
| **Random Forest** | — | — | — | — |

---

## 📚 References

1. **Ali et al. (2021)** — CRF-based Sindhi POS tagger; 90.34% accuracy on sentence-level news text.
2. **Mahar & Memon (2010)** — SVM-based Sindhi POS tagger; 97% on a 5-class WordNet subset.
3. **AMBILE Sindhi WordNet Corpus** — DOI: [10.21227/fy2b-6211](https://doi.org/10.21227/fy2b-6211)

---

## 📝 License

This project is submitted as academic coursework for the M.Sc. AIDE programme at the University of Pisa.
