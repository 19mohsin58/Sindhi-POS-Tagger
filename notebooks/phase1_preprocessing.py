# ================================================================
# SINDHI POS TAGGER — Phase 1: Data Preprocessing
# Authors: Mohsin Ali (721983), Zeynaddin Papakhov (721981)
# Course : Data Mining & Machine Learning — M.Sc. AIDE, UniPi
# ================================================================
# Paste each SECTION as a separate cell in Google Colab
# ================================================================

# ── CELL 1: Install & Mount ──────────────────────────────────────
"""
!pip install -q pandas numpy matplotlib seaborn scikit-learn scipy

from google.colab import drive
drive.mount('/content/drive')
"""

# ── CELL 2: Imports ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.dpi': 130, 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
PALETTE = ['#264653','#2a9d8f','#e9c46a','#f4a261',
           '#e76f51','#8ecae6','#219ebc','#023047']
VALID_TAGS = {'noun','verb','adj','adv','pro','pp','int','con'}
RANDOM_STATE = 42

# ── PATH: change to match your Drive folder ───────────────────────
DATASET_PATH = '/content/drive/MyDrive/sindhi-pos-tagger/data/Wordnet-Corpus 10-30-25.csv'
# Local path:
# DATASET_PATH = r'd:\MS AIDE Unipi\Data Mining and Machine Learning\Project\data\Wordnet-Corpus 10-30-25.csv'

print("✅ Imports ready.")

# ── CELL 3: Load Raw Data ────────────────────────────────────────
df_raw = pd.read_csv(DATASET_PATH, encoding='utf-8')

print("=" * 55)
print("STEP 1 — RAW DATASET OVERVIEW")
print("=" * 55)
print(f"Shape      : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"Memory     : {df_raw.memory_usage(deep=True).sum()/1e6:.1f} MB")
print(f"Columns    : {list(df_raw.columns)}")
print()
print("First 5 rows:")
display(df_raw.head(5))
print("\nLast 3 rows:")
display(df_raw.tail(3))

# ── CELL 4: Data Types ───────────────────────────────────────────
print("STEP 2 — DATA TYPES & NON-NULL COUNTS")
print(df_raw.dtypes.to_string())
print()
print(f"All columns are object (string) type — expected for a WordNet CSV.")
print(f"Note: '-' is used as a PLACEHOLDER for missing values (not NaN).")
print(f"      We must handle both NaN and '-' as missing.")

# ── CELL 5: Missing Value Analysis ───────────────────────────────
print("STEP 3 — MISSING VALUE ANALYSIS")
print("(Counting both NaN and '-' placeholder as missing)")

df_check = df_raw.replace('-', np.nan)
missing = pd.DataFrame({
    'Missing Count': df_check.isnull().sum(),
    'Missing %'    : (df_check.isnull().sum() / len(df_check) * 100).round(1),
    'Present Count': df_check.notnull().sum(),
}).sort_values('Missing %', ascending=False)
print(missing.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#e76f51' if p > 50 else '#e9c46a' if p > 10 else '#2a9d8f'
          for p in missing['Missing %']]
bars = ax.barh(missing.index, missing['Missing %'], color=colors, edgecolor='white')
for bar, pct in zip(bars, missing['Missing %']):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
            f'{pct:.1f}%', va='center', fontsize=9)
ax.axvline(90, color='red', ls='--', alpha=0.5, label='90% line')
ax.set_xlabel('Missing %')
ax.set_title('Missing Data per Column\nRed>50%  Orange>10%  Green<10%',
             fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('results/plots/01_missing_values.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
DECISION — COLUMNS TO DROP (insufficient data for modelling):
  hyp        : 100.0% missing  → DROP
  category   :  99.9% missing  → DROP
  tenses     :  96.5% missing  → DROP
  antonyms   :  97.9% missing  → DROP
  synonyms   :  91.0% missing  → DROP
  gender     :  90.7% missing  → DROP

COLUMNS TO KEEP:
  word       : our INPUT FEATURE (the Sindhi word string)
  tags       : our TARGET VARIABLE (POS label)
  invariants : 13% missing — kept for reference only
""")

# ── CELL 6: Tags Column Analysis ─────────────────────────────────
print("STEP 4 — TARGET VARIABLE (tags) ANALYSIS")

no_tag  = (df_raw['tags'] == '-').sum()
emp_tag = (df_raw['tags'].isna() | (df_raw['tags'] == '')).sum()
has_tag = (~df_raw['tags'].isin(['-','']) & df_raw['tags'].notna()).sum()
multi   = df_raw['tags'].str.contains(',', na=False).sum()

print(f"  Rows with tag '-' (missing) : {no_tag:,}  ({no_tag/len(df_raw)*100:.2f}%)")
print(f"  Rows with empty tag         : {int(emp_tag):,}")
print(f"  Rows WITH a tag             : {has_tag:,}  ({has_tag/len(df_raw)*100:.2f}%)")
print(f"  Multi-label rows (comma)    : {multi:,}  ({multi/len(df_raw)*100:.2f}%)")
print()
valid_rows = df_raw[~df_raw['tags'].isin(['-','']) & df_raw['tags'].notna()]
combo_counts = valid_rows['tags'].value_counts()
print(f"  Unique tag combinations: {len(combo_counts)}")
print(f"\n  Top 20 tag value_counts:")
print(combo_counts.head(20).to_string())

# ── CELL 7: CRITICAL ISSUE — 'v' vs 'verb' ───────────────────────
print("STEP 5 — CRITICAL ISSUE: 'v' vs 'verb' Tag Inconsistency")

atomic = Counter()
for t in df_raw['tags'].dropna():
    if t == '-': continue
    for p in t.split(','):
        atomic[p.strip()] += 1

print("\nAll unique atomic tag values found in dataset:")
print(f"  {'Tag':12s} {'Count':>8s}  {'In Valid Set?'}")
print(f"  {'-'*40}")
for tag, cnt in atomic.most_common():
    valid = '✅' if tag in VALID_TAGS else '❌ NOT VALID'
    print(f"  {tag:12s} {cnt:>8,}  {valid}")

print(f"""
FINDING:
  'v' appears {atomic.get('v',0):,} times but is NOT in the 8-class valid set.
  'verb' appears {atomic.get('verb',0)} times — it was never used.
  'v' is clearly an abbreviation of 'verb' used inconsistently.

DECISION: Rename all 'v' → 'verb' during preprocessing.
IMPACT   : Recovers {atomic.get('v',0):,} verb instances that would be LOST otherwise.
""")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
before = pd.Series(dict(atomic)).sort_values(ascending=False)
colors_b = ['#e76f51' if t not in VALID_TAGS else '#2a9d8f' for t in before.index]
axes[0].bar(before.index, before.values, color=colors_b, edgecolor='white')
axes[0].set_title('BEFORE Fix\n(Red = invalid tag)', fontweight='bold')
axes[0].set_ylabel('Count'); axes[0].tick_params(axis='x', rotation=45)
r = mpatches.Patch(color='#e76f51', label='Invalid')
g = mpatches.Patch(color='#2a9d8f', label='Valid')
axes[0].legend(handles=[g, r])

after_d = dict(atomic)
after_d['verb'] = after_d.get('verb',0) + after_d.pop('v',0)
after = pd.Series({k:v for k,v in after_d.items() if k in VALID_TAGS}).sort_values(ascending=False)
axes[1].bar(after.index, after.values, color='#2a9d8f', edgecolor='white')
axes[1].set_title('AFTER Fix (v→verb)\nAll 8 valid classes present', fontweight='bold')
axes[1].set_ylabel('Count'); axes[1].tick_params(axis='x', rotation=45)
plt.suptitle("Tag Inconsistency Fix: 'v' → 'verb'", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/02_tag_fix.png', dpi=150, bbox_inches='tight')
plt.show()

# ── CELL 8: Duplicate Analysis ───────────────────────────────────
print("STEP 6 — DUPLICATE ROWS ANALYSIS")

n_dup_rows = df_raw.duplicated(subset=['word','tags']).sum()
print(f"  Exact (word, tags) duplicates : {n_dup_rows:,}")
print()
print("Sample duplicate entries:")
dups = df_raw[df_raw.duplicated(subset=['word','tags'], keep=False)]
print(dups[['word_id','word','tags']].head(10).to_string(index=False))
print(f"""
DECISION: Drop duplicates keeping the first occurrence.
JUSTIFICATION:
  Exact (word, label) duplicates add zero new information.
  Worse, they cause DATA LEAKAGE — the same pair can appear
  in both train and test sets, inflating test performance.
""")

# ── CELL 9: Word Column Quality ──────────────────────────────────
print("STEP 7 — WORD COLUMN QUALITY CHECK")

words = df_raw['word'].astype(str).str.strip()
print(f"  Empty words    : {(words=='').sum()}")
print(f"  Single-char    : {(words.str.len()==1).sum()} (valid in Sindhi, e.g. conjunctions)")
print(f"  Very long (≥20): {(words.str.len()>=20).sum()}")
print(f"  Length range   : {words.str.len().min()} – {words.str.len().max()} chars")
print(f"  Mean length    : {words.str.len().mean():.2f} chars")

print("\nSingle-character words:")
print(df_raw[words.str.len()==1][['word_id','word','tags']].to_string(index=False))
print("\nVery long words (≥20 chars):")
print(df_raw[words.str.len()>=20][['word_id','word','tags']].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
lens = words[words!=''].str.len()
ax.hist(lens, bins=30, color='#264653', edgecolor='white', alpha=0.85)
ax.axvline(lens.mean(), color='#e76f51', lw=2, label=f'Mean={lens.mean():.1f}')
ax.axvline(lens.median(), color='#e9c46a', lw=2, ls='--', label=f'Median={lens.median():.0f}')
ax.set_xlabel('Word Length (characters)')
ax.set_ylabel('Frequency')
ax.set_title('Raw Word Length Distribution', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('results/plots/03_word_length_raw.png', dpi=150, bbox_inches='tight')
plt.show()
