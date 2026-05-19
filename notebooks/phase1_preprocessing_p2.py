# ── CELL 10: Full Preprocessing Pipeline ─────────────────────────
# (Continue in Colab after phase1_preprocessing.py cells)

print("STEP 8 — APPLY FULL PREPROCESSING PIPELINE")
print("=" * 55)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline with documented justification per step.

    Step 1 : Fix tag inconsistency  ('v' → 'verb')
    Step 2 : Drop rows with no tag  (dash or empty)
    Step 3 : Drop empty word rows
    Step 4 : Mark ambiguous words   (multi-tag → is_ambiguous flag)
    Step 5 : Explode multi-labels   (one row per label)
    Step 6 : Filter to 8 valid tags
    Step 7 : Drop duplicate pairs   (prevent data leakage)
    Step 8 : Reset index
    """
    df = df.copy()

    # STEP 1 — Fix 'v' → 'verb'
    df['tags'] = df['tags'].str.replace(r'\bv\b', 'verb', regex=True)
    step1_fixed = (df['tags'].str.contains(r'\bverb\b', na=False)).sum()

    # STEP 2 — Drop rows with no tag
    before = len(df)
    df = df[~df['tags'].isin(['-', '']) & df['tags'].notna()]
    step2_dropped = before - len(df)

    # STEP 3 — Drop empty words
    df['word'] = df['word'].astype(str).str.strip()
    before = len(df)
    df = df[df['word'] != '']
    step3_dropped = before - len(df)

    # STEP 4 — Mark ambiguous (multi-label) words
    df['is_ambiguous'] = df['tags'].str.contains(',')

    # STEP 5 — Explode multi-labels → one row per label
    df['label'] = df['tags'].str.split(',')
    df = df.explode('label')
    df['label'] = df['label'].str.strip()

    # STEP 6 — Keep only valid tags
    before = len(df)
    df = df[df['label'].isin(VALID_TAGS)]
    step6_dropped = before - len(df)

    # STEP 7 — Drop (word, label) duplicates
    df['word_len'] = df['word'].str.len()
    before = len(df)
    df = df.drop_duplicates(subset=['word', 'label'])
    step7_dropped = before - len(df)

    df = df[['word', 'label', 'is_ambiguous', 'word_len']].reset_index(drop=True)

    print("  Pipeline log:")
    print(f"    Step 1  v→verb fix          : {step1_fixed:,} rows now have 'verb'")
    print(f"    Step 2  No-tag rows dropped  : {step2_dropped:,}")
    print(f"    Step 3  Empty words dropped  : {step3_dropped:,}")
    print(f"    Step 4  Ambiguous flag added")
    print(f"    Step 5  Multi-label exploded")
    print(f"    Step 6  Invalid tags removed : {step6_dropped:,}")
    print(f"    Step 7  Duplicates removed   : {step7_dropped:,}")
    return df


df = preprocess(df_raw)

print()
print(f"  Raw rows          : {len(df_raw):,}")
print(f"  Cleaned instances : {len(df):,}")
print(f"  Unique words      : {df['word'].nunique():,}")
print(f"  POS classes       : {df['label'].nunique()}")
print(f"  Ambiguous pairs   : {df['is_ambiguous'].sum():,}  ({df['is_ambiguous'].mean()*100:.1f}%)")
print()
display(df.head(10))

# ── CELL 11: Before vs After Comparison ──────────────────────────
print("STEP 9 — BEFORE vs AFTER PREPROCESSING COMPARISON")

before_counts = Counter()
for t in df_raw['tags'].dropna():
    if t == '-': continue
    for p in t.split(','):
        p = p.strip()
        if p == 'v': p = 'verb'
        if p in VALID_TAGS:
            before_counts[p] += 1

after_counts = df['label'].value_counts().to_dict()

cdf = pd.DataFrame({'Before': pd.Series(before_counts),
                    'After' : pd.Series(after_counts)}).fillna(0).astype(int)
cdf['Removed'] = cdf['Before'] - cdf['After']
cdf = cdf.sort_values('Before', ascending=False)
print(cdf.to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
x = np.arange(len(cdf))
w = 0.35
axes[0].bar(x-w/2, cdf['Before'], w, label='Before', color='#e76f51', edgecolor='white')
axes[0].bar(x+w/2, cdf['After'],  w, label='After',  color='#2a9d8f', edgecolor='white')
axes[0].set_xticks(x); axes[0].set_xticklabels(cdf.index)
axes[0].set_title('Instance Counts: Before vs After', fontweight='bold')
axes[0].set_ylabel('Count'); axes[0].legend()

axes[1].bar(x-w/2, cdf['Before'], w, label='Before', color='#e76f51', edgecolor='white')
axes[1].bar(x+w/2, cdf['After'],  w, label='After',  color='#2a9d8f', edgecolor='white')
axes[1].set_yscale('log')
axes[1].set_xticks(x); axes[1].set_xticklabels(cdf.index)
axes[1].set_title('Same (Log Scale) — Reveals Rare Classes', fontweight='bold')
axes[1].set_ylabel('Count (log)'); axes[1].legend()

plt.suptitle('Preprocessing Impact per POS Class', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/04_before_after.png', dpi=150, bbox_inches='tight')
plt.show()

# ── CELL 12: Class Imbalance ──────────────────────────────────────
print("STEP 10 — CLASS IMBALANCE ANALYSIS")

lc  = df['label'].value_counts()
pct = lc / lc.sum() * 100
ratio = lc.iloc[0] / lc.iloc[-1]

print(f"  Most common  : {lc.index[0]} ({pct.iloc[0]:.2f}%)")
print(f"  Least common : {lc.index[-1]} ({pct.iloc[-1]:.4f}%)")
print(f"  Imbalance ratio : {ratio:.0f}:1")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(lc.index, lc.values, color=PALETTE, edgecolor='white')
axes[0].set_title('Counts (Absolute)', fontweight='bold')
axes[0].set_ylabel('Count')
for i,(l,c) in enumerate(lc.items()):
    axes[0].text(i, c+500, f'{c:,}', ha='center', fontsize=8, rotation=40)

axes[1].bar(pct.index, pct.values, color=PALETTE, edgecolor='white')
axes[1].set_title('Distribution (%)', fontweight='bold')
axes[1].set_ylabel('Percentage')
for i,(l,p) in enumerate(pct.items()):
    axes[1].text(i, p+0.3, f'{p:.2f}%', ha='center', fontsize=8, rotation=40)

axes[2].bar(lc.index, lc.values, color=PALETTE, edgecolor='white')
axes[2].set_yscale('log')
axes[2].set_title('Log Scale\n(Reveals rare classes)', fontweight='bold')
axes[2].set_ylabel('Count (log)')

plt.suptitle('Class Imbalance — Sindhi POS Corpus', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/05_class_imbalance.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"""
WHY IMBALANCE MATTERS:
  A naive classifier that always predicts 'noun' achieves {pct['noun']:.1f}% accuracy.
  This looks good but is completely useless.

SOLUTIONS APPLIED:
  1. class_weight='balanced' on ALL classifiers
     → Automatically up-weights rare classes during training.
  2. Macro-F1 as the primary evaluation metric
     → Treats all classes equally, regardless of frequency.
  3. Stratified K-Fold cross-validation
     → Preserves class proportions in every fold.
  4. Stratified train/test split
     → Ensures rare classes appear in both train and test.
""")

# ── CELL 13: Word Length Analysis per Tag ────────────────────────
print("STEP 11 — WORD LENGTH ANALYSIS BY POS TAG")

wl = df.groupby('label')['word_len'].agg(['mean','median','std','min','max']).round(2)
wl = wl.sort_values('mean', ascending=False)
print(wl.to_string())

# Kruskal-Wallis test
groups = [df[df['label']==lbl]['word_len'].values for lbl in VALID_TAGS if lbl in df['label'].values]
stat, p_val = stats.kruskal(*groups)
print(f"\nKruskal-Wallis H={stat:.2f}, p={p_val:.2e}")
if p_val < 0.05:
    print("→ REJECT H0: word length IS significantly different across POS tags.")
    print("  This confirms word length (captured via n-gram structure) is informative.")

order = df.groupby('label')['word_len'].median().sort_values(ascending=False).index
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

df.boxplot(column='word_len', by='label', ax=axes[0],
           patch_artist=True,
           boxprops=dict(facecolor='#8ecae6', color='#264653'),
           medianprops=dict(color='#e76f51', linewidth=2))
axes[0].set_title(f'Word Length by POS Tag\nKruskal-Wallis p={p_val:.2e}', fontweight='bold')
axes[0].set_xlabel('POS Tag'); axes[0].set_ylabel('Length (chars)')
axes[0].get_figure().suptitle('')

wl['mean'].sort_values().plot(kind='barh', ax=axes[1], color='#264653', edgecolor='white')
axes[1].set_title('Mean Word Length per POS Tag', fontweight='bold')
axes[1].set_xlabel('Mean Characters')
for i, v in enumerate(wl.sort_values('mean')['mean']):
    axes[1].text(v+0.05, i, f'{v:.2f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('results/plots/06_word_length_by_tag.png', dpi=150, bbox_inches='tight')
plt.show()

# ── CELL 14: Ambiguous Words Analysis ────────────────────────────
print("STEP 12 — AMBIGUOUS WORD ANALYSIS")

amb = df[df['is_ambiguous']]
amb_pct = (amb['label'].value_counts() / df['label'].value_counts() * 100).fillna(0).round(1)
total_per = df['label'].value_counts()
print(pd.DataFrame({'Ambiguous': amb['label'].value_counts(),
                    'Total': total_per, 'Ambig%': amb_pct}).to_string())

print("\nTop 10 most ambiguous words (most POS labels):")
word_nlabels = df.groupby('word')['label'].nunique().sort_values(ascending=False)
top_amb = word_nlabels[word_nlabels > 1].head(10)
for word, n in top_amb.items():
    labels = df[df['word']==word]['label'].unique()
    print(f"  {word:20s} → {n} tags: {list(labels)}")

fig, ax = plt.subplots(figsize=(10, 4))
amb_pct.sort_values(ascending=False).plot(kind='bar', ax=ax, color='#f4a261', edgecolor='white')
ax.set_title('Ambiguous Word-Label Pairs per POS Class (%)', fontweight='bold')
ax.set_ylabel('% of class instances that are ambiguous')
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(amb_pct.sort_values(ascending=False)):
    ax.text(i, v+0.3, f'{v:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('results/plots/07_ambiguous_words.png', dpi=150, bbox_inches='tight')
plt.show()

# ── CELL 15: Final Summary & Save ────────────────────────────────
print("=" * 60)
print("PREPROCESSING COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  Raw rows             : {len(df_raw):,}")
print(f"  Cleaned instances    : {len(df):,}")
print(f"  Reduction            : {(1 - len(df)/len(df_raw))*100:.1f}%")
print(f"  Unique words         : {df['word'].nunique():,}")
print(f"  Classes              : {sorted(df['label'].unique())}")
print()
print("  Issues resolved:")
print("    ✅ 'v' → 'verb' renamed")
print("    ✅ 793 no-tag rows dropped")
print("    ✅ 1 empty word dropped")
print("    ✅ Multi-labels exploded + is_ambiguous flag added")
print("    ✅ ~1,174 duplicate (word,label) pairs dropped")
print("    ✅ 7 high-missing columns dropped")
print()
print("  Class distribution (final):")
for lbl, cnt in df['label'].value_counts().items():
    bar = '█' * int(cnt / df['label'].value_counts().max() * 28)
    print(f"    {lbl:8s}: {cnt:>7,}  {bar}")

import os
os.makedirs('results', exist_ok=True)
df.to_csv('results/sindhi_pos_cleaned.csv', index=False, encoding='utf-8')
print("\n✅ Saved → results/sindhi_pos_cleaned.csv")
print("✅ All plots saved → results/plots/")
