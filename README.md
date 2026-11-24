# Cross-Script Multilingual Intent Classification (Anveshan Data Sprint 2.0)

**Team:** Tanisha (`Tani414`)  
**Leaderboard Result:** Rank 8 Finalist — **0.83678** Public Macro-F1 | **0.83712** Private Macro-F1 | **0.83695** Cumulative Macro-F1  

---

## 1. Pipeline Overview & Architecture

Our final inference pipeline combines a **fine-tuned `google/muril-base-cased` transformer backbone** with a **0M-neural-parameter Cross-Script & Phonetic TF-IDF linear classifier** via z-score standardized logit blending:

1. **Deterministic Cross-Script Normalization (`ITRANS` $\leftrightarrow$ `Devanagari`):**
   - Detects script per query and transliterates Romanized Hinglish inputs into canonical Devanagari so semantically identical utterances map to a shared subword and n-gram space.
2. **Neural Backbone (`google/muril-base-cased` — 237.64M parameters):**
   - **Last-4-Layer Aggregation:** Averages hidden states across the top 4 transformer encoder layers (`layers 9–12`) to capture both surface lexical cues and high-level intent semantics.
   - **Dual Pooling Head:** Concatenates attention-masked **Mean Pooling** (`768-d`) and learnable **Attention Pooling** (`768-d`) into a `1536-d` representation.
   - **Training Strategy:** 5-fold Stratified Cross-Validation with gentle prior logit adjustment ($\tau = 0.25$), label smoothing (`0.05`), cosine learning-rate warmup, and Stage-2 high-confidence pseudo-labeling (`conf >= 0.88`).
   - **Dual-Script Test-Time Augmentation (TTA):** Evaluates original query (`0.65`) and canonical Devanagari transliteration (`0.35`) through the single model checkpoint.
3. **Cross-Script Lexical Branch (TF-IDF + Logistic Regression):**
   - Combines word `(1, 2)` n-grams and character-boundary `char_wb (3, 5)` n-grams over `raw || canonical_devanagari` text to preserve exact entity stems and command morphology.
4. **Z-Score Standardized Logit Blending:**
   - Standardizes per-sample logits across all 56 classes (`(z - mean) / std`) for both branches and blends at `0.78 * MuRIL + 0.22 * Cross-Script TF-IDF`.

---

## 2. Cross-Script (`Hindi <-> Hinglish`) & Class 55 (Out-of-Scope) Handling

- **Cross-Script Alignment:** Romanized Hinglish queries (e.g., *"kal subah 7 baje ka alarm laga do"*) are deterministically transliterated to Devanagari (*"कल् सुभह् 7 बजे का अलार्म् लगा दो"*). Both scripts are jointly indexed in TF-IDF and evaluated via dual-script TTA in MuRIL, substantially closing the accuracy gap between Romanized (`0.8406` OOF Macro-F1) and Devanagari (`0.8385` OOF Macro-F1) queries.
- **Class 55 (Out-of-Scope) Calibration:** Rather than applying brittle post-hoc class-bias thresholds that overfit validation splits, Out-of-Scope calibration is handled inside the training objective via **prior logit adjustment ($\tau = 0.25$)** and **label smoothing (`0.05`)**. This keeps predicted Class 55 test prevalence aligned with the training distribution (`~6.8%` of test queries) and prevents false-positive drift on unseen data.

---

## 3. Model Parameter Count Proof (`<= 300M` Cap)

All inference components are verified to stay strictly within the **300,000,000** parameter ceiling:

| Component | Parameter Breakdown | Exact Count |
| :--- | :--- | ---: |
| `google/muril-base-cased` Encoder | Embeddings + 12 Transformer Layers | `237,556,224` |
| Custom Pooling & Classification Head | Attention (`768->1`) + Linear (`1536->56`) | `86,841` |
| **Total Neural Parameters** | `sum(p.numel() for p in model.parameters())` | **`237,643,065`** |
| Cross-Script TF-IDF Linear Classifier | `~145,000` sparse features $\times$ `56` classes | `8,120,000` |
| **Total Combined Inference Parameters** | **Neural + Classical Linear Weights** | **`245,763,065` (`<= 300M`)** |

Verification snippet included in `solution_pipeline.py`:
```python
total_neural_params = sum(p.numel() for p in model.parameters())
assert total_neural_params <= 300_000_000, f"Exceeded 300M cap: {total_neural_params}"
print(f"Verified Neural Params: {total_neural_params:,}")  # 237,643,065
