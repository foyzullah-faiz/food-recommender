# 🥗 Health-Personalized Food Recommender System

An end-to-end data science pipeline that recommends recipes based on medical conditions (diabetes, hypertension, vegan diet). Built as a university data science course project covering 10 of 11 work packages.

**Live Demo:** [food-recommender-26.streamlit.app](https://food-recommender-26.streamlit.app)

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| SVD Precision@1 | **51.4%** (51× better than random) |
| Two evaluation methods agree | ~51% |
| Recipe matrix | 186,490 × 37 dimensions |
| Training interactions | 464,257 |
| Cohen's Kappa | 1.000 (annotation agreement) |
| Optuna Hit@10 | ~65% after tuning |
| Health constraints | 100% enforced |
| W&B runs logged | 124 |

---

## 📦 Dataset & Data Sources

| Source | Description |
|--------|-------------|
| [Food.com Recipes & Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) | 231,637 recipes, 583K+ user interactions (2000–2018) |
| [USDA FoodData Central API](https://fdc.nal.usda.gov) | Scraped 2,000 recipes — adds fiber, potassium, calcium, iron |

---

## ✅ Work Packages Completed (10 / 11)

| Work Package | Implementation |
|---|---|
| **Data Scraping** | USDA FoodData Central REST API — `requests` library, 2,000 recipes, 89.6% hit rate |
| **Data Quality ★** | Percentile-based cleaning (top 0.5%), Atwater calorie validation, bias analysis |
| **Data Annotation** | Label Studio — 200 recipes annotated — Cohen's Kappa = 1.0 |
| **Vector Embeddings ★** | R ∈ ℝ^(186490×37) — 6 nutrition dims + 11 health labels + 20 MiniLM ingredient dims |
| **Recommender System ★** | SVD vs BPR hybrid — hard clinical constraints enforced after scoring |
| **Performance Evaluation ★** | Precision@k + Recall@k — two independent methods — 51.4% @ k=1 |
| **Hyperparameter Tuning** | Optuna TPE — 30 trials — n_factors=64, reg=0.003, lr=0.021 |
| **Experiments Logging ★** | Weights & Biases — 124 runs logged |
| **Perturbation Analysis** | Jaccard similarity — J ≥ 0.7 at σ=0.01 (robust) |
| **Frontend Application** | Streamlit — live health filtering, calorie proximity boost |

★ = starred work package

---

## 🏥 Clinical Health Labels

| Label | Threshold | Source |
|---|---|---|
| `diabetic_ok` | carbs ≤ 45g AND sugar ≤ 10g | ADA |
| `low_sodium` | sodium ≤ 400mg | WHO |
| `heart_healthy` | sat_fat ≤ 5g AND sodium ≤ 500mg | AHA |
| `high_protein` | protein ≥ 25g | Clinical |
| `low_fat` | total_fat ≤ 10g | Clinical |
| `vegetarian` | Food.com tags | Tags |
| `vegan` | Food.com tags | Tags |
| `gluten_free` | Food.com tags | Tags |

---

## 🧮 Core Formulas

```
Cosine Similarity:   cos(u,r) = (u·r) / (||u|| × ||r||)
SVD Factorization:   R ≈ U·Vᵀ  →  r̂ᵤᵢ = Uᵤ · Vᵢᵀ
Hybrid Score:        α·cos(u,r) + (1−α)·SVD(u,r)
Precision@k:         |relevant ∩ top-k| / k
Cohen's Kappa:       κ = (pₒ − pₑ) / (1 − pₑ)
Jaccard Similarity:  J(A,B) = |A∩B| / |A∪B|
```

---

## 🗂️ Repository Structure

```
food-recommender/
├── app.py                              # Streamlit frontend
├── food_recommender_COMPLETE_FINAL.ipynb  # Main notebook (all 13 sections)
├── requirements.txt                    # Python dependencies
├── data/
│   ├── usda_enrichment.csv            # USDA API scraped data (50KB)
│   ├── annotation_sample.csv          # 200 recipes for Label Studio
│   ├── label_studio_export.csv        # Annotated labels
│   ├── recipes_sample.csv             # 5K recipe sample for deployment
│   └── interactions_train.csv.zip     # Sample interactions
└── plots/
    ├── dataset_bias.png               # Rating distribution
    ├── cleaning_before_after.png      # Data cleaning results
    ├── embedding_space.png            # PCA recipe clusters
    ├── model_comparison_bar.png       # SVD vs BPR vs random
    ├── svd_vs_bpr.png                 # Precision@k + Recall@k curves
    ├── hyperparams.png                # Optuna search scatter plots
    ├── perturbation.png               # Jaccard robustness
    └── annotation_labels.png          # Health label distribution
```

> **Note:** Large data files (RAW_recipes.csv 281MB, recipes_clean.csv 238MB, model files) are excluded via `.gitignore` due to GitHub's 100MB limit. Raw data available on [Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews).

---

## ⚙️ Setup & Installation

```bash
# Create conda environment
conda create -n food_rec python=3.11 -y
conda activate food_rec

# Install dependencies
conda install -c conda-forge numpy=1.26 pandas=2.2 matplotlib scikit-learn jupyter ipykernel -y
pip install scikit-surprise optuna wandb requests cornac sentence-transformers streamlit plotly
```

---

## 🚀 Run

**Jupyter Notebook (full pipeline):**
```bash
conda activate food_rec
jupyter notebook
# Open: food_recommender_COMPLETE_FINAL.ipynb
# Run all cells top to bottom
```

**Streamlit App (local — full 186K recipes):**
```bash
conda activate food_rec
streamlit run app.py
```

**Live deployment (5K recipe sample):**
→ [food-recommender-26.streamlit.app](https://food-recommender-26.streamlit.app)

---

## 📊 Evaluation Protocol

Candidate-set evaluation — standard for implicit feedback recommender systems:

1. For each test user, take 1 positive recipe they actually interacted with
2. Sample 99 random negative recipes they never saw
3. Rank all 100 candidates using the SVD model
4. Measure Precision@k — does the positive appear in the top k?

**Random baseline = 1% at k=1. SVD achieves 51.4% — 51× better than random.**

Two independent methods (candidate-set ranking + leave-one-out) both agree at ~51%, confirming the evaluation methodology is correct.

---

## 🔬 Limitations & Future Work

**Current limitations:**
- SVD assumes linear latent factors — cannot capture non-linear interactions
- Cold start: new users get content-based recommendations only
- Binary implicit feedback — no explicit negative signal
- Single dataset (Food.com) — may not generalise across cuisines

**Future improvements:**
- Neural Collaborative Filtering (NCF) — expected Precision@1 ~60-70% ★★★
- User interaction embeddings — learn health profile from history, not sliders ★★★
- Tune BPR fairly with Optuna — honest comparison ★★
- Larger USDA scrape — high_fiber label currently only 0.1% ★

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy |
| ML / RecSys | scikit-surprise (SVD, BPR), scikit-learn |
| Embeddings | sentence-transformers (MiniLM-L6-v2), PCA |
| Tuning | Optuna TPE |
| Logging | Weights & Biases |
| Annotation | Label Studio |
| Frontend | Streamlit, Plotly |
| Scraping | requests (USDA REST API) |
