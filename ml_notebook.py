"""
Section 1: Machine Learning Pipeline
Success Prediction Model for Short-Form Video Content
"""

# ── 0. Imports ─────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, shap

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# ── 1. Load Data ────────────────────────────────────────────────────────────────
RAW_PATH = "Instagram_Data.csv"   # place the CSV in the same directory
df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

# ── 2. Preprocessing & Target Engineering ───────────────────────────────────────
# 2a. Boolean coercion (trending_audio may already be bool; guard either way)
df["trending_audio"] = df["trending_audio"].astype(int)   # XGBoost expects numeric

# 2b. Define pre-publish features ONLY – strict leakage prevention
PRE_FEATURES = [
    "reel_length_sec",
    "posting_time",
    "hook_strength_score",
    "caption_length",
    "hashtags_count",
    "trending_audio",
]

# 2c. Bin `reach` → target class (0=Low, 1=Moderate, 2=Viral)
#     Bottom 60 % → Low | Next 30 % (60-90 %) → Moderate | Top 10 % → Viral
bins   = [0, 0.60, 0.90, 1.0]
labels = [0, 1, 2]
df["target"] = pd.qcut(df["reach"], q=bins, labels=labels).astype(int)

print("\nClass distribution:")
print(df["target"].value_counts().sort_index()
      .rename({0: "Low (0)", 1: "Moderate (1)", 2: "Viral (2)"}))

# 2d. Build final modelling frame (drop ALL post-publish columns)
X = df[PRE_FEATURES].copy()
y = df["target"].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Missing values:\n{X.isnull().sum()}")

# ── 3. Exploratory Data Analysis ────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
LABEL_MAP = {0: "Low", 1: "Moderate", 2: "Viral"}

# 3a. Correlation heatmap (pre-publish features only)
fig, ax = plt.subplots(figsize=(8, 6))
corr = X.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap – Pre-Publish Features", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", dpi=150)
plt.show()
print("Saved: eda_correlation_heatmap.png")

# 3b. Boxplot – posting_time vs engagement category
plot_df = X[["posting_time"]].copy()
plot_df["Engagement Category"] = y.map(LABEL_MAP)

fig, ax = plt.subplots(figsize=(9, 5))
order = ["Low", "Moderate", "Viral"]
palette = {"Low": "#A8D5E2", "Moderate": "#F7B2AD", "Viral": "#B5EAD7"}
sns.boxplot(data=plot_df, x="Engagement Category", y="posting_time",
            order=order, palette=palette, linewidth=1.5, ax=ax)
ax.set_title("Posting Time Distribution by Engagement Category", fontsize=14, pad=10)
ax.set_xlabel("Engagement Category", fontsize=12)
ax.set_ylabel("Posting Hour (0–23)", fontsize=12)
plt.tight_layout()
plt.savefig("eda_posting_time_boxplot.png", dpi=150)
plt.show()
print("Saved: eda_posting_time_boxplot.png")

# ── 4. Train / Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

# ── 5. Pipeline Construction ────────────────────────────────────────────────────
NUMERIC_FEATURES = ["reel_length_sec", "posting_time",
                    "hook_strength_score", "caption_length", "hashtags_count"]

# Separate preprocessor (saved independently for the API)
preprocessor = Pipeline(steps=[
    ("scaler", StandardScaler())
])
# Fit on numeric columns of training set only
X_train_scaled = preprocessor.fit_transform(X_train[NUMERIC_FEATURES])
X_test_scaled  = preprocessor.transform(X_test[NUMERIC_FEATURES])

# Re-attach binary feature (trending_audio – no scaling needed)
import scipy.sparse as sp
X_train_full = np.hstack([X_train_scaled, X_train[["trending_audio"]].values])
X_test_full  = np.hstack([X_test_scaled,  X_test[["trending_audio"]].values])

feature_names_out = NUMERIC_FEATURES + ["trending_audio"]

# ── 6. GridSearchCV Hyperparameter Tuning ───────────────────────────────────────
base_xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)

param_grid = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth"    : [3, 5, 7],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_macro",
    verbose=1,
    n_jobs=-1,
)

print("\nRunning GridSearchCV (5-fold, f1_macro)…")
grid_search.fit(X_train_full, y_train)

print(f"\nBest params : {grid_search.best_params_}")
print(f"Best CV F1  : {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ── 7. Evaluation ───────────────────────────────────────────────────────────────
y_pred = best_model.predict(X_test_full)

print("\n── Classification Report ──────────────────────────────────")
print(classification_report(y_test, y_pred,
      target_names=["Low", "Moderate", "Viral"]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Low", "Moderate", "Viral"],
    cmap="Blues", ax=ax
)
ax.set_title("Confusion Matrix – XGBClassifier", fontsize=13)
plt.tight_layout()
plt.savefig("eval_confusion_matrix.png", dpi=150)
plt.show()
print("Saved: eval_confusion_matrix.png")

# ── 8. SHAP Beeswarm Summary Plot ───────────────────────────────────────────────
print("\nComputing SHAP values (TreeExplainer)…")
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_full)   # list of 3 arrays for multi-class

# Beeswarm for the "Viral" class (index 2)
fig, ax = plt.subplots(figsize=(9, 5))
shap.summary_plot(
    shap_values[2],
    X_test_full,
    feature_names=feature_names_out,
    plot_type="dot",
    show=False,
    color_bar=True,
)
plt.title("SHAP Beeswarm – Viral Class Feature Importance", fontsize=13, pad=10)
plt.tight_layout()
plt.savefig("shap_beeswarm_viral.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: shap_beeswarm_viral.png")

# ── 9. Export Artefacts ─────────────────────────────────────────────────────────
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("\n✅  Saved: xgb_model.pkl  |  preprocessor.pkl")
print("Pipeline complete.")
