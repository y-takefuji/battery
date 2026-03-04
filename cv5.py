import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import shap
from scipy.stats import spearmanr
from sklearn.cluster import FeatureAgglomeration
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load data
df = pd.read_csv('Battery_dataset.csv')
df = df.drop(columns=['battery_id'])
X = df.drop(columns=['RUL'])
y = df['RUL']
feature_names = X.columns.tolist()

# ── 2. Dataset shape & target distribution
print("=" * 55)
print("DATASET SHAPE")
print("=" * 55)
print(f"  Rows         : {df.shape[0]}")
print(f"  Columns      : {df.shape[1]}  (includes RUL)")
print(f"  Features     : {X.shape[1]}")
print()
print("=" * 55)
print("TARGET DISTRIBUTION  (RUL)")
print("=" * 55)
print(f"  Count        : {y.count()}")
print(f"  Mean         : {y.mean():.4f}")
print(f"  Std          : {y.std():.4f}")
print(f"  Min          : {y.min():.4f}")
print(f"  25%          : {y.quantile(0.25):.4f}")
print(f"  Median       : {y.median():.4f}")
print(f"  75%          : {y.quantile(0.75):.4f}")
print(f"  Max          : {y.max():.4f}")
print(f"  IQR          : {(y.quantile(0.75) - y.quantile(0.25)):.4f}")
print(f"  Range        : {(y.max() - y.min()):.4f}")
print(f"  CV (std/mean): {(y.std() / y.mean()):.4f}")
print("=" * 55)
print()

# ── 3. Helper: cross-validation (R2)
def cv_score(estimator, X_sub, y, cv=5):
    scores = cross_val_score(estimator, X_sub, y, cv=cv, scoring='r2')
    return round(float(np.mean(scores)), 4)

# ── 4. Fixed random indices for SHAP (100 instances)
np.random.seed(42)
shap_idx = np.random.choice(len(X), size=100, replace=False)

# ==============================================================================
# METHOD 1: Random Forest
# Step 1: Fit RF on FULL X -> rank all features -> Top 5
# Step 2: Remove highest (top1) from Top 5 -> X_red (4-feature reduced dataset)
# Step 3: Re-fit RF on X_red -> re-rank -> Top 4 from X_red
# Step 4: CV on Top 5 (from full X)
# ==============================================================================

# Step 1: fit on full X, select Top 5
rf_full        = RandomForestRegressor(random_state=42)
rf_full.fit(X, y)
rf_scores_full = pd.Series(rf_full.feature_importances_,
                            index=feature_names).sort_values(ascending=False)
top5_rf        = rf_scores_full.iloc[:5].index.tolist()   # Top 5 from full X
top1_rf        = top5_rf[0]                               # highest feature in Top 5

# Step 2: remove top1 from Top 5 -> 4-feature reduced dataset
X_red_rf       = X[top5_rf].drop(columns=[top1_rf])       # X_red has exactly 4 features

# Step 3: re-fit RF on X_red -> re-rank -> Top 4 from X_red
rf_red         = RandomForestRegressor(random_state=42)
rf_red.fit(X_red_rf, y)                                   # fit ONLY on 4-feature reduced dataset
rf_scores_red  = pd.Series(rf_red.feature_importances_,
                            index=X_red_rf.columns).sort_values(ascending=False)
top4_rf        = rf_scores_red.iloc[:4].index.tolist()    # Top 4 re-selected from X_red

# Step 4: CV on Top 5 only
cv5_rf = cv_score(RandomForestRegressor(random_state=42), X[top5_rf], y)

print(f"[RF]       Top 5 (full X)   : {top5_rf}")
print(f"[RF]       Top 1 removed    : {top1_rf}")
print(f"[RF]       Top 4 (from X_red, re-fit) : {top4_rf}")
print(f"[RF]       CV5 R2           : {cv5_rf}\n")

# ==============================================================================
# METHOD 2: RF-SHAP
# Step 1: Fit RF on FULL X -> mean|SHAP| -> Top 5
# Step 2: Remove highest (top1) from Top 5 -> X_red (4-feature reduced dataset)
# Step 3: Re-fit RF on X_red -> mean|SHAP| -> Top 4 from X_red
# Step 4: CV on Top 5 (from full X)
# ==============================================================================

# Step 1: fit on full X, compute mean|SHAP|, select Top 5
rf_shap_full        = RandomForestRegressor(random_state=42)
rf_shap_full.fit(X, y)
exp_rf_full         = shap.TreeExplainer(rf_shap_full)
sv_rf_full          = exp_rf_full.shap_values(X.iloc[shap_idx])
rf_shap_scores_full = pd.Series(np.abs(sv_rf_full).mean(axis=0),
                                 index=feature_names).sort_values(ascending=False)
top5_rf_shap        = rf_shap_scores_full.iloc[:5].index.tolist()  # Top 5 from full X
top1_rf_shap        = top5_rf_shap[0]                              # highest feature in Top 5

# Step 2: remove top1 from Top 5 -> 4-feature reduced dataset
X_red_rf_shap       = X[top5_rf_shap].drop(columns=[top1_rf_shap]) # X_red has exactly 4 features

# Step 3: re-fit RF on X_red -> mean|SHAP| -> Top 4 from X_red
rf_shap_red         = RandomForestRegressor(random_state=42)
rf_shap_red.fit(X_red_rf_shap, y)                                  # fit ONLY on 4-feature reduced dataset
exp_rf_red          = shap.TreeExplainer(rf_shap_red)
sv_rf_red           = exp_rf_red.shap_values(X_red_rf_shap.iloc[shap_idx])
rf_shap_scores_red  = pd.Series(np.abs(sv_rf_red).mean(axis=0),
                                 index=X_red_rf_shap.columns).sort_values(ascending=False)
top4_rf_shap        = rf_shap_scores_red.iloc[:4].index.tolist()   # Top 4 re-selected from X_red

# Step 4: CV on Top 5 only
cv5_rf_shap = cv_score(RandomForestRegressor(random_state=42), X[top5_rf_shap], y)

print(f"[RF-SHAP]  Top 5 (full X)   : {top5_rf_shap}")
print(f"[RF-SHAP]  Top 1 removed    : {top1_rf_shap}")
print(f"[RF-SHAP]  Top 4 (from X_red, re-fit) : {top4_rf_shap}")
print(f"[RF-SHAP]  CV5 R2           : {cv5_rf_shap}\n")

# ==============================================================================
# METHOD 3: XGBoost
# Step 1: Fit XGB on FULL X -> rank all features -> Top 5
# Step 2: Remove highest (top1) from Top 5 -> X_red (4-feature reduced dataset)
# Step 3: Re-fit XGB on X_red -> re-rank -> Top 4 from X_red
# Step 4: CV on Top 5 (from full X)
# ==============================================================================

# Step 1: fit on full X, select Top 5
xgb_full        = XGBRegressor(random_state=42)
xgb_full.fit(X, y)
xgb_scores_full = pd.Series(xgb_full.feature_importances_,
                             index=feature_names).sort_values(ascending=False)
top5_xgb        = xgb_scores_full.iloc[:5].index.tolist()  # Top 5 from full X
top1_xgb        = top5_xgb[0]                              # highest feature in Top 5

# Step 2: remove top1 from Top 5 -> 4-feature reduced dataset
X_red_xgb       = X[top5_xgb].drop(columns=[top1_xgb])    # X_red has exactly 4 features

# Step 3: re-fit XGB on X_red -> re-rank -> Top 4 from X_red
xgb_red         = XGBRegressor(random_state=42)
xgb_red.fit(X_red_xgb, y)                                  # fit ONLY on 4-feature reduced dataset
xgb_scores_red  = pd.Series(xgb_red.feature_importances_,
                             index=X_red_xgb.columns).sort_values(ascending=False)
top4_xgb        = xgb_scores_red.iloc[:4].index.tolist()   # Top 4 re-selected from X_red

# Step 4: CV on Top 5 only
cv5_xgb = cv_score(XGBRegressor(random_state=42), X[top5_xgb], y)

print(f"[XGB]      Top 5 (full X)   : {top5_xgb}")
print(f"[XGB]      Top 1 removed    : {top1_xgb}")
print(f"[XGB]      Top 4 (from X_red, re-fit) : {top4_xgb}")
print(f"[XGB]      CV5 R2           : {cv5_xgb}\n")

# ==============================================================================
# METHOD 4: XGB-SHAP
# Step 1: Fit XGB on FULL X -> mean|SHAP| -> Top 5
# Step 2: Remove highest (top1) from Top 5 -> X_red (4-feature reduced dataset)
# Step 3: Re-fit XGB on X_red -> mean|SHAP| -> Top 4 from X_red
# Step 4: CV on Top 5 (from full X)
# ==============================================================================

# Step 1: fit on full X, compute mean|SHAP|, select Top 5
xgb_shap_full        = XGBRegressor(random_state=42)
xgb_shap_full.fit(X, y)
exp_xgb_full         = shap.TreeExplainer(xgb_shap_full)
sv_xgb_full          = exp_xgb_full.shap_values(X.iloc[shap_idx])
xgb_shap_scores_full = pd.Series(np.abs(sv_xgb_full).mean(axis=0),
                                  index=feature_names).sort_values(ascending=False)
top5_xgb_shap        = xgb_shap_scores_full.iloc[:5].index.tolist()  # Top 5 from full X
top1_xgb_shap        = top5_xgb_shap[0]                              # highest feature in Top 5

# Step 2: remove top1 from Top 5 -> 4-feature reduced dataset
X_red_xgb_shap       = X[top5_xgb_shap].drop(columns=[top1_xgb_shap]) # X_red has exactly 4 features

# Step 3: re-fit XGB on X_red -> mean|SHAP| -> Top 4 from X_red
xgb_shap_red         = XGBRegressor(random_state=42)
xgb_shap_red.fit(X_red_xgb_shap, y)                                   # fit ONLY on 4-feature reduced dataset
exp_xgb_red          = shap.TreeExplainer(xgb_shap_red)
sv_xgb_red           = exp_xgb_red.shap_values(X_red_xgb_shap.iloc[shap_idx])
xgb_shap_scores_red  = pd.Series(np.abs(sv_xgb_red).mean(axis=0),
                                  index=X_red_xgb_shap.columns).sort_values(ascending=False)
top4_xgb_shap        = xgb_shap_scores_red.iloc[:4].index.tolist()    # Top 4 re-selected from X_red

# Step 4: CV on Top 5 only
cv5_xgb_shap = cv_score(XGBRegressor(random_state=42), X[top5_xgb_shap], y)

print(f"[XGB-SHAP] Top 5 (full X)   : {top5_xgb_shap}")
print(f"[XGB-SHAP] Top 1 removed    : {top1_xgb_shap}")
print(f"[XGB-SHAP] Top 4 (from X_red, re-fit) : {top4_xgb_shap}")
print(f"[XGB-SHAP] CV5 R2           : {cv5_xgb_shap}\n")

# ==============================================================================
# METHOD 5: Feature Agglomeration  (unsupervised, NO re-fit)
# Step 1: Fit FA on FULL X -> proximity scores -> Top 5
# Step 2: Remove highest (top1) from Top 5 scores (NO re-fit of FA)
# Step 3: Re-rank remaining 4 scores -> Top 4
# Step 4: CV on Top 5 (from full X)
# ==============================================================================

# Step 1: fit FA on full X, proximity scores for all features
n_clusters     = X.shape[1] // 2
fa             = FeatureAgglomeration(n_clusters=n_clusters)
fa.fit(X)
labels         = fa.labels_
X_arr          = X.values

proximity      = np.zeros(X.shape[1])
for cl in range(n_clusters):
    members    = np.where(labels == cl)[0]
    centroid   = X_arr[:, members].mean(axis=1, keepdims=True)
    msd        = np.mean((X_arr[:, members] - centroid) ** 2, axis=0)
    proximity[members] = 1.0 / (1.0 + msd)

fa_scores_full = pd.Series(proximity,
                            index=feature_names).sort_values(ascending=False)
top5_fa        = fa_scores_full.iloc[:5].index.tolist()   # Top 5 from full X
top1_fa        = top5_fa[0]                               # highest feature in Top 5

# Step 2: remove top1 from Top 5 scores, no re-fit (FA is unsupervised)
fa_scores_top5 = fa_scores_full[top5_fa]                  # scores of Top 5 only
fa_scores_red  = fa_scores_top5.drop(index=top1_fa)       # remove top1 -> 4 scores remain

# Step 3: re-rank remaining 4 scores -> Top 4
top4_fa        = fa_scores_red.iloc[:4].index.tolist()    # Top 4 re-ranked from reduced scores

# Step 4: CV on Top 5 only
cv5_fa = cv_score(RandomForestRegressor(random_state=42), X[top5_fa], y)

print(f"[FA]       Top 5 (full X)   : {top5_fa}")
print(f"[FA]       Top 1 removed    : {top1_fa}")
print(f"[FA]       Top 4 (re-ranked): {top4_fa}")
print(f"[FA]       CV5 R2           : {cv5_fa}\n")

# ==============================================================================
# METHOD 6: HVGS
# Step 1: Variance on FULL X -> Top 5
# Step 2: Remove highest (top1) from Top 5 -> X_red (4-feature reduced dataset)
# Step 3: Recompute variance on X_red -> Top 4 from X_red
# Step 4: CV on Top 5 (from full X)
# ==============================================================================

def hvgs_scores(X_in):
    return X_in.var().sort_values(ascending=False)

# Step 1
hvgs_scores_full = hvgs_scores(X)
top5_hvgs        = hvgs_scores_full.iloc[:5].index.tolist()  # Top 5 from full X
top1_hvgs        = top5_hvgs[0]                              # highest feature in Top 5

# Step 2: remove top1 from Top 5 -> 4-feature reduced dataset
X_red_hvgs       = X[top5_hvgs].drop(columns=[top1_hvgs])   # X_red has exactly 4 features

# Step 3: recompute variance on X_red -> Top 4 from X_red
hvgs_scores_red  = hvgs_scores(X_red_hvgs)
top4_hvgs        = hvgs_scores_red.iloc[:4].index.tolist()   # Top 4 re-selected from X_red

# Step 4: CV on Top 5 only
cv5_hvgs = cv_score(RandomForestRegressor(random_state=42), X[top5_hvgs], y)

print(f"[HVGS]     Top 5 (full X)   : {top5_hvgs}")
print(f"[HVGS]     Top 1 removed    : {top1_hvgs}")
print(f"[HVGS]     Top 4 (from X_red, recomputed) : {top4_hvgs}")
print(f"[HVGS]     CV5 R2           : {cv5_hvgs}\n")

# ==============================================================================
# METHOD 7: Spearman Correlation
# Step 1: |Spearman rho| on FULL X -> Top 5
# Step 2: Remove highest (top1) from Top 5 -> X_red (4-feature reduced dataset)
# Step 3: Recompute |Spearman rho| on X_red -> Top 4 from X_red
# Step 4: CV on Top 5 (from full X)
# ==============================================================================

def spearman_scores(X_in, y):
    return pd.Series(
        [abs(spearmanr(X_in[f], y).statistic) for f in X_in.columns],
        index=X_in.columns
    ).sort_values(ascending=False)

# Step 1
sp_scores_full = spearman_scores(X, y)
top5_sp        = sp_scores_full.iloc[:5].index.tolist()   # Top 5 from full X
top1_sp        = top5_sp[0]                               # highest feature in Top 5

# Step 2: remove top1 from Top 5 -> 4-feature reduced dataset
X_red_sp       = X[top5_sp].drop(columns=[top1_sp])       # X_red has exactly 4 features

# Step 3: recompute |Spearman rho| on X_red -> Top 4 from X_red
sp_scores_red  = spearman_scores(X_red_sp, y)
top4_sp        = sp_scores_red.iloc[:4].index.tolist()    # Top 4 re-selected from X_red

# Step 4: CV on Top 5 only
cv5_sp = cv_score(RandomForestRegressor(random_state=42), X[top5_sp], y)

print(f"[Spearman] Top 5 (full X)   : {top5_sp}")
print(f"[Spearman] Top 1 removed    : {top1_sp}")
print(f"[Spearman] Top 4 (from X_red, recomputed) : {top4_sp}")
print(f"[Spearman] CV5 R2           : {cv5_sp}\n")

# ==============================================================================
# Summary table: result.csv
# ==============================================================================
results = pd.DataFrame([
    {
        'Method'       : 'Random Forest',
        'CV Evaluator' : 'RandomForestRegressor',
        'Top5 Features': ', '.join(top5_rf),
        'CV5 R2'       : cv5_rf,
        'Top1 Removed' : top1_rf,
        'Top4 Features': ', '.join(top4_rf),
    },
    {
        'Method'       : 'RF-SHAP',
        'CV Evaluator' : 'RandomForestRegressor',
        'Top5 Features': ', '.join(top5_rf_shap),
        'CV5 R2'       : cv5_rf_shap,
        'Top1 Removed' : top1_rf_shap,
        'Top4 Features': ', '.join(top4_rf_shap),
    },
    {
        'Method'       : 'XGBoost',
        'CV Evaluator' : 'XGBRegressor',
        'Top5 Features': ', '.join(top5_xgb),
        'CV5 R2'       : cv5_xgb,
        'Top1 Removed' : top1_xgb,
        'Top4 Features': ', '.join(top4_xgb),
    },
    {
        'Method'       : 'XGB-SHAP',
        'CV Evaluator' : 'XGBRegressor',
        'Top5 Features': ', '.join(top5_xgb_shap),
        'CV5 R2'       : cv5_xgb_shap,
        'Top1 Removed' : top1_xgb_shap,
        'Top4 Features': ', '.join(top4_xgb_shap),
    },
    {
        'Method'       : 'Feature Agglomeration',
        'CV Evaluator' : 'RandomForestRegressor',
        'Top5 Features': ', '.join(top5_fa),
        'CV5 R2'       : cv5_fa,
        'Top1 Removed' : top1_fa,
        'Top4 Features': ', '.join(top4_fa),
    },
    {
        'Method'       : 'HVGS',
        'CV Evaluator' : 'RandomForestRegressor',
        'Top5 Features': ', '.join(top5_hvgs),
        'CV5 R2'       : cv5_hvgs,
        'Top1 Removed' : top1_hvgs,
        'Top4 Features': ', '.join(top4_hvgs),
    },
    {
        'Method'       : 'Spearman',
        'CV Evaluator' : 'RandomForestRegressor',
        'Top5 Features': ', '.join(top5_sp),
        'CV5 R2'       : cv5_sp,
        'Top1 Removed' : top1_sp,
        'Top4 Features': ', '.join(top4_sp),
    },
])

results.to_csv('result.csv', index=False)
print(results.to_string(index=False))
