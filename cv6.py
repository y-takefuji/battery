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
# Step 1: Fit RF on FULL X -> rank all features -> Top 6
# Step 2: Remove highest (top1) from Top 6 -> X_red (5-feature reduced dataset)
# Step 3: Re-fit RF on X_red -> re-rank -> Top 5 from X_red
# Step 4: CV on Top 6 (from full X)
# ==============================================================================

# Step 1: fit on full X, select Top 6
rf_full        = RandomForestRegressor(random_state=42)
rf_full.fit(X, y)
rf_scores_full = pd.Series(rf_full.feature_importances_,
                            index=feature_names).sort_values(ascending=False)
top6_rf        = rf_scores_full.iloc[:6].index.tolist()   # Top 6 from full X
top1_rf        = top6_rf[0]                               # highest feature in Top 6

# Step 2: remove top1 from Top 6 -> 5-feature reduced dataset
X_red_rf       = X[top6_rf].drop(columns=[top1_rf])       # X_red has exactly 5 features

# Step 3: re-fit RF on X_red -> re-rank -> Top 5 from X_red
rf_red         = RandomForestRegressor(random_state=42)
rf_red.fit(X_red_rf, y)                                   # fit ONLY on 5-feature reduced dataset
rf_scores_red  = pd.Series(rf_red.feature_importances_,
                            index=X_red_rf.columns).sort_values(ascending=False)
top5_rf        = rf_scores_red.iloc[:5].index.tolist()    # Top 5 re-selected from X_red

# Step 4: CV on Top 6 only
cv6_rf = cv_score(RandomForestRegressor(random_state=42), X[top6_rf], y)

print(f"[RF]       Top 6 (full X)   : {top6_rf}")
print(f"[RF]       Top 1 removed    : {top1_rf}")
print(f"[RF]       Top 5 (from X_red, re-fit) : {top5_rf}")
print(f"[RF]       CV6 R2           : {cv6_rf}\n")

# ==============================================================================
# METHOD 2: RF-SHAP
# Step 1: Fit RF on FULL X -> mean|SHAP| -> Top 6
# Step 2: Remove highest (top1) from Top 6 -> X_red (5-feature reduced dataset)
# Step 3: Re-fit RF on X_red -> mean|SHAP| -> Top 5 from X_red
# Step 4: CV on Top 6 (from full X)
# ==============================================================================

# Step 1: fit on full X, compute mean|SHAP|, select Top 6
rf_shap_full        = RandomForestRegressor(random_state=42)
rf_shap_full.fit(X, y)
exp_rf_full         = shap.TreeExplainer(rf_shap_full)
sv_rf_full          = exp_rf_full.shap_values(X.iloc[shap_idx])
rf_shap_scores_full = pd.Series(np.abs(sv_rf_full).mean(axis=0),
                                 index=feature_names).sort_values(ascending=False)
top6_rf_shap        = rf_shap_scores_full.iloc[:6].index.tolist()  # Top 6 from full X
top1_rf_shap        = top6_rf_shap[0]                              # highest feature in Top 6

# Step 2: remove top1 from Top 6 -> 5-feature reduced dataset
X_red_rf_shap       = X[top6_rf_shap].drop(columns=[top1_rf_shap]) # X_red has exactly 5 features

# Step 3: re-fit RF on X_red -> mean|SHAP| -> Top 5 from X_red
rf_shap_red         = RandomForestRegressor(random_state=42)
rf_shap_red.fit(X_red_rf_shap, y)                                  # fit ONLY on 5-feature reduced dataset
exp_rf_red          = shap.TreeExplainer(rf_shap_red)
sv_rf_red           = exp_rf_red.shap_values(X_red_rf_shap.iloc[shap_idx])
rf_shap_scores_red  = pd.Series(np.abs(sv_rf_red).mean(axis=0),
                                 index=X_red_rf_shap.columns).sort_values(ascending=False)
top5_rf_shap        = rf_shap_scores_red.iloc[:5].index.tolist()   # Top 5 re-selected from X_red

# Step 4: CV on Top 6 only
cv6_rf_shap = cv_score(RandomForestRegressor(random_state=42), X[top6_rf_shap], y)

print(f"[RF-SHAP]  Top 6 (full X)   : {top6_rf_shap}")
print(f"[RF-SHAP]  Top 1 removed    : {top1_rf_shap}")
print(f"[RF-SHAP]  Top 5 (from X_red, re-fit) : {top5_rf_shap}")
print(f"[RF-SHAP]  CV6 R2           : {cv6_rf_shap}\n")

# ==============================================================================
# METHOD 3: XGBoost
# Step 1: Fit XGB on FULL X -> rank all features -> Top 6
# Step 2: Remove highest (top1) from Top 6 -> X_red (5-feature reduced dataset)
# Step 3: Re-fit XGB on X_red -> re-rank -> Top 5 from X_red
# Step 4: CV on Top 6 (from full X)
# ==============================================================================

# Step 1: fit on full X, select Top 6
xgb_full        = XGBRegressor(random_state=42)
xgb_full.fit(X, y)
xgb_scores_full = pd.Series(xgb_full.feature_importances_,
                             index=feature_names).sort_values(ascending=False)
top6_xgb        = xgb_scores_full.iloc[:6].index.tolist()  # Top 6 from full X
top1_xgb        = top6_xgb[0]                              # highest feature in Top 6

# Step 2: remove top1 from Top 6 -> 5-feature reduced dataset
X_red_xgb       = X[top6_xgb].drop(columns=[top1_xgb])    # X_red has exactly 5 features

# Step 3: re-fit XGB on X_red -> re-rank -> Top 5 from X_red
xgb_red         = XGBRegressor(random_state=42)
xgb_red.fit(X_red_xgb, y)                                  # fit ONLY on 5-feature reduced dataset
xgb_scores_red  = pd.Series(xgb_red.feature_importances_,
                             index=X_red_xgb.columns).sort_values(ascending=False)
top5_xgb        = xgb_scores_red.iloc[:5].index.tolist()   # Top 5 re-selected from X_red

# Step 4: CV on Top 6 only
cv6_xgb = cv_score(XGBRegressor(random_state=42), X[top6_xgb], y)

print(f"[XGB]      Top 6 (full X)   : {top6_xgb}")
print(f"[XGB]      Top 1 removed    : {top1_xgb}")
print(f"[XGB]      Top 5 (from X_red, re-fit) : {top5_xgb}")
print(f"[XGB]      CV6 R2           : {cv6_xgb}\n")

# ==============================================================================
# METHOD 4: XGB-SHAP
# Step 1: Fit XGB on FULL X -> mean|SHAP| -> Top 6
# Step 2: Remove highest (top1) from Top 6 -> X_red (5-feature reduced dataset)
# Step 3: Re-fit XGB on X_red -> mean|SHAP| -> Top 5 from X_red
# Step 4: CV on Top 6 (from full X)
# ==============================================================================

# Step 1: fit on full X, compute mean|SHAP|, select Top 6
xgb_shap_full        = XGBRegressor(random_state=42)
xgb_shap_full.fit(X, y)
exp_xgb_full         = shap.TreeExplainer(xgb_shap_full)
sv_xgb_full          = exp_xgb_full.shap_values(X.iloc[shap_idx])
xgb_shap_scores_full = pd.Series(np.abs(sv_xgb_full).mean(axis=0),
                                  index=feature_names).sort_values(ascending=False)
top6_xgb_shap        = xgb_shap_scores_full.iloc[:6].index.tolist()  # Top 6 from full X
top1_xgb_shap        = top6_xgb_shap[0]                              # highest feature in Top 6

# Step 2: remove top1 from Top 6 -> 5-feature reduced dataset
X_red_xgb_shap       = X[top6_xgb_shap].drop(columns=[top1_xgb_shap]) # X_red has exactly 5 features

# Step 3: re-fit XGB on X_red -> mean|SHAP| -> Top 5 from X_red
xgb_shap_red         = XGBRegressor(random_state=42)
xgb_shap_red.fit(X_red_xgb_shap, y)                                   # fit ONLY on 5-feature reduced dataset
exp_xgb_red          = shap.TreeExplainer(xgb_shap_red)
sv_xgb_red           = exp_xgb_red.shap_values(X_red_xgb_shap.iloc[shap_idx])
xgb_shap_scores_red  = pd.Series(np.abs(sv_xgb_red).mean(axis=0),
                                  index=X_red_xgb_shap.columns).sort_values(ascending=False)
top5_xgb_shap        = xgb_shap_scores_red.iloc[:5].index.tolist()    # Top 5 re-selected from X_red

# Step 4: CV on Top 6 only
cv6_xgb_shap = cv_score(XGBRegressor(random_state=42), X[top6_xgb_shap], y)

print(f"[XGB-SHAP] Top 6 (full X)   : {top6_xgb_shap}")
print(f"[XGB-SHAP] Top 1 removed    : {top1_xgb_shap}")
print(f"[XGB-SHAP] Top 5 (from X_red, re-fit) : {top5_xgb_shap}")
print(f"[XGB-SHAP] CV6 R2           : {cv6_xgb_shap}\n")

# ==============================================================================
# METHOD 5: Feature Agglomeration  (unsupervised, NO re-fit)
# Step 1: Fit FA on FULL X -> proximity scores -> Top 6
# Step 2: Remove highest (top1) from Top 6 scores (NO re-fit of FA)
# Step 3: Re-rank remaining 5 scores -> Top 5
# Step 4: CV on Top 6 (from full X)
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
top6_fa        = fa_scores_full.iloc[:6].index.tolist()   # Top 6 from full X
top1_fa        = top6_fa[0]                               # highest feature in Top 6

# Step 2: remove top1 from Top 6 scores, no re-fit (FA is unsupervised)
fa_scores_top6 = fa_scores_full[top6_fa]                  # scores of Top 6 only
fa_scores_red  = fa_scores_top6.drop(index=top1_fa)       # remove top1 -> 5 scores remain

# Step 3: re-rank remaining 5 scores -> Top 5
top5_fa        = fa_scores_red.iloc[:5].index.tolist()    # Top 5 re-ranked from reduced scores

# Step 4: CV on Top 6 only
cv6_fa = cv_score(RandomForestRegressor(random_state=42), X[top6_fa], y)

print(f"[FA]       Top 6 (full X)   : {top6_fa}")
print(f"[FA]       Top 1 removed    : {top1_fa}")
print(f"[FA]       Top 5 (re-ranked): {top5_fa}")
print(f"[FA]       CV6 R2           : {cv6_fa}\n")

# ==============================================================================
# METHOD 6: HVGS
# Step 1: Variance on FULL X -> Top 6
# Step 2: Remove highest (top1) from Top 6 -> X_red (5-feature reduced dataset)
# Step 3: Recompute variance on X_red -> Top 5 from X_red
# Step 4: CV on Top 6 (from full X)
# ==============================================================================

def hvgs_scores(X_in):
    return X_in.var().sort_values(ascending=False)

# Step 1
hvgs_scores_full = hvgs_scores(X)
top6_hvgs        = hvgs_scores_full.iloc[:6].index.tolist()  # Top 6 from full X
top1_hvgs        = top6_hvgs[0]                              # highest feature in Top 6

# Step 2: remove top1 from Top 6 -> 5-feature reduced dataset
X_red_hvgs       = X[top6_hvgs].drop(columns=[top1_hvgs])   # X_red has exactly 5 features

# Step 3: recompute variance on X_red -> Top 5 from X_red
hvgs_scores_red  = hvgs_scores(X_red_hvgs)
top5_hvgs        = hvgs_scores_red.iloc[:5].index.tolist()   # Top 5 re-selected from X_red

# Step 4: CV on Top 6 only
cv6_hvgs = cv_score(RandomForestRegressor(random_state=42), X[top6_hvgs], y)

print(f"[HVGS]     Top 6 (full X)   : {top6_hvgs}")
print(f"[HVGS]     Top 1 removed    : {top1_hvgs}")
print(f"[HVGS]     Top 5 (from X_red, recomputed) : {top5_hvgs}")
print(f"[HVGS]     CV6 R2           : {cv6_hvgs}\n")

# ==============================================================================
# METHOD 7: Spearman Correlation
# Step 1: |Spearman rho| on FULL X -> Top 6
# Step 2: Remove highest (top1) from Top 6 -> X_red (5-feature reduced dataset)
# Step 3: Recompute |Spearman rho| on X_red -> Top 5 from X_red
# Step 4: CV on Top 6 (from full X)
# ==============================================================================

def spearman_scores(X_in, y):
    return pd.Series(
        [abs(spearmanr(X_in[f], y).statistic) for f in X_in.columns],
        index=X_in.columns
    ).sort_values(ascending=False)

# Step 1
sp_scores_full = spearman_scores(X, y)
top6_sp        = sp_scores_full.iloc[:6].index.tolist()   # Top 6 from full X
top1_sp        = top6_sp[0]                               # highest feature in Top 6

# Step 2: remove top1 from Top 6 -> 5-feature reduced dataset
X_red_sp       = X[top6_sp].drop(columns=[top1_sp])       # X_red has exactly 5 features

# Step 3: recompute |Spearman rho| on X_red -> Top 5 from X_red
sp_scores_red  = spearman_scores(X_red_sp, y)
top5_sp        = sp_scores_red.iloc[:5].index.tolist()    # Top 5 re-selected from X_red

# Step 4: CV on Top 6 only
cv6_sp = cv_score(RandomForestRegressor(random_state=42), X[top6_sp], y)

print(f"[Spearman] Top 6 (full X)   : {top6_sp}")
print(f"[Spearman] Top 1 removed    : {top1_sp}")
print(f"[Spearman] Top 5 (from X_red, recomputed) : {top5_sp}")
print(f"[Spearman] CV6 R2           : {cv6_sp}\n")

# ==============================================================================
# Summary table: result.csv
# ==============================================================================
results = pd.DataFrame([
    {
        'Method'       : 'Random Forest',
        'CV Evaluator' : 'RandomForestRegressor',
        'Top6 Features': ', '.join(top6_rf),
        'CV6 R2'       : cv6_rf,
        'Top1 Removed' : top1_rf,
        'Top5 Features': ', '.join(top5_rf),
    },
    {
        'Method'       : 'RF-SHAP',
        'CV Evaluator' : 'RandomForestRegressor',
        'Top6 Features': ', '.join(top6_rf_shap),
        'CV6 R2'       : cv6_rf_shap,
        'Top1 Removed' : top1_rf_shap,
        'Top5 Features': ', '.join(top5_rf_shap),
    },
    {
        'Method'       : 'XGBoost',
        'CV Evaluator' : 'XGBRegressor',
        'Top6 Features': ', '.join(top6_xgb),
        'CV6 R2'       : cv6_xgb,
        'Top1 Removed' : top1_xgb,
        'Top5 Features': ', '.join(top5_xgb),
    },
    {
        'Method'       : 'XGB-SHAP',
        'CV Evaluator' : 'XGBRegressor',
        'Top6 Features': ', '.join(top6_xgb_shap),
        'CV6 R2'       : cv6_xgb_shap,
        'Top1 Removed' : top1_xgb_shap,
        'Top5 Features': ', '.join(top5_xgb_shap),
    },
    {
        'Method'       : 'Feature Agglomeration',
        'CV Evaluator' : 'RandomForestRegressor',
        'Top6 Features': ', '.join(top6_fa),
        'CV6 R2'       : cv6_fa,
        'Top1 Removed' : top1_fa,
        'Top5 Features': ', '.join(top5_fa),
    },
    {
        'Method'       : 'HVGS',
        'CV Evaluator' : 'RandomForestRegressor',
        'Top6 Features': ', '.join(top6_hvgs),
        'CV6 R2'       : cv6_hvgs,
        'Top1 Removed' : top1_hvgs,
        'Top5 Features': ', '.join(top5_hvgs),
    },
    {
        'Method'       : 'Spearman',
        'CV Evaluator' : 'RandomForestRegressor',
        'Top6 Features': ', '.join(top6_sp),
        'CV6 R2'       : cv6_sp,
        'Top1 Removed' : top1_sp,
        'Top5 Features': ', '.join(top5_sp),
    },
])

results.to_csv('result.csv', index=False)
print(results.to_string(index=False))
