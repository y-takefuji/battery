import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from collections import Counter
from scipy import stats
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

N_REPEATS = 10
N_SPLITS = 5
BASE_SEED = 42

# ------------------------------------------------------------------
# 1. Load RAW data only (no encoding, no fitting of any transformer)
# ------------------------------------------------------------------
file_path = 'pnas.2214357120.sd01.xlsx'
raw_data = pd.read_excel(file_path, sheet_name='Dataset', header=[0, 1])

columns = []
for col in raw_data.columns:
    clean_col = [str(x) for x in col if pd.notna(x)]
    columns.append(' '.join(clean_col).strip())

df = pd.DataFrame(raw_data.values, columns=columns)

target_col = [c for c in df.columns if 'CE (%)' in c]
if not target_col:
    raise ValueError("Target variable 'CE (%)' not found in the dataset")
target_col = target_col[0]
print(f"Target variable column identified: {target_col}")

# Constant-value fill (uses no data statistics, so no leakage)
df = df.fillna(0)

y = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(float).reset_index(drop=True)
X_raw = df.drop(columns=[target_col]).reset_index(drop=True)

print("\nDataset shape:", df.shape)
print(y.describe())

# Column type detection (based on content type only, not target or fold statistics)
categorical_cols = []
for col in X_raw.columns:
    if X_raw[col].dtype == 'object' or X_raw[col].astype(str).str.contains('[a-zA-Z]').any():
        categorical_cols.append(col)
print(f"Categorical columns: {categorical_cols}")


# ------------------------------------------------------------------
# 2. Fold-wise preprocessing: fit on TRAIN only, apply to TEST
# ------------------------------------------------------------------
def fit_encoders(X_train):
    maps = {}
    for col in categorical_cols:
        cats = sorted(X_train[col].astype(str).unique())
        maps[col] = {c: i for i, c in enumerate(cats)}
    return maps


def apply_preprocessing(X_part, maps):
    out = pd.DataFrame(index=X_part.index)
    for col in X_raw.columns:
        if col in categorical_cols:
            out[col] = X_part[col].astype(str).map(maps[col]).fillna(-1).astype(float)
        else:
            out[col] = pd.to_numeric(X_part[col], errors='coerce').fillna(0).astype(float)
    return out


# ------------------------------------------------------------------
# 3. Model factory (no scaling for any algorithm)
# ------------------------------------------------------------------
def make_model(model_type, seed):
    if model_type == 'rf':
        return RandomForestRegressor(random_state=seed, n_jobs=-1)
    if model_type == 'xgb':
        return xgb.XGBRegressor(random_state=seed, n_jobs=-1)
    if model_type == 'mlp':
        return MLPRegressor(max_iter=1000, random_state=seed)
    raise ValueError(model_type)


# ------------------------------------------------------------------
# 4. Feature-ranking functions (receive TRAINING data only)
# ------------------------------------------------------------------
def rank_mlp(Xtr, ytr, seed):
    model = make_model('mlp', seed).fit(Xtr, ytr)
    rng = np.random.RandomState(seed)
    base = model.predict(Xtr)
    imp = []
    for c in Xtr.columns:
        Xp = Xtr.copy()
        Xp[c] = rng.permutation(Xp[c].values)
        imp.append(np.mean(np.abs(base - model.predict(Xp))))
    return list(pd.Series(imp, index=Xtr.columns).sort_values(ascending=False).index)


def rank_rf(Xtr, ytr, seed):
    m = make_model('rf', seed).fit(Xtr, ytr)
    return list(pd.Series(m.feature_importances_, index=Xtr.columns)
                .sort_values(ascending=False).index)


def rank_xgb(Xtr, ytr, seed):
    m = make_model('xgb', seed).fit(Xtr, ytr)
    return list(pd.Series(m.feature_importances_, index=Xtr.columns)
                .sort_values(ascending=False).index)


def rank_rf_shap(Xtr, ytr, seed):
    m = make_model('rf', seed).fit(Xtr, ytr)
    sv = shap.TreeExplainer(m).shap_values(Xtr)
    return list(pd.Series(np.abs(sv).mean(axis=0), index=Xtr.columns)
                .sort_values(ascending=False).index)


def rank_xgb_shap(Xtr, ytr, seed):
    m = make_model('xgb', seed).fit(Xtr, ytr)
    sv = shap.TreeExplainer(m).shap_values(Xtr)
    return list(pd.Series(np.abs(sv).mean(axis=0), index=Xtr.columns)
                .sort_values(ascending=False).index)


def rank_fa(Xtr, ytr, seed):
    n_clusters = max(2, min(Xtr.shape[1] - 1, 15))
    fa = FeatureAgglomeration(n_clusters=n_clusters).fit(Xtr)
    selected = []
    for cid in np.unique(fa.labels_):
        feats = Xtr.columns[fa.labels_ == cid]
        best_feat, best_corr = None, -1
        for f in feats:
            v = Xtr[f].values
            if np.std(v) == 0:
                corr = 0
            else:
                corr = abs(np.corrcoef(v, ytr.values)[0, 1])
                if np.isnan(corr):
                    corr = 0
            if corr > best_corr:
                best_corr, best_feat = corr, f
        selected.append((best_feat, best_corr))
    selected.sort(key=lambda t: t[1], reverse=True)
    ranked = [f for f, _ in selected]
    rest = [c for c in Xtr.columns if c not in ranked]
    return ranked + rest


def rank_hvgs(Xtr, ytr, seed):
    return list(Xtr.var().sort_values(ascending=False).index)


def rank_spearman(Xtr, ytr, seed):
    vals = {}
    for c in Xtr.columns:
        if Xtr[c].nunique() <= 1:
            vals[c] = 0
        else:
            r, _ = spearmanr(Xtr[c], ytr)
            vals[c] = 0 if np.isnan(r) else abs(r)
    return list(pd.Series(vals).sort_values(ascending=False).index)


METHODS = {
    # name: (ranking function, evaluation model)
    'MLP':          (rank_mlp,        'mlp'),
    'RF':           (rank_rf,         'rf'),
    'XGBoost':      (rank_xgb,        'xgb'),
    'RF_SHAP':      (rank_rf_shap,    'rf'),
    'XGBoost_SHAP': (rank_xgb_shap,   'xgb'),
    'FA':           (rank_fa,         'rf'),
    'HVGS':         (rank_hvgs,       'rf'),
    'Spearman':     (rank_spearman,   'rf'),
}


# ------------------------------------------------------------------
# 5. Helper: mean and 95% CI (t-distribution) over repeat-level scores
# ------------------------------------------------------------------
def mean_ci(scores):
    scores = np.asarray(scores, dtype=float)
    mean = scores.mean()
    sem = stats.sem(scores)
    if sem == 0 or np.isnan(sem):
        return mean, mean, mean
    lo, hi = stats.t.interval(0.95, df=len(scores) - 1, loc=mean, scale=sem)
    return mean, lo, hi


# ------------------------------------------------------------------
# 6. Repeated K-fold, leakage-free
#    split RAW data -> fit preprocessing on train -> rank features on train
#    -> fit model on train -> score on test (top-5 AND top-4 subsets)
# ------------------------------------------------------------------
def run_method(name, rank_fn, eval_model):
    repeat_scores_top5 = []
    repeat_scores_top4 = []
    top5_counter = Counter()
    top4_counter = Counter()
    n_folds_total = 0

    for rep in range(N_REPEATS):
        seed = BASE_SEED + rep
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        fold_scores_top5 = []
        fold_scores_top4 = []

        for tr_idx, te_idx in kf.split(X_raw):
            # Split RAW data first
            Xtr_raw, Xte_raw = X_raw.iloc[tr_idx], X_raw.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

            # Fit preprocessing on TRAIN only, apply to both
            maps = fit_encoders(Xtr_raw)
            Xtr = apply_preprocessing(Xtr_raw, maps)
            Xte = apply_preprocessing(Xte_raw, maps)

            # Top 5: ranking uses training fold ONLY
            ranked = rank_fn(Xtr, ytr, seed)
            top5 = ranked[:5]

            # Top 4 after removing best feature: training fold ONLY
            ranked_red = rank_fn(Xtr.drop(columns=[top5[0]]), ytr, seed)
            top4 = ranked_red[:4]

            # Evaluate top-5 subset on untouched test fold
            model5 = make_model(eval_model, seed).fit(Xtr[top5], ytr)
            fold_scores_top5.append(r2_score(yte, model5.predict(Xte[top5])))

            # Evaluate top-4 subset on untouched test fold
            model4 = make_model(eval_model, seed).fit(Xtr[top4], ytr)
            fold_scores_top4.append(r2_score(yte, model4.predict(Xte[top4])))

            top5_counter.update(top5)
            top4_counter.update(top4)
            n_folds_total += 1

        repeat_scores_top5.append(np.mean(fold_scores_top5))
        repeat_scores_top4.append(np.mean(fold_scores_top4))

    m5, lo5, hi5 = mean_ci(repeat_scores_top5)
    m4, lo4, hi4 = mean_ci(repeat_scores_top4)

    def fmt(counter, k):
        return ', '.join(f"{f} ({c}/{n_folds_total})"
                         for f, c in counter.most_common(k))

    return {
        'Method': name,
        'Top 5 R2 mean [95% CI]': f"{m5:.3f} [{lo5:.3f}, {hi5:.3f}]",
        'Top 4 R2 mean [95% CI]': f"{m4:.3f} [{lo4:.3f}, {hi4:.3f}]",
        'Top 5 features': fmt(top5_counter, 5),
        'Top 4 features (after removing best)': fmt(top4_counter, 4),
    }


rows = []
for name, (fn, mdl) in METHODS.items():
    print(f"Running {name} ...")
    rows.append(run_method(name, fn, mdl))

summary_df = pd.DataFrame(rows)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 300)
print("\nFeature Selection Results Summary "
      f"({N_REPEATS} repeats x {N_SPLITS}-fold CV, leakage-free, no scaling):")
print(summary_df.to_string(index=False))

summary_df.to_csv('result.csv', index=False)
print("\nResults saved to result.csv")