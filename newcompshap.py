import warnings
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')

N_REPEATS = 10
N_SPLITS = 5
BASE_SEED = 42
N_SHAP_SAMPLES = 100

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
df = pd.read_csv('organic_material_compatibility_dataset.csv')

X = df.drop('compatibility_label', axis=1).reset_index(drop=True)
y = df['compatibility_label'].reset_index(drop=True)

print("Dataset Shape:", df.shape)
print("\nTarget Distribution:")
print(y.value_counts())
print("\nTarget Distribution (proportions):")
print(y.value_counts(normalize=True))
print("\n" + "=" * 80 + "\n")


# ------------------------------------------------------------------
# 2. Model factory (no scaling for any algorithm)
# ------------------------------------------------------------------
def make_model(model_type, seed):
    if model_type == 'rf':
        return RandomForestClassifier(random_state=seed, n_jobs=-1)
    if model_type == 'xgb':
        return XGBClassifier(random_state=seed, eval_metric='logloss', n_jobs=-1)
    raise ValueError(model_type)


# ------------------------------------------------------------------
# 3. SHAP helper (handles list / 2D / 3D outputs)
# ------------------------------------------------------------------
def mean_abs_shap(model, X_sample):
    sv = shap.TreeExplainer(model).shap_values(X_sample)
    if isinstance(sv, list):
        arr = np.abs(sv[1]) if len(sv) > 1 else np.abs(sv[0])
    else:
        arr = np.abs(sv)
    if arr.ndim > 2:
        arr = arr[:, :, -1]
    out = arr.mean(axis=0)
    if out.ndim > 1:
        out = out.flatten()[:X_sample.shape[1]]
    return out


# ------------------------------------------------------------------
# 4. Feature-ranking functions (receive TRAINING data only)
# ------------------------------------------------------------------
def rank_rf(Xtr, ytr, seed):
    m = make_model('rf', seed).fit(Xtr, ytr)
    return list(pd.Series(m.feature_importances_, index=Xtr.columns)
                .sort_values(ascending=False).index)


def rank_xgb(Xtr, ytr, seed):
    m = make_model('xgb', seed).fit(Xtr, ytr)
    return list(pd.Series(m.feature_importances_, index=Xtr.columns)
                .sort_values(ascending=False).index)


def _shap_rank(model_type, Xtr, ytr, seed):
    m = make_model(model_type, seed).fit(Xtr, ytr)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xtr), size=min(N_SHAP_SAMPLES, len(Xtr)), replace=False)
    imp = mean_abs_shap(m, Xtr.iloc[idx])
    return list(pd.Series(imp, index=Xtr.columns).sort_values(ascending=False).index)


def rank_rf_shap(Xtr, ytr, seed):
    return _shap_rank('rf', Xtr, ytr, seed)


def rank_xgb_shap(Xtr, ytr, seed):
    return _shap_rank('xgb', Xtr, ytr, seed)


def rank_fa(Xtr, ytr, seed):
    """Original FA scoring: 0.9*variance + 0.1*(1/cluster size), train fold only."""
    n_clusters = max(2, min(5, Xtr.shape[1] - 1))
    fa = FeatureAgglomeration(n_clusters=n_clusters).fit(Xtr)
    labels = fa.labels_
    variances = Xtr.var()
    sizes = pd.Series(labels).value_counts()
    scores = []
    for i, col in enumerate(Xtr.columns):
        dist_score = 1.0 / sizes[labels[i]]
        scores.append(0.9 * variances[col] + 0.1 * dist_score)
    return list(pd.Series(scores, index=Xtr.columns)
                .sort_values(ascending=False).index)


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
    'RF':           (rank_rf,        'rf'),
    'XGBoost':      (rank_xgb,       'xgb'),
    'RF_SHAP':      (rank_rf_shap,   'rf'),
    'XGBoost_SHAP': (rank_xgb_shap,  'xgb'),
    'FA':           (rank_fa,        'rf'),
    'HVGS':         (rank_hvgs,      'rf'),
    'Spearman':     (rank_spearman,  'rf'),
}


# ------------------------------------------------------------------
# 5. Mean and 95% CI (t-distribution) over repeat-level scores
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
# 6. Repeated stratified K-fold, leakage-free
#    split -> rank features on train -> fit on train -> score on test
# ------------------------------------------------------------------
def run_method(name, rank_fn, eval_model):
    repeat_acc_top5, repeat_acc_top4 = [], []
    top5_counter, top4_counter = Counter(), Counter()
    n_folds_total = 0

    for rep in range(N_REPEATS):
        seed = BASE_SEED + rep
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        fold_acc5, fold_acc4 = [], []

        for tr_idx, te_idx in skf.split(X, y):
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

            # Top 5: ranking uses training fold ONLY
            ranked = rank_fn(Xtr, ytr, seed)
            top5 = ranked[:5]

            # Top 4 after removing best feature: training fold ONLY
            ranked_red = rank_fn(Xtr.drop(columns=[top5[0]]), ytr, seed)
            top4 = ranked_red[:4]

            # Evaluate both subsets on untouched test fold
            m5 = make_model(eval_model, seed).fit(Xtr[top5], ytr)
            fold_acc5.append(accuracy_score(yte, m5.predict(Xte[top5])))

            m4 = make_model(eval_model, seed).fit(Xtr[top4], ytr)
            fold_acc4.append(accuracy_score(yte, m4.predict(Xte[top4])))

            top5_counter.update(top5)
            top4_counter.update(top4)
            n_folds_total += 1

        repeat_acc_top5.append(np.mean(fold_acc5))
        repeat_acc_top4.append(np.mean(fold_acc4))

    m5, lo5, hi5 = mean_ci(repeat_acc_top5)
    m4, lo4, hi4 = mean_ci(repeat_acc_top4)

    def fmt(counter, k):
        return ', '.join(f"{f} ({c}/{n_folds_total})"
                         for f, c in counter.most_common(k))

    return {
        'Method': name,
        'Top 5 Accuracy mean [95% CI]': f"{m5:.3f} [{lo5:.3f}, {hi5:.3f}]",
        'Top 4 Accuracy mean [95% CI]': f"{m4:.3f} [{lo4:.3f}, {hi4:.3f}]",
        'Top 5 features': fmt(top5_counter, 5),
        'Top 4 features (after removing best)': fmt(top4_counter, 4),
    }


# ------------------------------------------------------------------
# 7. Run all methods and save
# ------------------------------------------------------------------
rows = []
for name, (fn, mdl) in METHODS.items():
    print(f"Running {name} ...")
    rows.append(run_method(name, fn, mdl))

summary_df = pd.DataFrame(rows)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 300)

print("\n" + "=" * 80)
print(f"SUMMARY TABLE ({N_REPEATS} repeats x {N_SPLITS}-fold stratified CV, "
      "leakage-free, no scaling)")
print("=" * 80)
print(summary_df.to_string(index=False))

summary_df.to_csv('result.csv', index=False)
print("\nResults saved to 'result.csv'")