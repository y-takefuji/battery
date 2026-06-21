import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import shap
from sklearn.cluster import FeatureAgglomeration
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('organic_material_compatibility_dataset.csv')

# Separate features and target
X = df.drop('compatibility_label', axis=1)
y = df['compatibility_label']

# Show shape and target distribution
print("Dataset Shape:", df.shape)
print("\nTarget Distribution:")
print(y.value_counts())
print("\nTarget Distribution (proportions):")
print(y.value_counts(normalize=True))
print("\n" + "="*80 + "\n")

# Helper function for cross-validation
def evaluate_features(X_subset, y, method_name):
    rf = RandomForestClassifier(random_state=42)
    scores = cross_val_score(rf, X_subset, y, cv=5, scoring='accuracy')
    mean_accuracy = scores.mean()
    print(f"{method_name} - 5-Fold CV Accuracy: {mean_accuracy:.4f}")
    return mean_accuracy

# Storage for results
results = []

# ============================================================================
# 1. Random Forest Feature Importance
# ============================================================================
print("1. Random Forest Feature Importance")
print("-" * 80)

# Fit RF on full dataset
rf_full = RandomForestClassifier(random_state=42)
rf_full.fit(X, y)
feature_importances_rf = pd.Series(rf_full.feature_importances_, index=X.columns)
top5_rf = feature_importances_rf.nlargest(5).index.tolist()
print(f"Top 5 features: {top5_rf}")

# CV with top 5
X_top5_rf = X[top5_rf]
cv_acc_rf = evaluate_features(X_top5_rf, y, "RF (Top 5)")

# Remove highest and refit
highest_rf = top5_rf[0]
X_reduced_rf = X.drop(columns=[highest_rf])
rf_reduced = RandomForestClassifier(random_state=42)
rf_reduced.fit(X_reduced_rf, y)
feature_importances_rf_reduced = pd.Series(rf_reduced.feature_importances_, index=X_reduced_rf.columns)
top4_rf = feature_importances_rf_reduced.nlargest(4).index.tolist()
print(f"Top 4 features (after removing {highest_rf}): {top4_rf}")
print()

results.append({
    'method': 'RF',
    'cv_accuracy': cv_acc_rf,
    'top5_features': ', '.join(top5_rf),
    'top4_features': ', '.join(top4_rf)
})

# ============================================================================
# 2. XGBoost Feature Importance
# ============================================================================
print("2. XGBoost Feature Importance")
print("-" * 80)

# Fit XGBoost on full dataset
xgb_full = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_full.fit(X, y)
feature_importances_xgb = pd.Series(xgb_full.feature_importances_, index=X.columns)
top5_xgb = feature_importances_xgb.nlargest(5).index.tolist()
print(f"Top 5 features: {top5_xgb}")

# CV with top 5
X_top5_xgb = X[top5_xgb]
cv_acc_xgb = evaluate_features(X_top5_xgb, y, "XGBoost (Top 5)")

# Remove highest and refit
highest_xgb = top5_xgb[0]
X_reduced_xgb = X.drop(columns=[highest_xgb])
xgb_reduced = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_reduced.fit(X_reduced_xgb, y)
feature_importances_xgb_reduced = pd.Series(xgb_reduced.feature_importances_, index=X_reduced_xgb.columns)
top4_xgb = feature_importances_xgb_reduced.nlargest(4).index.tolist()
print(f"Top 4 features (after removing {highest_xgb}): {top4_xgb}")
print()

results.append({
    'method': 'XGBoost',
    'cv_accuracy': cv_acc_xgb,
    'top5_features': ', '.join(top5_xgb),
    'top4_features': ', '.join(top4_xgb)
})

# ============================================================================
# 3. Random Forest with SHAP
# ============================================================================
print("3. Random Forest with SHAP")
print("-" * 80)

# Fit RF and compute SHAP values on 100 random instances
rf_shap = RandomForestClassifier(random_state=42)
rf_shap.fit(X, y)

# Select 100 random instances
np.random.seed(42)
sample_indices = np.random.choice(X.shape[0], size=min(100, X.shape[0]), replace=False)
X_sample = X.iloc[sample_indices]

# Compute SHAP values
explainer_rf = shap.TreeExplainer(rf_shap)
shap_values_rf = explainer_rf.shap_values(X_sample)

# For binary classification, handle SHAP output
if isinstance(shap_values_rf, list):
    # If list, use class 1 (positive class)
    shap_values_rf_array = np.abs(shap_values_rf[1])
else:
    # If single array, use it directly
    shap_values_rf_array = np.abs(shap_values_rf)

# Handle multi-dimensional output
if shap_values_rf_array.ndim > 2:
    # If 3D, take the last dimension (class 1)
    shap_values_rf_array = shap_values_rf_array[:, :, -1]

# Average absolute SHAP values across samples
mean_shap_rf = np.mean(shap_values_rf_array, axis=0)

# Ensure 1D array
if mean_shap_rf.ndim > 1:
    mean_shap_rf = mean_shap_rf.flatten()[:len(X.columns)]

shap_importance_rf = pd.Series(mean_shap_rf, index=X.columns)
top5_rf_shap = shap_importance_rf.nlargest(5).index.tolist()
print(f"Top 5 features: {top5_rf_shap}")

# CV with top 5
X_top5_rf_shap = X[top5_rf_shap]
cv_acc_rf_shap = evaluate_features(X_top5_rf_shap, y, "RF-SHAP (Top 5)")

# Remove highest and refit
highest_rf_shap = top5_rf_shap[0]
X_reduced_rf_shap = X.drop(columns=[highest_rf_shap])
rf_shap_reduced = RandomForestClassifier(random_state=42)
rf_shap_reduced.fit(X_reduced_rf_shap, y)

# Recompute SHAP on reduced dataset
X_sample_reduced_rf = X_reduced_rf_shap.iloc[sample_indices]
explainer_rf_reduced = shap.TreeExplainer(rf_shap_reduced)
shap_values_rf_reduced = explainer_rf_reduced.shap_values(X_sample_reduced_rf)

if isinstance(shap_values_rf_reduced, list):
    shap_values_rf_reduced_array = np.abs(shap_values_rf_reduced[1])
else:
    shap_values_rf_reduced_array = np.abs(shap_values_rf_reduced)

# Handle multi-dimensional output
if shap_values_rf_reduced_array.ndim > 2:
    shap_values_rf_reduced_array = shap_values_rf_reduced_array[:, :, -1]

mean_shap_rf_reduced = np.mean(shap_values_rf_reduced_array, axis=0)

# Ensure 1D array
if mean_shap_rf_reduced.ndim > 1:
    mean_shap_rf_reduced = mean_shap_rf_reduced.flatten()[:len(X_reduced_rf_shap.columns)]

shap_importance_rf_reduced = pd.Series(mean_shap_rf_reduced, index=X_reduced_rf_shap.columns)
top4_rf_shap = shap_importance_rf_reduced.nlargest(4).index.tolist()
print(f"Top 4 features (after removing {highest_rf_shap}): {top4_rf_shap}")
print()

results.append({
    'method': 'RF_SHAP',
    'cv_accuracy': cv_acc_rf_shap,
    'top5_features': ', '.join(top5_rf_shap),
    'top4_features': ', '.join(top4_rf_shap)
})

# ============================================================================
# 4. XGBoost with SHAP
# ============================================================================
print("4. XGBoost with SHAP")
print("-" * 80)

# Fit XGBoost and compute SHAP values on 100 random instances
xgb_shap = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_shap.fit(X, y)

# Compute SHAP values
explainer_xgb = shap.TreeExplainer(xgb_shap)
shap_values_xgb = explainer_xgb.shap_values(X_sample)

# Handle multi-dimensional output
shap_values_xgb_abs = np.abs(shap_values_xgb)
if shap_values_xgb_abs.ndim > 2:
    shap_values_xgb_abs = shap_values_xgb_abs[:, :, -1]

# Average absolute SHAP values
mean_shap_xgb = np.mean(shap_values_xgb_abs, axis=0)

# Ensure 1D array
if mean_shap_xgb.ndim > 1:
    mean_shap_xgb = mean_shap_xgb.flatten()[:len(X.columns)]

shap_importance_xgb = pd.Series(mean_shap_xgb, index=X.columns)
top5_xgb_shap = shap_importance_xgb.nlargest(5).index.tolist()
print(f"Top 5 features: {top5_xgb_shap}")

# CV with top 5
X_top5_xgb_shap = X[top5_xgb_shap]
cv_acc_xgb_shap = evaluate_features(X_top5_xgb_shap, y, "XGBoost-SHAP (Top 5)")

# Remove highest and refit
highest_xgb_shap = top5_xgb_shap[0]
X_reduced_xgb_shap = X.drop(columns=[highest_xgb_shap])
xgb_shap_reduced = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_shap_reduced.fit(X_reduced_xgb_shap, y)

# Recompute SHAP on reduced dataset
X_sample_reduced_xgb = X_reduced_xgb_shap.iloc[sample_indices]
explainer_xgb_reduced = shap.TreeExplainer(xgb_shap_reduced)
shap_values_xgb_reduced = explainer_xgb_reduced.shap_values(X_sample_reduced_xgb)

# Handle multi-dimensional output
shap_values_xgb_reduced_abs = np.abs(shap_values_xgb_reduced)
if shap_values_xgb_reduced_abs.ndim > 2:
    shap_values_xgb_reduced_abs = shap_values_xgb_reduced_abs[:, :, -1]

mean_shap_xgb_reduced = np.mean(shap_values_xgb_reduced_abs, axis=0)

# Ensure 1D array
if mean_shap_xgb_reduced.ndim > 1:
    mean_shap_xgb_reduced = mean_shap_xgb_reduced.flatten()[:len(X_reduced_xgb_shap.columns)]

shap_importance_xgb_reduced = pd.Series(mean_shap_xgb_reduced, index=X_reduced_xgb_shap.columns)
top4_xgb_shap = shap_importance_xgb_reduced.nlargest(4).index.tolist()
print(f"Top 4 features (after removing {highest_xgb_shap}): {top4_xgb_shap}")
print()

results.append({
    'method': 'XGBoost_SHAP',
    'cv_accuracy': cv_acc_xgb_shap,
    'top5_features': ', '.join(top5_xgb_shap),
    'top4_features': ', '.join(top4_xgb_shap)
})

# ============================================================================
# 5. Feature Agglomeration (FA)
# ============================================================================
print("5. Feature Agglomeration")
print("-" * 80)

# Apply Feature Agglomeration
fa = FeatureAgglomeration(n_clusters=5)
fa.fit(X)

# Get cluster labels for each feature
cluster_labels = fa.labels_

# Calculate variance for each feature
feature_variances = X.var()

# Calculate cluster distance score (inverse of cluster size as proxy)
cluster_sizes = pd.Series(cluster_labels).value_counts()
cluster_distance_scores = {}
for cluster_id in range(5):
    cluster_distance_scores[cluster_id] = 1.0 / cluster_sizes[cluster_id]

# Calculate combined score for each feature: 0.9 * variance + 0.1 * cluster_distance
feature_scores = []
for idx, col in enumerate(X.columns):
    cluster_id = cluster_labels[idx]
    variance_score = feature_variances[col]
    distance_score = cluster_distance_scores[cluster_id]
    combined_score = 0.9 * variance_score + 0.1 * distance_score
    feature_scores.append(combined_score)

# Create Series with combined scores
fa_importance = pd.Series(feature_scores, index=X.columns)

# Select top 5 features across all clusters
top5_fa = fa_importance.nlargest(5).index.tolist()
print(f"Top 5 features: {top5_fa}")

# CV with top 5
X_top5_fa = X[top5_fa]
cv_acc_fa = evaluate_features(X_top5_fa, y, "FA (Top 5)")

# Remove highest and refit
highest_fa = top5_fa[0]
X_reduced_fa = X.drop(columns=[highest_fa])

# Reapply FA on reduced dataset
fa_reduced = FeatureAgglomeration(n_clusters=4)
fa_reduced.fit(X_reduced_fa)

# Get cluster labels for reduced dataset
cluster_labels_reduced = fa_reduced.labels_

# Calculate variance for reduced features
feature_variances_reduced = X_reduced_fa.var()

# Calculate cluster distance score for reduced dataset
cluster_sizes_reduced = pd.Series(cluster_labels_reduced).value_counts()
cluster_distance_scores_reduced = {}
for cluster_id in range(4):
    cluster_distance_scores_reduced[cluster_id] = 1.0 / cluster_sizes_reduced[cluster_id]

# Calculate combined score for reduced features
feature_scores_reduced = []
for idx, col in enumerate(X_reduced_fa.columns):
    cluster_id = cluster_labels_reduced[idx]
    variance_score = feature_variances_reduced[col]
    distance_score = cluster_distance_scores_reduced[cluster_id]
    combined_score = 0.9 * variance_score + 0.1 * distance_score
    feature_scores_reduced.append(combined_score)

# Create Series with combined scores
fa_importance_reduced = pd.Series(feature_scores_reduced, index=X_reduced_fa.columns)

# Select top 4 features across all clusters
top4_fa = fa_importance_reduced.nlargest(4).index.tolist()
print(f"Top 4 features (after removing {highest_fa}): {top4_fa}")
print()

results.append({
    'method': 'FA',
    'cv_accuracy': cv_acc_fa,
    'top5_features': ', '.join(top5_fa),
    'top4_features': ', '.join(top4_fa)
})

# ============================================================================
# 6. Highly Variable Gene Selection (HVGS)
# ============================================================================
print("6. Highly Variable Gene Selection (HVGS)")
print("-" * 80)

# Calculate variance for each feature
feature_variances = X.var()
top5_hvgs = feature_variances.nlargest(5).index.tolist()
print(f"Top 5 features: {top5_hvgs}")

# CV with top 5 using RF
X_top5_hvgs = X[top5_hvgs]
cv_acc_hvgs = evaluate_features(X_top5_hvgs, y, "HVGS (Top 5)")

# Remove highest variance feature and reselect
highest_hvgs = top5_hvgs[0]
X_reduced_hvgs = X.drop(columns=[highest_hvgs])
feature_variances_reduced = X_reduced_hvgs.var()
top4_hvgs = feature_variances_reduced.nlargest(4).index.tolist()
print(f"Top 4 features (after removing {highest_hvgs}): {top4_hvgs}")
print()

results.append({
    'method': 'HVGS',
    'cv_accuracy': cv_acc_hvgs,
    'top5_features': ', '.join(top5_hvgs),
    'top4_features': ', '.join(top4_hvgs)
})

# ============================================================================
# 7. Spearman Correlation
# ============================================================================
print("7. Spearman Correlation")
print("-" * 80)

# Calculate Spearman correlation with target
spearman_scores = []
for col in X.columns:
    corr, _ = spearmanr(X[col], y)
    spearman_scores.append(abs(corr))

spearman_importance = pd.Series(spearman_scores, index=X.columns)
top5_spearman = spearman_importance.nlargest(5).index.tolist()
print(f"Top 5 features: {top5_spearman}")

# CV with top 5 using RF
X_top5_spearman = X[top5_spearman]
cv_acc_spearman = evaluate_features(X_top5_spearman, y, "Spearman (Top 5)")

# Remove highest and reselect
highest_spearman = top5_spearman[0]
X_reduced_spearman = X.drop(columns=[highest_spearman])

spearman_scores_reduced = []
for col in X_reduced_spearman.columns:
    corr, _ = spearmanr(X_reduced_spearman[col], y)
    spearman_scores_reduced.append(abs(corr))

spearman_importance_reduced = pd.Series(spearman_scores_reduced, index=X_reduced_spearman.columns)
top4_spearman = spearman_importance_reduced.nlargest(4).index.tolist()
print(f"Top 4 features (after removing {highest_spearman}): {top4_spearman}")
print()

results.append({
    'method': 'Spearman',
    'cv_accuracy': cv_acc_spearman,
    'top5_features': ', '.join(top5_spearman),
    'top4_features': ', '.join(top4_spearman)
})

# ============================================================================
# Create Summary Table and Save
# ============================================================================
print("="*80)
print("SUMMARY TABLE")
print("="*80)

results_df = pd.DataFrame(results)

# Format cv_accuracy to 4 significant digits
results_df['cv_accuracy'] = results_df['cv_accuracy'].apply(lambda x: float(f'{x:.4g}'))

print(results_df.to_string(index=False))
print()

# Save to CSV
results_df.to_csv('result.csv', index=False)
print("Results saved to 'result.csv'")