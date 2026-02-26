import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import FeatureAgglomeration
import shap
from scipy.stats import spearmanr

# Load the dataset
file_path = 'pnas.2214357120.sd01.xlsx'
raw_data = pd.read_excel(file_path, sheet_name='Dataset', header=[0, 1])

# Clean column names by combining the multi-index headers
columns = []
for col in raw_data.columns:
    clean_col = [str(x) for x in col if pd.notna(x)]
    columns.append(' '.join(clean_col).strip())

# Create DataFrame with cleaned column names
df = pd.DataFrame(raw_data.values, columns=columns)

# Identify target variable column
target_col = [col for col in df.columns if 'CE (%)' in col]
if target_col:
    target_col = target_col[0]
    print(f"Target variable column identified: {target_col}")
else:
    print("Target variable 'CE (%)' not found in the dataset")

# Fill missing values with 0
df = df.fillna(0)

# Display the shape of the dataset
print("\nDataset shape:")
print(df.shape)

# Display basic statistics of the target variable
print("\nBasic statistics of the target variable:")
print(df[target_col].describe())

# Convert all columns to appropriate types
categorical_cols = []
for col in df.columns:
    if col != target_col:  # Skip the target column
        # Check if column contains string values
        if df[col].dtype == 'object' or df[col].astype(str).str.contains('[a-zA-Z]').any():
            categorical_cols.append(col)
            # Convert to string first to ensure consistent encoding
            df[col] = df[col].astype(str)
            # Encode categorical values
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            print(f"Encoded categorical column: {col}")
        else:
            # Ensure numeric columns are float
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Replace any remaining NaNs with 0
            df[col] = df[col].fillna(0)

# Ensure target is numeric
df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)

# Print categorical columns found
print(f"Categorical columns encoded: {categorical_cols}")

# Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Verify no string values remain
print("\nFeature data types:")
print(X.dtypes)

# Function to perform 5-fold cross-validation with different models
def cross_validate_model(X, y, features, model_type='rf'):
    X_selected = X[features]
    
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'mlp':
        model = MLPRegressor(max_iter=1000, random_state=42)
    elif model_type == 'xgb':
        model = xgb.XGBRegressor(random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=kf, scoring='r2')
    return np.mean(scores)

# Dictionary to store results
results = {}

# 1. MLP Feature Importance
mlp = MLPRegressor(max_iter=1000, random_state=42)
mlp.fit(X, y)

# For MLP, we'll use a custom approach to get feature importance
# We'll calculate feature importance by measuring how model predictions change
# when we perturb each feature
feature_importance = []
for i in range(X.shape[1]):
    X_perturbed = X.copy()
    X_perturbed.iloc[:, i] = np.random.permutation(X_perturbed.iloc[:, i])
    y_pred_perturbed = mlp.predict(X_perturbed)
    y_pred_original = mlp.predict(X)
    importance = np.mean(np.abs(y_pred_original - y_pred_perturbed))
    feature_importance.append(importance)

# Create DataFrame with feature importances
mlp_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})
sorted_features = mlp_importance_df.sort_values('Importance', ascending=False)
top5_mlp = sorted_features.iloc[:5]['Feature'].tolist()

# Cross-validate with top 5 features using 5-fold CV with MLP
cv_score_mlp = cross_validate_model(X, y, top5_mlp, model_type='mlp')
print(f"\nMLP top 5 features: {top5_mlp}")
print(f"MLP cross-validation R² score: {cv_score_mlp:.4f}")
results['MLP'] = {'cv_score': cv_score_mlp, 'top5': top5_mlp}

# Remove highest feature
highest_feature_mlp = top5_mlp[0]
X_reduced_mlp = X.drop(highest_feature_mlp, axis=1)

# Refit MLP on reduced dataset
mlp = MLPRegressor(max_iter=1000, random_state=42)
mlp.fit(X_reduced_mlp, y)

# Recalculate feature importance on reduced dataset
feature_importance = []
for i in range(X_reduced_mlp.shape[1]):
    X_perturbed = X_reduced_mlp.copy()
    X_perturbed.iloc[:, i] = np.random.permutation(X_perturbed.iloc[:, i])
    y_pred_perturbed = mlp.predict(X_perturbed)
    y_pred_original = mlp.predict(X_reduced_mlp)
    importance = np.mean(np.abs(y_pred_original - y_pred_perturbed))
    feature_importance.append(importance)

# Create DataFrame with feature importances from reduced dataset
mlp_importance_df_reduced = pd.DataFrame({
    'Feature': X_reduced_mlp.columns,
    'Importance': feature_importance
})
sorted_features_reduced = mlp_importance_df_reduced.sort_values('Importance', ascending=False)
top4_mlp = sorted_features_reduced.iloc[:4]['Feature'].tolist()
results['MLP']['top4'] = top4_mlp

# 2. Random Forest Feature Importance
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)
rf_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})
sorted_features = rf_importance_df.sort_values('Importance', ascending=False)
top5_rf = sorted_features.iloc[:5]['Feature'].tolist()

# Cross-validate with top 5 features using 5-fold CV with RF
cv_score_rf = cross_validate_model(X, y, top5_rf, model_type='rf')
print(f"\nRF top 5 features: {top5_rf}")
print(f"RF cross-validation R² score: {cv_score_rf:.4f}")
results['RF'] = {'cv_score': cv_score_rf, 'top5': top5_rf}

# Remove highest feature
highest_feature_rf = top5_rf[0]
X_reduced_rf = X.drop(highest_feature_rf, axis=1)

# Refit RF on reduced dataset
rf = RandomForestRegressor(random_state=42)
rf.fit(X_reduced_rf, y)

# Get feature importance from reduced dataset
rf_importance_df_reduced = pd.DataFrame({
    'Feature': X_reduced_rf.columns,
    'Importance': rf.feature_importances_
})
sorted_features_reduced = rf_importance_df_reduced.sort_values('Importance', ascending=False)
top4_rf = sorted_features_reduced.iloc[:4]['Feature'].tolist()
results['RF']['top4'] = top4_rf

# 3. XGBoost Feature Importance
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X, y)
xgb_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
})
sorted_features = xgb_importance_df.sort_values('Importance', ascending=False)
top5_xgb = sorted_features.iloc[:5]['Feature'].tolist()

# Cross-validate with top 5 features using 5-fold CV with XGBoost
cv_score_xgb = cross_validate_model(X, y, top5_xgb, model_type='xgb')
print(f"\nXGBoost top 5 features: {top5_xgb}")
print(f"XGBoost cross-validation R² score: {cv_score_xgb:.4f}")
results['XGBoost'] = {'cv_score': cv_score_xgb, 'top5': top5_xgb}

# Remove highest feature
highest_feature_xgb = top5_xgb[0]
X_reduced_xgb = X.drop(highest_feature_xgb, axis=1)

# Refit XGBoost on reduced dataset
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_reduced_xgb, y)

# Get feature importance from reduced dataset
xgb_importance_df_reduced = pd.DataFrame({
    'Feature': X_reduced_xgb.columns,
    'Importance': xgb_model.feature_importances_
})
sorted_features_reduced = xgb_importance_df_reduced.sort_values('Importance', ascending=False)
top4_xgb = sorted_features_reduced.iloc[:4]['Feature'].tolist()
results['XGBoost']['top4'] = top4_xgb

# 4. RF with SHAP
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)
shap_importances = np.abs(shap_values).mean(axis=0)

rf_shap_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': shap_importances
})
sorted_features = rf_shap_importance_df.sort_values('Importance', ascending=False)
top5_rf_shap = sorted_features.iloc[:5]['Feature'].tolist()

# Cross-validate with top 5 features using 5-fold CV with RF
cv_score_rf_shap = cross_validate_model(X, y, top5_rf_shap, model_type='rf')
print(f"\nRF_SHAP top 5 features: {top5_rf_shap}")
print(f"RF_SHAP cross-validation R² score: {cv_score_rf_shap:.4f}")
results['RF_SHAP'] = {'cv_score': cv_score_rf_shap, 'top5': top5_rf_shap}

# Remove highest feature
highest_feature_rf_shap = top5_rf_shap[0]
X_reduced_rf_shap = X.drop(highest_feature_rf_shap, axis=1)

# Refit RF on reduced dataset
rf = RandomForestRegressor(random_state=42)
rf.fit(X_reduced_rf_shap, y)

# Get SHAP values on reduced dataset
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_reduced_rf_shap)
shap_importances = np.abs(shap_values).mean(axis=0)

rf_shap_importance_df_reduced = pd.DataFrame({
    'Feature': X_reduced_rf_shap.columns,
    'Importance': shap_importances
})
sorted_features_reduced = rf_shap_importance_df_reduced.sort_values('Importance', ascending=False)
top4_rf_shap = sorted_features_reduced.iloc[:4]['Feature'].tolist()
results['RF_SHAP']['top4'] = top4_rf_shap

# 5. XGBoost with SHAP
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X, y)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)
shap_importances = np.abs(shap_values).mean(axis=0)

xgb_shap_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': shap_importances
})
sorted_features = xgb_shap_importance_df.sort_values('Importance', ascending=False)
top5_xgb_shap = sorted_features.iloc[:5]['Feature'].tolist()

# Cross-validate with top 5 features using 5-fold CV with XGBoost
cv_score_xgb_shap = cross_validate_model(X, y, top5_xgb_shap, model_type='xgb')
print(f"\nXGBoost_SHAP top 5 features: {top5_xgb_shap}")
print(f"XGBoost_SHAP cross-validation R² score: {cv_score_xgb_shap:.4f}")
results['XGBoost_SHAP'] = {'cv_score': cv_score_xgb_shap, 'top5': top5_xgb_shap}

# Remove highest feature
highest_feature_xgb_shap = top5_xgb_shap[0]
X_reduced_xgb_shap = X.drop(highest_feature_xgb_shap, axis=1)

# Refit XGBoost on reduced dataset
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_reduced_xgb_shap, y)

# Get SHAP values on reduced dataset
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_reduced_xgb_shap)
shap_importances = np.abs(shap_values).mean(axis=0)

xgb_shap_importance_df_reduced = pd.DataFrame({
    'Feature': X_reduced_xgb_shap.columns,
    'Importance': shap_importances
})
sorted_features_reduced = xgb_shap_importance_df_reduced.sort_values('Importance', ascending=False)
top4_xgb_shap = sorted_features_reduced.iloc[:4]['Feature'].tolist()
results['XGBoost_SHAP']['top4'] = top4_xgb_shap

# 6. Feature Agglomeration
# Use FeatureAgglomeration to group similar features
n_clusters = min(len(X.columns) - 1, 15)  # Ensure we don't request more clusters than features
fa = FeatureAgglomeration(n_clusters=n_clusters)
fa.fit(X)

# For each cluster, identify the feature with the highest correlation with the target
selected_features = []
cluster_to_features = {}

# Map each feature to its cluster
for i, cluster_id in enumerate(fa.labels_):
    if cluster_id not in cluster_to_features:
        cluster_to_features[cluster_id] = []
    cluster_to_features[cluster_id].append((X.columns[i], i))

# From each cluster, select the feature most correlated with target
for cluster_id, features in cluster_to_features.items():
    best_corr = 0
    best_feature = None
    
    for feat_name, feat_idx in features:
        feat_values = X.iloc[:, feat_idx]
        corr = abs(np.corrcoef(feat_values, y)[0, 1])
        
        if corr > best_corr:
            best_corr = corr
            best_feature = feat_name
    
    if best_feature:
        selected_features.append((best_feature, best_corr))

# Sort by correlation and get top 5
selected_features.sort(key=lambda x: x[1], reverse=True)
top5_fa = [f[0] for f in selected_features[:5]]

# Cross-validate with top 5 features using 5-fold CV with RF
cv_score_fa = cross_validate_model(X, y, top5_fa, model_type='rf')
print(f"\nFA top 5 features: {top5_fa}")
print(f"FA cross-validation R² score: {cv_score_fa:.4f}")
results['FA'] = {'cv_score': cv_score_fa, 'top5': top5_fa}

# Remove highest feature
highest_feature_fa = top5_fa[0]
X_reduced_fa = X.drop(highest_feature_fa, axis=1)

# Rerun Feature Agglomeration on reduced dataset
fa = FeatureAgglomeration(n_clusters=n_clusters-1)  # One less cluster since we removed a feature
fa.fit(X_reduced_fa)

# For each cluster, identify the feature with the highest correlation with the target
selected_features = []
cluster_to_features = {}

# Map each feature to its cluster
for i, cluster_id in enumerate(fa.labels_):
    if cluster_id not in cluster_to_features:
        cluster_to_features[cluster_id] = []
    cluster_to_features[cluster_id].append((X_reduced_fa.columns[i], i))

# From each cluster, select the feature most correlated with target
for cluster_id, features in cluster_to_features.items():
    best_corr = 0
    best_feature = None
    
    for feat_name, feat_idx in features:
        feat_values = X_reduced_fa.iloc[:, feat_idx]
        corr = abs(np.corrcoef(feat_values, y)[0, 1])
        
        if corr > best_corr:
            best_corr = corr
            best_feature = feat_name
    
    if best_feature:
        selected_features.append((best_feature, best_corr))

# Sort by correlation and get top 4
selected_features.sort(key=lambda x: x[1], reverse=True)
top4_fa = [f[0] for f in selected_features[:4]]
results['FA']['top4'] = top4_fa

# 7. Highly Variable Gene Selection (HVGS)
# Calculate variance for each feature
variances = X.var()

# Create a DataFrame with feature variances
feature_variance_df = pd.DataFrame({
    'Feature': X.columns,
    'Variance': variances
})

# Sort by variance and select top 5
sorted_features = feature_variance_df.sort_values('Variance', ascending=False)
top5_hvgs = sorted_features.iloc[:5]['Feature'].tolist()

# Cross-validate with top 5 features using 5-fold CV with RF
cv_score_hvgs = cross_validate_model(X, y, top5_hvgs, model_type='rf')
print(f"\nHVGS top 5 features: {top5_hvgs}")
print(f"HVGS cross-validation R² score: {cv_score_hvgs:.4f}")
results['HVGS'] = {'cv_score': cv_score_hvgs, 'top5': top5_hvgs}

# Remove highest feature
highest_feature_hvgs = top5_hvgs[0]
X_reduced_hvgs = X.drop(highest_feature_hvgs, axis=1)

# Calculate variance for each feature in reduced dataset
variances = X_reduced_hvgs.var()

# Create a DataFrame with feature variances
feature_variance_df_reduced = pd.DataFrame({
    'Feature': X_reduced_hvgs.columns,
    'Variance': variances
})

# Sort by variance and select top 4
sorted_features_reduced = feature_variance_df_reduced.sort_values('Variance', ascending=False)
top4_hvgs = sorted_features_reduced.iloc[:4]['Feature'].tolist()
results['HVGS']['top4'] = top4_hvgs

# 8. Spearman Correlation
correlations = []
for col in X.columns:
    corr, _ = spearmanr(X[col], y)
    correlations.append((col, abs(corr)))

# Sort by correlation coefficient and select top 5
sorted_features = sorted(correlations, key=lambda x: x[1], reverse=True)
top5_spearman = [f[0] for f in sorted_features[:5]]

# Cross-validate with top 5 features using 5-fold CV with RF
cv_score_spearman = cross_validate_model(X, y, top5_spearman, model_type='rf')
print(f"\nSpearman top 5 features: {top5_spearman}")
print(f"Spearman cross-validation R² score: {cv_score_spearman:.4f}")
results['Spearman'] = {'cv_score': cv_score_spearman, 'top5': top5_spearman}

# Remove highest feature
highest_feature_spearman = top5_spearman[0]
X_reduced_spearman = X.drop(highest_feature_spearman, axis=1)

# Calculate correlations for reduced dataset
correlations = []
for col in X_reduced_spearman.columns:
    corr, _ = spearmanr(X_reduced_spearman[col], y)
    correlations.append((col, abs(corr)))

# Sort by correlation coefficient and select top 4
sorted_features_reduced = sorted(correlations, key=lambda x: x[1], reverse=True)
top4_spearman = [f[0] for f in sorted_features_reduced[:4]]
results['Spearman']['top4'] = top4_spearman

# Create summary table
summary_data = []
for method, result in results.items():
    summary_data.append({
        'Method': method,
        'CV R²': round(result['cv_score'], 4),
        'Top 5 Features': ', '.join(result['top5']),
        'Top 4 Features (after removing best)': ', '.join(result['top4'])
    })

summary_df = pd.DataFrame(summary_data)
print("\nFeature Selection Results Summary:")
print(summary_df)

# Save results to CSV file
summary_df.to_csv('result.csv', index=False)
print("\nResults saved to result.csv")
