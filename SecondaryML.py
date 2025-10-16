import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
from rdkit import RDLogger
import pickle
from datetime import datetime

# Suppress warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Load and prepare the dataset
print("=" * 80)
print(" " * 20 + "MODEL COMPARISON")
print(" " * 15 + "Ridge vs Random Forest vs Gradient Boosting")
print("=" * 80)

file_path = "2408 vbh_solubility_data_updated (1).xlsx"
df = pd.read_excel(file_path)
print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Find solubility columns
solubility_cols = [col for col in df.columns if
                   'log' in col.lower() and ('so' in col.lower() or 'solub' in col.lower())]


# Extract molecular descriptors from SMILES
def smiles_to_descriptors(smiles):
    """Extract chemical descriptors from SMILES string."""
    if pd.isna(smiles) or smiles == '':
        return pd.Series([np.nan] * 6)
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    return pd.Series([np.nan] * 6)
            else:
                return pd.Series([np.nan] * 6)

        descriptors = [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol)
        ]
        descriptors = [float(d) if d is not None else np.nan for d in descriptors]
        return pd.Series(descriptors)
    except Exception as e:
        return pd.Series([np.nan] * 6)


# Define descriptor names
cation_desc_names = ["Cat_MW_RDKit", "Cat_LogP", "Cat_TPSA", "Cat_HBD", "Cat_HBA", "Cat_RotBonds"]
anion_desc_names = ["An_MW_RDKit", "An_LogP", "An_TPSA", "An_HBD", "An_HBA", "An_RotBonds"]

# Extract descriptors
if any('cat' in col.lower() and 'smiles' in col.lower() for col in df.columns):
    cat_smiles_col = [col for col in df.columns if 'cat' in col.lower() and 'smiles' in col.lower()][0]
    df[cation_desc_names] = df[cat_smiles_col].apply(smiles_to_descriptors)

if any('an' in col.lower() and 'smiles' in col.lower() for col in df.columns):
    an_smiles_col = [col for col in df.columns if 'an' in col.lower() and 'smiles' in col.lower()][0]
    df[anion_desc_names] = df[an_smiles_col].apply(smiles_to_descriptors)


# Choose features and target variable
if solubility_cols:
    target_col = solubility_cols[0]
    print(f"Target: {target_col}")
    df_clean = df.dropna(subset=[target_col])
    print(f"Valid samples: {len(df_clean)}")
else:
    print("ERROR: No solubility columns found!")
    exit()

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
target_cols = [col for col in df_clean.columns if any(sol_name.lower() in col.lower()
                                                      for sol_name in ['log', 'solub'])]
features = [col for col in numeric_cols if col not in target_cols]

print(f"Features: {len(features)}")

X = df_clean[features].fillna(df_clean[features].median())
y = df_clean[target_col]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# Scale features for Ridge
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Ridge, Random Forest, Gradient Boosting
print("\n" + "=" * 80)
print("TRAINING MODELS...")
print("=" * 80)

models = {}

print("\nTraining Ridge Regression...")
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_scaled, y_train)
models['Ridge'] = {'model': ridge, 'name': 'Ridge Regression', 'requires_scaling': True, 'color': 'blue'}
print("✓ Ridge done")

print("\nTraining Random Forest...")
rf = RandomForestRegressor(
    n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
models['RandomForest'] = {'model': rf, 'name': 'Random Forest', 'requires_scaling': False, 'color': 'green'}
print("✓ RF done")

print("\nTraining Gradient Boosting...")
gbm = GradientBoostingRegressor(
    n_estimators=50, learning_rate=0.05, max_depth=2,
    min_samples_split=20, min_samples_leaf=10, subsample=0.7,
    max_features=0.5, random_state=42
)
gbm.fit(X_train, y_train)
models['GradientBoosting'] = {'model': gbm, 'name': 'Gradient Boosting', 'requires_scaling': False, 'color': 'red'}
print("✓ GBM done")


# Evaluate models
print("\n" + "=" * 80)
print("EVALUATING MODELS...")
print("=" * 80)

results = []
for name, model_info in models.items():
    print(f"\nEvaluating {model_info['name']}...")
    model = model_info['model']
    X_tr, X_te = (X_train_scaled, X_test_scaled) if model_info['requires_scaling'] else (X_train, X_test)

    # Training performance
    y_pred_train = model.predict(X_tr)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    # Test performance
    y_pred_test = model.predict(X_te)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation
    print("  Running 5-fold cross-validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X_tr, y_train, cv=kfold, scoring='r2')
    cv_rmse = -cross_val_score(model, X_tr, y_train, cv=kfold,
                               scoring='neg_root_mean_squared_error')

    model_info['predictions_train'] = y_pred_train
    model_info['predictions_test'] = y_pred_test

    results.append({
        'Model': model_info['name'],
        'Train_R2': r2_train, 'Train_RMSE': rmse_train, 'Train_MAE': mae_train,
        'Test_R2': r2_test, 'Test_RMSE': rmse_test, 'Test_MAE': mae_test,
        'CV_R2_Mean': cv_r2.mean(), 'CV_R2_Std': cv_r2.std(),
        'CV_RMSE_Mean': cv_rmse.mean(), 'CV_RMSE_Std': cv_rmse.std(),
        'Overfitting_Gap': abs(r2_train - r2_test),
        'CV_Stability': cv_rmse.std() / cv_rmse.mean()
    })

results_df = pd.DataFrame(results)


# Show summary of results
print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)

print("\nTraining Performance")
print(f"{'Model':<25} {'R²':<12} {'RMSE':<12} {'MAE':<12}")
for _, row in results_df.iterrows():
    print(f"{row['Model']:<25} {row['Train_R2']:<12.4f} {row['Train_RMSE']:<12.4f} {row['Train_MAE']:<12.4f}")

print("\nTest Performance")
print(f"{'Model':<25} {'R²':<12} {'RMSE':<12} {'MAE':<12}")
for _, row in results_df.iterrows():
    print(f"{row['Model']:<25} {row['Test_R2']:<12.4f} {row['Test_RMSE']:<12.4f} {row['Test_MAE']:<12.4f}")

print("\nCross-Validation Performance (5-Fold)")
for _, row in results_df.iterrows():
    print(f"{row['Model']:<25} {row['CV_R2_Mean']:.4f} ± {row['CV_R2_Std']:.4f}   "
          f"{row['CV_RMSE_Mean']:.4f} ± {row['CV_RMSE_Std']:.4f}")

print("\nGeneralization Analysis")
print(f"{'Model':<25} {'Overfitting Gap':<20} {'CV Stability':<20} {'Status':<15}")
for _, row in results_df.iterrows():
    gap = row['Overfitting_Gap']
    stability = row['CV_Stability']
    if gap < 0.1:
        status = "Excellent"
    elif gap < 0.2:
        status = "Good"
    elif gap < 0.3:
        status = "Fair"
    else:
        status = "Poor"
    print(f"{row['Model']:<25} {gap:<20.4f} {stability:<20.4f} {status:<15}")
# Visualization 1: Predicted vs Actual for each model
plt.figure(figsize=(12, 5))

for i, (name, model_info) in enumerate(models.items(), 1):
    plt.subplot(1, 3, i)
    y_pred = model_info['predictions_test']
    plt.scatter(y_test, y_pred, alpha=0.6, color=model_info['color'])
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f"{model_info['name']}\nPredicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

plt.tight_layout()
plt.show()


# Visualization 2: R² comparison bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(results_df['Model'], results_df['Test_R2'], color=['blue', 'green', 'red'])
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Model Comparison (Test R²)")
plt.ylabel("R² Score")
plt.ylim(min(0, results_df['Test_R2'].min() - 0.1), 1.05)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
             f"{height:.3f}", ha='center', va='bottom')

plt.show()
# Calculate residuals
residuals = abs(y_test - y_pred_test)

# Find the outlier index (in original DataFrame)
outlier_idx = residuals.idxmax()

# See the full original row for the outlier in your dataframe
outlier_row = df.loc[outlier_idx]
print(outlier_row)
import matplotlib.pyplot as plt

outlier_idx = 93
descriptors = features  # your feature list
target_col = 'log(So) Water'  # your target column name

# 1. Show outlier descriptors
print("Outlier descriptor values:")
print(df.loc[outlier_idx, descriptors])

# 2. Plot histogram for a few key descriptors to visually check extremes
key_descriptors = ['Cat_MW_RDKit', 'Cat_LogP', 'An_LogP', target_col]

for desc in key_descriptors:
    plt.figure(figsize=(6,4))
    plt.hist(df[desc].dropna(), bins=30, alpha=0.7, label='All data')
    plt.axvline(df.loc[outlier_idx, desc], color='red', linestyle='--', label='Outlier')
    plt.title(f"Distribution of {desc}")
    plt.legend()
    plt.show()

# 3. Check if y (target) for outlier is in unusual range
print(f"Outlier {target_col}: {df.loc[outlier_idx, target_col]}")
print(f"Dataset {target_col} range: {df[target_col].min()} - {df[target_col].max()}")
