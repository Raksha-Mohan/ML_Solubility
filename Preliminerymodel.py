import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import warnings
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')


# 1. Load dataset and inspect columns

file_path = "2408 vbh_solubility_data_updated (1).xlsx"
df = pd.read_excel(file_path)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# Find solubility columns
solubility_cols = [col for col in df.columns if
                   'log' in col.lower() and ('so' in col.lower() or 'solub' in col.lower())]
print(f"\nFound solubility columns: {solubility_cols}")



# Improved function to extract RDKit descriptors with error handling

def smiles_to_descriptors(smiles):
    """Extract basic chemical descriptors from SMILES string with robust error handling."""
    if pd.isna(smiles) or smiles == '':
        return pd.Series([np.nan] * 6)

    try:
        # Try to create molecule from SMILES
        mol = Chem.MolFromSmiles(str(smiles))

        # If failed, try to sanitize
        if mol is None:
            mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    return pd.Series([np.nan] * 6)
            else:
                return pd.Series([np.nan] * 6)

        # Calculate descriptors
        descriptors = [
            Descriptors.MolWt(mol),  # Molecular weight
            Crippen.MolLogP(mol),  # LogP
            rdMolDescriptors.CalcTPSA(mol),  # Topological polar surface area
            rdMolDescriptors.CalcNumHBD(mol),  # H-bond donors
            rdMolDescriptors.CalcNumHBA(mol),  # H-bond acceptors
            rdMolDescriptors.CalcNumRotatableBonds(mol)  # Rotatable bonds
        ]

        # Check for any None or invalid values
        descriptors = [float(d) if d is not None else np.nan for d in descriptors]
        return pd.Series(descriptors)

    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return pd.Series([np.nan] * 6)


# Define descriptor column names
cation_desc_names = ["Cat_MW_RDKit", "Cat_LogP", "Cat_TPSA", "Cat_HBD", "Cat_HBA", "Cat_RotBonds"]
anion_desc_names = ["An_MW_RDKit", "An_LogP", "An_TPSA", "An_HBD", "An_HBA", "An_RotBonds"]

# Check if SMILES columns exist
smiles_cols = [col for col in df.columns if 'smiles' in col.lower()]
print(f"\nFound SMILES columns: {smiles_cols}")

# Extract descriptors if SMILES columns exist
if any('cat' in col.lower() and 'smiles' in col.lower() for col in df.columns):
    cat_smiles_col = [col for col in df.columns if 'cat' in col.lower() and 'smiles' in col.lower()][0]
    print(f"Processing cation SMILES from column: {cat_smiles_col}")
    df[cation_desc_names] = df[cat_smiles_col].apply(smiles_to_descriptors)

if any('an' in col.lower() and 'smiles' in col.lower() for col in df.columns):
    an_smiles_col = [col for col in df.columns if 'an' in col.lower() and 'smiles' in col.lower()][0]
    print(f"Processing anion SMILES from column: {an_smiles_col}")
    df[anion_desc_names] = df[an_smiles_col].apply(smiles_to_descriptors)


# Select target variable

# Choosing target or use first available solubility column
if solubility_cols:
    target_col = solubility_cols[0]  # Use first solubility column found
    print(f"\nUsing target variable: {target_col}")

    # Remove rows where target is NaN
    initial_rows = len(df)
    df_clean = df.dropna(subset=[target_col])
    print(f"Removed {initial_rows - len(df_clean)} rows with missing target values")

else:
    print("No solubility columns found! Please check your dataset.")
    print("Available columns:", df.columns.tolist())
    exit()


# 4. Select features for ML
# Get all numeric columns
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

# Remove target columns from features
target_cols = [col for col in df_clean.columns if any(sol_name.lower() in col.lower() for sol_name in ['log', 'solub'])]
features = [col for col in numeric_cols if col not in target_cols]

print(f"\nUsing {len(features)} features for modeling")
print(f"Features: {features[:10]}...")  # Show first 10 features

# Prepare feature matrix and target
X = df_clean[features].fillna(df_clean[features].median())  # Use median for better handling of outliers
y = df_clean[target_col]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Target statistics:\n{y.describe()}")


# 5. Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# Train Random Forest Regressor

print("\nTraining Random Forest model...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,  # Prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)
model.fit(X_train, y_train)


# Evaluate model

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Training metrics
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)

# Test metrics
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Training RMSE: {rmse_train:.4f}")
print(f"Training R²:   {r2_train:.4f}")
print(f"Test RMSE:     {rmse_test:.4f}")
print(f"Test R²:       {r2_test:.4f}")

# Check for overfitting
if r2_train - r2_test > 0.2:
    print("\n Warning: Model may be overfitting (large gap between train and test R²)")
else:
    print("\n Good generalization (small gap between train and test performance)")


#Feature importance analysis

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 50)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))


# SHAP Analysis

if len(X_test) <= 1000:
    try:
        print("\nRunning SHAP analysis...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Create SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Save SHAP values
        shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
        shap_df.to_csv("shap_values.csv", index=False)
        print("SHAP values saved to 'shap_values.csv'")

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
else:
    print(f"Skipping SHAP analysis (dataset too large: {len(X_test)} samples)")


#Prediction vs Actual Plot

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_test, alpha=0.6, s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel(f'Actual {target_col}')
plt.ylabel(f'Predicted {target_col}')
plt.title(f'Actual vs Predicted {target_col}\nR² = {r2_test:.4f}, RMSE = {rmse_test:.4f}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE!")
print("=" * 50)
print(f" Model trained successfully")
print(f" Feature importance analysis completed")
print(f" Plots saved as PNG files")
if len(X_test) <= 1000:
    print(f" SHAP analysis completed")
print(f"\nModel is ready for predictions on new data!")