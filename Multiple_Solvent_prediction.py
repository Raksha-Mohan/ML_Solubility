import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from rdkit import RDLogger
import pickle
from datetime import datetime

# Suppress warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# CONFIGURATION
file_path = "2408 vbh_solubility_data_updated (1).xlsx"

target_solvents = [
    'log(So) Water', 'log(So) DMSO', 'log(So) DMAC', 'log(So) DMF',
    'log(So) MeNO2', 'log(So) MeCN', 'log(So) MeOH', 'log(So) Propanonitrile',
    'log(So) Nitroethane', 'log(So) Ethanol', 'log(So) 1-Propanol',
    'log(So) Acetone', 'log(So) 2-Propanol', 'log(So) 4-Methyl-2-Pentanone',
    'log(So) DCM', 'log(So) THF', 'log(So) Ethylacetate', 'log(So) Toluene',
    'log(So) n-Hexane'
]

# Gradient Boosting hyperparameters (optimized for minimal overfitting)
gbm_params = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 3,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'subsample': 0.7,
    'max_features': 0.5,
    'random_state': 42
}


# HELPER FUNCTIONS

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


def identify_outliers(y_true, y_pred, threshold=2.5):
    """Identify outliers based on residual analysis."""
    residuals = np.abs(y_true.values - y_pred)
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    outlier_mask = residuals > (mean_res + threshold * std_res)
    return outlier_mask, residuals


def plot_shap_summary(shap_values, X, feature_names, target_name, top_n=15):
    """Create SHAP summary plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      max_display=top_n, show=False)
    plt.title(f'SHAP Feature Importance - {target_name}', fontsize=14, pad=20)
    plt.tight_layout()
    return plt.gcf()


def plot_outlier_analysis(y_true, y_pred, residuals, outlier_mask, target_name):
    """Create comprehensive outlier visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Outlier Analysis - {target_name}', fontsize=16, y=1.00)

    y_true_arr = y_true.values

    # Plot 1: Predicted vs Actual with outliers highlighted
    ax1 = axes[0, 0]
    ax1.scatter(y_true_arr[~outlier_mask], y_pred[~outlier_mask],
                alpha=0.5, label='Normal', color='blue')
    if outlier_mask.sum() > 0:
        ax1.scatter(y_true_arr[outlier_mask], y_pred[outlier_mask],
                    color='red', s=100, alpha=0.7, label='Outliers', edgecolors='black')
    ax1.plot([y_true_arr.min(), y_true_arr.max()], [y_true_arr.min(), y_true_arr.max()],
             'k--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predictions with Outliers Highlighted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residual distribution
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    if outlier_mask.sum() > 0:
        threshold = np.mean(residuals) + 2.5 * np.std(residuals)
        ax2.axvline(threshold, color='red',
                    linestyle='--', linewidth=2, label='Outlier Threshold')
    ax2.set_xlabel('Absolute Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals vs Predicted
    ax3 = axes[1, 0]
    ax3.scatter(y_pred[~outlier_mask], residuals[~outlier_mask],
                alpha=0.5, color='blue')
    if outlier_mask.sum() > 0:
        ax3.scatter(y_pred[outlier_mask], residuals[outlier_mask],
                    color='red', s=100, alpha=0.7, edgecolors='black')
    ax3.axhline(np.mean(residuals) + 2.5 * np.std(residuals),
                color='red', linestyle='--', linewidth=2, label='Outlier Threshold')
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Absolute Residuals')
    ax3.set_title('Residuals vs Predictions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Outlier statistics (if any)
    ax4 = axes[1, 1]
    if outlier_mask.sum() > 0:
        outlier_stats = pd.DataFrame({
            'Metric': ['Count', 'Percentage', 'Mean Residual', 'Max Residual',
                       'Mean Actual', 'Mean Predicted'],
            'Value': [
                outlier_mask.sum(),
                f"{outlier_mask.sum() / len(outlier_mask) * 100:.1f}%",
                f"{residuals[outlier_mask].mean():.4f}",
                f"{residuals[outlier_mask].max():.4f}",
                f"{y_true_arr[outlier_mask].mean():.4f}",
                f"{y_pred[outlier_mask].mean():.4f}"
            ]
        })
        ax4.axis('off')
        table = ax4.table(cellText=outlier_stats.values,
                          colLabels=outlier_stats.columns,
                          cellLoc='left', loc='center',
                          colWidths=[0.5, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Outlier Statistics')
    else:
        ax4.text(0.5, 0.5, 'No Outliers Detected',
                 ha='center', va='center', fontsize=14)
        ax4.axis('off')

    plt.tight_layout()
    return fig


def analyze_outlier_features(outlier_mask, X, y_test, feature_names):
    """Analyze feature characteristics of outliers."""
    if outlier_mask.sum() == 0:
        return None

    # Get outlier indices in test set
    test_indices = y_test.index
    outlier_test_indices = test_indices[outlier_mask]

    outlier_features = X.loc[outlier_test_indices]

    # Create comparison with overall dataset
    comparison = pd.DataFrame({
        'Feature': feature_names,
        'Outlier_Mean': outlier_features[feature_names].mean().values,
        'Overall_Mean': X[feature_names].mean().values,
        'Outlier_Std': outlier_features[feature_names].std().values,
        'Overall_Std': X[feature_names].std().values
    })

    comparison['Mean_Diff'] = comparison['Outlier_Mean'] - comparison['Overall_Mean']
    comparison['Std_Diff'] = comparison['Outlier_Std'] - comparison['Overall_Std']
    comparison = comparison.sort_values('Mean_Diff', key=abs, ascending=False)

    return comparison, outlier_test_indices


# MAIN PIPELINE
print("=" * 80)
print(" " * 20 + "MULTI-TARGET GRADIENT BOOSTING PIPELINE")
print(" " * 25 + "with SHAP Analysis")
print("=" * 80)

# Load data
df = pd.read_excel(file_path)
print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Extract descriptors
cation_desc_names = ["Cat_MW_RDKit", "Cat_LogP", "Cat_TPSA", "Cat_HBD", "Cat_HBA", "Cat_RotBonds"]
anion_desc_names = ["An_MW_RDKit", "An_LogP", "An_TPSA", "An_HBD", "An_HBA", "An_RotBonds"]

if any('cat' in col.lower() and 'smiles' in col.lower() for col in df.columns):
    cat_smiles_col = [col for col in df.columns if 'cat' in col.lower() and 'smiles' in col.lower()][0]
    df[cation_desc_names] = df[cat_smiles_col].apply(smiles_to_descriptors)

if any('an' in col.lower() and 'smiles' in col.lower() for col in df.columns):
    an_smiles_col = [col for col in df.columns if 'an' in col.lower() and 'smiles' in col.lower()][0]
    df[anion_desc_names] = df[an_smiles_col].apply(smiles_to_descriptors)

# Prepare features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = [col for col in df.columns if any(sol in col.lower()
                                                 for sol in ['log', 'solub'])]
features = [col for col in numeric_cols if col not in exclude_cols]

print(f"Features extracted: {len(features)}")
print(f"Target solvents: {len(target_solvents)}")

# Storage for results
all_results = []
all_models = {}
all_shap_values = {}
all_outliers = {}

# TRAIN MODELS FOR EACH SOLVENT
for target_col in target_solvents:
    if target_col not in df.columns:
        print(f"\n⚠ Skipping {target_col} (not found in dataset)")
        continue

    print("\n" + "=" * 80)
    print(f"TARGET: {target_col}")
    print("=" * 80)

    # Prepare data
    df_clean = df.dropna(subset=[target_col])
    n_samples = len(df_clean)

    if n_samples < 20:
        print(f"⚠ Insufficient data ({n_samples} samples). Skipping...")
        continue

    print(f"Valid samples: {n_samples}")

    X = df_clean[features].fillna(df_clean[features].median())
    y = df_clean[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Train model
    print("\nTraining Gradient Boosting model...")
    model = GradientBoostingRegressor(**gbm_params)
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)

    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation
    print("Running 5-fold cross-validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    cv_rmse = -cross_val_score(model, X_train, y_train, cv=kfold,
                               scoring='neg_root_mean_squared_error')

    overfitting_gap = abs(train_r2 - test_r2)

    # Store results
    all_results.append({
        'Solvent': target_col,
        'N_Samples': n_samples,
        'Train_R2': train_r2,
        'Train_RMSE': train_rmse,
        'Train_MAE': train_mae,
        'Test_R2': test_r2,
        'Test_RMSE': test_rmse,
        'Test_MAE': test_mae,
        'CV_R2_Mean': cv_r2.mean(),
        'CV_R2_Std': cv_r2.std(),
        'CV_RMSE_Mean': cv_rmse.mean(),
        'CV_RMSE_Std': cv_rmse.std(),
        'Overfitting_Gap': overfitting_gap
    })

    # Print performance
    print(f"\nPerformance Summary:")
    print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
    print(f"  CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  Overfitting Gap: {overfitting_gap:.4f}")

    # SHAP ANALYSIS
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Store SHAP values
    all_shap_values[target_col] = {
        'explainer': explainer,
        'shap_values': shap_values,
        'X_test': X_test,
        'feature_names': features
    }

    # Plot SHAP summary
    plot_shap_summary(shap_values, X_test, features, target_col)
    plt.savefig(f'shap_summary_{target_col.replace(" ", "_").replace("(", "").replace(")", "")}.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    # OUTLIER DETECTION
    print("\nDetecting outliers...")
    outlier_mask, residuals = identify_outliers(y_test, y_pred_test)
    n_outliers = outlier_mask.sum()

    print(f"Outliers detected: {n_outliers} ({n_outliers / len(y_test) * 100:.1f}%)")

    if n_outliers > 0:
        # Analyze outlier features
        outlier_feature_analysis, outlier_indices = analyze_outlier_features(
            outlier_mask, X, y_test, features
        )

        print("\nTop 10 Feature Differences in Outliers:")
        print(outlier_feature_analysis.head(10)[['Feature', 'Mean_Diff', 'Outlier_Mean', 'Overall_Mean']])

        # Get outlier details from original dataframe
        df_outliers = df_clean.loc[outlier_indices]

        # Store outlier info
        all_outliers[target_col] = {
            'indices': outlier_indices.tolist(),
            'residuals': residuals[outlier_mask],
            'feature_analysis': outlier_feature_analysis,
            'df_outliers': df_outliers,
            'y_true': y_test.values[outlier_mask],
            'y_pred': y_pred_test[outlier_mask]
        }

        # Plot outlier analysis
        plot_outlier_analysis(y_test, y_pred_test, residuals, outlier_mask, target_col)
        plt.savefig(f'outlier_analysis_{target_col.replace(" ", "_").replace("(", "").replace(")", "")}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()
    else:
        all_outliers[target_col] = None

    # Store model
    all_models[target_col] = {
        'model': model,
        'features': features,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }

# SUMMARY RESULTS
results_df = pd.DataFrame(all_results)

print("\n" + "=" * 80)
print("OVERALL PERFORMANCE SUMMARY")
print("=" * 80)

print("\nTest Performance Ranking (by R²):")
results_sorted = results_df.sort_values('Test_R2', ascending=False)
print(results_sorted[['Solvent', 'N_Samples', 'Test_R2', 'Test_RMSE', 'Overfitting_Gap']].to_string(index=False))

print("\nGeneralization Quality:")
for _, row in results_sorted.iterrows():
    gap = row['Overfitting_Gap']
    if gap < 0.1:
        status = "Excellent ✓✓"
    elif gap < 0.2:
        status = "Good ✓"
    elif gap < 0.3:
        status = "Fair ~"
    else:
        status = "Poor ✗"
    print(f"{row['Solvent']:<30} Gap: {gap:.4f}  {status}")

# VISUALIZATIONS
# 1. Overall R² comparison
fig, ax = plt.subplots(figsize=(14, 8))
x_pos = np.arange(len(results_sorted))
bars = ax.barh(x_pos, results_sorted['Test_R2'], color='steelblue', alpha=0.8)
ax.set_yticks(x_pos)
ax.set_yticklabels(results_sorted['Solvent'])
ax.set_xlabel('Test R² Score', fontsize=12)
ax.set_title('Model Performance Across All Solvents', fontsize=14, pad=20)
ax.axvline(0.7, color='green', linestyle='--', alpha=0.5, label='Good (R²=0.7)')
ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (R²=0.5)')
ax.legend()
ax.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, results_sorted['Test_R2'])):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('overall_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Overfitting analysis
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(results_df['Test_R2'], results_df['Overfitting_Gap'],
           s=results_df['N_Samples'] * 2, alpha=0.6, c=results_df['Test_RMSE'],
           cmap='coolwarm', edgecolors='black')
ax.axhline(0.1, color='green', linestyle='--', label='Excellent generalization')
ax.axhline(0.2, color='orange', linestyle='--', label='Good generalization')
ax.set_xlabel('Test R²', fontsize=12)
ax.set_ylabel('Overfitting Gap (|Train R² - Test R²|)', fontsize=12)
ax.set_title('Generalization Analysis Across Solvents', fontsize=14, pad=20)
ax.legend()
ax.grid(True, alpha=0.3)
plt.colorbar(ax.collections[0], label='Test RMSE', ax=ax)
plt.tight_layout()
plt.savefig('generalization_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# SAVE RESULTS
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save models
with open(f'gbm_models_{timestamp}.pkl', 'wb') as f:
    pickle.dump(all_models, f)
print(f"\n✓ Models saved to: gbm_models_{timestamp}.pkl")

# Save results
results_df.to_csv(f'model_results_{timestamp}.csv', index=False)
print(f"✓ Results saved to: model_results_{timestamp}.csv")

# Save SHAP values
with open(f'shap_values_{timestamp}.pkl', 'wb') as f:
    pickle.dump(all_shap_values, f)
print(f"✓ SHAP values saved to: shap_values_{timestamp}.pkl")

# Save outlier information
with open(f'outliers_{timestamp}.pkl', 'wb') as f:
    pickle.dump(all_outliers, f)
print(f"✓ Outlier analysis saved to: outliers_{timestamp}.pkl")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
print(f"\nTotal models trained: {len(all_models)}")
print(f"Average Test R²: {results_df['Test_R2'].mean():.4f}")
print(f"Average Overfitting Gap: {results_df['Overfitting_Gap'].mean():.4f}")
print(f"\nBest performing solvent: {results_sorted.iloc[0]['Solvent']}")
print(f"  R²: {results_sorted.iloc[0]['Test_R2']:.4f}")
print(f"  RMSE: {results_sorted.iloc[0]['Test_RMSE']:.4f}")