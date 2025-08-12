
#!/usr/bin/env python3
"""
Laptop Pricing Analysis Dashboard

This script performs comprehensive analysis of laptop pricing data including:
- Data cleaning and preprocessing
- Correlation analysis
- Visualization of price relationships
- Statistical analysis

Author: Yash Patil
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def load_and_clean_data():
    """Load and clean the laptop pricing dataset."""
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
    df = pd.read_csv(url)
    return df

def main():
    """Main analysis function."""
    # Load data
    df = load_and_clean_data()
    
    # Remove extra unnamed index columns
    df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)   

    # Display DataFrame info
    print("--- Initial DataFrame Info ---")
    df.info()

    # Map numerical values to human-readable strings
    gpu_mapping = {1: "GTX 1050", 2: "RTX 3070", 3: "RTX 4080"}
    os_mapping = {1: "Windows", 2: "Linux"}
    category_mapping = {1: "Gaming", 2: "Business", 3: "Ultrabook", 4: "Workstation", 5: "Convertible"}

    df['GPU'] = df['GPU'].replace(gpu_mapping)
    df['OS'] = df['OS'].replace(os_mapping)
    df['Category'] = df['Category'].replace(category_mapping)

    # Quick data check
    print("Category values:", df['Category'].unique())
    print("GPU values:", df['GPU'].unique())
    print("OS values:", df['OS'].unique())
    print("\n--------------------------------------------------------------------------------DataFrame head-------------------------------------------------------------------------")
    print(df.head())

    # Set plot style
    sns.set(style="whitegrid")

    # Create results directory relative to script location
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # === 1. Basic Description ===
    print("\n--- Numeric Summary ---")
    print(df.describe().T)

    print("\n--- Categorical Summary ---")
    print(df.describe(include=['object']).T)

    # === 2. Correlation Matrix Heatmap ===
    plt.figure(figsize=(8, 6))
    corr = df.select_dtypes(include=[np.number]).corr()

    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Correlation Matrix Heatmap")

    plt.savefig(os.path.join(save_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # === 3. Scatter plots of numeric features vs Price ===
    numeric_features = ["CPU_frequency", "Screen_Size_inch", "Weight_pounds"]

    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    for ax, feature in zip(axes, numeric_features):
        sns.regplot(x=feature, y="Price", data=df, ax=ax)
        ax.set_title(f"{feature} vs Price")
        ax.set_ylim(0,)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scatter_plots.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # === 4. Boxplots for categorical/binned features ===
    categorical_features = ["Category", "GPU", "RAM_GB", "Storage_GB_SSD", "CPU_core", "OS"]

    fig, axes = plt.subplots(2, 3, figsize=(18,10))
    axes = axes.flatten()

    for ax, col in zip(axes, categorical_features):
        sns.boxplot(x=col, y="Price", data=df, ax=ax)
        ax.set_title(f"Price Distribution by {col}")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "boxplots.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # === 5. Grouped Heatmap (GPU x CPU_core) with average price ===
    grouped = df.groupby(['GPU', 'CPU_core'])['Price'].mean().unstack()

    plt.figure(figsize=(10,6))
    sns.heatmap(grouped, annot=True, fmt=".0f", cmap="RdBu_r", center=grouped.mean().mean())
    plt.title("Average Price by GPU and CPU Core Count")
    plt.ylabel("GPU")
    plt.xlabel("CPU Core Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "grouped_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()

# === 6. Enhanced Pearson Correlations with Price ===
def interpret_correlation(coef):
    """Interpret correlation coefficient strength and direction."""
    abs_coef = abs(coef)
    direction = "Positive" if coef > 0 else "Negative"
    
    if abs_coef >= 0.7:
        strength = "Strong"
    elif abs_coef >= 0.5:
        strength = "Moderate"
    elif abs_coef >= 0.3:
        strength = "Weak"
    else:
        strength = "Very Weak"
    
    return f"{direction} {strength}"

def interpret_p_value(p_val):
    """Interpret statistical significance."""
    if p_val < 0.001:
        return "Highly Significant (***)"
    elif p_val < 0.01:
        return "Very Significant (**)"
    elif p_val < 0.05:
        return "Significant (*)"
    else:
        return "Not Significant"

print("\n" + "="*80)
print("PEARSON CORRELATIONS WITH LAPTOP PRICE - DETAILED ANALYSIS")
print("="*80)

# Store correlation results for visualization
correlation_results = []

all_features = numeric_features + categorical_features
for param in all_features:
    # For categorical variables, encode to numeric for correlation calculation
    if df[param].dtype == 'object':
        encoded = pd.factorize(df[param])[0]
        coef, p_val = stats.pearsonr(encoded, df['Price'])
        feature_type = "Categorical"
    else:
        coef, p_val = stats.pearsonr(df[param], df['Price'])
        feature_type = "Numeric"
    
    # Store results
    correlation_results.append({
        'Feature': param,
        'Type': feature_type,
        'Correlation': coef,
        'P-value': p_val,
        'Interpretation': interpret_correlation(coef),
        'Significance': interpret_p_value(p_val)
    })
    
    # Print detailed interpretation
    print(f"\n📊 {param.upper()} ({feature_type})")
    print(f"   Correlation: {coef:.3f}")
    print(f"   Relationship: {interpret_correlation(coef)}")
    print(f"   Statistical Significance: {interpret_p_value(p_val)} (p = {p_val:.3g})")
    
    # Add practical interpretation
    if abs(coef) >= 0.3 and p_val < 0.05:
        if coef > 0:
            print(f"   💡 Insight: Higher {param} tends to be associated with HIGHER laptop prices")
        else:
            print(f"   💡 Insight: Higher {param} tends to be associated with LOWER laptop prices")
    else:
        print(f"   💡 Insight: {param} shows little to no reliable relationship with laptop prices")

# Create correlation results DataFrame for better visualization
corr_df = pd.DataFrame(correlation_results)
corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)

print("\n" + "="*80)
print("SUMMARY: FEATURES RANKED BY CORRELATION STRENGTH")
print("="*80)
print(corr_df[['Feature', 'Correlation', 'Interpretation', 'Significance']].to_string(index=False))

# === 7. Correlation Visualization ===
plt.figure(figsize=(12, 8))
colors = ['red' if corr < 0 else 'blue' for corr in corr_df['Correlation']]
bars = plt.barh(range(len(corr_df)), corr_df['Correlation'], color=colors, alpha=0.7)

# Add correlation values on bars
for i, (bar, corr) in enumerate(zip(bars, corr_df['Correlation'])):
    plt.text(corr + (0.02 if corr > 0 else -0.02), i, f'{corr:.3f}', 
             va='center', ha='left' if corr > 0 else 'right', fontweight='bold')

plt.yticks(range(len(corr_df)), corr_df['Feature'])
plt.xlabel('Correlation with Price')
plt.title('Feature Correlations with Laptop Price\n(Blue = Positive, Red = Negative)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='Weak threshold (±0.3)')
plt.axvline(x=-0.3, color='green', linestyle='--', alpha=0.5)
plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold (±0.5)')
plt.axvline(x=-0.5, color='orange', linestyle='--', alpha=0.5)
plt.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='Strong threshold (±0.7)')
plt.axvline(x=-0.7, color='red', linestyle='--', alpha=0.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "correlation_analysis.png"), dpi=300, bbox_inches='tight')
plt.show()


if __name__ == "__main__":
    main()