
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

import os

# Create results directory relative to script location
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

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

# === 6. Pearson Correlations with Price ===
print("\n--- Pearson Correlations with Price ---")
for param in numeric_features + categorical_features:
    # For categorical variables, encode to numeric for correlation calculation
    if df[param].dtype == 'object':
        encoded = pd.factorize(df[param])[0]
        coef, p_val = stats.pearsonr(encoded, df['Price'])
    else:
        coef, p_val = stats.pearsonr(df[param], df['Price'])
    print(f"{param}: Correlation = {coef:.3f}, p-value = {p_val:.3g}")


if __name__ == "__main__":
    main()