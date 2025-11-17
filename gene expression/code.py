import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------
# 1. ORIGINAL GENE EXPRESSION DATA
# -----------------------------------------
data = {
    "Gene": ["GeneA", "GeneB", "GeneC"],
    "Sample_1_Before": [100, 50, 200],
    "Sample_2_Before": [105, 55, 190],
    "Sample_3_After":  [110, 60, 180],
    "Sample_4_After":  [115, 65, 170]
}

df = pd.DataFrame(data)
print("Original Gene Expression Data:\n", df, "\n")

# -----------------------------------------
# 2. NORMALIZATION (LOG2 TRANSFORMATION)
# -----------------------------------------
before_cols = ["Sample_1_Before", "Sample_2_Before"]
after_cols = ["Sample_3_After", "Sample_4_After"]

# Log2 transform
df_log2 = df.copy()
df_log2[before_cols + after_cols] = np.log2(df[before_cols + after_cols])

print("Normalized Gene Expression Data (Log2):\n", df_log2[before_cols + after_cols], "\n")

# -----------------------------------------
# 3. T-TEST FOR EACH GENE
# -----------------------------------------
for i, row in df_log2.iterrows():
    gene_name = row["Gene"]
    before_vals = row[before_cols].values
    after_vals = row[after_cols].values

    t_stat, p_val = ttest_ind(before_vals, after_vals)

    print(f"Gene: {gene_name}")
    print(f"T-statistic: {round(t_stat, 3)}")
    print(f"P-value: {round(p_val, 3)}\n")

# -----------------------------------------
# 4. HEATMAP OF NORMALIZED EXPRESSION
# -----------------------------------------
# Prepare matrix for heatmap
heatmap_data = df_log2[before_cols + after_cols].values

plt.figure(figsize=(8, 5))
sns.heatmap(
    heatmap_data,
    cmap="viridis",
    annot=False,
    xticklabels=before_cols + after_cols,
    yticklabels=df_log2["Gene"],
    cbar_kws={"label": "Gene Expression (Log2)"}
)

plt.title("Heatmap of Normalized Gene Expression")
plt.xlabel("Samples")
plt.ylabel("Genes")
plt.show()
