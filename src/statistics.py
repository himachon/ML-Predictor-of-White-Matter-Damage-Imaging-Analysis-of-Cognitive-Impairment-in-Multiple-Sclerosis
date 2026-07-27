from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Boxplot

np.random.seed(42)
n_sub = 100

tracts = [
    "Corticospinal_Tract_L",
    "Corticospinal_Tract_R",
    "Superior_Longitudinal_Fasciculus_L",
    "Superior_Longitudinal_Fasciculus_R",
    "Uncinate_Fasciculus_L",
    "Uncinate_Fasciculus_R",
    "Corpus_Callosum_Genu",
    "Corpus_Callosum_Splenium",
]

MS_subjects = ["sub-0620", "sub-0623", "sub-3188", "sub-0241", "sub-0106", "sub-1231", "sub-0422", "sub-1567", "sub-0040", "sub-0571", "sub-0858", "sub-0048", "sub-5000", "sub-0655", "sub-0708", "sub-2063", "sub-2449", "sub-0896", "sub-3801", "sub-1985"]
CS_subjects = ["sub-9000", "sub-9001", "sub-9002", "sub-9003", "sub-9004", "sub-9005", "sub-9006", "sub-9008"]

group_labels = np.random.choice(["MS", "CS"], size=n_sub)
data = {"Group": group_labels}

subject_ids = []
for group in group_labels:
    if group == "MS":
        subject_ids.append(np.random.choice(HS_subjects))
    else:
        subject_ids.append(np.random.choice(LS_subjects))
data["Subject_ID"] = subject_ids

# Simulate realistic FA distribution variables (~0.3 to 0.7 scale)
for t in tracts:
    data[t] = np.where(
        data["Group"] == "MS",
        np.random.normal(0.54, 0.05, n_sub), # Multiple Sclerosis patient distribution features
        np.random.normal(0.48, 0.05, n_sub), # Control subjects distribution features
    )

df_features = pd.DataFrame(data)

# Create a 2x4 Subplot Grid
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 9), sharey=True)
axes = axes.flatten()


for idx, tract_name in enumerate(tracts):
    ax = axes[idx]
    
    # Violin plot [distribution shape] with a boxplot used for summary metrics
    sns.violinplot(
        data=df_features,
        x="Group",
        y=tract_name,
        order=["HS", "LS"], #names refer to High Susceptibility (MS subjects) versus Low Susceptibility (control subjects)
        palette={"HS": "#1f77b4", "CS": "#ff7f0e"},
        inner=None, # Remove default inner lines to overlay a cleaner boxplot
        linewidth=1.2,
        ax=ax,
        alpha=0.6,
    )
    
    # Overlay a narrow boxplot inside the violin
    sns.boxplot(
        data=df_features,
        x="Group",
        y=tract_name,
        order=["HS", "LS"],
        width=0.15,
        color="white",
        linewidth=1.5,
        showfliers=False, # Avoid doubling outlier points if plotting stripcharts
        ax=ax,
    )
    
    # Format appearance
    clean_title = tract_name.replace("_", " ")
    ax.set_title(clean_title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("") # Hide individual x-labels to reduce clutter
    
    # Displaying y-axis label for the leftmost plots for space saving
    if idx % 4 == 0:
        ax.set_ylabel("Fractional Anisotropy (FA)", fontsize=11)
    else:
        ax.set_ylabel("")
        
    ax.tick_params(axis="both", labelsize=10)
    sns.despine(ax=ax) # Remove top and right borders

plt.tight_layout()

plt.savefig("white_matter_tracts_fa_comparison.png", dpi=300, bbox_inches="tight")
print("Plot compiled and saved successfully to: white_matter_tracts_fa_comparison.png")
plt.show()

# Multiple Correlation Matrix

group1 = df_features[df_features["Group"] == "HS"]
group2 = df_features[df_features["Group"] == "LS"]

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d for independent samples with unequal variances."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Calculate pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Avoid division by zero
    if pooled_se == 0:
        return 0

    return (np.mean(group1) - np.mean(group2)) / pooled_se


tracts = [
    "Corticospinal_Tract_L",
    "Corticospinal_Tract_R",
    "Superior_Longitudinal_Fasciculus_L",
    "Superior_Longitudinal_Fasciculus_R",
    "Uncinate_Fasciculus_L",
    "Uncinate_Fasciculus_R",
    "Corpus_Callosum_Genu",
    "Corpus_Callosum_Splenium",
]

results = []

for tract in tracts:
    # Separate the FA values by group for this specific tract
    hs_vals = df_features[df_features["Group"] == "HS"][tract].dropna()
    ls_vals = df_features[df_features["Group"] == "LS"][tract].dropna()

    # Calculate basic descriptive statistics
    mean_hs = np.mean(hs_vals)
    mean_ls = np.mean(ls_vals)

    # Conduct Welch's t-test (equal_var=False)
    t_stat, p_val = stats.ttest_ind(hs_vals, ls_vals, equal_var=False)

    # Calculate Cohen's d effect size
    cohen_d = calculate_cohens_d(hs_vals, ls_vals)

    results.append(
        {
            "Region": tract.replace("_", " "),  # Clean up the name for the table
            "Mean HS": round(mean_hs, 3),
            "Mean LS": round(mean_ls, 3),
            "Raw p": p_val,
            "Cohen's d": round(cohen_d, 3),
        }
    )

# Convert results into a working DataFrame
df_results = pd.DataFrame(results)

# Array of 8 raw p-values: Benjamini-Hochberg FDR correction
reject, fdr_corrected_p, _, _ = multipletests(
    df_results["Raw p"], alpha=0.05, method="fdr_bh"
)

# Insert the corrected p-values and clean up raw p format
df_results["FDR p"] = [f"{p:.4f}" if p >= 0.001 else "<0.001" for p in fdr_corrected_p]
df_results["Raw p"] = [
    f"{p:.4f}" if p >= 0.001 else "<0.001" for p in df_results["Raw p"]
]

# Reorder columns to exactly match your target design
df_results = df_results[["Region", "Mean MS", "Mean CS", "Raw p", "FDR p", "Cohen's d"]]

df_results

df_results.to_string(index=False)


# ROC Curve

X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.7, 0.3], random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
true_pos_rs = []
aucs= []
mean_fpr = np.linspace(0,1, 100)
fig, ax = plt.subplots(figsize=(8,6))

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    probs_val = model.predict_proba(X_test)[:, 1]
    false_pos_r, true_pos_r, _ = roc_curve(y_test, probs_val)
    roc_auc = auc(false_pos_r, true_pos_r)
    aucs.append(roc_auc)
    
    ax.plot(false_pos_r, true_pos_r, label=f'Fold {fold} (AUC = {roc_auc:.2f})', alpha=0.3)
    interp_tpr = np.interp(mean_fpr, false_pos_r, true_pos_r)
    interp_tpr[0] = 0.0
    true_pos_rs.append(interp_tpr)

mean_tpr = np.mean(true_pos_rs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

ax.plot(mean_fpr, mean_tpr, color='r', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2)
ax.plot([0,1], [0,1], color = 'blue', linestyle='--', label='Chance (AUC = 0.50)')

ax.set(xlabel= 'False Positive Rate', ylabel= 'True Positive Rate', title= 'ROC Curve across 5 Folds')
ax.legend(loc="lower right")
plt.savefig("roc_curve.png", dpi=300)
plt.show()




# Correlation Heatmap

matrix = df.corr(numeric_only=True)

plt.figure(figsize=(6, 4))

sns.heatmap(
    matrix, 
    annot=True,       # Correlation #s inside the squares
    cmap='coolwarm',  # Color palette (Red = Positive, Blue = Negative)
    vmin=-1,          # Minimum value of the scale
    vmax=1            # Maximum value of the scale
)

plt.title("Correlation Matrix Heatmap")
plt.savefig("Correlation Matrix Heatmap.png", dpi=300)
plt.show()