"""
Supplementary analyses for the eye insulin/estrogen pipeline.
Outputs: Venn-like overlap, cross-dataset scatter, pathway barplots, eye-functional annotation table.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

OUTDIR = "eye_ie_results"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
lfc = pd.read_csv(os.path.join(OUTDIR, "fig1c_log2FC_matrix.tsv"), sep="\t", index_col=0)
genes = pd.read_csv("ncbi_curated_ie_gene_list.tsv", sep="\t", index_col="gene_id")
df = lfc.join(genes[["symbol", "pathway"]], how="left")

# Drop rows that are all NaN
df = df.dropna(how="all", subset=lfc.columns)

# ---------------------------------------------------------------------------
# 1. Eye-functional annotation table
# ---------------------------------------------------------------------------

eye_categories = {
    "visual_cycle": ["Rpe65", "Rbp4", "Lrat", "Rdh5", "Rdh12", "Abca4"],
    "photoreceptor": ["Rho", "Opn1sw", "Opn1mw", "Opn4", "Gnat1", "Gnat2", "Pde6a", "Cngb1"],
    "rpe": ["Rpe65", "Rbp4", "Mertk", "Tyrp1", "Otx2", "Mitf", "Best1"],
    "cornea_epithelium": ["Krt12", "Krt14", "Krt15", "Krt16", "Krt17", "Krt19", "Krt23", "Krt24"],
    "retinal_vessel": ["Tek", "Kdr", "Pdgfb", "Vegfa", "Angpt2", "Nos3"],
    "inflammation_immunity": ["Tnf", "Oas1a", "Oas1g", "Icam1", "Ccl2", "Il6", "Ifit1"],
    "metabolism_energy": ["Pdk4", "Fbp2", "Pck1", "Pfkm", "Ucp3", "Fasn", "Acaca", "Cpt1b", "Slc27a2", "Slc27a4", "Fabp5"],
    "estrogen_receptor": ["Esr1", "Esr2", "Gper1", "Esrra", "Esrrb", "Esrrg"],
    "insulin_receptor": ["Insr", "Irs1", "Irs2", "Irs4", "Pik3r1", "Akt1", "Akt2", "Mtor"],
}


def assign_eye_category(row):
    syms = [row["symbol"]] if isinstance(row["symbol"], str) else []
    matched = []
    for cat, symlist in eye_categories.items():
        if any(s in symlist for s in syms):
            matched.append(cat)
    return ";".join(matched) if matched else "other"


df["eye_function"] = df.apply(assign_eye_category, axis=1)

# Direction per dataset
def direction(x):
    if pd.isna(x):
        return "NA"
    if x > 0.5:
        return "UP"
    if x < -0.5:
        return "DN"
    return "NS"


for col in lfc.columns:
    df[f"dir_{col}"] = df[col].apply(direction)

# Save annotated table
annot_cols = ["symbol", "pathway", "eye_function"] + list(lfc.columns) + [f"dir_{c}" for c in lfc.columns]
df[annot_cols].to_csv(os.path.join(OUTDIR, "ie_genes_eye_annotated.tsv"), sep="\t")
print("Saved ie_genes_eye_annotated.tsv")

# ---------------------------------------------------------------------------
# 2. Venn-like overlap of genes with |lfc| > 0.5
# ---------------------------------------------------------------------------
sets = {}
for col in lfc.columns:
    sets[col] = set(df.index[df[col].abs() > 0.5])

# Compute overlaps for 3 sets
s1, s2, s3 = sets[lfc.columns[0]], sets[lfc.columns[1]], sets[lfc.columns[2]]
names = [c.replace("_", "\n") for c in lfc.columns]

# Counts
only1 = len(s1 - s2 - s3)
only2 = len(s2 - s1 - s3)
only3 = len(s3 - s1 - s2)
only12 = len((s1 & s2) - s3)
only13 = len((s1 & s3) - s2)
only23 = len((s2 & s3) - s1)
all123 = len(s1 & s2 & s3)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")

colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
# Draw three overlapping circles manually
centers = [(3.5, 5.5), (6.5, 5.5), (5.0, 3.2)]
radii = [2.6, 2.6, 2.6]
for c, r, col, name in zip(centers, radii, colors, names):
    circ = plt.Circle(c, r, color=col, alpha=0.3, ec=col, lw=2)
    ax.add_patch(circ)
    # label near each circle
    if c[0] < 5:
        ax.text(c[0]-1.5, c[1]+2.0, name, ha="center", va="center", fontsize=9, color=col, fontweight="bold")
    elif c[0] > 5:
        ax.text(c[0]+1.5, c[1]+2.0, name, ha="center", va="center", fontsize=9, color=col, fontweight="bold")
    else:
        ax.text(c[0], c[1]-3.0, name, ha="center", va="center", fontsize=9, color=col, fontweight="bold")

# Place counts
count_pos = {
    "only1": (2.2, 6.5),
    "only2": (7.8, 6.5),
    "only3": (5.0, 1.6),
    "12": (5.0, 6.5),
    "13": (3.8, 4.2),
    "23": (6.2, 4.2),
    "123": (5.0, 4.8),
}
counts = {
    "only1": only1, "only2": only2, "only3": only3,
    "12": only12, "13": only13, "23": only23, "123": all123,
}
for k, (x, y) in count_pos.items():
    ax.text(x, y, str(counts[k]), ha="center", va="center", fontsize=12, fontweight="bold", color="#333")

ax.set_title("Overlap of IE genes with |log2FC| > 0.5 across eye datasets", fontsize=13, pad=20)
plt.tight_layout()
venn_path = os.path.join(OUTDIR, "supp_venn_overlap.png")
plt.savefig(venn_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {venn_path}")

# ---------------------------------------------------------------------------
# 3. Pairwise scatter plots of log2FC
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
pairs = [(lfc.columns[0], lfc.columns[1]),
         (lfc.columns[0], lfc.columns[2]),
         (lfc.columns[1], lfc.columns[2])]

for ax, (c1, c2) in zip(axes, pairs):
    x = df[c1].values
    y = df[c2].values
    # remove NaN pairwise
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    ax.scatter(x, y, alpha=0.5, s=20, c="#555")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    # diagonal
    lim = max(abs(x).max(), abs(y).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=0.8, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(c1.replace("_", " "))
    ax.set_ylabel(c2.replace("_", " "))
    # correlation
    if len(x) > 2:
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"r = {r:.3f}")
    else:
        ax.set_title("r = NA")

plt.suptitle("Pairwise log2FC correlations between eye datasets (IE genes)", y=1.02, fontsize=13)
plt.tight_layout()
scatter_path = os.path.join(OUTDIR, "supp_pairwise_scatter.png")
plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {scatter_path}")

# ---------------------------------------------------------------------------
# 4. Pathway-level up/down barplot
# ---------------------------------------------------------------------------
summary_rows = []
for pathway, g in df.groupby("pathway"):
    for col in lfc.columns:
        up = (g[col] > 0.5).sum()
        dn = (g[col] < -0.5).sum()
        summary_rows.append({"pathway": pathway, "dataset": col.replace("_", " "), "up": up, "down": -dn})
summary = pd.DataFrame(summary_rows)

fig, ax = plt.subplots(figsize=(10, 6))
pathways = summary["pathway"].unique()
datasets = summary["dataset"].unique()
x = np.arange(len(pathways))
width = 0.25

colors_bar = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
for i, ds in enumerate(datasets):
    sub = summary[summary["dataset"] == ds]
    up_vals = sub.set_index("pathway").loc[pathways, "up"].values
    dn_vals = sub.set_index("pathway").loc[pathways, "down"].values
    offset = (i - 1) * width
    ax.bar(x + offset, up_vals, width, label=f"{ds} UP", color=colors_bar[i], alpha=0.85)
    ax.bar(x + offset, dn_vals, width, color=colors_bar[i], alpha=0.4)

ax.set_xticks(x)
ax.set_xticklabels(pathways, rotation=15, ha="right")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Number of genes (|log2FC| > 0.5)")
ax.set_title("Up- and down-regulated IE genes per pathway and dataset")
ax.legend(loc="upper right")
plt.tight_layout()
bar_path = os.path.join(OUTDIR, "supp_pathway_barplot.png")
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {bar_path}")

# ---------------------------------------------------------------------------
# 5. Eye-function category summary
# ---------------------------------------------------------------------------
eye_summary = []
for cat, g in df.groupby("eye_function"):
    if cat == "other":
        continue
    row = {"eye_function": cat, "n_genes": len(g)}
    for col in lfc.columns:
        row[f"mean_abs_lfc_{col}"] = g[col].abs().mean()
        row[f"up_{col}"] = (g[col] > 0.5).sum()
        row[f"dn_{col}"] = (g[col] < -0.5).sum()
    eye_summary.append(row)
eye_summary_df = pd.DataFrame(eye_summary)
eye_summary_df.to_csv(os.path.join(OUTDIR, "eye_function_summary.tsv"), sep="\t", index=False)
print("Saved eye_function_summary.tsv")

print("\nEye-function category counts:")
print(eye_summary_df[["eye_function", "n_genes"]].to_string(index=False))

print(f"\nAll supplementary outputs in: {os.path.abspath(OUTDIR)}")
