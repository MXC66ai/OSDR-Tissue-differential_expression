"""
Adapted pipeline from:
  "Spaceflight induces changes in gene expression profiles linked to insulin and estrogen"
  (Aydogan Mathyk et al., Commun Biol 2024)

Applied to mouse eye spaceflight datasets:
  - OSD-100  : C57BL/6J whole eye  (6 GC vs 6 FLT)
  - OSD-162  : BALB/c whole eye     (5 GC vs 5 FLT, BSL removed)
  - OSD-194  : BALB/c left retina   (4 GC vs 5 FLT, BSL removed)

Inputs (existing differential-expression tables):
  transcriptomics/results/OSD100/differential_expression_results.csv
  transcriptomics/results/OSD-162/GC&flt/differential_expression_results.csv
  transcriptomics/results/OSD-194/GC&flt/differential_expression_results.csv

Output: figures and tables under ./eye_ie_results/
"""
import os
import json
import urllib.request
import uuid
import numpy as np
import pandas as pd
from scipy import stats, cluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 0. Paths and parameters
# ---------------------------------------------------------------------------
DATASETS = {
    "OSD-100_whole_eye":  r"C:\Users\Administrator\.openclaw\workspace\transcriptomics\results\OSD100\differential_expression_results.csv",
    "OSD-162_whole_eye":  r"C:\Users\Administrator\.openclaw\workspace\transcriptomics\results\OSD-162\GC&flt\differential_expression_results.csv",
    "OSD-194_left_retina": r"C:\Users\Administrator\.openclaw\workspace\transcriptomics\results\OSD-194\GC&flt\differential_expression_results.csv",
}
GENE_LIST = "ncbi_curated_ie_gene_list.tsv"
OUTDIR = "eye_ie_results"
os.makedirs(OUTDIR, exist_ok=True)

LFC_MIN = 0.5   # keep genes with |log2FC| > 0.5 in at least one dataset
LFC_MAX = 2.0   # cap / exclude extreme values for the moderate-effect heatmap
N_CLUSTERS = 7

sns.set_theme(font_scale=0.9)


def read_de(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure required columns exist
    need = {"ENSEMBL", "SYMBOL", "log2FoldChange"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    # Remove duplicated ENSEMBL rows (keep first / max |lfc|)
    df = df.loc[df["log2FoldChange"].abs().groupby(df["ENSEMBL"]).idxmax()]
    return df.set_index("ENSEMBL")[["SYMBOL", "log2FoldChange"]]


# ---------------------------------------------------------------------------
# 1. Build gene × dataset log2FC matrix
# ---------------------------------------------------------------------------
print("Building log2FC matrix ...")
lfc_dict = {}
for name, path in DATASETS.items():
    df = read_de(path)
    lfc_dict[name] = df["log2FoldChange"].rename(name)

lfc_mat = pd.concat(lfc_dict, axis=1)  # rows = Ensembl, cols = datasets
print(f"  log2FC matrix: {lfc_mat.shape}")

# ---------------------------------------------------------------------------
# 2. Load insulin / estrogen / insulin-resistance gene list
# ---------------------------------------------------------------------------
print(f"Loading gene list {GENE_LIST} ...")
ncbi = pd.read_csv(GENE_LIST, sep="\t")
ncbi = ncbi.dropna(subset=["gene_id"]).drop_duplicates("gene_id")
print(f"  {len(ncbi)} reference genes")

# Intersect with DE matrix
common_ens = lfc_mat.index.intersection(ncbi["gene_id"])
lfc_subset = lfc_mat.loc[common_ens].copy()
ncbi = ncbi.set_index("gene_id").loc[common_ens]
print(f"  {len(lfc_subset)} reference genes present in eye DE results")

# ---------------------------------------------------------------------------
# 3. Moderate-effect filter (0.5 < |log2FC| < 2 in at least one dataset)
# ---------------------------------------------------------------------------
print("Applying moderate-effect filter ...")
mask = ((lfc_subset.abs() > LFC_MIN) & (lfc_subset.abs() < LFC_MAX)).any(axis=1)
lfc_filt = lfc_subset[mask].copy()
print(f"  {len(lfc_filt)} genes after 0.5<|log2FC|<2 filter")

if lfc_filt.empty:
    raise RuntimeError("No genes passed the filter. Try relaxing thresholds.")

# ---------------------------------------------------------------------------
# 4. Row-wise Z-score → hierarchical clustering (Euclidean + complete)
# ---------------------------------------------------------------------------
print("Z-score and clustering (Fig 1c) ...")
z_raw = stats.zscore(lfc_filt.values, axis=1, nan_policy="omit")
z = pd.DataFrame(z_raw, index=lfc_filt.index, columns=lfc_filt.columns)
# Drop genes that became NaN/constant (cannot be clustered)
z = z.dropna()
print(f"  {len(z)} genes after removing NaN/constant rows for clustering")
if z.empty:
    raise RuntimeError("No genes left after NaN removal for clustering.")

row_dist = pdist(z.values, metric="euclidean")
col_dist = pdist(z.T.values, metric="euclidean")
row_link = cluster.hierarchy.linkage(row_dist, method="complete")
col_link = cluster.hierarchy.linkage(col_dist, method="complete")

# Pathway annotation for rows
pathway_series = ncbi.loc[z.index, "pathway"]
pathway_palette = {
    "Insulin_resistance": "#4682B4",  # steelblue
    "Insulin_signaling":  "#FF6347",  # tomato
    "Estrogen_signaling": "#DAA520",  # goldenrod
    "Combination":        "#800080",  # purple
}
row_colors = pathway_series.map(pathway_palette)

# Fig 1c heatmap
g = sns.clustermap(
    z,
    row_linkage=row_link,
    col_linkage=col_link,
    cmap="RdBu_r",
    center=0,
    linewidths=0.0,
    figsize=(8, 12),
    row_colors=row_colors,
    dendrogram_ratio=(.18, .2),
    cbar_pos=(0.02, 0.82, 0.03, 0.15),
    vmin=-3, vmax=3
)
for label, color in pathway_palette.items():
    g.ax_row_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
g.ax_row_dendrogram.legend(loc="center left", title="Pathway", bbox_to_anchor=(1.05, 0.5))
g.ax_heatmap.set_xlabel("Eye spaceflight datasets")
g.ax_heatmap.set_ylabel("Insulin / estrogen / IR genes")
plt.suptitle("Fig 1c – Z-score heatmap of insulin/estrogen/IR genes in eye (0.5<|log2FC|<2)",
             y=1.02, fontsize=12)
fig1c_path = os.path.join(OUTDIR, "fig1c_ie_heatmap.png")
plt.savefig(fig1c_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved {fig1c_path}")

# Save the filtered matrix
lfc_filt.to_csv(os.path.join(OUTDIR, "fig1c_log2FC_matrix.tsv"), sep="\t")
z.to_csv(os.path.join(OUTDIR, "fig1c_zscore_matrix.tsv"), sep="\t")

# ---------------------------------------------------------------------------
# 5. Complete-case filter + 7 clusters  (Fig 1d)
# ---------------------------------------------------------------------------
print("Complete-case filter and 7-cluster cut (Fig 1d) ...")
complete_mask = ~lfc_filt.isna().any(axis=1)
lfc_645 = lfc_filt[complete_mask].copy()  # "645" is the paper's name; actual size may differ
print(f"  {len(lfc_645)} genes observed in all 3 datasets")

if len(lfc_645) < N_CLUSTERS:
    raise RuntimeError(f"Too few complete genes ({len(lfc_645)}) for {N_CLUSTERS} clusters.")

z2_raw = stats.zscore(lfc_645.values, axis=1, nan_policy="omit")
z2 = pd.DataFrame(z2_raw, index=lfc_645.index, columns=lfc_645.columns).dropna()
print(f"  {len(z2)} complete genes after removing NaN/constant rows for clustering")
if len(z2) < N_CLUSTERS:
    raise RuntimeError(f"Too few complete genes ({len(z2)}) for {N_CLUSTERS} clusters.")

row_dist2 = pdist(z2.values, metric="euclidean")
row_link2 = cluster.hierarchy.linkage(row_dist2, method="complete")
cluster_assign = cluster.hierarchy.fcluster(row_link2, t=N_CLUSTERS, criterion="maxclust")
cluster_series = pd.Series(cluster_assign, index=z2.index, name="Cluster")

# ---------------------------------------------------------------------------
# 6. GO-BP enrichment per cluster via Enrichr
# ---------------------------------------------------------------------------
print("GO-BP enrichment per cluster (Enrichr) ...")


def enrichr_top_term(symbols, bg="GO_Biological_Process_2023"):
    """Submit a gene list to Enrichr and return the top GO-BP term."""
    if not symbols:
        return "NA"
    boundary = uuid.uuid4().hex
    body_lines = [
        f"--{boundary}",
        'Content-Disposition: form-data; name="list"',
        "",
        "\n".join(symbols),
        f"--{boundary}",
        'Content-Disposition: form-data; name="description"',
        "",
        "eye_cluster",
        f"--{boundary}--",
        ""
    ]
    body = "\r\n".join(body_lines).encode()
    req = urllib.request.Request(
        "https://maayanlab.cloud/Enrichr/addList",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read().decode())
    except Exception as e:
        print(f"    Enrichr addList error: {e}")
        return "NA"
    uid = resp.get("userListId")
    if uid is None:
        return "NA"
    enrich_url = f"https://maayanlab.cloud/Enrichr/enrich?userListId={uid}&backgroundType={bg}"
    try:
        with urllib.request.urlopen(enrich_url, timeout=60) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        print(f"    Enrichr enrich error: {e}")
        return "NA"
    results = data.get(bg, [])
    if not results:
        return "NA"
    # results format: [rank, term, pval, oddsratio, ... , genes, adj_pval, ...]
    top = results[0]
    term = top[1]
    adj_p = top[6]
    if adj_p > 0.05:
        return f"{term} (padj={adj_p:.2f})"
    return term


symbol_map = ncbi["symbol"].to_dict()
cluster_go_term = {}
for cl in sorted(cluster_series.unique()):
    genes = cluster_series[cluster_series == cl].index.tolist()
    symbols = [symbol_map.get(g, "") for g in genes]
    symbols = [s for s in symbols if s and isinstance(s, str)]
    print(f"  Cluster {cl}: {len(symbols)} symbols")
    top = enrichr_top_term(symbols)
    cluster_go_term[cl] = top
    print(f"    top GO: {top}")

# ---------------------------------------------------------------------------
# 7. Fig 1d heatmap with cluster + GO annotations
# ---------------------------------------------------------------------------
print("Plotting Fig 1d ...")
cluster_palette = sns.color_palette("Set2", n_colors=N_CLUSTERS)
cluster_color_map = {cl: cluster_palette[i] for i, cl in enumerate(sorted(cluster_series.unique()))}
row_colors_cl = cluster_series.map(cluster_color_map)

g2 = sns.clustermap(
    z2,
    row_linkage=row_link2,
    col_linkage=col_link,
    cmap="RdBu_r",
    center=0,
    linewidths=0.0,
    figsize=(8, 12),
    row_colors=row_colors_cl,
    dendrogram_ratio=(.18, .2),
    cbar_pos=(0.02, 0.82, 0.03, 0.15),
    vmin=-3, vmax=3
)

# Add cluster/GO labels on row dendrogram
for cl, term in cluster_go_term.items():
    first_gene = cluster_series[cluster_series == cl].index[0]
    # Position is tricky because clustermap reorders rows; use index lookup
    y_pos = np.where(g2.data2d.index == first_gene)[0]
    if len(y_pos) == 0:
        continue
    y_pos = y_pos[0]
    g2.ax_row_dendrogram.text(
        -0.05, y_pos / len(z2),
        f"C{cl}: {term}",
        transform=g2.ax_row_dendrogram.transAxes,
        rotation=0,
        va="center",
        ha="right",
        fontsize=8,
        color=cluster_color_map[cl]
    )

g2.ax_heatmap.set_xlabel("Eye spaceflight datasets")
g2.ax_heatmap.set_ylabel(f"{len(z2)} insulin/estrogen/IR genes (all datasets)")
plt.suptitle("Fig 1d – Z-score heatmap of complete insulin/estrogen/IR genes (7 clusters)",
             y=1.02, fontsize=12)
fig1d_path = os.path.join(OUTDIR, "fig1d_ie_heatmap_clusters.png")
plt.savefig(fig1d_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved {fig1d_path}")

# Save cluster assignments and matrices
lfc_645.to_csv(os.path.join(OUTDIR, "fig1d_log2FC_matrix.tsv"), sep="\t")
z2.to_csv(os.path.join(OUTDIR, "fig1d_zscore_matrix.tsv"), sep="\t")
cluster_df = pd.DataFrame({
    "ensembl_id": cluster_series.index,
    "symbol": [symbol_map.get(g, "") for g in cluster_series.index],
    "cluster": cluster_series.values,
    "top_GO_BP": [cluster_go_term[c] for c in cluster_series.values]
})
cluster_df.to_csv(os.path.join(OUTDIR, "cluster_assignments.tsv"), sep="\t", index=False)

# ---------------------------------------------------------------------------
# 8. Summary table per dataset
# ---------------------------------------------------------------------------
summary_rows = []
for ds in lfc_subset.columns:
    s = lfc_subset[ds]
    summary_rows.append({
        "dataset": ds,
        "IE_genes_tested": s.notna().sum(),
        "IE_genes_0.5<|lfc|<2": ((s.abs() > 0.5) & (s.abs() < 2)).sum(),
        "IE_genes_|lfc|>0.5": (s.abs() > 0.5).sum(),
        "mean_abs_lfc": s.abs().mean(),
        "median_abs_lfc": s.abs().median()
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(OUTDIR, "dataset_summary.tsv"), sep="\t", index=False)
print("\nDataset summary:")
print(summary.to_string(index=False))

print(f"\nAll outputs written to: {os.path.abspath(OUTDIR)}")
