# 基本科学计算
conda install -c conda-forge pandas numpy scipy seaborn matplotlib

# RNA‑seq 差异分析（DESeq2 的 Python 实现）
pip install pydeseq2   # 依赖: torch, tqdm

# 基因注释（Ensembl → Symbol）
pip install mygene

# Gene Set Enrichment / GSEA
pip install gseapy   # 包含 fast‑GSEA（fGSEA）实现

# GO 富集（clusterProfiler 的 Python 替代）
pip install goatools   # 需要本地 GO annotation (gene2go) 文件

# 画热图（ComplexHeatmap 的近似实现）
pip install seaborn-clustermap   # 可选，直接用 seaborn.clustermap
# -------------------------------------------------
# 0. 环境准备
# -------------------------------------------------
import os
import pandas as pd
import numpy as np
import glob
import math
from pathlib import Path

# 统计 & 聚类
from scipy import stats, spatial, cluster
from scipy.spatial.distance import pdist, squareform

# 绘图
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# DESeq2 (Python implementation)
from pydeseq2 import DESeq2
from pydeseq2.dds import DeseqDataSet

# 基因注释
from mygene import MyGeneInfo

# GSEA / GO
import gseapy as gp
from goatools.obo_parser import GODag
from goatools.associations import read_ncbi_gene2go
from goatools.go_enrichment import GOEnrichmentStudy
# -------------------------------------------------
# 1. 读取 RSEM 计数（已转为整数）
# -------------------------------------------------
count_dir = Path("rsem_counts")
count_files = sorted(count_dir.glob("*.genes.results"))

def read_rsem(fpath):
    """读取一个 RSEM *.genes.results，返回 gene_id、expected_count（向上取整）"""
    df = pd.read_csv(fpath, sep="\t", header=0,
                     usecols=["gene_id", "expected_count"])
    # 向上取整（ceil），确保每个计数为整数
    df["expected_count"] = np.ceil(df["expected_count"]).astype(int)
    return df

# 读取全部组织
counts_list = [read_rsem(f) for f in count_files]

# 为每个 DataFrame 设定列名 = 组织名（从文件名去掉后缀）
tissue_names = [f.stem.replace(".genes", "") for f in count_files]
for df, name in zip(counts_list, tissue_names):
    df.rename(columns={"expected_count": name}, inplace=True)

# 合并计数矩阵（外连接，缺失值填 0）
all_counts = counts_list[0]
for df in counts_list[1:]:
    all_counts = pd.merge(all_counts, df, on="gene_id", how="outer")

# 替换 NA 为 0，设置 gene_id 为行索引
all_counts.set_index("gene_id", inplace=True)
all_counts.fillna(0, inplace=True).astype(int)
# -------------------------------------------------
# 2. 构建实验设计（sample_info）
# -------------------------------------------------
samples = all_counts.columns.tolist()                 # 所有列＝样本名
# 示例：前 5 为 flight，后 5 为 ground；实际请用真实分组信息
condition = (["flight"] * 5) + (["ground"] * 5)

# 组织顺序（必须与样本排列对应；这里假设每组织 2 样本，flight+ground）
tissues = (["Liver", "Liver",
            "Soleus", "Soleus",
            "EDL", "EDL",
            "Adrenal", "Adrenal",
            "Kidney", "Kidney",
            "Eye", "Eye",
            "Gastrocnemius", "Gastrocnemius",
            "Quadriceps", "Quadriceps",
            "TibialisAnterior", "TibialisAnterior"][:len(samples)])

sample_info = pd.DataFrame({
    "sample": samples,
    "condition": condition,
    "tissue": tissues
})
sample_info.set_index("sample", inplace=True)
# -------------------------------------------------
# 3. 差异表达分析（每组织单独）
# -------------------------------------------------
# pydeseq2 需要一个 “count matrix” + “metadata”。我们在每一次循环里只保留当前组织的列。
def run_deseq2(counts_sub, meta_sub):
    """
    counts_sub : DataFrame（基因 × 样本），只包含同一组织的两组（flight / ground）
    meta_sub   : DataFrame（样本 × 元信息），对应 counts_sub 的列顺序
    返回：DataFrame（基因 × log2FC, pvalue, padj）
    """
    dds = DeseqDataSet(
        counts=counts_sub,
        design_factors=meta_sub["condition"],  # 只用 condition 进行比较
        ref_factor="ground"                    # 将 ground 设为参考
    )
    dds.deseq2()
    # 生成 results 对象（默认 Wald test）
    res = dds.results()
    # 只保留 log2FoldChange 与 padj
    out = res[["log2FoldChange", "padj"]].copy()
    out.columns = ["logFC", "padj"]
    return out

# 为每个组织跑一次，并把 logFC 收集到一个大矩阵
logFC_dict = {}
for tissue in sample_info["tissue"].unique():
    # 取该组织的样本（flight + ground）
    cols = sample_info[sample_info["tissue"] == tissue].index.tolist()
    counts_sub = all_counts[cols]
    meta_sub   = sample_info.loc[cols]

    # 运行 DESeq2
    res_df = run_deseq2(counts_sub, meta_sub)

    # 保存 logFC（行索引为 gene_id）
    logFC_dict[tissue] = res_df["logFC"]

# 合并所有组织的 logFC → gene × tissue 矩阵
logFC_mat = pd.concat(logFC_dict, axis=1)          # 列名 = 组织名
logFC_mat.head()
# -------------------------------------------------
# 4. 基因注释
# -------------------------------------------------
mg = MyGeneInfo()

# 取出所有出现的 Ensembl 基因 ID
ensembl_ids = logFC_mat.index.tolist()
query = mg.querymany(ensembl_ids,
                    scopes="ensembl.gene",
                    fields="symbol",
                    species="mouse",
                    returnall=True)

# 将查询结果转成字典 {ensembl_id: symbol}
symbol_map = {item["query"]: item.get("symbol", "") for item in query["out"]}

# 添加一列 gene_symbol 到 logFC_mat（方便后续筛选）
logFC_mat["gene_symbol"] = logFC_mat.index.map(symbol_map)
logFC_mat.head()
# -------------------------------------------------
# 5. 读取 NCBI curated gene list (1970)
# -------------------------------------------------
ncbi_df = pd.read_csv("NCBI_curated_1970.tsv", sep="\t")    # 必须含 gene_id (= Ensembl)
# 只保留在我们差异分析矩阵里出现的基因
keep = ncbi_df["gene_id"].isin(logFC_mat.index)
ncbi_genes = ncbi_df[keep].copy()

# 把 1970 基因的 logFC 抽出来（去掉 symbol 列）
logFC_subset = logFC_mat.loc[ncbi_genes["gene_id"], ncbi_genes.columns[:-1]]   # 仅保留组织列
logFC_subset.head()
# -------------------------------------------------
# 6. Fig 1c：Z‑score & 热图
# -------------------------------------------------
# 1) 行‑wise Z‑score（对每个基因在 9 种组织的 logFC 标准化）
z_mat = stats.zscore(logFC_subset, axis=1, nan_policy="omit")
# 注意：stats.zscore 会返回 np.ndarray；把它转回 DataFrame 便于后面使用
z_df = pd.DataFrame(z_mat,
                    index=logFC_subset.index,
                    columns=logFC_subset.columns)

# 2) 层次聚类（欧氏距离 + 完全链接）
row_dist = pdist(z_df.values, metric="euclidean")
col_dist = pdist(z_df.T.values, metric="euclidean")

row_link = cluster.hierarchy.linkage(row_dist, method="complete")
col_link = cluster.hierarchy.linkage(col_dist, method="complete")

# 3) 行注释——通路归属
#   ncbi_genes 包含列：gene_id、pathway（如 "Insulin_resistance", "Insulin_signaling", "Estrogen_signaling", "Combination")
pathway_series = ncbi_genes.set_index("gene_id")["pathway"]
# 把 pathway 映射到颜色
pathway_palette = {
    "Insulin_resistance": "#4682B4",   # steelblue
    "Insulin_signaling":   "#FF6347",   # tomato
    "Estrogen_signaling": "#DAA520",   # goldenrod
    "Combination":         "#800080",   # purple
}
row_colors = pathway_series.map(pathway_palette)

# 4) 画热图（seaborn.clustermap）
g = sns.clustermap(z_df,
                   row_linkage=row_link,
                   col_linkage=col_link,
                   row_cluster=True,
                   col_cluster=True,
                   cmap="RdBu_r",
                   linewidths=0.0,
                   figsize=(10, 12),
                   row_colors=row_colors,
                   dendrogram_ratio=(.2, .2),
                   cbar_pos=(0.02, 0.8, 0.03, 0.18))

# 添加图例（手动绘制颜色块）
for label, color in pathway_palette.items():
    g.ax_row_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
g.ax_row_dendrogram.legend(loc="center left", title="Pathway", bbox_to_anchor=(1.0, 0.5))

g.ax_heatmap.set_xlabel("RR‑1 Tissues")
g.ax_heatmap.set_ylabel("1970 NCBI curated genes")
plt.suptitle("Fig 1c – Z‑score heatmap (1970 genes)", y=1.02, fontsize=14)
plt.show()
# -------------------------------------------------
# 7. Fig 1d：完整 645 基因子集
# -------------------------------------------------
# 1) 过滤：仅保留每个基因在所有组织都有非缺失的 logFC
complete_mask = ~logFC_subset.isna().any(axis=1)
logFC_645 = logFC_subset[complete_mask]

# 2) Z‑score
z_645 = stats.zscore(logFC_645, axis=1, nan_policy="omit")
z_645_df = pd.DataFrame(z_645,
                        index=logFC_645.index,
                        columns=logFC_645.columns)

# 3) 行聚类 + 切割成 7 簇
row_dist_645 = pdist(z_645_df.values, metric="euclidean")
row_link_645 = cluster.hierarchy.linkage(row_dist_645, method="complete")
cluster_assign = cluster.hierarchy.fcluster(row_link_645, t=7, criterion="maxclust")

# 把聚类结果转为 Series（索引=gene_id，值=cluster_id）
cluster_series = pd.Series(cluster_assign, index=z_645_df.index, name="Cluster")

# 4) 对每个簇做 GO‑BP 富集（使用 goatools）
# 读取 GO DAG 与 gene2go 注释文件（请提前下载 gene2go）
obodag = GODag("gene2go")  # 实际路径为 GO OBO 文件，如 "go-basic.obo"
gene2go = read_ncbi_gene2go("gene2go")   # 这里使用同名文件，实际请指向下载的 gene2go

# 准备 GOEnrichmentStudy 对象
goeaobj = GOEnrichmentStudy(
    list(gene2go.keys()),   # 背景基因集合
    gene2go,
    obodag,
    methods=["fisher"],
    alpha=0.05,
    multiple_test_correction="fdr_bh"
)

# 对每个簇运行 GO 富集，取显著性最高的 GO term 作为注释（与 Fig S1 对齐）
cluster_go_term = {}
for cl in sorted(cluster_series.unique()):
    genes_in_cluster = cluster_series[cluster_series == cl].index.tolist()
    # GOEA 需要使用 **基因 Symbol**（或 Ensembl） → 这里转成 Symbol
    symbols = [symbol_map.get(g, "") for g in genes_in_cluster]
    # 删除空字符
    symbols = [s for s in symbols if s]
    goea_results = goeaobj.run_study(symbols)
    # 只保留显著 (pFDR <0.05) 的结果
    sig = [r for r in goea_results if r.p_fdr_bh < 0.05]
    if sig:
        # 取最显著的 GO term 名称（Description）
        top_term = sig[0].name
    else:
        top_term = "NA"
    cluster_go_term[cl] = top_term

# 把簇号 → GO term 转成颜色映射字典（可自行挑配色）
cluster_palette = sns.color_palette("Set2", n_colors=7)
cluster_color_map = {cl: cluster_palette[i] for i, cl in enumerate(sorted(cluster_series.unique()))}
row_colors_cluster = cluster_series.map(cluster_color_map)

# 5) 画热图（行颜色 = 簇号 + GO 注释）
g2 = sns.clustermap(z_645_df,
                    row_linkage=row_link_645,
                    col_linkage=col_link,          # 使用上一节算好的列聚类
                    row_cluster=True,
                    col_cluster=True,
                    cmap="RdBu_r",
                    linewidths=0.0,
                    figsize=(10, 12),
                    row_colors=row_colors_cluster,
                    dendrogram_ratio=(.2, .2),
                    cbar_pos=(0.02, 0.8, 0.03, 0.18))

# 为每个簇添加文字标签（使用轴 ticks）
for cl, term in cluster_go_term.items():
    # 取该簇的第一个基因所在的行索引（在热图中的位置）
    first_gene = cluster_series[cluster_series == cl].index[0]
    y_pos = np.where(z_645_df.index == first_gene)[0][0]
    g2.ax_row_dendrogram.text(-0.05, y_pos,
                               f"Cluster {cl}: {term}",
                               rotation=0,
                               va="center",
                               ha="right",
                               fontsize=9,
                               color=cluster_color_map[cl])

g2.ax_heatmap.set_xlabel("RR‑1 Tissues")
g2.ax_heatmap.set_ylabel("645 genes (present in all tissues)")
plt.suptitle("Fig 1d – Z‑score heatmap of 645 fully‑observed genes", y=1.02, fontsize=14)
plt.show()
