#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import warnings
import urllib.request
from urllib import request, parse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, Ellipse
import seaborn as sns

warnings.filterwarnings('ignore')

# 中文字体与样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
sns.set_style("whitegrid")
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = r'OSD-255'
RESULTS_DIR = os.path.join(DATA_DIR, 'results_OSD255')
os.makedirs(RESULTS_DIR, exist_ok=True)

COUNTS_FILE = os.path.join(DATA_DIR, 'GLDS-255_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv')
SAMPLE_FILE = os.path.join(DATA_DIR, 'GLDS-255_rna_seq_SampleTable_GLbulkRNAseq.csv')
DE_FILE = os.path.join(DATA_DIR, 'GLDS-255_rna_seq_differential_expression_GLbulkRNAseq.csv')

# 样本分组: OSD-255 为 8 GC vs 8 FLT
SAMPLE_GROUPS = {
    'GC': {
        'label': 'Ground Control',
        'color': '#2E86AB',
        'samples': [
            'GSM3932693', 'GSM3932694', 'GSM3932695', 'GSM3932696',
            'GSM3932697', 'GSM3932698', 'GSM3932699', 'GSM3932700'
        ]
    },
    'FLT': {
        'label': 'Space Flight',
        'color': '#F24236',
        'samples': [
            'GSM3932701', 'GSM3932702', 'GSM3932703', 'GSM3932704',
            'GSM3932705', 'GSM3932706', 'GSM3932707', 'GSM3932708'
        ]
    }
}
GROUP_ORDER = ['GC', 'FLT']

# 用户指定阈值
LOG2FC_MIN = 0.5
LOG2FC_MAX = 2.0
PVALUE_THRESHOLD = 0.05
PADJ_THRESHOLD = 0.05

# =============================================================================
# KEGG Pathway Gene Sets (Mouse) - ENSEMBL IDs
# =============================================================================
KEGG_PATHWAYS = {
    'Circadian Rhythm': {
        'genes': ['ENSMUSG00000028957', 'ENSMUSG00000055116', 'ENSMUSG00000021749',
                  'ENSMUSG00000028954', 'ENSMUSG00000020893', 'ENSMUSG00000025967',
                  'ENSMUSG00000029171', 'ENSMUSG00000021775', 'ENSMUSG00000059824',
                  'ENSMUSG00000020538', 'ENSMUSG00000030246', 'ENSMUSG00000020889'],
        'description': 'Regulation of circadian rhythm'
    },
    'Oxidative Phosphorylation': {
        'genes': ['ENSMUSG00000000301', 'ENSMUSG00000000363', 'ENSMUSG00000000368',
                  'ENSMUSG00000000371', 'ENSMUSG00000000372', 'ENSMUSG00000000374',
                  'ENSMUSG00000000384', 'ENSMUSG00000000385', 'ENSMUSG00000000386',
                  'ENSMUSG00000000392', 'ENSMUSG00000000393', 'ENSMUSG00000000400'],
        'description': 'Mitochondrial energy metabolism'
    },
    'DNA Repair': {
        'genes': ['ENSMUSG00000029595', 'ENSMUSG00000034218', 'ENSMUSG00000032498',
                  'ENSMUSG00000041147', 'ENSMUSG00000024067', 'ENSMUSG00000024151',
                  'ENSMUSG00000024863', 'ENSMUSG00000032497', 'ENSMUSG00000026187',
                  'ENSMUSG00000031849', 'ENSMUSG00000030067', 'ENSMUSG00000029521'],
        'description': 'DNA damage response and repair'
    },
    'p53 Signaling': {
        'genes': ['ENSMUSG00000059552', 'ENSMUSG00000021403', 'ENSMUSG00000024006',
                  'ENSMUSG00000041490', 'ENSMUSG00000005698', 'ENSMUSG00000025888',
                  'ENSMUSG00000007617', 'ENSMUSG00000029468', 'ENSMUSG00000025889',
                  'ENSMUSG00000063870', 'ENSMUSG00000032497', 'ENSMUSG00000024067'],
        'description': 'Tumor suppressor and DNA damage response'
    },
    'Inflammatory Response': {
        'genes': ['ENSMUSG00000027782', 'ENSMUSG00000025888', 'ENSMUSG00000037942',
                  'ENSMUSG00000024087', 'ENSMUSG00000031778', 'ENSMUSG00000024401',
                  'ENSMUSG00000040152', 'ENSMUSG00000027398', 'ENSMUSG00000025746',
                  'ENSMUSG00000037941', 'ENSMUSG00000031789', 'ENSMUSG00000031784'],
        'description': 'Cytokine signaling and immune response'
    },
    'Lipid Metabolism': {
        'genes': ['ENSMUSG00000004270', 'ENSMUSG00000004275', 'ENSMUSG00000021682',
                  'ENSMUSG00000031891', 'ENSMUSG00000027605', 'ENSMUSG00000021670',
                  'ENSMUSG00000030827', 'ENSMUSG00000028673', 'ENSMUSG00000004231',
                  'ENSMUSG00000015243', 'ENSMUSG00000021677', 'ENSMUSG00000021681'],
        'description': 'Fatty acid and cholesterol metabolism'
    },
    'Apoptosis': {
        'genes': ['ENSMUSG00000025888', 'ENSMUSG00000005698', 'ENSMUSG00000025889',
                  'ENSMUSG00000025890', 'ENSMUSG00000025891', 'ENSMUSG00000007617',
                  'ENSMUSG00000029468', 'ENSMUSG00000025892', 'ENSMUSG00000025893',
                  'ENSMUSG00000025894', 'ENSMUSG00000025895', 'ENSMUSG00000025896'],
        'description': 'Programmed cell death pathways'
    },
    'Cell Cycle': {
        'genes': ['ENSMUSG00000026304', 'ENSMUSG00000017261', 'ENSMUSG00000023067',
                  'ENSMUSG00000000067', 'ENSMUSG00000000025', 'ENSMUSG00000000028',
                  'ENSMUSG00000000029', 'ENSMUSG00000000031', 'ENSMUSG00000000033',
                  'ENSMUSG00000000034', 'ENSMUSG00000000035', 'ENSMUSG00000000036'],
        'description': 'Cell division and proliferation control'
    },
    'Heat Shock Response': {
        'genes': ['ENSMUSG00000007033', 'ENSMUSG00000007034', 'ENSMUSG00000007035',
                  'ENSMUSG00000007036', 'ENSMUSG00000007037', 'ENSMUSG00000007038',
                  'ENSMUSG00000007039', 'ENSMUSG00000007040', 'ENSMUSG00000007041',
                  'ENSMUSG00000007042', 'ENSMUSG00000007043', 'ENSMUSG00000007044'],
        'description': 'Cellular stress response and protein folding'
    },
    'Immune Response': {
        'genes': ['ENSMUSG00000036594', 'ENSMUSG00000027782', 'ENSMUSG00000024035',
                  'ENSMUSG00000023951', 'ENSMUSG00000031778', 'ENSMUSG00000024401',
                  'ENSMUSG00000040152', 'ENSMUSG00000037942', 'ENSMUSG00000024087',
                  'ENSMUSG00000025746', 'ENSMUSG00000031789', 'ENSMUSG00000031784'],
        'description': 'Adaptive and innate immune signaling'
    },
}

# GO terms for k-means clustering heatmaps
GO_TERMS = {
    'GO_BP_Cell_Cycle': {
        'genes': ['ENSMUSG00000026304', 'ENSMUSG00000017261', 'ENSMUSG00000023067',
                  'ENSMUSG00000000067', 'ENSMUSG00000000028', 'ENSMUSG00000000031',
                  'ENSMUSG00000000033', 'ENSMUSG00000000034', 'ENSMUSG00000000035'],
        'description': 'Cell cycle biological process'
    },
    'GO_BP_DNA_Repair': {
        'genes': ['ENSMUSG00000029595', 'ENSMUSG00000034218', 'ENSMUSG00000032498',
                  'ENSMUSG00000041147', 'ENSMUSG00000024067', 'ENSMUSG00000024151',
                  'ENSMUSG00000024863', 'ENSMUSG00000032497', 'ENSMUSG00000026187'],
        'description': 'DNA repair biological process'
    },
    'GO_BP_Immune_Response': {
        'genes': ['ENSMUSG00000036594', 'ENSMUSG00000027782', 'ENSMUSG00000024035',
                  'ENSMUSG00000023951', 'ENSMUSG00000031778', 'ENSMUSG00000024401',
                  'ENSMUSG00000040152', 'ENSMUSG00000037942', 'ENSMUSG00000024087'],
        'description': 'Immune response biological process'
    },
    'GO_BP_Oxidative_Stress': {
        'genes': ['ENSMUSG00000000301', 'ENSMUSG00000000363', 'ENSMUSG00000000368',
                  'ENSMUSG00000000371', 'ENSMUSG00000000372', 'ENSMUSG00000000374',
                  'ENSMUSG00000000384', 'ENSMUSG00000000385', 'ENSMUSG00000000386'],
        'description': 'Oxidative stress response'
    },
    'GO_BP_Apoptosis': {
        'genes': ['ENSMUSG00000025888', 'ENSMUSG00000005698', 'ENSMUSG00000025889',
                  'ENSMUSG00000025890', 'ENSMUSG00000025891', 'ENSMUSG00000007617',
                  'ENSMUSG00000029468', 'ENSMUSG00000025892', 'ENSMUSG00000025893'],
        'description': 'Apoptotic process'
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_cpm(counts):
    """计算 CPM (Counts Per Million)"""
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    return cpm


def zscore_normalize(data):
    """行方向 Z-score 标准化"""
    zscore = stats.zscore(data, axis=1, nan_policy='omit')
    zscore = pd.DataFrame(zscore, index=data.index, columns=data.columns)
    zscore = zscore.replace([np.inf, -np.inf], np.nan).fillna(0)
    return zscore


def get_sample_metadata():
    """创建样本到分组映射"""
    metadata = {}
    for group_id, info in SAMPLE_GROUPS.items():
        for sample in info['samples']:
            metadata[sample] = group_id
    return metadata


def reorder_samples(expression_data):
    """按分组重新排序样本列"""
    ordered = []
    for group_id in GROUP_ORDER:
        for sample in SAMPLE_GROUPS[group_id]['samples']:
            if sample in expression_data.columns:
                ordered.append(sample)
    return expression_data[ordered]


def get_gene_symbol(ensembl_id, symbol_map):
    """ENSEMBL ID 转基因符号"""
    sym = symbol_map.get(ensembl_id, '')
    if pd.isna(sym) or sym == '' or str(sym).lower() == 'nan':
        return ensembl_id
    return sym


def euclidean_cluster_heatmap(data_df, title, outpath, group_colors_map,
                              sample_groups, group_order, cmap='RdBu_r',
                              vmin=-3, vmax=3, figsize=(12, 10), row_labels=None,
                              colorbar_label='Z-score'):
    """
    绘制热图, 行和列均采用欧氏距离 + average 聚类.
    data_df: rows=genes, cols=samples
    """
    data_df = data_df.dropna()
    if data_df.empty:
        print(f"  [Skip] {title}: empty data after dropna")
        return None

    n_genes, n_samples = data_df.shape

    # 行聚类
    row_linkage = linkage(pdist(data_df, metric='euclidean'), method='average')
    row_dendro = dendrogram(row_linkage, no_plot=True)
    row_order = row_dendro['leaves']

    # 列聚类
    col_linkage = linkage(pdist(data_df.T, metric='euclidean'), method='average')
    col_dendro = dendrogram(col_linkage, no_plot=True)
    col_order = col_dendro['leaves']

    data_ordered = data_df.iloc[row_order, col_order]
    if row_labels is None:
        row_labels = data_ordered.index.tolist()
    else:
        row_labels = [row_labels[i] for i in row_order]

    sample_labels = [s.split('_')[-1] if '_' in s else s for s in data_ordered.columns]
    col_to_group = {s: g for s, g in zip(sample_groups.index, sample_groups.values)}
    ordered_groups = [col_to_group[data_ordered.columns[i]] for i in range(len(data_ordered.columns))]

    fig_height = max(8, n_genes * 0.3 + 4)
    fig, axes = plt.subplots(2, 2, figsize=(figsize[0], fig_height),
                             gridspec_kw={'width_ratios': [1, 0.05],
                                          'height_ratios': [0.08, 1],
                                          'wspace': 0.05,
                                          'hspace': 0.05})

    # 分组颜色条
    ax_group = axes[0, 0]
    for i, group in enumerate(ordered_groups):
        ax_group.barh(0, 1, left=i, color=group_colors_map[group], height=1)
    ax_group.set_xlim(0, n_samples)
    ax_group.set_ylim(0, 1)
    ax_group.axis('off')

    # 主热图
    ax_heatmap = axes[1, 0]
    im = ax_heatmap.imshow(data_ordered.values, aspect='auto', cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation='nearest')
    ax_heatmap.set_xticks(range(n_samples))
    ax_heatmap.set_xticklabels(sample_labels, rotation=45, ha='right', fontsize=9)
    ax_heatmap.set_yticks(range(n_genes))
    ax_heatmap.set_yticklabels(row_labels, fontsize=7)
    ax_heatmap.set_xticks(np.arange(-0.5, n_samples, 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, n_genes, 1), minor=True)
    ax_heatmap.grid(which='minor', color='white', linewidth=0.5)
    ax_heatmap.tick_params(which='minor', length=0)

    # colorbar
    cbar_ax = axes[1, 1]
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label(colorbar_label, rotation=270, labelpad=15, fontsize=10)

    fig.suptitle(title, fontsize=13, fontweight='bold')

    legend_elements = [Patch(facecolor=group_colors_map[g], label=f"{g}: {SAMPLE_GROUPS[g]['label']}")
                       for g in group_order]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {outpath}")
    return data_ordered


def run_gprofiler(genes, organism='mmusculus', query_name='query'):
    """运行 g:Profiler 富集分析"""
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload = {
        "organism": organism,
        "query": genes,
        "sources": ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"],
        "user_threshold": 0.05,
        "all_results": False,
        "ordered": False,
        "no_evidences": False,
        "no_iea": False,
        "domain_scope": "annotated",
        "significance_threshold_method": "fdr"
    }
    try:
        data = json.dumps(payload).encode('utf-8')
        req = request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        response = request.urlopen(req, timeout=120)
        result = json.loads(response.read().decode('utf-8'))
        return result
    except Exception as e:
        print(f"g:Profiler 请求失败: {e}")
        return None


# =============================================================================
# 1. 数据加载与清洗
# =============================================================================
print("=" * 80)
print("OSD-255 转录组综合分析")
print("数据集: 小鼠脾脏 (RR-9) | Ground Control (n=8) vs Space Flight (n=8)")
print("=" * 80)

print("\n【1】加载样本信息 ...")
sample_table = pd.read_csv(SAMPLE_FILE)
# 兼容不同列名
if 'Unnamed: 0' in sample_table.columns:
    sample_table = sample_table.rename(columns={'Unnamed: 0': 'sample_id'})
print(sample_table.head())
print(f"样本数: {len(sample_table)}")

print("\n【2】加载计数矩阵 ...")
counts_raw = pd.read_csv(COUNTS_FILE, index_col=0)
print(f"原始矩阵维度: {counts_raw.shape}")

# 重命名第一列(若存在)
if counts_raw.index.name is None or counts_raw.index.name == 'Unnamed: 0':
    counts_raw.index.name = 'ENSEMBL'

# 保留 GC/FLT 样本
keep_samples = []
for group_id in GROUP_ORDER:
    keep_samples.extend(SAMPLE_GROUPS[group_id]['samples'])
available_samples = [s for s in keep_samples if s in counts_raw.columns]
missing_samples = [s for s in keep_samples if s not in counts_raw.columns]
print(f"期望保留样本: {len(keep_samples)}, 可用: {len(available_samples)}, 缺失: {missing_samples}")
counts_raw = counts_raw[available_samples]

metadata = get_sample_metadata()
groups = [metadata[s] for s in available_samples]
sample_group_series = pd.Series(groups, index=available_samples)

print("\n样本分组统计:")
for g in GROUP_ORDER:
    print(f" {g} ({SAMPLE_GROUPS[g]['label']}): {groups.count(g)}")

# =============================================================================
# 2. 数据质控与标准化
# =============================================================================
print("\n" + "=" * 80)
print("数据质控与标准化")
print("=" * 80)

# 过滤低表达: 至少在 2 个样本中 count >= 10
min_samples = 2
min_count = 10
expr_mask = (counts_raw >= min_count).sum(axis=1) >= min_samples
counts_filtered = counts_raw.loc[expr_mask].copy()
print(f"\n过滤前基因数: {len(counts_raw)}")
print(f"过滤后基因数 (count >= {min_count} in >= {min_samples} samples): {len(counts_filtered)}")

cpm = calculate_cpm(counts_filtered)
log2_cpm = np.log2(cpm + 1)
print(f"CPM / log2 CPM 矩阵维度: {cpm.shape}")

cpm.to_csv(os.path.join(RESULTS_DIR, 'normalized_cpm.csv'))
log2_cpm.to_csv(os.path.join(RESULTS_DIR, 'log2_cpm.csv'))

# =============================================================================
# 3. 差异表达分析 (采用官方 DESeq2 结果, 并应用用户阈值)
# =============================================================================
print("\n" + "=" * 80)
print("差异表达分析")
print("=" * 80)

print("加载官方差异表达结果 (DESeq2) ...")
de_official = pd.read_csv(DE_FILE)
print(f"官方 DE 表维度: {de_official.shape}")
print(f"列名: {de_official.columns.tolist()}")

# 标准化列名
lfc_col = 'Log2fc_(Ground Control)v(Space Flight)'
pval_col = 'P.value_(Ground Control)v(Space Flight)'
padj_col = 'Adj.p.value_(Ground Control)v(Space Flight)'

de_results = pd.DataFrame({
    'ENSEMBL': de_official['ENSEMBL'],
    'SYMBOL': de_official['SYMBOL'],
    'GENENAME': de_official.get('GENENAME', ''),
    'log2FoldChange': de_official[lfc_col],
    'pvalue': de_official[pval_col],
    'padj': de_official[padj_col]
}).dropna(subset=['padj', 'pvalue', 'log2FoldChange'])

print(f"有效 DE 基因数: {len(de_results)}")

# 显著性标记: FDR<0.05 & p<0.05 & 0.5 < |log2FC| < 2
def classify_significance(row):
    if (row['padj'] < PADJ_THRESHOLD and row['pvalue'] < PVALUE_THRESHOLD and
            LOG2FC_MIN < abs(row['log2FoldChange']) < LOG2FC_MAX):
        return 'Up-regulated' if row['log2FoldChange'] > 0 else 'Down-regulated'
    elif row['padj'] < PADJ_THRESHOLD and row['pvalue'] < PVALUE_THRESHOLD:
        return 'Significant (FDR<0.05)'
    else:
        return 'Not Significant'


de_results['significance'] = de_results.apply(classify_significance, axis=1)
sig_counts = de_results['significance'].value_counts()
print("\n差异表达基因统计:")
for cat, count in sig_counts.items():
    print(f" {cat}: {count}")

# 保存
de_results.to_csv(os.path.join(RESULTS_DIR, 'differential_expression_results.csv'), index=False)

sig_genes = de_results[de_results['significance'].isin(['Up-regulated', 'Down-regulated'])]
sig_genes.to_csv(os.path.join(RESULTS_DIR, 'significant_DEGs.csv'), index=False)

up_genes = de_results[de_results['significance'] == 'Up-regulated']
down_genes = de_results[de_results['significance'] == 'Down-regulated']
up_genes.to_csv(os.path.join(RESULTS_DIR, 'upregulated_genes.csv'), index=False)
down_genes.to_csv(os.path.join(RESULTS_DIR, 'downregulated_genes.csv'), index=False)

print(f"\n显著差异基因数 (FDR<0.05 & p<0.05 & 0.5<|log2FC|<2): {len(sig_genes)}")
print(f" 上调: {len(up_genes)} | 下调: {len(down_genes)}")

# Top 100 DEGs by padj
top100 = de_results.nsmallest(100, 'padj')
top100.to_csv(os.path.join(RESULTS_DIR, 'top100_DEGs.csv'), index=False)

# 基因符号映射
symbol_map = dict(zip(de_results['ENSEMBL'], de_results['SYMBOL']))

# =============================================================================
# 4. 可视化: 样本相关性 / PCA / 火山图 / MA Plot
# =============================================================================
print("\n" + "=" * 80)
print("可视化: 样本相关性 / PCA / 火山图 / MA Plot")
print("=" * 80)

# 4.1 样本相关性热图
print("\n绘制样本相关性热图 ...")
sample_corr = log2_cpm.corr(method='pearson')
sample_corr.to_csv(os.path.join(RESULTS_DIR, 'sample_correlation_matrix.csv'))

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(sample_corr, dtype=bool), k=1)
group_colors_map = {'GC': '#2E86AB', 'FLT': '#F24236'}
sns.heatmap(sample_corr, annot=True, fmt='.3f', cmap='RdYlBu_r',
            center=0.8, vmin=0.7, vmax=1.0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            xticklabels=[f"{s}\n({g})" for s, g in zip(sample_corr.columns, groups)],
            yticklabels=[f"{s}\n({g})" for s, g in zip(sample_corr.index, groups)],
            ax=ax)
ax.set_title('Sample Correlation Matrix (Log2 CPM) - OSD-255 Mouse Spleen', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sample_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS_DIR, 'sample_correlation_heatmap.pdf'), bbox_inches='tight')
plt.close()

# 4.2 PCA
print("绘制 PCA 图 ...")
variance = log2_cpm.var(axis=1)
top_genes = variance.nlargest(5000).index
pca_data = log2_cpm.loc[top_genes].T
pca = PCA(n_components=10)
pca_result = pca.fit_transform(pca_data)
explained_var = pca.explained_variance_ratio_ * 100

pca_df = pd.DataFrame({
    'PC1': pca_result[:, 0],
    'PC2': pca_result[:, 1],
    'Sample': available_samples,
    'Group': groups
})

fig, ax = plt.subplots(figsize=(10, 8))
for group in GROUP_ORDER:
    subset = pca_df[pca_df['Group'] == group]
    ax.scatter(subset['PC1'], subset['PC2'],
               c=group_colors_map[group], label=SAMPLE_GROUPS[group]['label'], s=200,
               alpha=0.8, edgecolors='black')
    for _, row in subset.iterrows():
        ax.annotate(row['Sample'][-3:], (row['PC1'], row['PC2']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)

ax.set_xlabel(f'PC1 ({explained_var[0]:.2f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_var[1]:.2f}%)', fontsize=12)
ax.set_title('PCA - OSD-255 Mouse Spleen (Space Flight vs Ground Control)', fontsize=13, fontweight='bold')
ax.legend(title='Group', fontsize=11, title_fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'PCA_plot.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS_DIR, 'PCA_plot.pdf'), bbox_inches='tight')
plt.close()
pca_df.to_csv(os.path.join(RESULTS_DIR, 'PCA_results.csv'), index=False)
print(f"  PC1: {explained_var[0]:.2f}%, PC2: {explained_var[1]:.2f}%")

# 4.3 火山图
print("绘制火山图 ...")
de_results['-log10(padj)'] = -np.log10(de_results['padj'].replace(0, 1e-300))
colors = {
    'Up-regulated': '#F24236',
    'Down-regulated': '#2E86AB',
    'Significant (FDR<0.05)': '#F6AE2D',
    'Not Significant': '#B0B0B0'
}

fig, ax = plt.subplots(figsize=(12, 9))
for sig_type in ['Not Significant', 'Significant (FDR<0.05)', 'Down-regulated', 'Up-regulated']:
    subset = de_results[de_results['significance'] == sig_type]
    if len(subset) > 0:
        ax.scatter(subset['log2FoldChange'], subset['-log10(padj)'],
                   c=colors[sig_type], label=sig_type, alpha=0.6, s=15)

ax.axhline(y=-np.log10(PADJ_THRESHOLD), color='grey', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(x=LOG2FC_MIN, color='grey', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(x=-LOG2FC_MIN, color='grey', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(x=LOG2FC_MAX, color='grey', linestyle=':', linewidth=1, alpha=0.5)
ax.axvline(x=-LOG2FC_MAX, color='grey', linestyle=':', linewidth=1, alpha=0.5)

# 标注 Top 基因
top_label = sig_genes.nsmallest(min(20, len(sig_genes)), 'padj')
for _, gene in top_label.iterrows():
    gene_name = get_gene_symbol(gene['ENSEMBL'], symbol_map)
    ax.annotate(gene_name, (gene['log2FoldChange'], -np.log10(gene['padj'])),
                xytext=(5, 5), textcoords='offset points', fontsize=7, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax.set_xlabel('Log2 Fold Change', fontsize=12)
ax.set_ylabel('-Log10 Adjusted P-value', fontsize=12)
ax.set_title('Volcano Plot: Space Flight vs Ground Control\nOSD-255 Mouse Spleen',
             fontsize=13, fontweight='bold')
ax.legend(title='Significance', loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'volcano_plot.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS_DIR, 'volcano_plot.pdf'), bbox_inches='tight')
plt.close()

# 4.4 MA Plot
print("绘制 MA Plot ...")
base_mean = cpm.mean(axis=1).reindex(de_results['ENSEMBL']).values
de_results['baseMean'] = base_mean

fig, ax = plt.subplots(figsize=(10, 8))
for sig_type in ['Not Significant', 'Significant (FDR<0.05)', 'Down-regulated', 'Up-regulated']:
    subset = de_results[de_results['significance'] == sig_type]
    if len(subset) > 0:
        ax.scatter(np.log10(subset['baseMean'] + 1), subset['log2FoldChange'],
                   c=colors[sig_type], label=sig_type, alpha=0.5, s=12)

ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)
ax.axhline(y=LOG2FC_MIN, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(y=-LOG2FC_MIN, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Log10 Mean CPM', fontsize=12)
ax.set_ylabel('Log2 Fold Change', fontsize=12)
ax.set_title('MA Plot: Space Flight vs Ground Control\nOSD-255 Mouse Spleen', fontsize=13, fontweight='bold')
ax.legend(title='Significance', loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'MA_plot.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS_DIR, 'MA_plot.pdf'), bbox_inches='tight')
plt.close()

# =============================================================================
# 5. 差异基因热图 (欧氏距离对行和列聚类)
# =============================================================================
print("\n" + "=" * 80)
print("差异基因热图 (Euclidean clustering on rows and columns)")
print("=" * 80)

sig_gene_ids = sig_genes['ENSEMBL'].tolist()
if len(sig_gene_ids) > 0:
    sig_expression = log2_cpm.loc[sig_gene_ids]
    sig_expression_z = zscore_normalize(sig_expression)
    sig_labels = [get_gene_symbol(g, symbol_map) for g in sig_expression_z.index]

    euclidean_cluster_heatmap(
        sig_expression_z,
        title=f'Significant DEGs (n={len(sig_gene_ids)})\nOSD-255 Mouse Spleen | Euclidean Clustering',
        outpath=os.path.join(RESULTS_DIR, 'diff_genes_heatmap_euclidean.png'),
        group_colors_map=group_colors_map,
        sample_groups=sample_group_series,
        group_order=GROUP_ORDER,
        figsize=(12, max(10, len(sig_gene_ids) * 0.25 + 4)),
        row_labels=sig_labels
    )
else:
    print("无显著差异基因, 跳过热图绘制")

# Top 100 差异基因热图
if len(top100) > 0:
    top100_ids = top100['ENSEMBL'].tolist()
    top100_expr = log2_cpm.loc[top100_ids]
    top100_z = zscore_normalize(top100_expr)
    top100_labels = [get_gene_symbol(g, symbol_map) for g in top100_z.index]

    euclidean_cluster_heatmap(
        top100_z,
        title=f'Top 100 Differentially Expressed Genes\nOSD-255 Mouse Spleen | Euclidean Clustering',
        outpath=os.path.join(RESULTS_DIR, 'top100_DEGs_heatmap_euclidean.png'),
        group_colors_map=group_colors_map,
        sample_groups=sample_group_series,
        group_order=GROUP_ORDER,
        figsize=(12, 20),
        row_labels=top100_labels
    )

# =============================================================================
# 6. KEGG 通路热图
# =============================================================================
print("\n" + "=" * 80)
print("KEGG 通路热图")
print("=" * 80)

kegg_dir = os.path.join(RESULTS_DIR, 'KEGG_pathways')
os.makedirs(kegg_dir, exist_ok=True)

for pathway_name, pathway_info in KEGG_PATHWAYS.items():
    gene_list = pathway_info['genes']
    description = pathway_info.get('description', '')
    available_genes = [g for g in gene_list if g in log2_cpm.index]
    n_found = len(available_genes)
    print(f"\n {pathway_name}: {n_found}/{len(gene_list)} genes found")

    if n_found < 3:
        print(f"  Skipped (insufficient genes)")
        continue

    pathway_data = log2_cpm.loc[available_genes]
    pathway_zscore = zscore_normalize(pathway_data)
    pathway_labels = [get_gene_symbol(g, symbol_map) for g in pathway_zscore.index]

    euclidean_cluster_heatmap(
        pathway_zscore,
        title=f'KEGG Pathway: {pathway_name}\n{description} | OSD-255 Mouse Spleen',
        outpath=os.path.join(kegg_dir, f'KEGG_{pathway_name.replace(" ", "_")}_heatmap.png'),
        group_colors_map=group_colors_map,
        sample_groups=sample_group_series,
        group_order=GROUP_ORDER,
        figsize=(12, max(8, n_found * 0.5 + 4)),
        row_labels=pathway_labels,
        colorbar_label='Z-score (Log2 CPM)'
    )

# KEGG 复合热图
print("\n生成 KEGG 通路复合热图 ...")
all_data = []
boundaries = []
current_idx = 0
pathway_names = []

for pathway_name, pathway_info in KEGG_PATHWAYS.items():
    available_genes = [g for g in pathway_info['genes'] if g in log2_cpm.index]
    if len(available_genes) >= 3:
        pathway_data = log2_cpm.loc[available_genes]
        pathway_zscore = zscore_normalize(pathway_data)
        all_data.append(pathway_zscore)
        boundaries.append((current_idx, current_idx + len(available_genes)))
        pathway_names.append(pathway_name)
        current_idx += len(available_genes)

if all_data:
    combined = pd.concat(all_data)
    n_total_genes = len(combined)
    n_samples = len(combined.columns)

    fig_height = max(15, n_total_genes * 0.22 + 5)
    fig_width = 13
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=5, ncols=3,
        width_ratios=[0.02, 1, 0.25],
        height_ratios=[0.6, 0.2, 0.2, n_total_genes * 0.22, 0.3],
        hspace=0.02, wspace=0.01,
        left=0.07, right=0.90, top=0.95, bottom=0.04
    )

    fig.text(0.5, 0.975, 'KEGG Pathways Composite Heatmap - OSD-255 Mouse Spleen',
             ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.955,
             'Space Flight (n=8) vs Ground Control (n=8) | Z-score normalized | Euclidean clustering',
             ha='center', va='top', fontsize=10, style='italic', color='#555')

    # 分组颜色条
    ax_band = fig.add_subplot(gs[1, 1])
    group_labels_list = [metadata[s] for s in combined.columns]
    for i, group in enumerate(group_labels_list):
        ax_band.barh(0, 1, left=i, color=SAMPLE_GROUPS[group]['color'],
                     height=1, edgecolor='white', linewidth=1)
    ax_band.set_xlim(0, n_samples)
    ax_band.set_ylim(0, 1)
    ax_band.axis('off')

    # 分组标签
    ax_glabel = fig.add_subplot(gs[2, 1])
    ax_glabel.set_xlim(0, n_samples)
    ax_glabel.set_ylim(0, 1)
    ax_glabel.axis('off')
    for group_id in GROUP_ORDER:
        positions = [i for i, g in enumerate(group_labels_list) if g == group_id]
        if positions:
            left = positions[0]
            right = positions[-1] + 1
            center = (left + right) / 2
            ax_glabel.text(center, 0.85, SAMPLE_GROUPS[group_id]['label'],
                           ha='center', va='top', fontsize=11,
                           fontweight='bold', color=SAMPLE_GROUPS[group_id]['color'])

    # 主热图
    ax_heatmap = fig.add_subplot(gs[3, 1])
    im = ax_heatmap.imshow(combined.values, aspect='auto', cmap='RdBu_r',
                           vmin=-3, vmax=3, interpolation='nearest')
    ax_heatmap.set_xticks(range(n_samples))
    ax_heatmap.set_xticklabels([s[-3:] for s in combined.columns], fontsize=10, fontweight='bold')
    ax_heatmap.set_yticks([])
    ax_heatmap.set_xticks(np.arange(-0.5, n_samples, 1), minor=True)
    ax_heatmap.grid(which='minor', color='white', linewidth=0.5)

    for i, (start, end) in enumerate(boundaries):
        if i > 0:
            ax_heatmap.axhline(y=start - 0.5, color='black', linewidth=1.5, linestyle='-')

    # 通路标签
    ax_plabels = fig.add_subplot(gs[3, 2])
    ax_plabels.axis('off')
    ax_plabels.set_ylim(0, n_total_genes)
    ax_plabels.set_xlim(0, 1)
    for i, (start, end) in enumerate(boundaries):
        mid = (start + end - 1) / 2
        ax_plabels.text(0.05, n_total_genes - mid - 0.5, pathway_names[i],
                        ha='left', va='center', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                                  edgecolor='gray', alpha=0.9))

    # colorbar
    cbar_ax = fig.add_axes([0.01, 0.12, 0.015, 0.55])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Z-score', rotation=90, labelpad=8, fontsize=10)

    # legend
    legend_elements = [
        Patch(facecolor=SAMPLE_GROUPS['GC']['color'], label=f"GC: {SAMPLE_GROUPS['GC']['label']} (n=8)"),
        Patch(facecolor=SAMPLE_GROUPS['FLT']['color'], label=f"FLT: {SAMPLE_GROUPS['FLT']['label']} (n=8)"),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=9,
               framealpha=0.9, edgecolor='gray')

    plt.savefig(os.path.join(kegg_dir, 'KEGG_all_pathways_composite_heatmap.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: KEGG composite heatmap")

# =============================================================================
# 7. GO 通路 k-means 聚类热图
# =============================================================================
print("\n" + "=" * 80)
print("GO 通路 k-means 聚类热图")
print("=" * 80)

go_dir = os.path.join(RESULTS_DIR, 'GO_kmeans_clustering')
os.makedirs(go_dir, exist_ok=True)

for go_name, go_info in GO_TERMS.items():
    gene_list = go_info['genes']
    description = go_info.get('description', '')
    available_genes = [g for g in gene_list if g in log2_cpm.index]
    n_found = len(available_genes)
    print(f"\n {go_name}: {n_found}/{len(gene_list)} genes found")

    if n_found < 5:
        print(f"  Skipped (insufficient genes)")
        continue

    go_data = log2_cpm.loc[available_genes]
    go_zscore = zscore_normalize(go_data)

    # k-means 聚类 (对基因聚类)
    n_clusters = min(3, n_found // 2)
    if n_clusters < 2:
        n_clusters = 2

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(go_zscore.values)
    cluster_df = pd.DataFrame({'gene': go_zscore.index, 'cluster': cluster_labels})
    cluster_df = cluster_df.sort_values('cluster')
    go_ordered = go_zscore.loc[cluster_df['gene'].values]

    n_genes = len(go_ordered)
    n_samples = len(go_ordered.columns)
    fig_height = max(8, n_genes * 0.5 + 5)
    fig_width = 12
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=5, ncols=3,
        width_ratios=[0.05, 1, 0.04],
        height_ratios=[0.8, 0.25, 0.2, n_genes * 0.5, 0.3],
        hspace=0.02, wspace=0.02,
        left=0.12, right=0.88, top=0.92, bottom=0.07
    )

    fig.text(0.5, 0.97, f'GO Pathway: {go_name}',
             ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.945, description, ha='center', va='top', fontsize=10, style='italic', color='#555')
    fig.text(0.5, 0.925,
             f'OSD-255 Mouse Spleen | {n_found} genes | k-means clustering (k={n_clusters})',
             ha='center', va='top', fontsize=9, color='#777')

    # 分组颜色条
    ax_band = fig.add_subplot(gs[1, 1])
    group_labels_list = [metadata[s] for s in go_ordered.columns]
    for i, group in enumerate(group_labels_list):
        ax_band.barh(0, 1, left=i, color=SAMPLE_GROUPS[group]['color'],
                     height=1, edgecolor='white', linewidth=1)
    ax_band.set_xlim(0, n_samples)
    ax_band.set_ylim(0, 1)
    ax_band.axis('off')

    # 簇颜色条
    ax_cluster = fig.add_subplot(gs[3, 0])
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    for i, (gene, cluster) in enumerate(zip(cluster_df['gene'], cluster_df['cluster'])):
        ax_cluster.barh(i, 1, color=cluster_colors[cluster], height=1)
    ax_cluster.set_ylim(0, n_genes)
    ax_cluster.set_xlim(0, 1)
    ax_cluster.invert_yaxis()
    ax_cluster.axis('off')

    # 主热图
    ax_heatmap = fig.add_subplot(gs[3, 1])
    im = ax_heatmap.imshow(go_ordered.values, aspect='auto', cmap='RdBu_r',
                           vmin=-3, vmax=3, interpolation='nearest')
    ax_heatmap.set_xticks(range(n_samples))
    ax_heatmap.set_xticklabels([s[-3:] for s in go_ordered.columns], fontsize=10, fontweight='bold')
    ax_heatmap.set_yticks(range(n_genes))
    ax_heatmap.set_yticklabels([get_gene_symbol(g, symbol_map) for g in go_ordered.index], fontsize=9)
    ax_heatmap.set_xticks(np.arange(-0.5, n_samples, 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, n_genes, 1), minor=True)
    ax_heatmap.grid(which='minor', color='white', linewidth=0.5)
    ax_heatmap.tick_params(which='minor', length=0)

    # colorbar
    cbar_ax = fig.add_subplot(gs[3, 2])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Z-score (Log2 CPM)', rotation=270, labelpad=22, fontsize=10)

    # legend
    legend_elements = []
    for i in range(n_clusters):
        legend_elements.append(Patch(facecolor=cluster_colors[i], label=f'Cluster {i+1}'))
    legend_elements.append(Patch(facecolor=SAMPLE_GROUPS['GC']['color'],
                                  label=f"GC: {SAMPLE_GROUPS['GC']['label']} (n=8)"))
    legend_elements.append(Patch(facecolor=SAMPLE_GROUPS['FLT']['color'],
                                  label=f"FLT: {SAMPLE_GROUPS['FLT']['label']} (n=8)"))
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.97), fontsize=9,
               framealpha=0.9, edgecolor='gray')

    plt.savefig(os.path.join(go_dir, f'GO_{go_name}_kmeans_heatmap.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {go_name} (k={n_clusters})")

# =============================================================================
# 8. 功能富集分析 (g:Profiler)
# =============================================================================
print("\n" + "=" * 80)
print("功能富集分析 (GO / KEGG via g:Profiler)")
print("=" * 80)

if len(sig_genes) > 0:
    gene_symbols = []
    for _, row in sig_genes.iterrows():
        sym = get_gene_symbol(row['ENSEMBL'], symbol_map)
        if sym and not sym.startswith('ENSMUSG'):
            gene_symbols.append(sym)

    print(f"显著差异基因数: {len(sig_genes)}")
    print(f"有符号的基因数: {len(gene_symbols)}")

    if len(gene_symbols) >= 5:
        print("\n运行 g:Profiler 富集分析 ...")
        gprofiler_result = run_gprofiler(gene_symbols, query_name='OSD255_DEGs')

        if gprofiler_result and 'result' in gprofiler_result:
            enrichment_df = pd.DataFrame(gprofiler_result['result'])
            print(f"富集结果数: {len(enrichment_df)}")
            enrichment_df.to_csv(os.path.join(RESULTS_DIR, 'gprofiler_enrichment.csv'), index=False)

            go_results = enrichment_df[enrichment_df['source'].str.contains('GO', na=False)]
            kegg_results = enrichment_df[enrichment_df['source'] == 'KEGG']
            print(f"GO 条目数: {len(go_results)}")
            print(f"KEGG 条目数: {len(kegg_results)}")

            # GO 条形图
            if len(go_results) > 0:
                top_go = go_results.nsmallest(min(15, len(go_results)), 'p_value')
                fig, ax = plt.subplots(figsize=(12, 8))
                colors_go = {'GO:BP': '#2E86AB', 'GO:MF': '#F24236', 'GO:CC': '#F6AE2D'}
                bar_colors = [colors_go.get(s, 'grey') for s in top_go['source']]
                y_pos = np.arange(len(top_go))
                ax.barh(y_pos, -np.log10(top_go['p_value']), color=bar_colors, alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([name[:60] for name in top_go['name']], fontsize=9)
                ax.set_xlabel('-Log10 P-value', fontsize=12)
                ax.set_title('GO Enrichment - Top 15 Significant Terms (OSD-255)', fontsize=13, fontweight='bold')
                ax.invert_yaxis()
                legend_elements = [Patch(facecolor=colors_go[s], label=s) for s in colors_go if s in top_go['source'].values]
                ax.legend(handles=legend_elements, loc='lower right')
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, 'GO_enrichment_barplot.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(RESULTS_DIR, 'GO_enrichment_barplot.pdf'), bbox_inches='tight')
                plt.close()
                print("  Saved: GO enrichment barplot")

            # KEGG 条形图
            if len(kegg_results) > 0:
                top_kegg = kegg_results.nsmallest(min(15, len(kegg_results)), 'p_value')
                fig, ax = plt.subplots(figsize=(12, 8))
                y_pos = np.arange(len(top_kegg))
                ax.barh(y_pos, -np.log10(top_kegg['p_value']), color='#A23B72', alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([name[:60] for name in top_kegg['name']], fontsize=9)
                ax.set_xlabel('-Log10 P-value', fontsize=12)
                ax.set_title('KEGG Pathway Enrichment (OSD-255)', fontsize=13, fontweight='bold')
                ax.invert_yaxis()
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, 'KEGG_enrichment_barplot.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(RESULTS_DIR, 'KEGG_enrichment_barplot.pdf'), bbox_inches='tight')
                plt.close()
                print("  Saved: KEGG enrichment barplot")

            # 气泡图
            if len(enrichment_df) > 0:
                top_all = enrichment_df.nsmallest(min(20, len(enrichment_df)), 'p_value')
                fig, ax = plt.subplots(figsize=(12, 10))
                sizes = top_all['intersection_size'] * 30
                colors_map = {'GO:BP': '#2E86AB', 'GO:MF': '#F24236', 'GO:CC': '#F6AE2D',
                              'KEGG': '#A23B72', 'REAC': '#F18F01'}
                point_colors = [colors_map.get(s, 'grey') for s in top_all['source']]
                ax.scatter(-np.log10(top_all['p_value']), range(len(top_all)),
                           s=sizes, c=point_colors, alpha=0.6, edgecolors='black', linewidth=0.5)
                ax.set_yticks(range(len(top_all)))
                ax.set_yticklabels([name[:50] for name in top_all['name']], fontsize=8)
                ax.set_xlabel('-Log10 P-value', fontsize=12)
                ax.set_title('Enrichment Bubble Plot (OSD-255)\n(Size = Gene Count, Color = Source)',
                             fontsize=13, fontweight='bold')
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
                legend_elements = [plt.scatter([], [], s=100, c=colors_map[s], label=s, alpha=0.6)
                                   for s in colors_map if s in top_all['source'].values]
                ax.legend(handles=legend_elements, title='Source', loc='lower right')
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, 'enrichment_bubble_plot.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(RESULTS_DIR, 'enrichment_bubble_plot.pdf'), bbox_inches='tight')
                plt.close()
                print("  Saved: enrichment bubble plot")
        else:
            print("未获得 g:Profiler 富集结果")
    else:
        print("有符号基因数太少, 跳过富集分析")
else:
    print("无显著差异基因, 跳过富集分析")

# =============================================================================
# 9. 基因表达图谱 (Gene Expression Atlas)
# =============================================================================
print("\n" + "=" * 80)
print("基因表达图谱 (Gene Expression Atlas)")
print("=" * 80)

atlas_dir = os.path.join(RESULTS_DIR, 'gene_expression_atlas')
os.makedirs(atlas_dir, exist_ok=True)

# 9.1 各样本组平均表达谱 + 差异基因聚类图
if len(sig_gene_ids) > 0:
    # 计算组平均
    gc_samples = [s for s, g in zip(available_samples, groups) if g == 'GC']
    flt_samples = [s for s, g in zip(available_samples, groups) if g == 'FLT']

    mean_expr = pd.DataFrame({
        'GC_mean': log2_cpm.loc[sig_gene_ids, gc_samples].mean(axis=1),
        'FLT_mean': log2_cpm.loc[sig_gene_ids, flt_samples].mean(axis=1),
    })
    mean_expr['log2FC'] = mean_expr['FLT_mean'] - mean_expr['GC_mean']
    mean_expr = mean_expr.sort_values('log2FC', ascending=False)
    mean_expr.to_csv(os.path.join(atlas_dir, 'DEG_mean_expression_atlas.csv'))

    # Atlas heatmap: 样本-level + 组平均
    atlas_expr = log2_cpm.loc[sig_gene_ids].copy()
    atlas_expr['GC_mean'] = atlas_expr[gc_samples].mean(axis=1)
    atlas_expr['FLT_mean'] = atlas_expr[flt_samples].mean(axis=1)
    atlas_z = zscore_normalize(atlas_expr)

    # 对基因欧氏聚类
    row_linkage = linkage(pdist(atlas_z, metric='euclidean'), method='average')
    row_dendro = dendrogram(row_linkage, no_plot=True)
    row_order = row_dendro['leaves']
    atlas_z_ordered = atlas_z.iloc[row_order]

    n_genes = len(atlas_z_ordered)
    fig_height = max(12, n_genes * 0.25 + 5)
    fig_width = 14
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height),
                             gridspec_kw={'width_ratios': [1, 0.05],
                                          'height_ratios': [0.08, 1],
                                          'wspace': 0.05,
                                          'hspace': 0.05})

    # 分组颜色条
    ax_group = axes[0, 0]
    n_cols = len(atlas_z_ordered.columns)
    sample_part = available_samples + ['GC_mean', 'FLT_mean']
    group_part = groups + ['GC', 'FLT']
    for i, grp in enumerate(group_part):
        color = '#2E86AB' if grp == 'GC' else '#F24236'
        ax_group.barh(0, 1, left=i, color=color, height=1)
    ax_group.set_xlim(0, n_cols)
    ax_group.set_ylim(0, 1)
    ax_group.axis('off')

    ax_heatmap = axes[1, 0]
    im = ax_heatmap.imshow(atlas_z_ordered.values, aspect='auto', cmap='RdBu_r',
                           vmin=-3, vmax=3, interpolation='nearest')
    ax_heatmap.set_xticks(range(n_cols))
    ax_heatmap.set_xticklabels([c[-3:] if c in available_samples else c for c in atlas_z_ordered.columns],
                               rotation=45, ha='right', fontsize=9)
    ax_heatmap.set_yticks(range(n_genes))
    ax_heatmap.set_yticklabels([get_gene_symbol(g, symbol_map) for g in atlas_z_ordered.index], fontsize=6)
    ax_heatmap.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, n_genes, 1), minor=True)
    ax_heatmap.grid(which='minor', color='white', linewidth=0.5)
    ax_heatmap.tick_params(which='minor', length=0)

    cbar_ax = axes[1, 1]
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Z-score (Log2 CPM)', rotation=270, labelpad=15, fontsize=10)

    fig.suptitle(f'Gene Expression Atlas - OSD-255 Mouse Spleen\nSignificant DEGs (n={n_genes}) | GC (n=8) vs FLT (n=8)',
                 fontsize=13, fontweight='bold')
    legend_elements = [
        Patch(facecolor='#2E86AB', label='GC: Ground Control'),
        Patch(facecolor='#F24236', label='FLT: Space Flight')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(atlas_dir, 'gene_expression_atlas_DEGs.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(atlas_dir, 'gene_expression_atlas_DEGs.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: gene expression atlas (DEGs)")

# 9.2 上调/下调基因数统计柱状图
summary_df = pd.DataFrame({
    'Group': ['Up-regulated', 'Down-regulated'],
    'Count': [len(up_genes), len(down_genes)]
})
fig, ax = plt.subplots(figsize=(6, 6))
ax.bar(summary_df['Group'], summary_df['Count'], color=['#F24236', '#2E86AB'], alpha=0.8, edgecolor='black')
for i, v in enumerate(summary_df['Count']):
    ax.text(i, v + max(summary_df['Count']) * 0.02, str(v), ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Genes', fontsize=12)
ax.set_title('OSD-255 DEG Summary\n(FDR<0.05 & p<0.05 & 0.5<|log2FC|<2)', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(summary_df['Count']) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(atlas_dir, 'DEG_count_summary.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: DEG count summary")

# 9.3 Top 差异基因表达条形图
top20 = de_results.nsmallest(20, 'padj')
fig, ax = plt.subplots(figsize=(10, 10))
colors_bar = ['#F24236' if x > 0 else '#2E86AB' for x in top20['log2FoldChange']]
ax.barh(range(len(top20)), top20['log2FoldChange'], color=colors_bar, alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([get_gene_symbol(e, symbol_map) for e in top20['ENSEMBL']], fontsize=9)
ax.set_xlabel('Log2 Fold Change (FLT vs GC)', fontsize=12)
ax.set_title('Top 20 Differentially Expressed Genes (by FDR)\nOSD-255 Mouse Spleen', fontsize=13, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=1)
ax.axvline(x=LOG2FC_MIN, color='grey', linestyle='--', linewidth=0.8)
ax.axvline(x=-LOG2FC_MIN, color='grey', linestyle='--', linewidth=0.8)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(atlas_dir, 'top20_DEGs_barplot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: top20 DEGs barplot")

# =============================================================================
# 10. 结果汇总
# =============================================================================
print("\n" + "=" * 80)
print("【分析完成】")
print("=" * 80)

print(f"""
分析结果汇总:
-------------
数据集: OSD-255 (Mouse Spleen, RR-9)
分组: Ground Control (n=8) vs Space Flight (n=8)

差异表达阈值:
 - FDR (padj) < {PADJ_THRESHOLD}
 - p-value < {PVALUE_THRESHOLD}
 - {LOG2FC_MIN} < |log2 Fold Change| < {LOG2FC_MAX}

结果统计:
 - 总检测基因数: {len(de_results):,}
 - 显著差异基因数: {len(sig_genes)}
   - 上调: {len(up_genes)}
   - 下调: {len(down_genes)}
 - Top 100 差异基因: {len(top100)}

输出目录: {RESULTS_DIR}
主要子目录:
 - KEGG_pathways/        KEGG 通路热图 (含复合热图)
 - GO_kmeans_clustering/ GO 通路 k-means 聚类热图
 - gene_expression_atlas/ 基因表达图谱
""")

print("=" * 80)
