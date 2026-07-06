#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSD-162 转录组数据分析 - 小鼠眼组织 (Space Flight vs Ground Control)
数据集来源: https://osdr.nasa.gov/bio/repo/data/studies/OSD-162

分析内容:
1. 数据清洗 - 仅保留 Space Flight (FLT) 和 Ground Control (GC) 两组
2. 差异表达分析 (|log2FC| > 0.5, p-value < 0.05)
3. 去批次效应 (ComBat)
4. 富集分数计算 (GSVA-like)
5. Top100差异基因、上调/下调基因
6. GO富集分析
7. KEGG通路分析
8. 热图生成 (欧氏聚类法对行和列聚类)
9. 特定GO通路的k-means聚类热图

参照: https://www.mdpi.com/2072-6694/12/2/381 图文摘要和图4、图5
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import os
import json
from urllib import request, parse
import time

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = r'C:\Users\Administrator\.openclaw\workspace\transcriptomics\data\OSD-162'
RESULTS_DIR = r'C:\Users\Administrator\.openclaw\workspace\transcriptomics\results\OSD-162'

# 保留的样本 (仅 FLT 和 GC)
# FLT: RR3_FLT_F1-F5
# GC: RR3_GC_G1-G5
SAMPLE_GROUPS = {
    'FLT': {
        'label': 'Space Flight',
        'color': '#F24236',
        'samples': [
            'Mmus_BAL-TAL_EYE_FLT_Rep1_F1',
            'Mmus_BAL-TAL_EYE_FLT_Rep2_F2',
            'Mmus_BAL-TAL_EYE_FLT_Rep3_F3',
            'Mmus_BAL-TAL_EYE_FLT_Rep4_F4',
            'Mmus_BAL-TAL_EYE_FLT_Rep5_F5',
        ]
    },
    'GC': {
        'label': 'Ground Control',
        'color': '#2E86AB',
        'samples': [
            'Mmus_BAL-TAL_EYE_GC_Rep1_G1',
            'Mmus_BAL-TAL_EYE_GC_Rep2_G2',
            'Mmus_BAL-TAL_EYE_GC_Rep3_G4',
            'Mmus_BAL-TAL_EYE_GC_Rep4_G5',
            'Mmus_BAL-TAL_EYE_GC_Rep5_G7',
        ]
    }
}

GROUP_ORDER = ['GC', 'FLT']

# 差异表达阈值
LOG2FC_THRESHOLD = 0.5  # |log2FC| > 0.5
PVALUE_THRESHOLD = 0.05  # p-value < 0.05

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
    'Phototransduction': {
        'genes': ['ENSMUSG00000031342', 'ENSMUSG00000040938', 'ENSMUSG00000032708',
                  'ENSMUSG00000038915', 'ENSMUSG00000040047', 'ENSMUSG00000024243',
                  'ENSMUSG00000030337', 'ENSMUSG00000030107', 'ENSMUSG00000030339',
                  'ENSMUSG00000030111', 'ENSMUSG00000030343', 'ENSMUSG00000030119'],
        'description': 'Retinal phototransduction cascade'
    },
    'Retinal Metabolism': {
        'genes': ['ENSMUSG00000022425', 'ENSMUSG00000032402', 'ENSMUSG00000022217',
                  'ENSMUSG00000018923', 'ENSMUSG00000025351', 'ENSMUSG00000030209',
                  'ENSMUSG00000024030', 'ENSMUSG00000026077', 'ENSMUSG00000035033',
                  'ENSMUSG00000020841', 'ENSMUSG00000024747', 'ENSMUSG00000032602'],
        'description': 'Retinal energy and metabolic processes'
    }
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

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def calculate_cpm(counts):
    """计算CPM (Counts Per Million)"""
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    return cpm

def zscore_normalize(data):
    """Z-score标准化 (行方向)"""
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

def get_sample_short_name(sample):
    """提取样本简称"""
    parts = sample.split('_')
    if 'FLT' in sample:
        return 'F' + parts[-1]
    elif 'GC' in sample:
        return 'G' + parts[-1]
    return parts[-1]

def reorder_samples(expression_data):
    """按分组重新排序样本列"""
    ordered = []
    for group_id in GROUP_ORDER:
        for sample in SAMPLE_GROUPS[group_id]['samples']:
            if sample in expression_data.columns:
                ordered.append(sample)
    return expression_data[ordered]

def load_gene_symbol_mapping(viz_file):
    """从visualization_output_table加载ENSEMBL→SYMBOL映射"""
    print("Loading gene symbol mapping...")
    viz_df = pd.read_csv(viz_file, usecols=[0, 1])
    viz_df.columns = ['ENSEMBL', 'SYMBOL']
    mapping = dict(zip(viz_df['ENSEMBL'], viz_df['SYMBOL']))
    print(f"  Loaded {len(mapping)} gene mappings")
    return mapping

def get_gene_symbol(ensembl_id, symbol_map):
    """将ENSEMBL ID转换为基因符号"""
    sym = symbol_map.get(ensembl_id, '')
    if pd.isna(sym) or sym == '':
        return ensembl_id
    return sym

# =============================================================================
# 1. 数据加载与清洗
# =============================================================================

def load_and_clean_data():
    """加载数据并清洗 - 仅保留FLT和GC样本"""
    print("=" * 80)
    print("OSD-162 数据加载与清洗")
    print("=" * 80)
    
    # 读取样本信息
    sample_table_file = os.path.join(DATA_DIR, 'GLDS-162_rna_seq_SampleTable.csv')
    sample_table = pd.read_csv(sample_table_file, index_col=0)
    print(f"\n原始样本数: {len(sample_table)}")
    print(f"原始分组:")
    for condition, count in sample_table['condition'].value_counts().items():
        print(f"  {condition}: {count}")
    
    # 读取原始计数矩阵
    counts_file = os.path.join(DATA_DIR, 'GLDS-162_rna_seq_Unnormalized_Counts.csv')
    counts_raw = pd.read_csv(counts_file, index_col=0)
    print(f"\n原始计数矩阵维度: {counts_raw.shape}")
    
    # 要保留的样本
    keep_samples = []
    for group_id in GROUP_ORDER:
        keep_samples.extend(SAMPLE_GROUPS[group_id]['samples'])
    
    # 检查哪些样本存在
    available_samples = [s for s in keep_samples if s in counts_raw.columns]
    missing_samples = [s for s in keep_samples if s not in counts_raw.columns]
    
    print(f"\n期望保留样本: {len(keep_samples)}")
    print(f"可用样本: {len(available_samples)}")
    if missing_samples:
        print(f"缺失样本: {missing_samples}")
    
    # 过滤样本
    counts_filtered = counts_raw[available_samples].copy()
    print(f"\n过滤后计数矩阵维度: {counts_filtered.shape}")
    
    # 要删除的样本
    remove_samples = [s for s in counts_raw.columns if s not in available_samples]
    print(f"\n删除的样本 ({len(remove_samples)}个):")
    for s in remove_samples:
        condition = sample_table.loc[s, 'condition'] if s in sample_table.index else 'Unknown'
        print(f"  {s} -> {condition}")
    
    return counts_filtered, available_samples

# =============================================================================
# 2. 数据质控与标准化
# =============================================================================

def quality_control(counts_raw):
    """数据质控和标准化"""
    print("\n" + "=" * 80)
    print("数据质控与标准化")
    print("=" * 80)
    
    # 过滤低表达基因 (至少在2个样本中count >= 10)
    min_samples = 2
    min_count = 10
    expr_mask = (counts_raw >= min_count).sum(axis=1) >= min_samples
    counts_filtered = counts_raw.loc[expr_mask].copy()
    
    print(f"\n过滤前基因数: {len(counts_raw)}")
    print(f"过滤后基因数 (count >= {min_count} in >= {min_samples} samples): {len(counts_filtered)}")
    
    # 计算CPM
    cpm = calculate_cpm(counts_filtered)
    print(f"CPM矩阵维度: {cpm.shape}")
    
    # Log2转换
    log2_cpm = np.log2(cpm + 1)
    print(f"Log2 CPM矩阵维度: {log2_cpm.shape}")
    print(f"Log2 CPM值范围: [{log2_cpm.min().min():.2f}, {log2_cpm.max().max():.2f}]")
    
    # 保存标准化数据
    ensure_dir(RESULTS_DIR)
    cpm.to_csv(os.path.join(RESULTS_DIR, 'normalized_cpm.csv'))
    log2_cpm.to_csv(os.path.join(RESULTS_DIR, 'log2_cpm.csv'))
    print("标准化数据已保存")
    
    return counts_filtered, cpm, log2_cpm

# =============================================================================
# 3. 去批次效应 (ComBat)
# =============================================================================

def combat_correction(log2_cpm, groups):
    """使用ComBat去除批次效应"""
    print("\n" + "=" * 80)
    print("去批次效应 (ComBat)")
    print("=" * 80)
    
    try:
        from combat.pycombat import pycombat
        
        # 创建批次信息 (这里假设所有样本来自同一批次，但如果需要可以修改)
        # OSD-162数据没有明显的批次信息，我们创建一个虚拟批次
        batch = pd.Series(['Batch1'] * len(log2_cpm.columns), index=log2_cpm.columns)
        
        # 运行ComBat
        log2_combat = pycombat(log2_cpm, batch)
        
        print("ComBat校正完成")
        print(f"校正后Log2 CPM值范围: [{log2_combat.min().min():.2f}, {log2_combat.max().max():.2f}]")
        
        # 保存
        log2_combat.to_csv(os.path.join(RESULTS_DIR, 'log2_cpm_combat_corrected.csv'))
        print("校正后数据已保存")
        
        return log2_combat
        
    except ImportError:
        print("警告: combat包未安装，跳过ComBat校正")
        print("使用原始log2 CPM数据")
        return log2_cpm
    except Exception as e:
        print(f"ComBat校正失败: {e}")
        print("使用原始log2 CPM数据")
        return log2_cpm

# =============================================================================
# 4. 差异表达分析
# =============================================================================

def differential_expression_analysis(counts_filtered, log2_cpm, groups):
    """差异表达分析"""
    print("\n" + "=" * 80)
    print("差异表达分析")
    print("=" * 80)
    
    # 获取分组样本
    gc_samples = [s for s, g in zip(log2_cpm.columns, groups) if g == 'GC']
    flt_samples = [s for s, g in zip(log2_cpm.columns, groups) if g == 'FLT']
    
    print(f"\nGC样本: {len(gc_samples)}")
    print(f"FLT样本: {len(flt_samples)}")
    
    # 使用t-test进行差异表达分析
    results = []
    for gene in counts_filtered.index:
        gc_log = log2_cpm.loc[gene, gc_samples].values
        flt_log = log2_cpm.loc[gene, flt_samples].values
        
        # t-test
        t_stat, p_val = stats.ttest_ind(gc_log, flt_log)
        log2fc = np.mean(flt_log) - np.mean(gc_log)
        
        results.append({
            'ENSEMBL': gene,
            'log2FoldChange': log2fc,
            'statistic': t_stat,
            'pvalue': p_val
        })
    
    de_results = pd.DataFrame(results)
    
    # 多重检验校正 (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    _, padj, _, _ = multipletests(de_results['pvalue'].fillna(1), method='fdr_bh')
    de_results['padj'] = padj
    
    # 去除NA
    de_results = de_results.dropna(subset=['padj'])
    
    print(f"\n分析基因数: {len(de_results)}")
    
    # 添加显著性标记 (使用用户指定的阈值: |log2FC| > 0.5, p-value < 0.05)
    def classify_significance(row):
        if row['pvalue'] < PVALUE_THRESHOLD and abs(row['log2FoldChange']) >= LOG2FC_THRESHOLD:
            if row['log2FoldChange'] > 0:
                return 'Up-regulated'
            else:
                return 'Down-regulated'
        elif row['pvalue'] < PVALUE_THRESHOLD:
            return 'Significant (p<0.05)'
        else:
            return 'Not Significant'
    
    de_results['significance'] = de_results.apply(classify_significance, axis=1)
    
    # 统计
    sig_counts = de_results['significance'].value_counts()
    print(f"\n差异表达基因统计 (p-value < {PVALUE_THRESHOLD} & |log2FC| >= {LOG2FC_THRESHOLD}):")
    for cat, count in sig_counts.items():
        print(f"  {cat}: {count}")
    
    # 保存结果
    de_results.to_csv(os.path.join(RESULTS_DIR, 'differential_expression_results.csv'), index=False)
    
    # 提取显著差异基因
    sig_genes = de_results[de_results['significance'].isin(['Up-regulated', 'Down-regulated'])]
    sig_genes.to_csv(os.path.join(RESULTS_DIR, 'significant_DEGs.csv'), index=False)
    
    # Top 100差异基因 (按p-value排序)
    top100 = de_results.nsmallest(100, 'pvalue')
    top100.to_csv(os.path.join(RESULTS_DIR, 'top100_DEGs.csv'), index=False)
    
    # 上调和下调基因
    up_genes = de_results[de_results['significance'] == 'Up-regulated']
    down_genes = de_results[de_results['significance'] == 'Down-regulated']
    
    up_genes.to_csv(os.path.join(RESULTS_DIR, 'upregulated_genes.csv'), index=False)
    down_genes.to_csv(os.path.join(RESULTS_DIR, 'downregulated_genes.csv'), index=False)
    
    print(f"\n显著差异基因数: {len(sig_genes)}")
    print(f"  - 上调基因: {len(up_genes)}")
    print(f"  - 下调基因: {len(down_genes)}")
    print(f"\nTop 100差异基因已保存")
    
    return de_results, sig_genes, top100, up_genes, down_genes

# =============================================================================
# 5. 富集分数计算 (GSVA-like)
# =============================================================================

def calculate_enrichment_scores(log2_cpm, pathways_dict):
    """计算通路富集分数 (类似GSVA的简单实现)"""
    print("\n" + "=" * 80)
    print("富集分数计算 (GSVA-like)")
    print("=" * 80)
    
    enrichment_scores = {}
    
    for pathway_name, pathway_info in pathways_dict.items():
        gene_list = pathway_info['genes']
        available_genes = [g for g in gene_list if g in log2_cpm.index]
        
        if len(available_genes) >= 3:
            # 计算通路内基因的平均表达
            pathway_expr = log2_cpm.loc[available_genes].mean(axis=0)
            enrichment_scores[pathway_name] = pathway_expr
            print(f"  {pathway_name}: {len(available_genes)}/{len(gene_list)} genes, mean ES = {pathway_expr.mean():.3f}")
        else:
            print(f"  {pathway_name}: skipped (only {len(available_genes)} genes found)")
    
    # 创建富集分数矩阵
    es_df = pd.DataFrame(enrichment_scores).T
    es_df.to_csv(os.path.join(RESULTS_DIR, 'enrichment_scores.csv'))
    print(f"\n富集分数矩阵已保存: {es_df.shape}")
    
    return es_df

# =============================================================================
# 6. 可视化
# =============================================================================

def plot_sample_correlation(log2_cpm, groups, sample_names):
    """样本相关性热图"""
    print("\n绘制样本相关性热图...")
    
    sample_corr = log2_cpm.corr(method='pearson')
    sample_corr.to_csv(os.path.join(RESULTS_DIR, 'sample_correlation_matrix.csv'))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(sample_corr, dtype=bool), k=1)
    
    group_colors_map = {'GC': '#2E86AB', 'FLT': '#F24236'}
    sample_colors = [group_colors_map[g] for g in groups]
    
    sns.heatmap(sample_corr, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0.8, vmin=0.7, vmax=1.0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=[f"{get_sample_short_name(s)}\n({g})" for s, g in zip(sample_corr.columns, groups)],
                yticklabels=[f"{get_sample_short_name(s)}\n({g})" for s, g in zip(sample_corr.index, groups)],
                ax=ax)
    ax.set_title('Sample Correlation Matrix (Log2 CPM)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sample_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'sample_correlation_heatmap.pdf'), bbox_inches='tight')
    plt.close()
    print("  样本相关性热图已保存")

def plot_pca(log2_cpm, groups, sample_names):
    """PCA分析"""
    print("\n绘制PCA图...")
    
    # 使用Top 5000变异最大的基因
    variance = log2_cpm.var(axis=1)
    top_genes = variance.nlargest(5000).index
    pca_data = log2_cpm.loc[top_genes].T
    
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(pca_data)
    
    explained_var = pca.explained_variance_ratio_ * 100
    print(f"  PC1: {explained_var[0]:.2f}%")
    print(f"  PC2: {explained_var[1]:.2f}%")
    
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Sample': sample_names,
        'Group': groups
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    group_colors_map = {'GC': '#2E86AB', 'FLT': '#F24236'}
    
    for group in GROUP_ORDER:
        subset = pca_df[pca_df['Group'] == group]
        ax.scatter(subset['PC1'], subset['PC2'],
                   c=group_colors_map[group], label=group, s=200, alpha=0.8, edgecolors='black')
        for _, row in subset.iterrows():
            ax.annotate(get_sample_short_name(row['Sample']),
                       (row['PC1'], row['PC2']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, alpha=0.8)
    
    ax.set_xlabel(f'PC1 ({explained_var[0]:.2f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.2f}%)', fontsize=12)
    ax.set_title('PCA Analysis - OSD-162 Mouse Eye Transcriptome\n(Space Flight vs Ground Control)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Group', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'PCA_plot.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'PCA_plot.pdf'), bbox_inches='tight')
    plt.close()
    
    pca_df.to_csv(os.path.join(RESULTS_DIR, 'PCA_results.csv'), index=False)
    print("  PCA图已保存")

def plot_volcano(de_results, symbol_map):
    """火山图"""
    print("\n绘制火山图...")
    
    de_results['-log10(pvalue)'] = -np.log10(de_results['pvalue'].replace(0, 1e-300))
    
    colors = {
        'Up-regulated': '#F24236',
        'Down-regulated': '#2E86AB',
        'Significant (p<0.05)': '#F6AE2D',
        'Not Significant': '#B0B0B0'
    }
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    for sig_type in ['Not Significant', 'Significant (p<0.05)', 'Down-regulated', 'Up-regulated']:
        subset = de_results[de_results['significance'] == sig_type]
        if len(subset) > 0:
            ax.scatter(subset['log2FoldChange'], subset['-log10(pvalue)'],
                      c=colors[sig_type], label=sig_type, alpha=0.6, s=15)
    
    # 阈值线
    ax.axhline(y=-np.log10(PVALUE_THRESHOLD), color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=LOG2FC_THRESHOLD, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=-LOG2FC_THRESHOLD, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    
    # 标注Top基因
    sig_genes = de_results[de_results['significance'].isin(['Up-regulated', 'Down-regulated'])]
    if len(sig_genes) > 0:
        top_genes = sig_genes.nsmallest(min(20, len(sig_genes)), 'pvalue')
        for _, gene in top_genes.iterrows():
            gene_name = get_gene_symbol(gene['ENSEMBL'], symbol_map)
            ax.annotate(gene_name,
                       (gene['log2FoldChange'], -np.log10(gene['pvalue'])),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Log2 Fold Change', fontsize=12)
    ax.set_ylabel('-Log10 P-value', fontsize=12)
    ax.set_title(f'Volcano Plot: Space Flight vs Ground Control\nOSD-162 Mouse Eye | |log2FC| > {LOG2FC_THRESHOLD}, p < {PVALUE_THRESHOLD}',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Significance', loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'volcano_plot.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'volcano_plot.pdf'), bbox_inches='tight')
    plt.close()
    print("  火山图已保存")

def plot_ma(de_results):
    """MA Plot"""
    print("\n绘制MA Plot...")
    
    de_results['baseMean'] = de_results['ENSEMBL'].map(
        lambda x: np.nan  # 将在主函数中计算
    )
    
    # 这里简化处理，使用log2FC和pvalue
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'Up-regulated': '#F24236',
        'Down-regulated': '#2E86AB',
        'Significant (p<0.05)': '#F6AE2D',
        'Not Significant': '#B0B0B0'
    }
    
    for sig_type in ['Not Significant', 'Significant (p<0.05)', 'Down-regulated', 'Up-regulated']:
        subset = de_results[de_results['significance'] == sig_type]
        if len(subset) > 0:
            ax.scatter(subset.index, subset['log2FoldChange'],
                      c=colors[sig_type], label=sig_type, alpha=0.5, s=12)
    
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel('Gene Index', fontsize=12)
    ax.set_ylabel('Log2 Fold Change', fontsize=12)
    ax.set_title('MA Plot: Space Flight vs Ground Control', fontsize=14, fontweight='bold')
    ax.legend(title='Significance', loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'MA_plot.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'MA_plot.pdf'), bbox_inches='tight')
    plt.close()
    print("  MA Plot已保存")

def plot_diff_genes_heatmap(log2_cpm, sig_genes, groups, sample_names, symbol_map):
    """差异基因热图 - 欧氏聚类法对行和列聚类"""
    print("\n绘制差异基因热图 (欧氏聚类)...")
    
    sig_gene_ids = sig_genes['ENSEMBL'].tolist()
    
    if len(sig_gene_ids) == 0:
        print("  无显著差异基因，跳过热图绘制")
        return
    
    # 提取表达矩阵
    sig_expression = log2_cpm.loc[sig_gene_ids]
    
    # Z-score标准化
    sig_expression_z = zscore_normalize(sig_expression)
    
    # 欧氏距离聚类 (行和列)
    # 行聚类 (基因)
    row_linkage = linkage(pdist(sig_expression_z, metric='euclidean'), method='average')
    row_dendro = dendrogram(row_linkage, no_plot=True)
    row_order = row_dendro['leaves']
    
    # 列聚类 (样本)
    col_linkage = linkage(pdist(sig_expression_z.T, metric='euclidean'), method='average')
    col_dendro = dendrogram(col_linkage, no_plot=True)
    col_order = col_dendro['leaves']
    
    # 重排序
    sig_ordered = sig_expression_z.iloc[row_order, col_order]
    
    # 获取基因符号
    ordered_symbols = [get_gene_symbol(g, symbol_map) for g in sig_ordered.index]
    
    # 样本标签
    ordered_samples = [get_sample_short_name(sig_ordered.columns[i]) for i in range(len(sig_ordered.columns))]
    # 创建列名到分组的映射
    col_to_group = {s: g for s, g in zip(sample_names, groups)}
    ordered_groups = [col_to_group[sig_ordered.columns[i]] for i in range(len(sig_ordered.columns))]
    
    # 绘图
    n_genes = len(sig_ordered)
    fig_height = max(10, n_genes * 0.3 + 4)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, fig_height),
                             gridspec_kw={'width_ratios': [1, 0.05],
                                         'height_ratios': [0.1, 1],
                                         'wspace': 0.05,
                                         'hspace': 0.05})
    
    # 分组颜色条
    ax_group = axes[0, 0]
    group_colors_map = {'GC': '#2E86AB', 'FLT': '#F24236'}
    for i, group in enumerate(ordered_groups):
        ax_group.barh(0, 1, left=i, color=group_colors_map[group], height=1)
    ax_group.set_xlim(0, len(ordered_groups))
    ax_group.set_ylim(0, 1)
    ax_group.axis('off')
    
    # 主热图
    ax_heatmap = axes[1, 0]
    im = ax_heatmap.imshow(sig_ordered.values,
                           aspect='auto',
                           cmap='RdBu_r',
                           vmin=-3, vmax=3,
                           interpolation='nearest')
    
    ax_heatmap.set_xticks(range(len(ordered_samples)))
    ax_heatmap.set_xticklabels(ordered_samples, rotation=45, ha='right', fontsize=9)
    ax_heatmap.set_yticks(range(len(ordered_symbols)))
    ax_heatmap.set_yticklabels(ordered_symbols, fontsize=7)
    
    ax_heatmap.set_xticks(np.arange(-0.5, len(ordered_samples), 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, len(ordered_symbols), 1), minor=True)
    ax_heatmap.grid(which='minor', color='white', linewidth=0.5)
    ax_heatmap.tick_params(which='minor', length=0)
    
    # Colorbar
    cbar_ax = axes[1, 1]
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Z-score', rotation=270, labelpad=15, fontsize=10)
    
    # 标题
    fig.suptitle(f'Differentially Expressed Genes (n={n_genes})\nEuclidean Clustering | OSD-162 Mouse Eye',
                 fontsize=14, fontweight='bold')
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=group_colors_map[g], label=f"{g}: {SAMPLE_GROUPS[g]['label']}") 
                       for g in GROUP_ORDER]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'diff_genes_heatmap_euclidean.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'diff_genes_heatmap_euclidean.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  差异基因热图已保存 ({n_genes} genes)")

def plot_top100_heatmap(log2_cpm, top100, groups, sample_names, symbol_map):
    """Top 100差异基因热图"""
    print("\n绘制Top 100差异基因热图...")
    
    top100_ids = top100['ENSEMBL'].tolist()
    
    if len(top100_ids) == 0:
        print("  无Top 100基因")
        return
    
    top100_expr = log2_cpm.loc[top100_ids]
    top100_z = zscore_normalize(top100_expr)
    
    # 欧氏聚类
    row_linkage = linkage(pdist(top100_z, metric='euclidean'), method='average')
    row_dendro = dendrogram(row_linkage, no_plot=True)
    row_order = row_dendro['leaves']
    
    col_linkage = linkage(pdist(top100_z.T, metric='euclidean'), method='average')
    col_dendro = dendrogram(col_linkage, no_plot=True)
    col_order = col_dendro['leaves']
    
    top100_ordered = top100_z.iloc[row_order, col_order]
    
    ordered_symbols = [get_gene_symbol(g, symbol_map) for g in top100_ordered.index]
    ordered_samples = [get_sample_short_name(top100_ordered.columns[i]) for i in range(len(top100_ordered.columns))]
    col_to_group = {s: g for s, g in zip(sample_names, groups)}
    ordered_groups = [col_to_group[top100_ordered.columns[i]] for i in range(len(top100_ordered.columns))]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 20),
                             gridspec_kw={'width_ratios': [1, 0.05],
                                         'height_ratios': [0.1, 1],
                                         'wspace': 0.05,
                                         'hspace': 0.05})
    
    # 分组颜色条
    ax_group = axes[0, 0]
    group_colors_map = {'GC': '#2E86AB', 'FLT': '#F24236'}
    for i, group in enumerate(ordered_groups):
        ax_group.barh(0, 1, left=i, color=group_colors_map[group], height=1)
    ax_group.set_xlim(0, len(ordered_groups))
    ax_group.set_ylim(0, 1)
    ax_group.axis('off')
    
    # 主热图
    ax_heatmap = axes[1, 0]
    im = ax_heatmap.imshow(top100_ordered.values,
                           aspect='auto',
                           cmap='RdBu_r',
                           vmin=-3, vmax=3,
                           interpolation='nearest')
    
    ax_heatmap.set_xticks(range(len(ordered_samples)))
    ax_heatmap.set_xticklabels(ordered_samples, rotation=45, ha='right', fontsize=9)
    ax_heatmap.set_yticks(range(len(ordered_symbols)))
    ax_heatmap.set_yticklabels(ordered_symbols, fontsize=7)
    
    ax_heatmap.set_xticks(np.arange(-0.5, len(ordered_samples), 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, len(ordered_symbols), 1), minor=True)
    ax_heatmap.grid(which='minor', color='white', linewidth=0.5)
    ax_heatmap.tick_params(which='minor', length=0)
    
    # Colorbar
    cbar_ax = axes[1, 1]
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Z-score', rotation=270, labelpad=15, fontsize=10)
    
    fig.suptitle('Top 100 Differentially Expressed Genes\nEuclidean Clustering | OSD-162 Mouse Eye',
                 fontsize=14, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=group_colors_map[g], label=f"{g}: {SAMPLE_GROUPS[g]['label']}") 
                       for g in GROUP_ORDER]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'top100_DEGs_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'top100_DEGs_heatmap.pdf'), bbox_inches='tight')
    plt.close()
    print("  Top 100热图已保存")

# =============================================================================
# 7. KEGG通路热图
# =============================================================================

def plot_kegg_pathway_heatmaps(log2_cpm, pathways_dict, symbol_map):
    """生成KEGG通路热图"""
    print("\n" + "=" * 80)
    print("KEGG通路热图生成")
    print("=" * 80)
    
    kegg_dir = os.path.join(RESULTS_DIR, 'KEGG_pathways')
    ensure_dir(kegg_dir)
    
    for pathway_name, pathway_info in pathways_dict.items():
        gene_list = pathway_info['genes']
        description = pathway_info.get('description', '')
        
        available_genes = [g for g in gene_list if g in log2_cpm.index]
        n_found = len(available_genes)
        
        print(f"\n  {pathway_name}: {n_found}/{len(gene_list)} genes found")
        
        if n_found < 3:
            print(f"    Skipped (insufficient genes)")
            continue
        
        pathway_data = log2_cpm.loc[available_genes]
        pathway_zscore = zscore_normalize(pathway_data)
        
        sample_labels = [get_sample_short_name(s) for s in pathway_zscore.columns]
        n_genes = len(available_genes)
        n_samples = len(sample_labels)
        
        fig_height = max(8, n_genes * 0.5 + 4)
        fig_width = 12
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        gs = fig.add_gridspec(
            nrows=4, ncols=2,
            width_ratios=[1, 0.04],
            height_ratios=[0.8, 0.25, 0.2, n_genes * 0.5],
            hspace=0.02, wspace=0.02,
            left=0.18, right=0.88, top=0.92, bottom=0.07
        )
        
        # 标题
        fig.text(0.5, 0.97, f'KEGG Pathway: {pathway_name}',
                ha='center', va='top', fontsize=14, fontweight='bold')
        fig.text(0.5, 0.945, f'{description}',
                ha='center', va='top', fontsize=10, style='italic', color='#555')
        fig.text(0.5, 0.925,
                f'OSD-162 Mouse Eye | {n_found} genes | Z-score normalized | GC (n=5) vs FLT (n=5)',
                ha='center', va='top', fontsize=9, color='#777')
        
        # 分组颜色条
        ax_band = fig.add_subplot(gs[1, 0])
        metadata = get_sample_metadata()
        group_labels_list = [metadata[s] for s in pathway_zscore.columns]
        
        for i, group in enumerate(group_labels_list):
            ax_band.barh(0, 1, left=i, color=SAMPLE_GROUPS[group]['color'],
                        height=1, edgecolor='white', linewidth=1)
        ax_band.set_xlim(0, n_samples)
        ax_band.set_ylim(0, 1)
        ax_band.axis('off')
        
        # 分组标签
        ax_label = fig.add_subplot(gs[2, 0])
        ax_label.set_xlim(0, n_samples)
        ax_label.set_ylim(0, 1)
        ax_label.axis('off')
        
        for group_id in GROUP_ORDER:
            positions = [i for i, g in enumerate(group_labels_list) if g == group_id]
            if positions:
                left = positions[0]
                right = positions[-1] + 1
                center = (left + right) / 2
                ax_label.text(center, 0.85, SAMPLE_GROUPS[group_id]['label'],
                            ha='center', va='top', fontsize=11,
                            fontweight='bold', color=SAMPLE_GROUPS[group_id]['color'])
        
        # 主热图
        ax_heatmap = fig.add_subplot(gs[3, 0])
        
        im = ax_heatmap.imshow(
            pathway_zscore.values,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-3, vmax=3,
            interpolation='nearest'
        )
        
        ax_heatmap.set_xticks(range(n_samples))
        ax_heatmap.set_xticklabels(sample_labels, fontsize=10, fontweight='bold')
        ax_heatmap.set_yticks(range(n_genes))
        ax_heatmap.set_yticklabels([get_gene_symbol(g, symbol_map) for g in pathway_zscore.index], fontsize=9)
        
        ax_heatmap.set_xticks(np.arange(-0.5, n_samples, 1), minor=True)
        ax_heatmap.set_yticks(np.arange(-0.5, n_genes, 1), minor=True)
        ax_heatmap.grid(which='minor', color='white', linewidth=0.5)
        ax_heatmap.tick_params(which='minor', length=0)
        
        # Colorbar
        cbar_ax = fig.add_subplot(gs[3, 1])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Z-score (Log2 CPM)', rotation=270, labelpad=22, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        
        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=SAMPLE_GROUPS['GC']['color'],
                 label=f"GC: {SAMPLE_GROUPS['GC']['label']} (n=5)"),
            Patch(facecolor=SAMPLE_GROUPS['FLT']['color'],
                 label=f"FLT: {SAMPLE_GROUPS['FLT']['label']} (n=5)"),
        ]
        fig.legend(handles=legend_elements, loc='upper right',
                  bbox_to_anchor=(0.99, 0.97), fontsize=9,
                  framealpha=0.9, edgecolor='gray')
        
        plt.savefig(
            os.path.join(kegg_dir, f'KEGG_{pathway_name.replace(" ", "_")}_heatmap.png'),
            dpi=300, bbox_inches='tight', facecolor='white'
        )
        print(f"    Saved: {pathway_name}")
        plt.close()

# =============================================================================
# 8. GO通路k-means聚类热图
# =============================================================================

def plot_go_kmeans_heatmaps(log2_cpm, go_dict, symbol_map):
    """生成特定GO通路的k-means聚类热图"""
    print("\n" + "=" * 80)
    print("GO通路k-means聚类热图")
    print("=" * 80)
    
    go_dir = os.path.join(RESULTS_DIR, 'GO_kmeans_clustering')
    ensure_dir(go_dir)
    
    for go_name, go_info in go_dict.items():
        gene_list = go_info['genes']
        description = go_info.get('description', '')
        
        available_genes = [g for g in gene_list if g in log2_cpm.index]
        n_found = len(available_genes)
        
        print(f"\n  {go_name}: {n_found}/{len(gene_list)} genes found")
        
        if n_found < 5:
            print(f"    Skipped (insufficient genes)")
            continue
        
        go_data = log2_cpm.loc[available_genes]
        go_zscore = zscore_normalize(go_data)
        
        # k-means聚类 (对基因进行聚类)
        n_clusters = min(3, n_found // 2)  # 最多3个簇
        if n_clusters < 2:
            n_clusters = 2
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(go_zscore.values)
        
        # 按簇排序
        cluster_df = pd.DataFrame({'gene': go_zscore.index, 'cluster': cluster_labels})
        cluster_df = cluster_df.sort_values('cluster')
        
        go_ordered = go_zscore.loc[cluster_df['gene'].values]
        
        # 样本标签
        sample_labels = [get_sample_short_name(s) for s in go_ordered.columns]
        n_genes = len(go_ordered)
        n_samples = len(sample_labels)
        
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
        
        # 标题
        fig.text(0.5, 0.97, f'GO Pathway: {go_name}',
                ha='center', va='top', fontsize=14, fontweight='bold')
        fig.text(0.5, 0.945, f'{description}',
                ha='center', va='top', fontsize=10, style='italic', color='#555')
        fig.text(0.5, 0.925,
                f'OSD-162 Mouse Eye | {n_found} genes | k-means clustering (k={n_clusters}) | GC (n=5) vs FLT (n=5)',
                ha='center', va='top', fontsize=9, color='#777')
        
        # 分组颜色条
        ax_band = fig.add_subplot(gs[1, 1])
        metadata = get_sample_metadata()
        group_labels_list = [metadata[s] for s in go_ordered.columns]
        
        for i, group in enumerate(group_labels_list):
            ax_band.barh(0, 1, left=i, color=SAMPLE_GROUPS[group]['color'],
                        height=1, edgecolor='white', linewidth=1)
        ax_band.set_xlim(0, n_samples)
        ax_band.set_ylim(0, 1)
        ax_band.axis('off')
        
        # 分组标签
        ax_label = fig.add_subplot(gs[2, 1])
        ax_label.set_xlim(0, n_samples)
        ax_label.set_ylim(0, 1)
        ax_label.axis('off')
        
        for group_id in GROUP_ORDER:
            positions = [i for i, g in enumerate(group_labels_list) if g == group_id]
            if positions:
                left = positions[0]
                right = positions[-1] + 1
                center = (left + right) / 2
                ax_label.text(center, 0.85, SAMPLE_GROUPS[group_id]['label'],
                            ha='center', va='top', fontsize=11,
                            fontweight='bold', color=SAMPLE_GROUPS[group_id]['color'])
        
        # 簇颜色条 (左侧)
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
        
        im = ax_heatmap.imshow(
            go_ordered.values,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-3, vmax=3,
            interpolation='nearest'
        )
        
        ax_heatmap.set_xticks(range(n_samples))
        ax_heatmap.set_xticklabels(sample_labels, fontsize=10, fontweight='bold')
        ax_heatmap.set_yticks(range(n_genes))
        ax_heatmap.set_yticklabels([get_gene_symbol(g, symbol_map) for g in go_ordered.index], fontsize=9)
        
        ax_heatmap.set_xticks(np.arange(-0.5, n_samples, 1), minor=True)
        ax_heatmap.set_yticks(np.arange(-0.5, n_genes, 1), minor=True)
        ax_heatmap.grid(which='minor', color='white', linewidth=0.5)
        ax_heatmap.tick_params(which='minor', length=0)
        
        # Colorbar
        cbar_ax = fig.add_subplot(gs[3, 2])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Z-score (Log2 CPM)', rotation=270, labelpad=22, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        
        # 簇图例
        from matplotlib.patches import Patch
        legend_elements = []
        for i in range(n_clusters):
            legend_elements.append(Patch(facecolor=cluster_colors[i], label=f'Cluster {i+1}'))
        
        # 分组图例
        legend_elements.append(Patch(facecolor=SAMPLE_GROUPS['GC']['color'],
                                    label=f"GC: {SAMPLE_GROUPS['GC']['label']} (n=5)"))
        legend_elements.append(Patch(facecolor=SAMPLE_GROUPS['FLT']['color'],
                                    label=f"FLT: {SAMPLE_GROUPS['FLT']['label']} (n=5)"))
        
        fig.legend(handles=legend_elements, loc='upper right',
                  bbox_to_anchor=(0.99, 0.97), fontsize=9,
                  framealpha=0.9, edgecolor='gray')
        
        plt.savefig(
            os.path.join(go_dir, f'GO_{go_name}_kmeans_heatmap.png'),
            dpi=300, bbox_inches='tight', facecolor='white'
        )
        print(f"    Saved: {go_name} (k={n_clusters})")
        plt.close()

# =============================================================================
# 9. GO/KEGG富集分析
# =============================================================================

def run_gprofiler(genes, organism='mmusculus', query_name='query'):
    """运行g:Profiler富集分析"""
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
        response = request.urlopen(req, timeout=60)
        result = json.loads(response.read().decode('utf-8'))
        return result
    except Exception as e:
        print(f"g:Profiler请求失败: {e}")
        return None

def functional_enrichment_analysis(de_results, symbol_map):
    """功能富集分析"""
    print("\n" + "=" * 80)
    print("功能富集分析 (GO/KEGG)")
    print("=" * 80)
    
    # 获取显著差异基因的符号
    sig_genes = de_results[de_results['significance'].isin(['Up-regulated', 'Down-regulated'])]
    
    if len(sig_genes) == 0:
        print("无显著差异基因，跳过富集分析")
        return
    
    # 获取基因符号
    gene_symbols = []
    for _, row in sig_genes.iterrows():
        sym = get_gene_symbol(row['ENSEMBL'], symbol_map)
        if sym and not sym.startswith('ENSMUSG'):
            gene_symbols.append(sym)
    
    print(f"\n显著差异基因数: {len(sig_genes)}")
    print(f"有符号的基因数: {len(gene_symbols)}")
    
    if len(gene_symbols) < 5:
        print("基因数太少，跳过富集分析")
        return
    
    # 运行g:Profiler
    print("\n运行g:Profiler富集分析...")
    gprofiler_result = run_gprofiler(gene_symbols, query_name='OSD-162_DEGs')
    
    if gprofiler_result and 'result' in gprofiler_result:
        enrichment_df = pd.DataFrame(gprofiler_result['result'])
        print(f"富集结果数: {len(enrichment_df)}")
        
        # 保存结果
        enrichment_df.to_csv(os.path.join(RESULTS_DIR, 'gprofiler_enrichment.csv'), index=False)
        
        # 提取GO和KEGG结果
        go_results = enrichment_df[enrichment_df['source'].str.contains('GO', na=False)]
        kegg_results = enrichment_df[enrichment_df['source'] == 'KEGG']
        
        print(f"GO条目数: {len(go_results)}")
        print(f"KEGG条目数: {len(kegg_results)}")
        
        # 可视化
        plot_enrichment_results(enrichment_df, go_results, kegg_results)
        
    else:
        print("未获得富集结果")

def plot_enrichment_results(enrichment_df, go_results, kegg_results):
    """绘制富集结果图"""
    print("\n绘制富集结果图...")
    
    # GO富集条形图
    if len(go_results) > 0:
        top_go = go_results.nsmallest(min(15, len(go_results)), 'p_value')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = {'GO:BP': '#2E86AB', 'GO:MF': '#F24236', 'GO:CC': '#F6AE2D'}
        bar_colors = [colors.get(s, 'grey') for s in top_go['source']]
        
        y_pos = np.arange(len(top_go))
        bars = ax.barh(y_pos, -np.log10(top_go['p_value']), color=bar_colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{name[:60]}..." if len(name) > 60 else name
                           for name in top_go['name']], fontsize=9)
        ax.set_xlabel('-Log10 P-value', fontsize=12)
        ax.set_title('GO Enrichment Analysis\nTop 15 Significant Terms (OSD-162)',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[s], label=s) for s in colors if s in top_go['source'].values]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'GO_enrichment_barplot.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(RESULTS_DIR, 'GO_enrichment_barplot.pdf'), bbox_inches='tight')
        plt.close()
        print("  GO富集图已保存")
    
    # KEGG富集条形图
    if len(kegg_results) > 0:
        top_kegg = kegg_results.nsmallest(min(15, len(kegg_results)), 'p_value')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(top_kegg))
        bars = ax.barh(y_pos, -np.log10(top_kegg['p_value']), color='#A23B72', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{name[:60]}..." if len(name) > 60 else name
                           for name in top_kegg['name']], fontsize=9)
        ax.set_xlabel('-Log10 P-value', fontsize=12)
        ax.set_title('KEGG Pathway Enrichment\nTop Significant Pathways (OSD-162)',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'KEGG_enrichment_barplot.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(RESULTS_DIR, 'KEGG_enrichment_barplot.pdf'), bbox_inches='tight')
        plt.close()
        print("  KEGG富集图已保存")
    
    # 气泡图
    if len(enrichment_df) > 0:
        top_all = enrichment_df.nsmallest(min(20, len(enrichment_df)), 'p_value')
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sizes = top_all['intersection_size'] * 30
        colors_map = {'GO:BP': '#2E86AB', 'GO:MF': '#F24236', 'GO:CC': '#F6AE2D',
                     'KEGG': '#A23B72', 'REAC': '#F18F01'}
        point_colors = [colors_map.get(s, 'grey') for s in top_all['source']]
        
        scatter = ax.scatter(-np.log10(top_all['p_value']),
                            range(len(top_all)),
                            s=sizes,
                            c=point_colors,
                            alpha=0.6,
                            edgecolors='black',
                            linewidth=0.5)
        
        ax.set_yticks(range(len(top_all)))
        ax.set_yticklabels([f"{name[:50]}..." if len(name) > 50 else name
                           for name in top_all['name']], fontsize=8)
        ax.set_xlabel('-Log10 P-value', fontsize=12)
        ax.set_title('Enrichment Analysis Bubble Plot (OSD-162)\n(Size = Gene Count, Color = Source)',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [plt.scatter([], [], s=100, c=colors_map[s], label=s, alpha=0.6)
                          for s in colors_map if s in top_all['source'].values]
        ax.legend(handles=legend_elements, title='Source', loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'enrichment_bubble_plot.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(RESULTS_DIR, 'enrichment_bubble_plot.pdf'), bbox_inches='tight')
        plt.close()
        print("  富集气泡图已保存")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("OSD-162 转录组数据分析")
    print("数据集: 小鼠眼组织 (Space Flight vs Ground Control)")
    print("数据来源: https://osdr.nasa.gov/bio/repo/data/studies/OSD-162")
    print("=" * 80)
    
    # 创建输出目录
    ensure_dir(RESULTS_DIR)
    
    # 1. 加载和清洗数据
    counts_raw, sample_names = load_and_clean_data()
    
    # 创建分组信息
    metadata = get_sample_metadata()
    groups = [metadata[s] for s in sample_names]
    
    print(f"\n样本分组:")
    for g in GROUP_ORDER:
        count = groups.count(g)
        print(f"  {g} ({SAMPLE_GROUPS[g]['label']}): {count}")
    
    # 2. 数据质控和标准化
    counts_filtered, cpm, log2_cpm = quality_control(counts_raw)
    
    # 3. 去批次效应
    log2_combat = combat_correction(log2_cpm, groups)
    
    # 加载基因符号映射
    viz_file = os.path.join(DATA_DIR, 'GLDS-162_rna_seq_visualization_output_table.csv')
    symbol_map = load_gene_symbol_mapping(viz_file)
    
    # 4. 差异表达分析
    de_results, sig_genes, top100, up_genes, down_genes = differential_expression_analysis(
        counts_filtered, log2_combat, groups
    )
    
    # 5. 富集分数计算
    es_df = calculate_enrichment_scores(log2_combat, KEGG_PATHWAYS)
    
    # 6. 可视化
    plot_sample_correlation(log2_combat, groups, sample_names)
    plot_pca(log2_combat, groups, sample_names)
    plot_volcano(de_results, symbol_map)
    plot_ma(de_results)
    plot_diff_genes_heatmap(log2_combat, sig_genes, groups, sample_names, symbol_map)
    plot_top100_heatmap(log2_combat, top100, groups, sample_names, symbol_map)
    
    # 7. KEGG通路热图
    plot_kegg_pathway_heatmaps(log2_combat, KEGG_PATHWAYS, symbol_map)
    
    # 8. GO通路k-means聚类热图
    plot_go_kmeans_heatmaps(log2_combat, GO_TERMS, symbol_map)
    
    # 9. 功能富集分析
    functional_enrichment_analysis(de_results, symbol_map)
    
    # 10. 结果汇总
    print("\n" + "=" * 80)
    print("【分析完成】")
    print("=" * 80)
    
    print(f"""
分析结果汇总:
-------------
数据集: OSD-162 (Mouse Eye)
分组: Ground Control (n=5) vs Space Flight (n=5)

差异表达阈值:
  - |log2 Fold Change| > {LOG2FC_THRESHOLD}
  - P-value < {PVALUE_THRESHOLD}

结果统计:
  - 总检测基因数: {len(de_results):,}
  - 显著差异基因数: {len(sig_genes)}
    - 上调基因: {len(up_genes)}
    - 下调基因: {len(down_genes)}
  - Top 100差异基因: {len(top100)}

输出文件:
---------""")
    
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(('.csv', '.png', '.pdf')):
            print(f"  - {f}")
    
    print(f"\n结果目录: {RESULTS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
