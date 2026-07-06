#!/usr/bin/env python3
"""
KEGG Pathway Heatmap Analysis for OSD-194 (GLDS-194) Mouse Eye (Retina) Transcriptome
v5: 从源数据重新清洗 + 仅保留Left retina样本

数据源：
  - GLDS-194_rna_seq_Normalized_Counts.csv (Normalized counts, 13 samples)
  - GLDS-194_rna_seq_visualization_output_table.csv (含ENSEMBL→SYMBOL映射 + DEG结果)
  - GLDS-194_rna_seq_SampleTable.csv (样本元数据)

样本命名规则：
  LRTN = Left retina (保留)
  RRTN = Right retina (删除)

样本分组（仅保留Left retina）：
  - BSL: Basal Control
      LRTN: B7 (1 sample, 全部 RRTN: B8/B9/B10 被删除)
  - GC:  Ground Control
      LRTN: G6, G8, G9 (3 samples, RRTN: G10 被删除)
  - FLT: Space Flight
      LRTN: F6, F7, F8, F9, F10 (5 samples, 全部 LRTN)

比较焦点 (按用户研究设计)：
  Ground Control (n=3) vs Space Flight (n=5)
  (Basal Control 仅1个样本, 不纳入主比较)

Author: OpenClaw Assistant
Date: 2026-06-30
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




from matplotlib.patches import Patch
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = r'C:\Users\Administrator\.openclaw\workspace\transcriptomics\data\OSD-194'
NORMALIZED_COUNTS_FILE = os.path.join(DATA_DIR, 'GLDS-194_rna_seq_Normalized_Counts.csv')
VIZ_OUTPUT_FILE = os.path.join(DATA_DIR, 'GLDS-194_rna_seq_visualization_output_table.csv')
SAMPLE_TABLE_FILE = os.path.join(DATA_DIR, 'GLDS-194_rna_seq_SampleTable.csv')
OUTPUT_DIR = r'C:\Users\Administrator\.openclaw\workspace\transcriptomics\results\OSD194_KEGG_heatmaps_v5_LRTN'

# =============================================================================
# 样本组定义 — 仅保留Left retina (LRTN)
# =============================================================================

# 比较组: Ground Control vs Space Flight
# 两者均有足够Left retina样本

SAMPLE_GROUPS = {
    'BSL': {
        'label': 'Basal Control',
        'color': '#808080',  # 灰色
        'samples': [
            'Mmus_BAL-TAL_LRTN_BSL_Rep1_B7',
        ]
    },
    'GC': {
        'label': 'Ground Control',
        'color': '#2E86AB',  # 蓝色
        'samples': [
            'Mmus_BAL-TAL_LRTN_GC_Rep1_G6',
            'Mmus_BAL-TAL_LRTN_GC_Rep2_G8',
            'Mmus_BAL-TAL_LRTN_GC_Rep3_G9',
        ]
    },
    'FLT': {
        'label': 'Space Flight',
        'color': '#A23B72',  # 紫色
        'samples': [
            'Mmus_BAL-TAL_LRTN_FLT_Rep1_F6',
            'Mmus_BAL-TAL_LRTN_FLT_Rep2_F7',
            'Mmus_BAL-TAL_LRTN_FLT_Rep3_F8',
            'Mmus_BAL-TAL_LRTN_FLT_Rep4_F9',
            'Mmus_BAL-TAL_LRTN_FLT_Rep5_F10',
        ]
    },
}

# 主比较组顺序: GC -> FLT (BSL不参与聚类，仅作为参考)
GROUP_ORDER = ['BSL', 'GC', 'FLT']


# =============================================================================
# KEGG Pathway Gene Sets (Mouse) — ENSEMBL IDs
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
    'Muscle Atrophy': {
        'genes': ['ENSMUSG00000022817', 'ENSMUSG00000037608', 'ENSMUSG00000029472',
                 'ENSMUSG00000031934', 'ENSMUSG00000029474', 'ENSMUSG00000029475',
                 'ENSMUSG00000029476', 'ENSMUSG00000029477', 'ENSMUSG00000029478',
                 'ENSMUSG00000029479', 'ENSMUSG00000029480', 'ENSMUSG00000029481'],
        'description': 'Muscle protein degradation'
    },
    'Bone Remodeling': {
        'genes': ['ENSMUSG00000028199', 'ENSMUSG00000028200', 'ENSMUSG00000028201',
                 'ENSMUSG00000029314', 'ENSMUSG00000029315', 'ENSMUSG00000029316',
                 'ENSMUSG00000029317', 'ENSMUSG00000029318', 'ENSMUSG00000029319',
                 'ENSMUSG00000029320', 'ENSMUSG00000029321', 'ENSMUSG00000029322'],
        'description': 'Bone formation and resorption markers'
    },
    # 视网膜特有通路
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_gene_symbol_mapping(viz_file):
    """从visualization_output_table.csv加载ENSEMBL→SYMBOL映射"""
    print("Loading gene symbol mapping...")
    mapping_df = pd.read_csv(viz_file, usecols=[0, 1])
    mapping_df.columns = ['ENSEMBL', 'SYMBOL']
    mapping = dict(zip(mapping_df['ENSEMBL'], mapping_df['SYMBOL']))
    print(f"  Loaded {len(mapping)} gene mappings")
    return mapping


def get_gene_symbol(ensembl_id, symbol_map):
    """将ENSEMBL ID转换为基因符号"""
    sym = symbol_map.get(ensembl_id, '')
    if pd.isna(sym) or sym == '':
        return ensembl_id
    return sym


def zscore_normalize(data):
    """Z-score normalize data across samples (row-wise)."""
    zscore = stats.zscore(data, axis=1, nan_policy='omit')
    zscore = pd.DataFrame(zscore, index=data.index, columns=data.columns)
    zscore = zscore.replace([np.inf, -np.inf], np.nan).fillna(0)
    return zscore


def get_sample_metadata():
    """Create sample to group mapping."""
    metadata = {}
    for group_id, info in SAMPLE_GROUPS.items():
        for sample in info['samples']:
            metadata[sample] = group_id
    return metadata


def reorder_samples(expression_data):
    """Reorder columns: BSL → GC → FLT"""
    ordered = []
    for group_id in GROUP_ORDER:
        for sample in SAMPLE_GROUPS[group_id]['samples']:
            if sample in expression_data.columns:
                ordered.append(sample)
    return expression_data[ordered]


def get_sample_short_name(sample):
    """提取样本简称，如 B7, G6, F6"""
    return sample.split('_')[-1]


# =============================================================================
# HEATMAP GENERATION
# =============================================================================

def create_pathway_heatmap(pathway_name, pathway_info, expression_data, symbol_map, output_dir):
    """Create a heatmap for a single KEGG pathway."""
    
    gene_list = pathway_info['genes']
    description = pathway_info.get('description', '')
    
    available_genes = [g for g in gene_list if g in expression_data.index]
    n_found = len(available_genes)
    
    print(f"\n  {pathway_name}: {n_found}/{len(gene_list)} genes found")
    
    if n_found < 3:
        print(f"    Skipped (insufficient genes)")
        return None
    
    pathway_data = expression_data.loc[available_genes]
    pathway_zscore = zscore_normalize(pathway_data)
    
    sample_labels = [get_sample_short_name(s) for s in pathway_zscore.columns]
    
    # Hierarchical clustering
    row_linkage = linkage(pdist(pathway_zscore), method='average')
    row_dendro = dendrogram(row_linkage, no_plot=True)
    row_order = row_dendro['leaves']
    pathway_ordered = pathway_zscore.iloc[row_order]
    ordered_symbols = [get_gene_symbol(pathway_ordered.index[i], symbol_map) for i in range(len(pathway_ordered.index))]
    
    n_genes = len(ordered_symbols)
    n_samples = len(sample_labels)
    
    # Figure dimensions
    fig_height = max(7, n_genes * 0.5 + 4.0)
    fig_width = 11
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Layout: 4 rows
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        width_ratios=[1, 0.04],
        height_ratios=[0.8, 0.25, 0.2, n_genes * 0.5],
        hspace=0.02, wspace=0.02,
        left=0.18, right=0.88, top=0.92, bottom=0.07
    )
    
    # Title
    fig.text(0.5, 0.97, f'KEGG Pathway: {pathway_name}',
             ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.945, f'{description}',
             ha='center', va='top', fontsize=10, style='italic', color='#555')
    fig.text(0.5, 0.925,
             f'OSD-194 Mouse Eye (Left Retina) | {n_found} genes | Z-score normalized | GC (n=3) vs FLT (n=5)',
             ha='center', va='top', fontsize=9, color='#777')
    
    # Row 1: Group color band
    ax_band = fig.add_subplot(gs[1, 0])
    metadata = get_sample_metadata()
    group_labels_list = [metadata[s] for s in pathway_zscore.columns]
    
    for i, group in enumerate(group_labels_list):
        ax_band.barh(0, 1, left=i, color=SAMPLE_GROUPS[group]['color'],
                    height=1, edgecolor='white', linewidth=1)
    ax_band.set_xlim(0, n_samples)
    ax_band.set_ylim(0, 1)
    ax_band.axis('off')
    
    # Row 2: Group text labels below band
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
            ax_label.plot([left + 0.1, right - 0.1], [0.3, 0.3],
                         color=SAMPLE_GROUPS[group_id]['color'], linewidth=2)
            ax_label.plot([left + 0.1, left + 0.1], [0.3, 0.5],
                         color=SAMPLE_GROUPS[group_id]['color'], linewidth=2)
            ax_label.plot([right - 0.1, right - 0.1], [0.3, 0.5],
                         color=SAMPLE_GROUPS[group_id]['color'], linewidth=2)
    
    # Row 3: Main heatmap
    ax_heatmap = fig.add_subplot(gs[3, 0])
    
    im = ax_heatmap.imshow(
        pathway_ordered.values,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-3, vmax=3,
        interpolation='nearest'
    )
    
    ax_heatmap.set_xticks(range(n_samples))
    ax_heatmap.set_xticklabels(sample_labels, fontsize=10, fontweight='bold')
    ax_heatmap.set_yticks(range(n_genes))
    ax_heatmap.set_yticklabels(ordered_symbols, fontsize=9)
    
    ax_heatmap.set_xticks(np.arange(-0.5, n_samples, 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, n_genes, 1), minor=True)
    ax_heatmap.grid(which='minor', color='white', linewidth=0.5)
    ax_heatmap.tick_params(which='minor', length=0)
    
    # Colorbar
    cbar_ax = fig.add_subplot(gs[3, 1])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Z-score (Log2 CPM)', rotation=270, labelpad=22, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    # Legend
    legend_elements = [
        Patch(facecolor=SAMPLE_GROUPS['BSL']['color'],
              label=f"BSL: {SAMPLE_GROUPS['BSL']['label']} (n=1)"),
        Patch(facecolor=SAMPLE_GROUPS['GC']['color'],
              label=f"GC: {SAMPLE_GROUPS['GC']['label']} (n=3)"),
        Patch(facecolor=SAMPLE_GROUPS['FLT']['color'],
              label=f"FLT: {SAMPLE_GROUPS['FLT']['label']} (n=5)"),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.99, 0.97), fontsize=9,
               framealpha=0.9, edgecolor='gray')
    
    plt.savefig(
        os.path.join(output_dir, f'KEGG_{pathway_name.replace(" ", "_")}_heatmap.png'),
        dpi=300, bbox_inches='tight', facecolor='white'
    )
    print(f"    Saved: {pathway_name}")
    plt.close()


def create_composite_heatmap(expression_data, pathways_dict, symbol_map, output_dir):
    """Create composite heatmap of all pathways."""
    
    print("\nGenerating composite heatmap...")
    
    all_data = []
    boundaries = []
    current_idx = 0
    pathway_names = []
    
    for pathway_name, pathway_info in pathways_dict.items():
        available_genes = [g for g in pathway_info['genes'] if g in expression_data.index]
        
        if len(available_genes) >= 3:
            pathway_data = expression_data.loc[available_genes]
            pathway_zscore = zscore_normalize(pathway_data)
            
            all_data.append(pathway_zscore)
            boundaries.append((current_idx, current_idx + len(available_genes)))
            pathway_names.append(pathway_name)
            current_idx += len(available_genes)
    
    if not all_data:
        print("No pathway data available!")
        return None
    
    combined = pd.concat(all_data)
    n_total_genes = len(combined)
    n_samples = len(combined.columns)
    sample_labels = [get_sample_short_name(s) for s in combined.columns]
    
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
    
    # Title
    fig.text(0.5, 0.975, 'KEGG Pathways Composite Heatmap - OSD-194 Mouse Eye (Left Retina)',
             ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.955,
             'Space Flight (n=5) vs Ground Control (n=3) | Basal Control (n=1) | Z-score normalized',
             ha='center', va='top', fontsize=10, style='italic', color='#555')
    fig.text(0.5, 0.937,
             f'9 Left retina samples | Red = High, Blue = Low',
             ha='center', va='top', fontsize=9, color='#777')
    
    # Row 1: Color band
    ax_band = fig.add_subplot(gs[1, 1])
    metadata = get_sample_metadata()
    group_labels_list = [metadata[s] for s in combined.columns]
    
    for i, group in enumerate(group_labels_list):
        ax_band.barh(0, 1, left=i, color=SAMPLE_GROUPS[group]['color'],
                    height=1, edgecolor='white', linewidth=1)
    ax_band.set_xlim(0, n_samples)
    ax_band.set_ylim(0, 1)
    ax_band.axis('off')
    
    # Row 2: Group text labels
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
            ax_glabel.plot([left + 0.1, right - 0.1], [0.3, 0.3],
                         color=SAMPLE_GROUPS[group_id]['color'], linewidth=2)
            ax_glabel.plot([left + 0.1, left + 0.1], [0.3, 0.5],
                         color=SAMPLE_GROUPS[group_id]['color'], linewidth=2)
            ax_glabel.plot([right - 0.1, right - 0.1], [0.3, 0.5],
                         color=SAMPLE_GROUPS[group_id]['color'], linewidth=2)
    
    # Row 3: Main heatmap
    ax_heatmap = fig.add_subplot(gs[3, 1])
    
    im = ax_heatmap.imshow(
        combined.values,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-3, vmax=3,
        interpolation='nearest'
    )
    
    ax_heatmap.set_xticks(range(n_samples))
    ax_heatmap.set_xticklabels(sample_labels, fontsize=10, fontweight='bold')
    ax_heatmap.tick_params(axis='y', length=0)
    ax_heatmap.set_yticks([])
    
    for i, (start, end) in enumerate(boundaries):
        if i > 0:
            ax_heatmap.axhline(y=start - 0.5, color='black', linewidth=1.5, linestyle='-')
    
    ax_heatmap.set_xticks(np.arange(-0.5, n_samples, 1), minor=True)
    ax_heatmap.grid(which='minor', color='white', linewidth=0.5)
    
    # Pathway labels on right
    ax_plabels = fig.add_subplot(gs[3, 2])
    ax_plabels.axis('off')
    ax_plabels.set_ylim(0, n_total_genes)
    ax_plabels.set_xlim(0, 1)
    
    for i, (start, end) in enumerate(boundaries):
        mid = (start + end - 1) / 2
        ax_plabels.text(0.05, n_total_genes - mid - 0.5,
                       pathway_names[i],
                       ha='left', va='center', fontsize=8,
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2',
                                facecolor='lightyellow',
                                edgecolor='gray',
                                alpha=0.9))
    
    # Colorbar
    cbar_ax = fig.add_axes([0.01, 0.12, 0.015, 0.55])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Z-score', rotation=90, labelpad=8, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    # Legend
    legend_elements = [
        Patch(facecolor=SAMPLE_GROUPS['BSL']['color'],
              label=f"BSL: {SAMPLE_GROUPS['BSL']['label']} (n=1)"),
        Patch(facecolor=SAMPLE_GROUPS['GC']['color'],
              label=f"GC: {SAMPLE_GROUPS['GC']['label']} (n=3)"),
        Patch(facecolor=SAMPLE_GROUPS['FLT']['color'],
              label=f"FLT: {SAMPLE_GROUPS['FLT']['label']} (n=5)"),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.99, 0.98), fontsize=9,
               framealpha=0.9, edgecolor='gray')
    
    plt.savefig(
        os.path.join(output_dir, 'KEGG_all_pathways_composite_heatmap.png'),
        dpi=300, bbox_inches='tight', facecolor='white'
    )
    print(f"Saved: composite heatmap")
    plt.close()
    return combined


def create_top_degs_heatmap(expression_data, viz_data, symbol_map, n_top=50, output_dir='.'):
    """Create heatmap of top differentially expressed genes from visualization output table."""
    
    print("\nGenerating top DEGs heatmap...")
    
    if viz_data is None:
        print("Visualization data not available, skipping...")
        return None
    
    # 使用 "Ground Control vs Space Flight" 对比 (主研究问题)
    lfc_col = 'Log2fc_(Ground Control)v(Space Flight)'
    padj_col = 'Adj.p.value_(Ground Control)v(Space Flight)'
    
    if lfc_col not in viz_data.columns:
        # 尝试反向对比
        lfc_col = 'Log2fc_(Space Flight)v(Ground Control)'
        padj_col = 'Adj.p.value_(Space Flight)v(Ground Control)'
    
    if lfc_col not in viz_data.columns:
        print(f"Could not find LFC column. Available: {[c for c in viz_data.columns if 'Log2fc' in c]}")
        return None
    
    deg_data = viz_data[['ENSEMBL', 'SYMBOL', lfc_col, padj_col]].copy()
    deg_data = deg_data.dropna(subset=[lfc_col])
    
    # 按Log2FC绝对值排序
    deg_data['abs_lfc'] = deg_data[lfc_col].abs()
    
    # 筛选显著DEG: adj.p < 0.05 AND |log2FC| > 1
    sig_degs = deg_data[(deg_data[padj_col] < 0.05) & (deg_data['abs_lfc'] > 1.0)].sort_values('abs_lfc', ascending=False)
    
    print(f"  Significant DEGs (adj.p<0.05 & |log2FC|>1): {len(sig_degs)}")
    
    if len(sig_degs) >= n_top:
        top_degs = sig_degs.head(n_top)
        print(f"  Using top {n_top} by |log2FC|")
    elif len(sig_degs) >= 10:
        top_degs = sig_degs
        print(f"  Using all {len(sig_degs)} significant DEGs")
    else:
        # 放宽阈值
        sig_degs = deg_data[deg_data[padj_col] < 0.05].sort_values('abs_lfc', ascending=False)
        print(f"  Relaxed: using top {n_top} from {len(sig_degs)} DEGs (adj.p<0.05)")
        top_degs = sig_degs.head(n_top)
    
    available_degs = [g for g in top_degs['ENSEMBL'].values if g in expression_data.index]
    
    if len(available_degs) < 5:
        print(f"Only {len(available_degs)} DEGs found in expression data, skipping...")
        return None
    
    deg_expression = expression_data.loc[available_degs]
    
    # 过滤低表达基因
    min_samples_expressed = 3
    min_count = 10
    expr_mask = (deg_expression > min_count).sum(axis=1) >= min_samples_expressed
    deg_expression = deg_expression.loc[expr_mask]
    
    # 同步更新top_degs
    top_degs = top_degs.set_index('ENSEMBL').loc[deg_expression.index].reset_index()
    
    print(f"  After low-expression filter (>=10 in >=3 samples): {len(deg_expression)} genes")
    
    if len(deg_expression) < 5:
        print(f"  Too few genes after filtering, skipping...")
        return None
    
    # Log2(x+1) transform后再Z-score
    deg_log2 = np.log2(deg_expression + 1)
    deg_zscore = zscore_normalize(deg_log2)
    
    n_degs = len(deg_zscore)
    n_samples = len(deg_zscore.columns)
    sample_labels = [get_sample_short_name(s) for s in deg_zscore.columns]
    
    # Clustering
    row_linkage = linkage(pdist(deg_zscore), method='average')
    row_dendro = dendrogram(row_linkage, no_plot=True)
    row_order = row_dendro['leaves']
    deg_ordered = deg_zscore.iloc[row_order]
    
    # Build symbol lookup
    symbol_lookup = {}
    if 'ENSEMBL' in top_degs.columns:
        for _, row in top_degs.iterrows():
            sym_val = row.get('SYMBOL', '')
            if pd.notna(sym_val) and str(sym_val).strip() != '' and str(sym_val).strip().lower() != 'nan':
                symbol_lookup[row['ENSEMBL']] = str(sym_val)
    
    ordered_symbols = []
    for g in deg_ordered.index:
        if g in symbol_lookup:
            ordered_symbols.append(symbol_lookup[g])
        else:
            ordered_symbols.append(get_gene_symbol(g, symbol_map))
    
    fig_height = max(10, n_degs * 0.35 + 4.5)
    fig_width = 11
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        width_ratios=[1, 0.04],
        height_ratios=[0.8, 0.25, 0.2, n_degs * 0.35],
        hspace=0.02, wspace=0.02,
        left=0.18, right=0.88, top=0.92, bottom=0.07
    )
    
    fig.text(0.5, 0.97,
             f'Top {n_degs} Differentially Expressed Genes',
             ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.945,
             f'GC vs FLT | OSD-194 Mouse Eye (Left Retina only)',
             ha='center', va='top', fontsize=10, style='italic', color='#555')
    fig.text(0.5, 0.925,
             f'Z-score Normalized | BSL (n=1) | GC (n=3) | FLT (n=5)',
             ha='center', va='top', fontsize=9, color='#777')
    
    # Group color band
    ax_band = fig.add_subplot(gs[1, 0])
    metadata = get_sample_metadata()
    group_labels_list = [metadata[s] for s in deg_zscore.columns]
    for i, group in enumerate(group_labels_list):
        ax_band.barh(0, 1, left=i, color=SAMPLE_GROUPS[group]['color'],
                    height=1, edgecolor='white', linewidth=1)
    ax_band.set_xlim(0, n_samples)
    ax_band.set_ylim(0, 1)
    ax_band.axis('off')
    
    # Group text labels
    ax_glabel = fig.add_subplot(gs[2, 0])
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
            ax_glabel.plot([left + 0.1, right - 0.1], [0.3, 0.3],
                         color=SAMPLE_GROUPS[group_id]['color'], linewidth=2)
            ax_glabel.plot([left + 0.1, left + 0.1], [0.3, 0.5],
                         color=SAMPLE_GROUPS[group_id]['color'], linewidth=2)
            ax_glabel.plot([right - 0.1, right - 0.1], [0.3, 0.5],
                         color=SAMPLE_GROUPS[group_id]['color'], linewidth=2)
    
    # Heatmap
    ax_heatmap = fig.add_subplot(gs[3, 0])
    
    im = ax_heatmap.imshow(deg_ordered.values, aspect='auto',
                          cmap='RdBu_r', vmin=-3, vmax=3,
                          interpolation='nearest')
    
    ax_heatmap.set_xticks(range(n_samples))
    ax_heatmap.set_xticklabels(sample_labels, fontsize=10, fontweight='bold')
    ax_heatmap.set_yticks(range(n_degs))
    ax_heatmap.set_yticklabels(ordered_symbols, fontsize=8)
    
    ax_heatmap.set_xticks(np.arange(-0.5, n_samples, 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, n_degs, 1), minor=True)
    ax_heatmap.grid(which='minor', color='white', linewidth=0.5)
    ax_heatmap.tick_params(which='minor', length=0)
    
    cbar_ax = fig.add_subplot(gs[3, 1])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Z-score (Log2 CPM)', rotation=270, labelpad=22, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    legend_elements = [
        Patch(facecolor=SAMPLE_GROUPS['BSL']['color'],
              label=f"BSL: {SAMPLE_GROUPS['BSL']['label']} (n=1)"),
        Patch(facecolor=SAMPLE_GROUPS['GC']['color'],
              label=f"GC: {SAMPLE_GROUPS['GC']['label']} (n=3)"),
        Patch(facecolor=SAMPLE_GROUPS['FLT']['color'],
              label=f"FLT: {SAMPLE_GROUPS['FLT']['label']} (n=5)"),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.99, 0.97), fontsize=9,
               framealpha=0.9, edgecolor='gray')
    
    plt.savefig(os.path.join(output_dir, 'top_DEGs_heatmap.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: top DEGs heatmap ({n_degs} genes)")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("OSD-194 KEGG Pathway Heatmap Analysis (v5 - Left Retina only)")
    print("Source data: GLDS-194 NASA OSDR")
    print("Sample selection: Left Retina (LRTN) only | Right Retina (RRTN) excluded")
    print("Groups: Basal Control (n=1), Ground Control (n=3), Space Flight (n=5)")
    print("Main comparison: Ground Control vs Space Flight")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # === 1. Load source data ===
    print("\n--- LOADING SOURCE DATA ---")
    
    # Normalized counts
    print("Loading normalized counts...")
    norm_counts = pd.read_csv(NORMALIZED_COUNTS_FILE, index_col=0)
    print(f"  Shape: {norm_counts.shape}")
    print(f"  Samples (all): {len(norm_counts.columns)}")
    for c in norm_counts.columns:
        is_lrtn = 'LRTN' in c
        marker = 'L' if is_lrtn else 'R'
        print(f"    [{marker}] {c}")
    
    # Gene symbol mapping
    symbol_map = load_gene_symbol_mapping(VIZ_OUTPUT_FILE)
    
    # Visualization output table
    print("\nLoading visualization output table (DEG results)...")
    viz_data = pd.read_csv(VIZ_OUTPUT_FILE)
    print(f"  Shape: {viz_data.shape}")
    
    # === 2. Data validation: select only LRTN samples ===
    print("\n--- DATA VALIDATION (Left Retina filtering) ---")
    
    # Filter: only keep LRTN samples
    lrtn_samples = [c for c in norm_counts.columns if 'LRTN' in c]
    rrtn_samples = [c for c in norm_counts.columns if 'RRTN' in c]
    
    print(f"  Total samples in source: {len(norm_counts.columns)}")
    print(f"  Left retina (LRTN) samples: {len(lrtn_samples)}")
    print(f"  Right retina (RRTN) samples: {len(rrtn_samples)} [EXCLUDED]")
    
    norm_counts_lrtn = norm_counts[lrtn_samples].copy()
    print(f"  Filtered expression matrix shape: {norm_counts_lrtn.shape}")
    
    # === 3. Verify expected samples are present ===
    expected_samples = []
    for group_id in GROUP_ORDER:
        expected_samples.extend(SAMPLE_GROUPS[group_id]['samples'])
    
    missing = [s for s in expected_samples if s not in norm_counts_lrtn.columns]
    if missing:
        print(f"  WARNING: {len(missing)} expected samples not found:")
        for s in missing:
            print(f"    {s}")
        # Use intersection
        actual_expected = [s for s in expected_samples if s in norm_counts_lrtn.columns]
        print(f"  Using {len(actual_expected)} available samples")
    else:
        print(f"  All {len(expected_samples)} expected Left retina samples found [OK]")
    
    # === 4. Reorder samples ===
    expression_data = reorder_samples(norm_counts_lrtn)
    print(f"\nFinal expression data shape: {expression_data.shape}")
    print(f"Sample order (Left retina only):")
    for i, s in enumerate(expression_data.columns):
        group = get_sample_metadata()[s]
        print(f"  {i+1}. {get_sample_short_name(s)} -> {SAMPLE_GROUPS[group]['label']}")
    
    # === 5. Generate individual pathway heatmaps ===
    print("\n--- GENERATING PATHWAY HEATMAPS ---")
    for pathway_name, pathway_info in KEGG_PATHWAYS.items():
        create_pathway_heatmap(pathway_name, pathway_info, expression_data, symbol_map, OUTPUT_DIR)
    
    # === 6. Composite heatmap ===
    print("\n--- GENERATING COMPOSITE HEATMAP ---")
    create_composite_heatmap(expression_data, KEGG_PATHWAYS, symbol_map, OUTPUT_DIR)
    
    # === 7. Top DEGs heatmap ===
    print("\n--- GENERATING TOP DEGs HEATMAP ---")
    create_top_degs_heatmap(expression_data, viz_data, symbol_map, n_top=50, output_dir=OUTPUT_DIR)
    
    # === 8. Summary ===
    summary = []
    for pathway_name, pathway_info in KEGG_PATHWAYS.items():
        gene_list = pathway_info['genes']
        available = [g for g in gene_list if g in expression_data.index]
        summary.append({
            'Pathway': pathway_name,
            'Total_Genes': len(gene_list),
            'Found_Genes': len(available),
            'Percentage': f"{len(available)/len(gene_list)*100:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'pathway_summary.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("Pathway Summary:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # Save a report of which samples were used
    report_path = os.path.join(OUTPUT_DIR, 'data_filtering_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("OSD-194 Data Filtering Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source: {NORMALIZED_COUNTS_FILE}\n")
        f.write(f"Original samples: {len(norm_counts.columns)}\n")
        f.write(f"Excluded (RRTN - Right retina): {len(rrtn_samples)}\n")
        for s in rrtn_samples:
            f.write(f"  - {s}\n")
        f.write(f"\nRetained (LRTN - Left retina): {len(lrtn_samples)}\n")
        for s in lrtn_samples:
            group = get_sample_metadata().get(s, 'Unknown')
            f.write(f"  [{group}] {s}\n")
    
    print(f"\nFiltering report: {report_path}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
