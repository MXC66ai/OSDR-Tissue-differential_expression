#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import ssl
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS_DIR = r'E:\maxc\OSD\OSD255\results_OSD255'
SIG_FILE = os.path.join(RESULTS_DIR, 'significant_DEGs.csv')
'''
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
'''

def get_gene_symbol(ensembl_id, symbol_map):
    sym = symbol_map.get(ensembl_id, '')
    if pd.isna(sym) or sym == '' or str(sym).lower() == 'nan':
        return ensembl_id
    return sym


def run_gprofiler(genes, organism='mmusculus', query_name='OSD255_DEGs'):
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
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        response = urllib.request.urlopen(req, timeout=120, context=ssl_context)
        result = json.loads(response.read().decode('utf-8'))
        return result
    except Exception as e:
        print(f"g:Profiler 请求失败: {e}")
        return None


print("=" * 80)
print("OSD-255 功能富集分析 (SSL verify disabled)")
print("=" * 80)

sig_genes = pd.read_csv(SIG_FILE)
print(f"显著差异基因数: {len(sig_genes)}")

symbol_map = dict(zip(sig_genes['ENSEMBL'], sig_genes['SYMBOL']))
gene_symbols = []
for _, row in sig_genes.iterrows():
    sym = get_gene_symbol(row['ENSEMBL'], symbol_map)
    if sym and not sym.startswith('ENSMUSG'):
        gene_symbols.append(sym)

print(f"有符号的基因数: {len(gene_symbols)}")
print(f"前10个基因: {gene_symbols[:10]}")

if len(gene_symbols) >= 5:
    print("\n运行 g:Profiler 富集分析 ...")
    gprofiler_result = run_gprofiler(gene_symbols)

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
            print("  Saved: GO_enrichment_barplot")

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
            print("  Saved: KEGG_enrichment_barplot")

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
            print("  Saved: enrichment_bubble_plot")
    else:
        print("未获得 g:Profiler 富集结果")
else:
    print("有符号基因数太少, 跳过富集分析")

print("\n富集分析完成.")
