| Dataset | Variety | Organization | Group | Significantly differentially expressed genes (padj < 0.05) | Source | 
|-----------|---------|---------------|------------|----------------------------|------------|
  | OSD-100 | C57BL/6J | Whole eye | GC n=6 vs FLT n=6             |       193 | RR-1 (37 days ISS) |
| OSD-162 | BALB/c | Whole eye | GC n=5 vs FLT n=5 (BSL excluded) | 0   | RR-3 (41–42 days ISS) |
| OSD-194 | BALB/c | Left retina | GC n=4 vs FLT n=5 (BSL excluded) | 0 | RR-3 (41–42 days ISS) | 

Main script:
- build_ie_gene_list.py: Generates ncbi_curated_ie_gene_list.tsv
- eye_ie_pipeline.py: Generates heatmaps, clustering, GO enrichment tables, and outputs them to eye_ie_results/
- eye_ie_supplementary.py: Supplements Venn diagrams, scatter plots, pathway bar charts, and eye function annotation tables.
  
Running the Python code in this repository will produce the following files:
- fig1c_ie_heatmap.png: Z-score heatmap of insulin/estrogen/IR genes in 3 eye datasets (in the style of Fig 1c)
- fig1d_ie_heatmap_clusters.png: 7-cluster annotated heatmap (in the style of Fig 1d)
- supp_venn_overlap.png: Overlapping graph of IE genes in the 3 datasets
- supp_pairwise_scatter.png: Pairwise scatter plot of log2FC and correlation
- supp_pathway_barplot.png: Bar chart of up/down gene numbers for each pathway
- ie_genes_eye_annotated.tsv: Complete gene table with eye function annotations
- eye_function_summary.tsv: Statistical summary of eye functions
- cluster_assignments.tsv: Cluster assignment and most significant GO-BP term for each gene
- dataset_summary.tsv: Statistical summary of IE pathways in each dataset
- fig1c_log2FC_matrix.tsv / fig1d_log2FC_matrix.tsv: Original log2FC matrices
