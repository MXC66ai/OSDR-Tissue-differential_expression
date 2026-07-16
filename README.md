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
List of abbreviation：
 ie：Insulin and Estrogen

References:
[1] Pan, S., Du, S., Ma, Z. et al. A transparent PVA-based polymer sunscreen protects retinal tissue from ultraviolet damage. Nat Commun (2026). https://doi.org/10.1038/s41467-026-73448-8
[2] Skovsø, S., Panzhinskiy, E., Kolic, J. et al. Beta-cell specific Insr deletion promotes insulin hypersecretion and improves glucose tolerance prior to global insulin resistance. Nat Commun 13, 735 (2022). https://doi.org/10.1038/s41467-022-28039-8
[3] Hafemeister C, Satija R. Normalization and variance stabilization of single-cell RNA-seq data using regularized negative binomial regression. Genome Biol. 2019 Dec 23;20(1):296. doi: 10.1186/s13059-019-1874-1. PMID: 31870423; PMCID: PMC6927181.
[4] McDonald, J. Tyson, Robert Stainforth, Jack Miller, Thomas Cahill, Willian A. da Silveira, Komal S. Rathi, Gary Hardiman, Deanne Taylor, Sylvain V. Costes, Vinita Chauhan, and et al. 2020. "NASA GeneLab Platform Utilized for Biological Response to Space Radiation in Animal Models" Cancers 12, no. 2: 381. https://doi.org/10.3390/cancers12020381
[5] Arnold C. Differential Gene Expression Analysis of Rodent Mammary Tissue Reveals Dysregulation of Greb1 and an Age-Dependent Response to Spaceflight.
[6] Purushothaman I. Dysregulation of the OGF-OGFr Axis Determines Onset and Severity of Ocular Surface Complications in Type 1 Diabetes.
[7] Roy U, Hadad R, Rodriguez AA, Saju A, Roy D, Gil M, Keane RW, Scott RT, Mao XW, de Rivero Vaccari JP. Effects of Space Flight on Inflammasome Activation in the Brain of Mice. Cells. 2025 Mar 12;14(6):417.
[8] Rauscher FG, Elze T, Francke M, Martinez-Perez ME, Li Y, Wirkner K, Tönjes A, Engel C, Thiery J, Blüher M, Stumvoll M. Glucose tolerance and insulin resistance/sensitivity associate with retinal layer characteristics: the LIFE-Adult-Study. Diabetologia. 2024 May;67(5):928-39.
[9] Zheng Z, Yu X. Insulin resistance in the retina: possible implications for certain ocular diseases. Frontiers in Endocrinology. 2024 Jun 17;15:1415521.
[10] Arnold C, Casaletto J, Heller P. Spaceflight disrupts gene expression of estrogen signaling in rodent mammary tissue. Medical Research Archives. 2024 Sep 11;12(3).
[11] Grandke F, Rishik S, Wagner V, Engel A, Ludwig N, Calcuttawala K, Kern F, Keller V, Krawczyk M, Stodieck L, Ferguson V. MiRNAs shape mouse age-independent tissue adaptation to spaceflight via ECM and developmental pathways. Nature Communications. 2026 Feb 5.
[12] Siew K, Nestler KA, Nelson C, D’Ambrosio V, Zhong C, Li Z, Grillo A, Wan ER, Patel V, Overbey E, Kim J. Cosmic kidney disease: an integrated pan-omic, physiological and morphological study into spaceflight-induced renal dysfunction. Nature communications. 2024 Jun 11;15(1):4923.
[13] Lee R, Rayhun A, Kim JK, Meydan C, Beheshti A, Sporn K, Kumar R, Calixte J, McNerney MW, Shah J, Waisberg E. Genomic Analysis of Cardiovascular Diseases Utilizing Space Omics and Medical Atlas. Genes. 2025 Aug 25;16(9):996.
[14] Sadaki S, Fujita R, Hayashi T, Nakamura A, Okamura Y, Fuseya S, Hamada M, Warabi E, Kuno A, Ishii A, Muratani M. Large Maf transcription factor family is a major regulator of fast type IIb myofiber determination. Cell reports. 2023 Apr 25;42(4).
[15] Camera A, Tabetah M, Castañeda V, Kim J, Galsinh AS, Haro-Vinueza A, Salinas I, Seylani A, Arif S, Das S, Mori MA. Aging and putative frailty biomarkers are altered by spaceflight. Scientific Reports. 2024 Jun 11;14(1):13098.
[16] Mathyk BA, Tabetah M, Karim R, Zaksas V, Kim J, Anu RI, Muratani M, Tasoula A, Singh RS, Chen YK, Overbey E. Spaceflight induces changes in gene expression profiles linked to insulin and estrogen. Communications Biology. 2024 Jun 11;7(1):692.
[17] Casaletto JA, Zhao T, Yeung J, Ansari A, Raj A, Mishra A, Fry A, Sun K, Lendahl S, Guan W, Lee A. Machine Learning Ensemble Reveals Age-Specific Responses of Murine Mammary Tissue to Spaceflight With Relevance to Breast Cancer: An Observational Study. bioRxiv. 2025 Feb 23:2025-02.
[18] Scotti MM, Wilson BK, Bubenik JL, Yu F, Swanson MS, Allen JB. Spaceflight effects on human vascular smooth muscle cell phenotype and function. npj Microgravity. 2024 Mar 28;10(1):41.
[19] Mao X, Stanbouly S, Holley J, Pecaut M, Crapo J. Evidence of spaceflight-induced adverse effects on photoreceptors and retinal function in the mouse eye. International journal of molecular sciences. 2023 Apr 17;24(8):7362.
[20] Lee AG, Mader TH, Gibson CR, Tarver W, Rabiei P, Riascos RF, Galdamez LA, Brunstetter T. Spaceflight associated neuro-ocular syndrome (SANS) and the neuro-ophthalmologic effects of microgravity: a review and an update. npj Microgravity. 2020 Feb 7;6(1):7.
[21] Titone R, Zhu M, Robertson DM. Insulin mediates de novo nuclear accumulation of the IGF-1/insulin Hybrid Receptor in corneal epithelial cells. Scientific reports. 2018 Mar 12;8(1):4378.
[22] Faiq MA, Sengupta T, Nath M, Velpandian T, Saluja D, Dada R, Dada T, Chan KC. Ocular manifestations of central insulin resistance. Neural Regeneration Research. 2023 May 1;18(5):1139-46.
[23] Järgen P, Dietrich A, Herling AW, Hammes HP, Wohlfart P. The role of insulin resistance in experimental diabetic retinopathy—Genetic and molecular aspects. PloS one. 2017 Jun 2;12(6):e0178658.
[24] Mason CE, Green J, Adamopoulos KI, Afshin EE, Baechle JJ, Basner M, Bailey SM, Bielski L, Borg J, Borg J, Broddrick JT. A second space age spanning omics, platforms and medicine across orbits. Nature. 2024 Aug 29;632(8027):995-1008.
