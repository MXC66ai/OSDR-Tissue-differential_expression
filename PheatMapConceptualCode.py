import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import cluster
from matplotlib import colormaps # Used for color scales

# ==================================================
# 步骤 1: 加载数据和创建示例矩阵 (假设 data_matrix 是你的输入 mat)
# data_matrix: 此处应替换为实际数据，必须是DataFrame或NumPy Array格式
# 假定我们有 50 个基因 (rows) 和 8 个样本 (cols) 的表达值。
np.random.seed(42)
num_genes = 50
num_samples = 8
data_matrix = pd.DataFrame(
    np.random.rand(num_genes, num_samples),
    index=[f'Gene_{i}' for i in range(num_genes)],
    columns=[f'Sample_{j}' for j in range(num_samples)]
)

# ==================================================
# 步骤 2: 设置绘图参数 (对应 R 代码中的复杂设置)
heatmap_params = {
    'cmap': 'RdYlBu_r',  # RdYlBu 的反转版本，更常用
    'metric': 'euclidean', # 距离度量
    'method': 'ward',       # 或者 'average' / 'complete' 等聚类方法
    'figsize': (10, 12),    # 设置图表尺寸
    'linewidths': 0.5,      # 列间线宽，对应 border_color
    'annot_k': None         # 如果需要显示数值，可设置此项，否则设为None
}

# ==================================================
# 步骤 3: 使用 seaborn.clustermap 函数生成热图 (最接近 pheatmap 的函数)
print("正在使用 seaborn.clustermap 生成热图...")

plt.figure(figsize=heatmap_params['figsize'])

# **核心功能调用**：seaborn.clustermap 会自动处理聚类和绘图流程，极大地简化了R的复杂参数。
sns.clustermap(
    data_matrix,
    cmap=heatmap_params['cmap'],
    metric=heatmap_params['metric'],
    method=heatmap_params['method'], # 对行和列都进行层次聚类
    figsize=heatmap_params['figsize']
)

plt.suptitle("Heatmap Visualization of Omics Data (Python Equivalent)", y=1.02, fontsize=16)
plt.show()

