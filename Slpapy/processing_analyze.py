import csv
import os
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.cluster import KMeans

sc.settings.verbosity = 3  # 设置日志等级: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=720, facecolor='white')


def resave(input_file, head):
    # 指定输入和输出文件路径
    input_file = str(input_file)
    output_file = f'{input_file}' + "-1.csv"

    # 增加字段大小限制
    csv.field_size_limit(104857600)  # 1MB

    # 打开输入和输出文件
    with open(input_file, 'r', encoding='utf-8') as input_csv_file, open(output_file, 'w', newline='',
                                                                         encoding='utf-8') as output_csv_file:
        # 创建CSV读取器和写入器对象
        input_csv_reader = csv.reader(input_csv_file, delimiter=';')
        output_csv_writer = csv.writer(output_csv_file)

        # 跳过前6行
        for _ in range(int(head)):
            next(input_csv_reader)

        # 逐行读取并写入新的CSV文件
        for row in input_csv_reader:
            output_csv_writer.writerow(row)


def Data_reconstruction(TIC, XY):
    TIC = str(TIC)
    XY = str(XY)
    # 文件修整重排
    resave(TIC, 9)
    resave(XY, 7)
    # 重新读取并转换保存为AnnData对象
    df = pd.read_csv(f'{TIC}' + "-1.csv", header=1, index_col='m/z', low_memory=False)
    df1 = pd.read_csv(f'{XY}' + "-1.csv", header=1, index_col='Spot index', low_memory=False)
    # 建一个基本的 AnnData 对象
    counts = csr_matrix(df.T, dtype=np.float32)
    adata = ad.AnnData(counts)
    # 为x和y轴提供索引
    adata.obs_names = df.columns
    adata.var['m/z'] = df.index
    print(adata.obs_names[:])
    print(adata.var_names[:])
    # 标注坐标位置
    adata.obs["X"] = df1.x.values
    adata.obs["Y"] = df1.y.values
    adata.write(f'{XY}' + ".h5ad")
    return adata


def XYadata(adata):
    df = preprocessing.normalize(np.c_[np.array(adata.obs['X']), np.array(adata.obs['Y'])])
    counts = csr_matrix(df, dtype=np.float32)
    XYmatrix = ad.AnnData(counts)
    # 为x和y轴提供索引
    XYmatrix.obs_names = adata.obs_names
    XYmatrix.var_names = ['X', 'Y']
    t = adata.obs
    adata = ad.concat([adata, XYmatrix], axis=1)
    adata.obs = t
    adata.obsm['X_spacial'] = np.array(adata.obs.loc[:, 'X':'Y'])
    return adata


def pp_analyze(adata):
    # 显示在所有细胞中在每个单细胞中产生最高计数分数的脂质
    # sc.pl.highest_expr_genes(adata, n_top=20, save='_highest_expr_protein.png')
    # 小提琴图：表达基因的数量，每个细胞包含的表达量，线粒体基因表达量的百分比。
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # 将线粒体基因标记为 mt
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None,
                               log1p=False, inplace=True)
    sc.pl.violin(adata, 'total_counts', jitter=0.4, multi_panel=True,
                 save='_n_genes_by_counts_and_total_counts_and_pct_counts_mt.png')
    # 归一化，使得不同细胞样本间可比
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # 存储数据
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(adata, save='_highly_variable_lipid.png')
    adata = adata[:, adata.var.highly_variable]
    # 回归每个细胞的总计数和表达的线粒体基因的百分比的影响。
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    adata.write('./pp_done.h5ad')
    return adata


def se_analyze(adata):
    # 将每个基因缩放到单位方差。阈值超过标准偏差 10。
    sc.pp.scale(adata, max_value=10)
    # 绘制 PCA 图降维
    sc.tl.pca(adata, svd_solver='arpack')  #
    # Neighborhood graph使用数据矩阵的 PCA 表示来计算细胞的邻域图
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50, )
    # sc.pp.neighbors(adata, n_neighbors=20, n_pcs=30, )#YONG
    sc.tl.umap(adata, min_dist=0.3, alpha=10, spread=1)  #
    # Leiden 图聚类
    # 计算
    sc.tl.leiden(adata, resolution=0.3)
    return adata


def Spatial_map(adata, cls):
    cls = str(cls)
    sc.pl.embedding(adata, basis='X_spacial', color=f'{cls}', frameon=False, save=f'_spacial_{cls}.png')
    """
    如需原始比例则在embedding 458行后加上
    axs.axis('equal')
    """


def Basic_processing_flow(
        adata: ad.AnnData,
        *,
        use_spacial: bool = False,
        orgknn: bool = False,
) -> Optional[ad.AnnData]:
    adata = pp_analyze(adata)
    adata.obsm['X_spacial'] = np.array(adata.obs.loc[:, 'X':'Y'])
    if use_spacial == True:
        adata = XYadata(adata)
        adata = se_analyze(adata)
        sc.pl.pca_variance_ratio(adata, log=True, save='_spacial.png')
        sc.pl.umap(adata, color=['leiden'], save='_spacial.png')
        sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
        sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save='_spacial_Wilcoxon.png')
        Spatial_map(adata, 'leiden')
    else:
        adata = se_analyze(adata)
        sc.pl.pca_variance_ratio(adata, log=True, save='_general.png')
        sc.pl.umap(adata, color=['leiden'], save='_general.png')
        sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
        sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save='_Wilcoxon.png')
        adata.obs['leidenXY'] = adata.obs['leiden']
        Spatial_map(adata, 'leidenXY')

    if orgknn == True:
        adata = se_analyze(adata)
        estimator = KMeans(n_clusters=adata.obs['leiden'].values.codes.max() + 1)  # 构造聚类器
        estimator.fit(
            np.c_[adata.X, preprocessing.normalize(np.c_[np.array(adata.obs['X']), np.array(adata.obs['Y'])])])
        label_pred = estimator.labels_  # 获取聚类标签
        adata.obs['KNNlableXY'] = label_pred.T
        adata.obs['KNNlableXY'] = adata.obs['KNNlableXY'].astype('category')
        sc.tl.rank_genes_groups(adata, 'KNNlableXY', method='wilcoxon')
        sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save='_KXY_Wilcoxon.png')
        sc.pl.pca_variance_ratio(adata, log=True, save='_KXY.png')
        sc.pl.umap(adata, color=['KNNlableXY'], save='_KXY.png')
        Spatial_map(adata, 'KNNlableXY')
    adata.write('./analyze_done.h5ad')
    return adata




