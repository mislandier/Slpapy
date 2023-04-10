import csv
import os
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def resave(input_file, head):
    # 指定输入和输出文件路径
    input_file = str(input_file)
    output_file = f'{input_file}' + "-1.csv"

    # 增加字段大小限制
    csv.field_size_limit(104857600)  # 100MB

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
    resave(TIC, 10)
    resave(XY, 8)
    # 重新读取并转换保存为AnnData对象
    mz = pd.DataFrame()
    df = pd.read_csv(f'{TIC}' + "-1.csv", header=0, index_col='m/z', low_memory=False)
    df1 = pd.read_csv(f'{XY}' + "-1.csv", header=0, index_col='Spot index', low_memory=False)
    os.remove(f'{TIC}' + "-1.csv")
    os.remove(f'{XY}' + "-1.csv")
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
