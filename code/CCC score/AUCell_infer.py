#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pyscenic.utils import modules_from_adjacencies
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from arboreto.algo import grnboost2
from arboreto.utils import load_tf_names
from dask.distributed import Client, LocalCluster
import psutil
import gc
import time
from ctxcore.genesig import Regulon
import dask.config

warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -------------------- 路径配置 --------------------
EXPR_PATH = r'/data/wuwei/cell/BERT/case_study/HNSCC/expr_matrix.csv'
META_PATH = r'/data/wuwei/cell/BERT/case_study/HNSCC/cell_annotations.csv'
LR_PATH = r'/data/wuwei/cell/BERT/out/five-fold-cross-validation/auc/dataset1/predictions+known_lri.csv'
TF_LIST_PATH = r'/data/wuwei/cell/BERT/dataset/tf_motif_rankDB/allTFs_hg38.csv'
FEATHER_PATH = r'/data/wuwei/cell/BERT/dataset/tf_motif_rankDB/hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather'
MOTIF_ANNO = r'/data/wuwei/cell/BERT/dataset/tf_motif_rankDB/motifs-v9-nr.hgnc-m0.001-o0.0.tbl'

# -------------------- 参数配置 --------------------
THRESH_GENES = 500
THRESH_COUNTS = 1000
THRESH_LR_TF = 0.8
MIN_TF_ACTIVITY = 0.1
COMMUNICATION_THRESHOLD = 0.15
MEMORY_LIMIT = "40GB"
NUM_WORKERS = 4
MIN_GENE_EXPRESSION = 0.1
MIN_CELLS_WITH_GENE = 10
MIN_CELL_PERCENT = 0.05
MIN_EXPRESSION_PERCENTILE = 10
MIN_REGULON_COVERAGE = 0.5
MAX_REGULON_GENES = 100
MIN_REGULON_GENES = 5

def optimize_memory(df):
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype('float32')
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='unsigned')
        elif pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype('category')
    return df

def load_and_validate_inputs():
    """加载并校验输入数据，返回 expr, meta, lr"""
    print("🧪 加载数据...")
    expr = pd.read_csv(EXPR_PATH, index_col=0, skiprows=[1, 2, 3, 4, 5])
    expr.index = expr.index.str.strip("'")
    expr = optimize_memory(expr.astype(float))

    meta = pd.read_csv(META_PATH, index_col=0, usecols=['Cell', 'classified as cancer cell', 'non-cancer cell type'])
    meta = optimize_memory(meta)

    lr = pd.read_csv(LR_PATH)
    lr = optimize_memory(lr)

    print(f"✅ 原始数据 | 基因: {expr.shape[0]} | 细胞: {expr.shape[1]}")
    print(f"✅ 元数据   | 细胞: {meta.shape[0]}")
    print(f"✅ LR对     | 条目: {lr.shape[0]}")

    # 确保索引对齐
    if not expr.columns.equals(meta.index):
        print("⚠️ 表达矩阵与元数据的 cell 顺序不一致，尝试对齐...")
        common_cells = expr.columns.intersection(meta.index)
        expr = expr[common_cells]
        meta = meta.loc[common_cells]
        print(f"对齐后 | 基因: {expr.shape[0]} | 细胞: {expr.shape[1]}")

    # 检查空数据
    if expr.empty or meta.empty:
        raise ValueError("❌ 表达矩阵或元数据为空！")

    return expr, meta, lr

def filter_genes_debug(expr):
    print("🧬 进行基因过滤...")
    gene_means = expr.mean(axis=1)
    gene_n_cells = (expr > 0).sum(axis=1)
    total_cells = expr.shape[1]
    min_cells = max(50, int(total_cells * MIN_CELL_PERCENT))
    mean_threshold = np.percentile(gene_means, MIN_EXPRESSION_PERCENTILE)
    print(f"过滤阈值 | 最小细胞数: {min_cells} | 平均表达分位数: {mean_threshold:.4f}")
    good_genes = (gene_means >= mean_threshold) & (gene_n_cells >= min_cells)
    expr_filt = expr.loc[good_genes]
    print(f"保留基因: {expr_filt.shape[0]}/{expr.shape[0]} ({expr_filt.shape[0]/expr.shape[0]*100:.1f}%)")
    if expr_filt.empty:
        raise ValueError("❌ 基因过滤后没有基因剩余！")
    return expr_filt

def filter_cells_debug(expr, meta):
    print("🔍 细胞质控...")
    n_genes = (expr > 0).sum(axis=0)
    n_counts = expr.sum(axis=0)
    qc_metrics = pd.DataFrame({'n_genes': n_genes, 'n_counts': n_counts})
    qc_metrics.to_csv('cell_qc_metrics.csv')
    good_cells = (n_genes >= THRESH_GENES) & (n_counts >= THRESH_COUNTS)
    good_cells = good_cells.reindex(meta.index, fill_value=False)
    expr_filt = expr.loc[:, good_cells]
    meta_filt = meta.loc[good_cells]
    if expr_filt.empty:
        raise ValueError("❌ 细胞过滤后没有细胞剩余！")
    expr_filt = filter_genes_debug(expr_filt)
    expr_filt = expr_filt.astype('float32')
    print(f"✅ 质控完成 | 细胞: {expr_filt.shape[1]} | 基因: {expr_filt.shape[0]}")
    return expr_filt, meta_filt

def prepare_grn_inputs(expr_filt):
    tf_names_raw = load_tf_names(TF_LIST_PATH)
    tf_names = [str(tf).strip().strip("'").upper() for tf in tf_names_raw]
    gene_names = [str(g).strip().strip("'").upper() for g in expr_filt.index]
    expr_filt.index = gene_names  # 覆盖原始索引
    common_tfs = [tf for tf in tf_names if tf in gene_names]
    print(f"✅ TF列表 | 原始: {len(tf_names)} | 存在于表达矩阵: {len(common_tfs)}")
    if not common_tfs:
        print("❗ 前10个表达矩阵基因示例:", gene_names[:10])
        print("❗ 前10个TF示例:", tf_names[:10])
        raise ValueError("❌ 没有TF存在于表达矩阵中，请检查基因名格式！")
    return common_tfs, gene_names

def main():
    total_ram = psutil.virtual_memory().total // (1024 ** 3)
    print(f"系统资源 | CPU: {psutil.cpu_count()} | 内存: {total_ram}GB")

    # 1. 加载和校验
    expr, meta, lr = load_and_validate_inputs()
    expr_filt, meta_filt = filter_cells_debug(expr, meta)
    del expr, meta
    gc.collect()

    # 2. 准备 GRN 输入
    common_tfs, gene_names = prepare_grn_inputs(expr_filt)

    # 3. 启动 Dask
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=1,
                           memory_limit=MEMORY_LIMIT, silence_logs=30)
    client = Client(cluster)
    print(f"✅ Dask已启动 | 仪表盘: {cluster.dashboard_link}")

    print("✅ 基因名格式检查 | 前5个基因:", expr_filt.index[:5])
    print("✅ TF名格式检查   | 前5个TF:", common_tfs[:5])

    try:
        # 4. GRN 推断
        print("🧬 开始 GRN 推断...")
        expr_clean = expr_filt.astype(float).values
        print(f"传入 grnboost2 的 TF 数量: {len(common_tfs)}")
        print(f"传入 grnboost2 的基因数量: {len(expr_filt.index)}")

        print("🧪 检查表达方差...")
        variances = expr_filt.var(axis=1)
        print(f"最小方差: {variances.min():.6f}")
        print(f"最大方差: {variances.max():.6f}")
        print(f"零方差基因数: {(variances == 0).sum()}")

        if variances.max() < 1e-6:
            raise ValueError("❌ 所有基因表达方差过低，无法建模！")

        # 强制检查 NaN/Inf
        expr_filt = expr_filt.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
        expr_filt = expr_filt.astype(np.float32)

        print("✅ 清理 NaN/Inf 后 | 基因: {} | 细胞: {}".format(expr_filt.shape[0], expr_filt.shape[1]))
        print("是否有 NaN:", np.isnan(expr_filt.values).any())
        print("是否有 Inf:", np.isinf(expr_filt.values).any())

        adj = grnboost2(
            expression_data=expr_filt.T.astype(float),
            gene_names=expr_filt.index.astype(str).tolist(),
            tf_names=common_tfs,
            client_or_address=client,
            verbose=True,
            seed=42,
        )
        print(f"✅ GRN 推断完成 | 边数: {adj.shape[0]}")

        # 5. 其余步骤省略（与原始脚本相同）
        feather_db = RankingDatabase(FEATHER_PATH, 'hg38')
        modules = modules_from_adjacencies(adj, expr_filt.T)
        regulons = []
        total_mod = len(modules)
        for i, mod in enumerate(modules):
            try:
                df = prune2df([feather_db], [mod], MOTIF_ANNO, client_or_address=client)
                if not df.empty:
                    regs = df2regulons(df)
                    for reg in regs:
                        genes = list(reg.genes)[:MAX_REGULON_GENES]
                        present = [g for g in genes if g in expr_filt.index]
                        cov = len(present) / len(genes) if genes else 0
                        if cov >= MIN_REGULON_COVERAGE and len(present) >= MIN_REGULON_GENES:
                            new_reg = Regulon(
                                reg.name,
                                reg.gene2weight,
                                reg.transcription_factor,
                                reg.context
                            )
                            regulons.append(new_reg)

            except Exception as e:
                print(f"模块 {i} 处理失败: {e}")

            # ===== 每 50 个模块手动回收一次 =====
            if (i + 1) % 50 == 0:
                gc.collect()
                client.run(lambda: gc.collect())  # 所有 worker 一起收
                print(f"↻ 已处理 {i + 1}/{total_mod} 模块，强制 GC 一次")

        print(f"✅ 调控子构建完成 | 原始: {len(modules)} | 有效: {len(regulons)}")
        auc_mtx = aucell(expr_filt.T, regulons, auc_threshold=0.05, noweights=True, num_workers=NUM_WORKERS)
        auc_mtx.to_csv('auc_mtx.csv')
        print(f"✅ TF活性计算完成 | TF: {auc_mtx.shape[1]} | 细胞: {auc_mtx.shape[0]}")

        # -------------- 导出 regulons.csv --------------
        regulons_df = pd.DataFrame([
            {"tf": reg.transcription_factor, "gene": g, "weight": w}
            for reg in regulons
            for g, w in reg.gene2weight.items()
        ])
        regulons_df.to_csv('regulons.csv', index=False)
        print("✅ regulons.csv 已导出完成")

        # -------------- 一次性保存中间结果 --------------
        adj.to_pickle('adj.pkl')  # GRN 边表
        with open('modules.pkl', 'wb') as f:
            pickle.dump(modules, f)  # 模块列表
        with open('regulons.pkl', 'wb') as f:
            pickle.dump(regulons, f)  # regulon 对象
        print("✅ 已保存 adj.pkl / modules.pkl / regulons.pkl")

    except Exception as e:
        print("❌ 分析出错:", e)
        import traceback
        traceback.print_exc()
    finally:
        client.close()
        cluster.close()
        print("✅ Dask已关闭")

if __name__ == '__main__':
    dask.config.set({
        "distributed.worker.memory.target": 0.5,  # 50 % 开始 spill
        "distributed.worker.memory.spill": 0.7,  # 70 % 强制 spill
        "distributed.worker.memory.pause": 0.98,  # 90 % 才暂停
        "distributed.worker.memory.terminate": 0.99  # 98 % 才 kill
    })
    main()
