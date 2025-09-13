import os
import pandas as pd
import numpy as np
import gc
import time

# -------------------- 路径配置 --------------------
AUC_MTX_PATH = 'auc_mtx.csv'  # TF活性矩阵文件
EXPR_FILT_PATH = 'expr_filtered.csv'  # 过滤后的表达矩阵
META_FILT_PATH = 'meta_filtered.csv'  # 过滤后的元数据
LR_PAIRS_PATH = '../out/five-fold-cross-validation/auc/dataset1/predictions+known_lri.csv'  # 配体-受体对列表
OUTPUT_MATRIX_PATH = 'cell_type_communication_matrix.csv'
OUTPUT_TOP3_PATH = 'top3_ligand_receptor_communication.csv'

# -------------------- 参数配置 --------------------
MIN_TF_ACTIVITY = 0.01  # TF活性最小值阈值
MIN_EXPRESSION = 0.05  # 最小表达水平阈值
TOP_N = 3  # 每个细胞类型对保留的Top N配体-受体对

# 定义标准细胞类型名称
STANDARD_CELL_TYPES = {
    "fibroblast": "Fibroblasts",
    "b cell": "B cells",
    "myocyte": "Myocytes",
    "macrophage": "Macrophages",
    "endothelial": "Endothelial cells",
    "t cell": "T cells",
    "dendritic": "Dendritic cells",
    "mast": "Mast cells"
}


def standardize_cell_type(cell_type):
    """标准化细胞类型名称"""
    cell_type_lower = cell_type.lower().strip()
    if cell_type_lower in STANDARD_CELL_TYPES:
        return STANDARD_CELL_TYPES[cell_type_lower]
    return "HNSCC cancer cells"


def load_data():
    """加载所有必要数据"""
    print("📂📂📂📂📂📂📂📂 加载数据...")
    start_time = time.time()

    # 加载TF活性矩阵
    auc_mtx = pd.read_csv(AUC_MTX_PATH, index_col=0)
    print(f"✅ TF活性矩阵 | TF数量: {auc_mtx.shape[1]} | 细胞数量: {auc_mtx.shape[0]}")

    # 加载过滤后的表达矩阵
    expr_filt = pd.read_csv(EXPR_FILT_PATH, index_col=0)
    expr_filt = expr_filt.fillna(0)
    print(f"✅ 表达矩阵 | 基因数量: {expr_filt.shape[0]} | 细胞数量: {expr_filt.shape[1]}")

    # 加载过滤后的元数据
    meta_filt = pd.read_csv(META_FILT_PATH, index_col=0)
    print(f"✅ 元数据 | 细胞数量: {meta_filt.shape[0]}")

    # 标准化细胞类型名称
    if 'non-cancer cell type' in meta_filt.columns:
        meta_filt['standard_cell_type'] = meta_filt['non-cancer cell type'].apply(standardize_cell_type)
        print("📊 标准化后的细胞类型统计:")
        print(meta_filt['standard_cell_type'].value_counts())
    else:
        print("⚠️ 警告: 元数据中没有 'non-cancer cell type' 列")
        meta_filt['standard_cell_type'] = "HNSCC cancer cells"

    # 打印癌细胞比例
    if 'classified as cancer cell' in meta_filt.columns:
        cancer_cells = meta_filt['classified as cancer cell'].sum()
        print(f"📊 癌细胞数量: {cancer_cells} ({cancer_cells / meta_filt.shape[0] * 100:.1f}%)")
    else:
        print("⚠️ 警告: 元数据中没有 'classified as cancer cell' 列")

    # 加载配体-受体对
    lr_pairs = pd.read_csv(LR_PAIRS_PATH)
    print(f"✅ L-R对 | 数量: {lr_pairs.shape[0]}")

    # 重命名列
    lr_pairs = lr_pairs.rename(columns={
        'ligand_gene': 'Ligand',
        'receptor_gene': 'Receptor'
    })

    elapsed = time.time() - start_time
    print(f"✅ 数据加载完成 | 耗时: {elapsed:.1f}秒")

    return auc_mtx, expr_filt, meta_filt, lr_pairs


def calculate_cell_communication(auc_mtx, expr_filt, meta_filt, lr_pairs):
    """计算所有细胞类型之间的通讯得分"""
    print("🧪🧪🧪🧪🧪🧪🧪🧪 开始计算细胞通讯得分...")
    start_time = time.time()

    # 1. 准备数据 - 使用标准化细胞类型
    cell_types = meta_filt['standard_cell_type'].copy()
    unique_cell_types = cell_types.unique()
    print(f"📊📊📊📊 唯一细胞类型: {len(unique_cell_types)}种")

    # 打印细胞类型列表
    print("📊 细胞类型列表:")
    for cell_type in unique_cell_types:
        print(f"- {cell_type}")

    # 2. 计算细胞类型平均表达量
    print("📊📊📊📊📊📊📊📊 计算细胞类型平均表达...")
    cell_type_expression = {}
    for cell_type in unique_cell_types:
        cell_ids = cell_types[cell_types == cell_type].index
        valid_cell_ids = [cid for cid in cell_ids if cid in expr_filt.columns]

        if len(valid_cell_ids) == 0:
            cell_type_expression[cell_type] = pd.Series(0, index=expr_filt.index)
            continue

        expr_subset = expr_filt[valid_cell_ids].astype(float)
        expr_subset = np.log1p(expr_subset)
        cell_type_expression[cell_type] = expr_subset.mean(axis=1)

    # 3. 初始化得分矩阵
    scores_matrix = pd.DataFrame(
        np.zeros((len(unique_cell_types), len(unique_cell_types))),
        index=unique_cell_types, columns=unique_cell_types
    )
    print(f"📊 通讯矩阵维度: {scores_matrix.shape}")

    # 4. 初始化Top3配体-受体得分存储
    top3_data = {}

    # 5. 计算TF活性代表值
    print("📊📊📊📊 计算TF活性代表值...")
    try:
        auc_mtx_values = auc_mtx.astype(float).values
        tf_mean_activities = np.log1p(auc_mtx_values).mean(axis=0)
        global_tf_act_score = tf_mean_activities.mean()
        print(f"✅ TF活性代表值: {global_tf_act_score}")
    except Exception as e:
        print(f"⚠️ 计算TF活性时出错: {e}")
        global_tf_act_score = 0

    # 应用TF活性过滤
    if global_tf_act_score < MIN_TF_ACTIVITY:
        print(f"⚠️ 警告: TF活性代表值 {global_tf_act_score:.4f} < 阈值 {MIN_TF_ACTIVITY}")
    else:
        print(f"✅ TF活性代表值满足阈值")

    # 6. 预过滤L-R对
    valid_lr_pairs = []
    for idx, row in lr_pairs.iterrows():
        ligand = str(row['Ligand']).strip()
        receptor = str(row['Receptor']).strip()

        if ligand in expr_filt.index and receptor in expr_filt.index:
            valid_lr_pairs.append({
                'Ligand': ligand,
                'Receptor': receptor
            })

    valid_lr_df = pd.DataFrame(valid_lr_pairs)
    print(f"📊📊📊📊 有效L-R对: {len(valid_lr_df)}/{len(lr_pairs)} "
          f"({len(valid_lr_df) / len(lr_pairs) * 100:.1f}%)")

    # 7. 分块处理L-R对
    chunk_size = 50
    total_pairs = len(valid_lr_df)

    for chunk_start in range(0, total_pairs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_pairs)
        chunk = valid_lr_df.iloc[chunk_start:chunk_end]

        for _, pair in chunk.iterrows():
            ligand = pair['Ligand']
            receptor = pair['Receptor']

            # 跳过如果TF活性不满足阈值
            if global_tf_act_score < MIN_TF_ACTIVITY:
                continue

            lr_scores = []

            for sender_type in unique_cell_types:
                ligand_expr = cell_type_expression[sender_type].get(ligand, 0)
                if isinstance(ligand_expr, pd.Series):
                    ligand_expr = ligand_expr.iloc[0] if not ligand_expr.empty else 0

                if ligand_expr < MIN_EXPRESSION:
                    continue

                for receiver_type in unique_cell_types:
                    receptor_expr = cell_type_expression[receiver_type].get(receptor, 0)
                    if isinstance(receptor_expr, pd.Series):
                        receptor_expr = receptor_expr.iloc[0] if not receptor_expr.empty else 0

                    if receptor_expr < MIN_EXPRESSION:
                        continue

                    base_score = np.sqrt(ligand_expr * receptor_expr)
                    weighted_score = base_score * global_tf_act_score

                    # 累加到总得分矩阵
                    scores_matrix.loc[sender_type, receiver_type] += weighted_score

                    # 存储当前L-R对在此细胞类型对上的得分
                    lr_scores.append({
                        'sender': sender_type,
                        'receiver': receiver_type,
                        'score': weighted_score,
                        'ligand': ligand,
                        'receptor': receptor
                    })

            # 更新Top3数据
            for score_entry in lr_scores:
                sender = score_entry['sender']
                receiver = score_entry['receiver']
                key = (sender, receiver)

                if key not in top3_data:
                    top3_data[key] = []

                top3_data[key].append((
                    score_entry['score'],
                    score_entry['ligand'],
                    score_entry['receptor']
                ))

        gc.collect()
        print(f"🔄🔄🔄🔄 已完成 {chunk_end}/{total_pairs} L-R对处理")

    # 8. 归一化得分矩阵
    if scores_matrix.sum().sum() > 0:
        max_score = scores_matrix.values.max()
        if max_score > 0:
            normalized_scores = scores_matrix / max_score
        else:
            normalized_scores = scores_matrix.copy()
            print("⚠️ 警告: 所有通讯得分均为0")
    else:
        normalized_scores = scores_matrix.copy()
        print("⚠️ 警告: 所有通讯得分均为0")

    # 9. 提取Top3配体-受体对
    top3_df = extract_top3_ligand_receptors(top3_data, TOP_N)

    # 10. 保存结果
    normalized_scores.to_csv(OUTPUT_MATRIX_PATH)
    top3_df.to_csv(OUTPUT_TOP3_PATH)

    elapsed = time.time() - start_time
    print(f"✅ 细胞通讯得分计算完成 | 耗时: {elapsed:.1f}秒")

    print("🧪🧪 生成严格过滤后的LRI文件...")

    # 使用集合自动去重
    filtered_lri_pairs = set()

    # 只收集在癌细胞相关通讯中出现的L-R对
    for (sender, receiver), scores in top3_data.items():
        if "HNSCC cancer cells" not in [sender, receiver]:
            continue

        for score, ligand, receptor in scores:
            # 确保表达量满足阈值
            sender_expr = cell_type_expression[sender].get(ligand, 0)
            receiver_expr = cell_type_expression[receiver].get(receptor, 0)

            if (isinstance(sender_expr, (pd.Series, np.ndarray)) and
                    isinstance(receiver_expr, (pd.Series, np.ndarray))):
                sender_expr = sender_expr.mean()
                receiver_expr = receiver_expr.mean()

            if sender_expr >= MIN_EXPRESSION and receiver_expr >= MIN_EXPRESSION:
                filtered_lri_pairs.add((ligand, receptor))

    # 转换为DataFrame并保存
    filtered_lri_df = pd.DataFrame(filtered_lri_pairs, columns=['Ligand', 'Receptor'])
    filtered_lri_df.to_csv('filtered_lri_pairs.csv', index=False)

    print(f"✅ 保存严格过滤后的LRI对至: filtered_lri_pairs.csv")
    print(f"📊 过滤后LRI数量: {len(filtered_lri_df)} (原始数量: {len(lr_pairs)})")

    # 确保只返回两个值以匹配原有调用
    return normalized_scores, top3_df


def extract_top3_ligand_receptors(top3_data, n=3):
    """提取所有细胞类型通讯的Top N配体-受体对（仅保留涉及癌细胞的通讯）"""
    print("📊📊📊📊 提取Top3配体-受体对（仅癌细胞相关）...")

    # 准备结果数据结构
    results = []

    # 处理每个细胞类型对
    for (sender, receiver), scores in top3_data.items():
        # 只保留涉及癌细胞的通讯对
        if "HNSCC cancer cells" not in [sender, receiver]:
            continue

        # 按得分降序排序
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)

        # 取前N个
        top_n = sorted_scores[:n]

        # 添加到结果
        for i, (score, ligand, receptor) in enumerate(top_n):
            cell_pair = f"{sender}-{receiver}"
            ligand_receptor = f"{ligand}-{receptor}"

            results.append({
                'Ligand-Receptor': ligand_receptor,
                'Cell-Type-Pair': cell_pair,
                'Score': score,
                'Rank': i + 1
            })

    # 转换为DataFrame
    result_df = pd.DataFrame(results)

    # 转换为要求的格式: 行为配体-受体对，列为细胞类型对
    if not result_df.empty:
        pivot_df = result_df.pivot_table(
            index='Ligand-Receptor',
            columns='Cell-Type-Pair',
            values='Score',
            aggfunc='first'
        )

        # 填充NaN为0
        pivot_df = pivot_df.fillna(0)

        # 添加总分列
        pivot_df['Total_Score'] = pivot_df.sum(axis=1)

        # 按总分排序
        pivot_df = pivot_df.sort_values('Total_Score', ascending=False)

        print(f"✅ 提取完成 | 配体-受体对数量: {pivot_df.shape[0]} | 细胞类型对数量: {pivot_df.shape[1] - 1}")
        return pivot_df
    else:
        print("⚠️ 警告: 没有找到有效的配体-受体对得分")
        return pd.DataFrame()


def main():
    print("=" * 50)
    print("🧪🧪🧪🧪🧪🧪🧪🧪 细胞通讯得分计算器")
    print("=" * 50)

    # 1. 加载数据
    auc_mtx, expr_filt, meta_filt, lr_pairs = load_data()

    # 2. 计算并保存结果
    communication_matrix, top3_df = calculate_cell_communication(auc_mtx, expr_filt, meta_filt, lr_pairs)

    # 3. 单独生成过滤后的LRI文件
    print("\n🧪🧪 单独生成最终的过滤LRI文件...")
    # 从top3_df中提取所有唯一的配体-受体对
    all_lr_pairs = set()
    for col in top3_df.columns:
        if col == 'Total_Score':
            continue
        for lr_pair in top3_df.index:
            if top3_df.loc[lr_pair, col] > 0:
                ligand, receptor = lr_pair.split('-', 1)
                all_lr_pairs.add((ligand, receptor))

    final_filtered_lri = pd.DataFrame(all_lr_pairs, columns=['Ligand', 'Receptor'])
    final_filtered_lri.to_csv('final_filtered_lri_pairs.csv', index=False)

    print(f"✅ 最终过滤后的LRI对数量: {len(final_filtered_lri)}")

    # 打印结果摘要
    print("\n📊 通讯矩阵摘要:")
    print(communication_matrix)

    print("\n📊 Top3配体-受体对摘要:")
    print(top3_df.head())

    print("=" * 50)
    print("✅ 分析完成！结果已保存至:")
    print(f"- 细胞类型通讯矩阵: {OUTPUT_MATRIX_PATH}")
    print(f"- Top3配体-受体通讯得分: {OUTPUT_TOP3_PATH}")
    print("=" * 50)


if __name__ == '__main__':
    main()
