import pandas as pd
import torch
import re
import os
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import time
import sys

# ==== 设置环境 ====
print("📦 蛋白质特征提取程序启动...")
start_time = time.time()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 使用设备: {device}")

# 设置模型路径
MODEL_PATH = "../../PorstT5"
print(f"🔍 模型路径: {os.path.abspath(MODEL_PATH)}")

# 检查模型是否存在
if not os.path.exists(MODEL_PATH):
    print(f"❌ 错误: 模型目录不存在: {MODEL_PATH}")
    print("请确认模型下载到正确位置")
    sys.exit(1)

# 检查模型文件
model_files = os.listdir(MODEL_PATH)
required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
missing_files = [f for f in required_files if f not in model_files]

if missing_files:
    print(f"❌ 模型文件不完整，缺少文件: {', '.join(missing_files)}")
    print("请重新下载完整模型")
    sys.exit(1)
else:
    print("✅ 模型文件完整")

# ==== 加载模型 ====
print("\n🚀 正在加载ProstT5模型...")

# 尝试加载tokenizer
try:
    print("  1. 加载tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_PATH,
        do_lower_case=False,
        legacy=False  # 禁用旧版行为
    )
    print("  ✅ tokenizer加载成功")
except Exception as e:
    print(f"  ❌ tokenizer加载失败: {str(e)}")
    print("  尝试使用旧版tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_PATH,
        do_lower_case=False,
        legacy=True
    )
    print("  ✅ 旧版tokenizer加载成功")

# 尝试加载主模型
try:
    print("  2. 加载主模型...")
    model_encoder = T5EncoderModel.from_pretrained(MODEL_PATH).to(device)
    print(f"  ✅ 主模型加载成功 ({model_encoder.__class__.__name__})")
except Exception as e:
    print(f"  ❌ 模型加载失败: {str(e)}")
    sys.exit(1)

# 设置模型精度
if device.type == 'cuda':
    model_encoder = model_encoder.half()
    print("  🔄 模型设置为半精度 (float16)")
else:
    print("  🔄 CPU模式使用单精度 (float32)")

# 测试模型推理
print("  3. 测试模型推理...")
try:
    test_seq = "MKKLLFAIVLVVLTLLTAEGPSQPTR"
    inputs = tokenizer(
        f"<AA2fold> {' '.join(list(test_seq))}",
        return_tensors="pt",
        add_special_tokens=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        output = model_encoder(**inputs)

    print(f"  ✅ 模型测试成功! 输出形状: {output.last_hidden_state.shape}")
except Exception as e:
    print(f"  ❌ 模型推理测试失败: {str(e)}")
    sys.exit(1)

# ==== 设置数据路径 ====
DATA_DIR = "./dataset1"
LIGAND_FILE = os.path.join(DATA_DIR, "ligand.csv")
RECEPTOR_FILE = os.path.join(DATA_DIR, "receptor.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "features_1")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n📂 数据文件路径:")
print(f"  配体数据: {os.path.abspath(LIGAND_FILE)}")
print(f"  受体数据: {os.path.abspath(RECEPTOR_FILE)}")
print(f"  输出目录: {os.path.abspath(OUTPUT_DIR)}")

# 检查数据文件
for path in [LIGAND_FILE, RECEPTOR_FILE]:
    if not os.path.exists(path):
        print(f"❌ 错误: 文件不存在: {os.path.abspath(path)}")
        sys.exit(1)


# ==== 核心功能 ====
def preprocess_sequence(seq):
    """预处理蛋白质序列"""
    seq = re.sub(r"[UZOB]", "X", seq.upper())
    return " ".join(list(seq))


def extract_global_features(seq, max_length=1024):
    """提取蛋白质的全局特征向量"""
    # 处理超长序列
    original_seq = seq
    if len(seq) > max_length:
        seq = seq[:max_length] + "..."  # 截断但保留后缀表示

    # 预处理序列
    processed = preprocess_sequence(seq)

    # 添加前缀（AA到3Di）
    full_seq = "<AA2fold> " + processed

    # 编码输入
    inputs = tokenizer(
        full_seq,
        return_tensors="pt",
        add_special_tokens=True,
        padding='max_length',
        max_length=min(1024, len(processed) + 10),  # 序列长度+缓冲
        truncation=True
    ).to(device)

    # 提取嵌入
    with torch.no_grad():
        output = model_encoder(**inputs)

    # 获取所有非填充token的嵌入
    embeddings = output.last_hidden_state[0]

    # 计算全局特征（均值池化）
    global_feature = embeddings.mean(dim=0).cpu().numpy()

    return global_feature

def process_protein_file(file_path, prefix):
    """处理蛋白质文件并提取特征"""
    print(f"\n🔍 开始处理 {prefix} 数据: {os.path.basename(file_path)}")

    # 读取CSV文件
    try:
        print(f"  读取数据文件...")
        # 假设文件有表头：protein_name 和 sequence
        df = pd.read_csv(file_path)
        print(f"  ✅ 读取完成! 共 {len(df)} 条蛋白质序列")
        print(f"  样本数据: {df.iloc[0]['protein_name']} - {df.iloc[0]['sequence'][:30]}...")
    except Exception as e:
        print(f"  ❌ 文件读取失败: {str(e)}")
        print(f"  尝试无表头读取...")
        try:
            df = pd.read_csv(file_path, header=None, names=["protein_name", "sequence"])
            print(f"  ✅ 无表头读取成功! 共 {len(df)} 条序列")
        except:
            print(f"  ❌ 无法读取文件! 跳过处理。")
            return

    # 确保序列是字符串
    df['sequence'] = df['sequence'].astype(str)

    # 准备结果存储
    global_features = []
    protein_names = []
    error_count = 0
    processed_count = 0

    # 进度条设置
    print("  ⏳ 开始特征提取...")
    progress_bar = tqdm(df.iterrows(), total=len(df), desc=f"提取{prefix}特征")

    # 处理每条序列
    for idx, row in progress_bar:
        name = row["protein_name"]
        seq = row["sequence"]

        protein_names.append(name)
        progress_bar.set_postfix({
            "蛋白质": name[:10] + "..." if len(name) > 10 else name,
            "长度": len(seq),
            "错误": error_count
        })

        try:
            # 对非常长的序列进行警告
            if len(seq) > 1500:
                print(f"\n  ⚠️ 警告: {name} 序列超长 ({len(seq)}个残基)")

            # 提取特征
            feat = extract_global_features(seq)
            global_features.append(feat)
            processed_count += 1

        except torch.cuda.OutOfMemoryError:
            print(f"\n  ❗️ CUDA显存不足! 跳过蛋白质 {name} (长度: {len(seq)})")
            global_features.append(np.zeros(1024))
            error_count += 1
            # GPU内存清理
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n  ❗️ 处理{name}时出错: {str(e)}")
            global_features.append(np.zeros(1024))
            error_count += 1

        # 进度报告
        if processed_count % 10 == 0:
            # 显存使用情况报告
            if device.type == 'cuda':
                mem_used = torch.cuda.memory_allocated() / 1024 ** 2
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
                progress_bar.set_postfix({
                    "进度": f"{idx + 1}/{len(df)}",
                    "GPU内存": f"{mem_used:.0f}/{mem_total:.0f} MB"
                })

    # 创建DataFrame保存结果
    result_df = pd.DataFrame(global_features)
    result_df.insert(0, "protein_name", protein_names)

    # 列名格式化
    feature_columns = [f"feat_{i}" for i in range(1024)]
    result_df.columns = ["protein_name"] + feature_columns

    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, f"{prefix}_global_features.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ {prefix}全局特征已保存: {os.path.basename(output_path)}")
    print(f"  处理完成! 总序列: {len(df)}, 成功: {processed_count}, 失败: {error_count}")

    return result_df


# ==== 主程序 ====
if __name__ == "__main__":
    print("\n" + "=" * 50)

    # 先处理一个样本测试
    print("\n🧪 运行测试样本...")
    test_protein = "ENSP00000002829"
    test_sequence = "MLVAGLLLSVLSLTAIGAPSPTQODLIPATPVPVLSFLSKELKATCTAIPFFHLQLTYLLTLLKGDHRMDVYGCSKVDYLSLSLIDHNEEPPLIHHWAAAGTGGTGGKDSKVDYLSKFLVSLKFDGLVSL"

    try:
        print(f"  蛋白质: {test_protein}")
        print(f"  序列长度: {len(test_sequence)}")

        test_feature = extract_global_features(test_sequence)
        print(f"  ✅ 测试成功! 特征向量长度: {len(test_feature)}")

    except Exception as e:
        print(f"  ❌ 测试失败: {str(e)}")
        print("  请在解决此问题后再继续处理主文件")
        sys.exit(1)

    # 处理主文件
    print("\n" + "=" * 50)
    print("🚀 开始处理主文件...")

    ligand_result = process_protein_file(LIGAND_FILE, "ligand")

    print("\n" + "=" * 50)

    receptor_result = process_protein_file(RECEPTOR_FILE, "receptor")

    # 完成统计
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"🎉 所有处理完成! 总用时: {total_time / 60:.2f} 分钟")

    if ligand_result is not None:
        print(f"配体数量: {len(ligand_result)}")
    if receptor_result is not None:
        print(f"受体数量: {len(receptor_result)}")

    # GPU内存使用统计
    if device.type == 'cuda':
        print(f"峰值GPU内存使用: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
