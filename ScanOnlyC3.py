import numpy as np
import re
import ast
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from Run import GetCorrCum, EFFICIENCY_ARRAY, Get_Raw_Cumulants_From_CorrectedCumulants

def _read_array_from_text(text, name):
    m = re.search(rf'{name}\s*=\s*np\.array\(\s*(\[[^\]]*\])\s*\)', text, re.S)
    if m:
        return np.array(ast.literal_eval(m.group(1)))
    m = re.search(rf'{name}\s*=\s*(\[[^\]]*\])', text, re.S)
    if m:
        return np.array(ast.literal_eval(m.group(1)))
    return None

# 定义滑动窗口标准差函数
def sliding_std(data, window_size):
    if len(data) < window_size:
        return np.array([])
    stds = np.array([np.std(data[i:i+window_size]) for i in range(len(data) - window_size + 1)])
    return stds

# 定义IQR异常值检测函数
def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1 * IQR
    upper_bound = Q3 + 1 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers

# 定义C3值的范围
scan_weight_dir = './ScanOnlyC3'
file_pattern = re.compile(r'fit_results_([\d.]+)\.txt')

c3_values = set()
import os
# 遍历目录中的所有文件
for filename in os.listdir(scan_weight_dir):
    match = file_pattern.match(filename)
    if match:
        c3 = float(match.group(1))
        c3_values.add(c3)

# 转换为排序后的数组
range_c3 = np.array(sorted(c3_values))
print(range_c3)
c3_cut_lower = 16.625
c3_cut_upper = 16.79
range_c3 = range_c3[(range_c3 >= c3_cut_lower) & (range_c3 <= c3_cut_upper)]
# exit()
# 创建图形
fig, axes = plt.subplots(4, 4, figsize=(22, 10))
axes = axes.flatten()

# 为不同C3值使用不同颜色
colors = 'red'
colors1 = 'blue'
# 存储结果
chi2_list = []
weight_list = []
C1_list = []
C2_list = []
C3_list = []
C4_list = []
K41_list = []

for i, c3 in enumerate(range_c3):
    path = f'./ScanOnlyC3/fit_results_{c3:.4f}.txt'
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()

    m = re.search(r'chi2\s*=\s*([0-9.+-eE]+)', txt)
    chi2_list.append(float(m.group(1)) if m else np.nan)


    weights = _read_array_from_text(txt, 'weights')
    weight_list.append(weights)
    C1_list.append(_read_array_from_text(txt, 'C1_Fit_corr_DE_data'))
    C2_list.append(_read_array_from_text(txt, 'C2_Fit_corr_DE_data'))
    C3_list.append(_read_array_from_text(txt, 'C3_Fit_corr_DE_data'))
    C4_list.append(_read_array_from_text(txt, 'C4_Fit_corr_DE_data'))
weight_last = np.array([w[-1] for w in weight_list])
weight_last_1 = np.array([w[-2] for w in weight_list])
# Optionally convert lists of arrays to 2D numpy arrays (rows = files)
C1_array = np.vstack([a for a in C1_list if a is not None]) if any(a is not None for a in C1_list) else None
C2_array = np.vstack([a for a in C2_list if a is not None]) if any(a is not None for a in C2_list) else None
C3_array = np.vstack([a for a in C3_list if a is not None]) if any(a is not None for a in C3_list) else None
C4_array = np.vstack([a for a in C4_list if a is not None]) if any(a is not None for a in C4_list) else None
C1_array = np.vstack([a for a in C1_list if a is not None]) if any(a is not None for a in C1_list) else None
C2_array = np.vstack([a for a in C2_list if a is not None]) if any(a is not None for a in C2_list) else None
C3_array = np.vstack([a for a in C3_list if a is not None]) if any(a is not None for a in C3_list) else None
C4_array = np.vstack([a for a in C4_list if a is not None]) if any(a is not None for a in C4_list) else None
chi2_array = np.array(chi2_list)
C1_last_raw = np.zeros(len(range_c3))
C2_last_raw = np.zeros(len(range_c3))
C3_last_raw = np.zeros(len(range_c3))
C4_last_raw = np.zeros(len(range_c3))
C1_first_raw = np.zeros(len(range_c3))
C2_first_raw = np.zeros(len(range_c3))
C3_first_raw = np.zeros(len(range_c3))
C4_first_raw = np.zeros(len(range_c3))
C1_first_corr = np.zeros(len(range_c3))
C2_first_corr = np.zeros(len(range_c3))
C3_first_corr = np.zeros(len(range_c3))
C4_first_corr = np.zeros(len(range_c3))
C1_last_corr = np.zeros(len(range_c3))
C2_last_corr = np.zeros(len(range_c3))
C3_last_corr = np.zeros(len(range_c3))
C4_last_corr = np.zeros(len(range_c3))
for i in range(len(range_c3)):
    C1, C2, C3, C4 = Get_Raw_Cumulants_From_CorrectedCumulants(C1_array[i, :], C2_array[i, :], C3_array[i, :], C4_array[i, :])
    # C1, C2, C3, C4 = GetCorrCum(C1_array[i, :], C2_array[i, :], C3_array[i, :], C4_array[i, :])
    C1_last_corr[i] = C1[-1]
    C2_last_corr[i] = C2[-1]
    C3_last_corr[i] = C3[-1]
    C4_last_corr[i] = C4[-1]
    C1_first_corr[i] = C1[0]
    C2_first_corr[i] = C2[0]
    C3_first_corr[i] = C3[0]
    C4_first_corr[i] = C4[0]
    C1_first_raw[i] = C1_array[i, 0]
    C2_first_raw[i] = C2_array[i, 0]
    C3_first_raw[i] = C3_array[i, 0]
    C4_first_raw[i] = C4_array[i, 0]
    C1_last_raw[i] = C1_array[i, -1]
    C2_last_raw[i] = C2_array[i, -1]
    C3_last_raw[i] = C3_array[i, -1]
    C4_last_raw[i] = C4_array[i, -1]
# exit()
x = list(range_c3)
# C1_last, C2_last, C3_last, C4_last = GetCorrCum(C1_last, C2_last, C3_last, C4_last)
K4_last_corr = C4_last_corr - 6*C3_last_corr + 11*C2_last_corr - 6*C1_last_corr
K4_last_raw = C4_last_raw - 6*C3_last_raw + 11*C2_last_raw - 6*C1_last_raw
K41_last_corr = K4_last_corr/C1_last_corr
K41_last_raw = K4_last_raw/C1_last_raw
# 绘制图形
axes[0].plot(x, chi2_array, marker='o', linestyle='-', color=colors, 
                linewidth=1.0, markersize=5, label=f'corr')
axes[1].plot(x, C1_last_corr, marker='s', linestyle='-', color=colors, 
                linewidth=1.0, markersize=5, label=f'corr')
axes[1].plot(x, C1_last_raw, marker='s', linestyle='-', color=colors1, 
                linewidth=1.0, markersize=5, label=f'raw')
axes[2].plot(x, C2_last_corr, marker='^', linestyle='-', color=colors, 
                linewidth=1.0, markersize=5, label=f'corr')
axes[2].plot(x, C2_last_raw, marker='^', linestyle='-', color=colors1, 
                linewidth=1.0, markersize=5, label=f'raw') 
axes[3].plot(x, C3_last_corr, marker='D', linestyle='-', color=colors,  
                linewidth=1.0, markersize=5, label=f'corr')
axes[3].plot(x, C3_last_raw, marker='D', linestyle='-', color=colors1, 
                linewidth=1.0, markersize=5, label=f'raw') 
axes[4].plot(x, C4_last_corr, marker='v', linestyle='-', color=colors, 
                linewidth=1.0, markersize=5, label=f'corr')
axes[4].plot(x, C4_last_raw, marker='v', linestyle='-', color=colors1, 
                linewidth=1.0, markersize=5, label=f'raw') 
axes[5].plot(x, C2_last_corr/C1_last_corr, marker='o', linestyle='-', color=colors, 
                linewidth=1.0, markersize=5, label=f'corr')
axes[5].plot(x, C2_last_raw/C1_last_raw, marker='o', linestyle='-', color=colors1, 
                linewidth=1.0, markersize=5, label=f'raw')
axes[6].plot(x, K41_last_corr, marker='o', linestyle='-', color=colors, 
                linewidth=1.0, markersize=5, label=f'corr')
axes[6].plot(x, K41_last_raw, marker='o', linestyle='-', color=colors1, 
                linewidth=1.0, markersize=5, label=f'raw')
axes[7].plot(x, C4_last_corr/C2_last_corr, marker='o', linestyle='-', color=colors, 
                linewidth=1.0, markersize=5, label=f'corr')
axes[7].plot(x, C4_last_raw/C2_last_raw, marker='o', linestyle='-', color=colors1, 
                linewidth=1.0, markersize=5, label=f'raw')
axes[8].plot(x, C4_first_corr/C2_first_corr, marker='o', linestyle='-', color=colors, 
                linewidth=1.0, markersize=5, label=f'corr')
axes[8].plot(x, C4_first_raw/C2_first_raw, marker='o', linestyle='-', color=colors1, 
                linewidth=1.0, markersize=5, label=f'raw')
axes[9].plot(x, C2_first_corr/C1_first_corr, marker='o', linestyle='-', color=colors, 
                linewidth=1.0, markersize=5, label=f'corr')
axes[9].plot(x, C2_first_raw/C1_first_raw, marker='o', linestyle='-', color=colors1, 
                linewidth=1.0, markersize=5, label=f'raw')
axes[10].plot(x, C3_first_corr/C1_first_corr, marker='o', linestyle='-', color=colors, 
                linewidth=1.0, markersize=5, label=f'corr')
axes[10].plot(x, C3_first_raw/C1_first_raw, marker='o', linestyle='-', color=colors1, 
                linewidth=1.0, markersize=5, label=f'raw')
axes[11].plot(x, C3_last_raw/C2_last_raw, marker='o', linestyle='-', color=colors1, 
                linewidth=1.0, markersize=5, label=f'raw')
axes[11].plot(x, C3_last_corr/C2_last_corr, marker='o', linestyle='-', color=colors, 
                linewidth=1.0, markersize=5, label=f'corr')
axes[12].plot(x, weight_last/weight_last_1, marker='o', linestyle='-', color=colors, 
                linewidth=1.0, markersize=5, label=f'corr')
# 设置子图标题和标签
axes[0].set_title('Chi2 vs Weight', fontsize=14)
axes[0].set_ylabel('Chi2', fontsize=12)
axes[0].grid(True, alpha=0.4)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

axes[1].set_title('C1 (last) vs Weight', fontsize=14)
axes[1].set_ylabel('C1', fontsize=12)
axes[1].grid(True, alpha=0.4)
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

axes[2].set_title('C2 (last) vs Weight', fontsize=14)
axes[2].set_ylabel('C2', fontsize=12)
axes[2].grid(True, alpha=0.4)
axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

axes[3].set_title('C3 (last) vs Weight', fontsize=14)
axes[3].set_ylabel('C3', fontsize=12)
axes[3].set_xlabel('Weight', fontsize=12)
axes[3].grid(True, alpha=0.4)
axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

axes[4].set_title('C4 (last) vs Weight', fontsize=14)
axes[4].set_ylabel('C4', fontsize=12)
axes[4].set_xlabel('Weight', fontsize=12)
axes[4].grid(True, alpha=0.4)
axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 隐藏第6个子图
# axes[5].axis('off')
axes[5].set_title('C2/C1 (last) vs Weight', fontsize=14)
axes[5].set_xlabel('Weight', fontsize=12)
axes[5].grid(True, alpha=0.4)
axes[5].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

axes[6].set_title('K41 (last) vs Weight', fontsize=14)
axes[6].set_ylabel('K41', fontsize=12)
axes[6].set_xlabel('Weight', fontsize=12)
axes[6].grid(True, alpha=0.4)
axes[6].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

axes[7].set_title('C4/C2 (last) vs Weight', fontsize=14)
axes[7].set_ylabel('C4/C2', fontsize=12)
axes[7].set_xlabel('Weight', fontsize=12)
axes[7].grid(True, alpha=0.4)
axes[7].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

axes[8].set_title('C4/C2 (first) vs Weight', fontsize=14)
axes[8].set_ylabel('C4/C2', fontsize=12)
axes[8].set_xlabel('Weight', fontsize=12)
axes[8].grid(True, alpha=0.4)
axes[8].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

axes[9].set_title('C2/C1 (first) vs Weight', fontsize=14)
axes[9].set_ylabel('C2/C1', fontsize=12)
axes[9].set_xlabel('Weight', fontsize=12)
axes[9].grid(True, alpha=0.4)
axes[9].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

axes[10].set_title('C3/C1 (first) vs Weight', fontsize=14)
axes[10].set_ylabel('C3/C1', fontsize=12)
axes[10].set_xlabel('Weight', fontsize=12)
axes[10].grid(True, alpha=0.4)
axes[10].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

axes[11].set_title('C3/C2 (first) vs Weight', fontsize=14)
axes[11].set_ylabel('C3/C2', fontsize=12)
axes[11].set_xlabel('Weight', fontsize=12)
axes[11].grid(True, alpha=0.4)
axes[11].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

axes[12].set_title('Weight (last) vs Weight (last-1)', fontsize=14)
axes[12].set_ylabel('Weight (last)/Weight (last-1)', fontsize=12)
axes[12].set_xlabel('Weight', fontsize=12)
axes[12].grid(True, alpha=0.4)
axes[12].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 调整布局，为图例留出空间
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# 保存和显示图形
plt.savefig('./ScanOnlyC3/ScanOnlyC3_tmp.png', dpi=300, bbox_inches='tight')
# plt.show()
R21 = C2_last_corr/C1_last_corr
R42 = C4_last_corr/C2_last_corr
# 异常值检测和过滤
print("\n=== 异常值检测结果 ===")
r21_outliers = detect_outliers_iqr(R21)
r42_outliers = detect_outliers_iqr(R42)

print(f"R21异常值索引: {np.where(r21_outliers)[0]}")
print(f"R21异常值: {R21[r21_outliers]}")
print(f"R42异常值索引: {np.where(r42_outliers)[0]}")
print(f"R42异常值: {R42[r42_outliers]}")

# 创建过滤后的数据
R21_filtered = R21.copy()
R42_filtered = R42.copy()

# 标记异常值的索引
outlier_indices = np.union1d(np.where(r21_outliers)[0], np.where(r42_outliers)[0])
print(f"\n合并后的异常值索引: {outlier_indices}")

# 计算R42last和R21的波动稳定性（使用过滤后的数据）
window_size = 5  # 滑动窗口大小
min_stable_points = 3  # 稳定区域的最小点数
std_threshold = 0.1  # 波动阈值（可根据实际情况调整）

# 如果存在异常值，使用过滤后的数据；否则使用原始数据
if len(outlier_indices) > 0:
    # 为了保持数据长度一致，我们将异常值替换为附近的正常值（使用线性插值）
    valid_indices = np.setdiff1d(np.arange(len(R21)), outlier_indices)
    R21_clean = np.interp(np.arange(len(R21)), valid_indices, R21[valid_indices])
    R42_clean = np.interp(np.arange(len(R42)), valid_indices, R42[valid_indices])
    
    # 计算滑动窗口标准差
    r42_stds = sliding_std(R42_clean, window_size)
    r21_stds = sliding_std(R21_clean, window_size)
    
    print(f"\n已使用插值方法处理异常值，共处理 {len(outlier_indices)} 个异常点")
else:
    # 计算滑动窗口标准差
    r42_stds = sliding_std(R42, window_size)
    r21_stds = sliding_std(R21, window_size)
    R21_clean = R21.copy()
    R42_clean = R42.copy()

# 识别R42的稳定区域
r42_stable_regions = []
current_region = []
for i, std in enumerate(r42_stds):
    if std < std_threshold:
        current_region.append(i)
    else:
        if len(current_region) >= min_stable_points:
            r42_stable_regions.append(current_region)
        current_region = []
if len(current_region) >= min_stable_points:
    r42_stable_regions.append(current_region)

# 识别R21的稳定区域
r21_stable_regions = []
current_region = []
for i, std in enumerate(r21_stds):
    if std < std_threshold:
        current_region.append(i)
    else:
        if len(current_region) >= min_stable_points:
            r21_stable_regions.append(current_region)
        current_region = []
if len(current_region) >= min_stable_points:
    r21_stable_regions.append(current_region)

# 计算稳定区域的平均值
def calculate_stable_average(data, stable_regions, window_size):
    if not stable_regions:
        return None, []
    
    all_stable_points = []
    for region in stable_regions:
        # 将窗口索引转换为原始数据索引
        for i in region:
            # 窗口中心位置
            center_idx = i + window_size // 2
            all_stable_points.append(center_idx)
    
    # 去重并排序
    all_stable_points = sorted(list(set(all_stable_points)))
    
    # 计算平均值
    stable_values = data[all_stable_points]
    return np.mean(stable_values), all_stable_points

# 获取R42和R21的稳定区域平均值
r42_stable_avg, r42_stable_points = calculate_stable_average(R42_clean, r42_stable_regions, window_size)
r21_stable_avg, r21_stable_points = calculate_stable_average(R21_clean, r21_stable_regions, window_size)

# 找到同时满足R42和R21稳定的区域
if r42_stable_points and r21_stable_points:
    common_stable_points = list(set(r42_stable_points) & set(r21_stable_points))
    common_stable_points.sort()
    
    if common_stable_points:
        # 计算共同稳定区域的平均值
        common_r42_avg = np.mean(R42_clean[common_stable_points])
        common_r21_avg = np.mean(R21_clean[common_stable_points])
        common_weights = range_c3[common_stable_points]
        optimal_weight = np.mean(common_weights)
        
        print(f"\n=== 最佳点分析结果（过滤异常值后）===")
        print(f"R42稳定区域: {r42_stable_points}")
        print(f"R42稳定区域平均值: {r42_stable_avg:.6f}")
        print(f"R21稳定区域: {r21_stable_points}")
        print(f"R21稳定区域平均值: {r21_stable_avg:.6f}")
        print(f"共同稳定区域: {common_stable_points}")
        print(f"共同稳定区域R42平均值: {common_r42_avg:.6f}")
        print(f"共同稳定区域R21平均值: {common_r21_avg:.6f}")
        print(f"最佳Weight值: {optimal_weight:.6f}")
        
        # 绘制稳定区域和最佳点
        plt.figure(figsize=(15, 6))
        
        # R42稳定区域图
        plt.subplot(1, 2, 1)
        plt.plot(range_c3, R42, 'o-', label='R42 raw data')
        plt.plot(range_c3[outlier_indices], R42[outlier_indices], 'ro', label='R42 outliers', alpha=0.6)
        plt.plot(range_c3[r42_stable_points], R42_clean[r42_stable_points], 'go', label='R42 stable points')
        plt.axhline(y=r42_stable_avg, color='r', linestyle='--', label=f'R42 stable average: {r42_stable_avg:.4f}')
        plt.axvline(x=optimal_weight, color='g', linestyle='--', label=f'Best Weight: {optimal_weight:.4f}')
        plt.title('R42 Stability Analysis (Filtered)')
        plt.xlabel('Weight')
        plt.ylabel('R42')
        plt.legend()
        plt.grid(True, alpha=0.4)
        
        # R21稳定区域图
        plt.subplot(1, 2, 2)
        plt.plot(range_c3, R21, 'o-', label='R21 raw data')
        plt.plot(range_c3[outlier_indices], R21[outlier_indices], 'ro', label='R21 outliers', alpha=0.6)
        plt.plot(range_c3[r21_stable_points], R21_clean[r21_stable_points], 'go', label='R21 stable points')
        plt.axhline(y=r21_stable_avg, color='r', linestyle='--', label=f'R21 stable average: {r21_stable_avg:.4f}')
        plt.axvline(x=optimal_weight, color='g', linestyle='--', label=f'Best Weight: {optimal_weight:.4f}')
        plt.title('R21 Stability Analysis (Filtered)')
        plt.xlabel('Weight')
        plt.ylabel('R21')
        plt.legend()
        plt.grid(True, alpha=0.4)
        
        plt.tight_layout()
        # plt.savefig('./ScanOnlyC3/ScanOnlyC3_stability_analysis_filtered.png', dpi=300, bbox_inches='tight')

# 高斯分布拟合分析
print("\n=== 高斯分布拟合分析 ===")

# 准备用于拟合的数据
# 首先使用过滤后的R21和R42数据（去除异常值）
valid_indices = np.setdiff1d(np.arange(len(R21)), outlier_indices)
R21_for_fit = R21[valid_indices]
R42_for_fit = R42[valid_indices]
weights_for_fit = range_c3[valid_indices]

# 对R21进行高斯拟合
if len(R21_for_fit) > 2:
    # 使用scipy.stats.norm进行拟合
    r21_mean, r21_std = norm.fit(R21_for_fit)
    # 计算拟合的相关统计量
    r21_pdf = norm.pdf(R21_for_fit, r21_mean, r21_std)
    
    print(f"\nR21高斯拟合结果:")
    print(f"均值 (最佳值): {r21_mean:.6f}")
    print(f"标准差: {r21_std:.6f}")
    print(f"数据点数量: {len(R21_for_fit)}")

# 对R42进行高斯拟合
if len(R42_for_fit) > 2:
    r42_mean, r42_std = norm.fit(R42_for_fit)
    r42_pdf = norm.pdf(R42_for_fit, r42_mean, r42_std)
    
    print(f"\nR42高斯拟合结果:")
    print(f"均值 (最佳值): {r42_mean:.6f}")
    print(f"标准差: {r42_std:.6f}")
    print(f"数据点数量: {len(R42_for_fit)}")

# 可视化高斯拟合结果
plt.figure(figsize=(15, 12))

# R21高斯拟合图
if len(R21_for_fit) > 2:
    plt.subplot(2, 2, 1)
    # 直方图
    n, bins, patches = plt.hist(R21_for_fit, bins='auto', density=True, alpha=0.6, color='g', label='R21 Histogram')
    # 拟合的高斯曲线
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, r21_mean, r21_std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit: $\mu={r21_mean:.4f}$, $\sigma={r21_std:.4f}$')
    plt.title('R21 Gaussian Distribution Fit')
    plt.xlabel('R21 Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.4)

# R42高斯拟合图
if len(R42_for_fit) > 2:
    plt.subplot(2, 2, 2)
    n, bins, patches = plt.hist(R42_for_fit, bins='auto', density=True, alpha=0.6, color='b', label='R42 Histogram')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, r42_mean, r42_std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit: $\mu={r42_mean:.4f}$, $\sigma={r42_std:.4f}$')
    plt.title('R42 Gaussian Distribution Fit')
    plt.xlabel('R42 Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.4)

# R21与Weight关系图（带高斯拟合点）
if len(R21_for_fit) > 2:
    plt.subplot(2, 2, 3)
    plt.plot(range_c3, R21, 'o-', label='R21 raw data')
    plt.plot(range_c3[outlier_indices], R21[outlier_indices], 'ro', label='Outliers', alpha=0.6)
    plt.axhline(y=r21_mean, color='k', linestyle='--', linewidth=2, label=f'Gaussian Mean: {r21_mean:.4f}')
    plt.fill_between(range_c3, r21_mean - r21_std, r21_mean + r21_std, color='gray', alpha=0.2, label=f'1σ Range')
    plt.title('R21 vs Weight with Gaussian Fit')
    plt.xlabel('Weight')
    plt.ylabel('R21')
    plt.legend()
    plt.grid(True, alpha=0.4)

# R42与Weight关系图（带高斯拟合点）
if len(R42_for_fit) > 2:
    plt.subplot(2, 2, 4)
    plt.plot(range_c3, R42, 'o-', label='R42 raw data')
    plt.plot(range_c3[outlier_indices], R42[outlier_indices], 'ro', label='Outliers', alpha=0.6)
    plt.axhline(y=r42_mean, color='k', linestyle='--', linewidth=2, label=f'Gaussian Mean: {r42_mean:.4f}')
    plt.fill_between(range_c3, r42_mean - r42_std, r42_mean + r42_std, color='gray', alpha=0.2, label=f'1σ Range')
    plt.title('R42 vs Weight with Gaussian Fit')
    plt.xlabel('Weight')
    plt.ylabel('R42')
    plt.legend()
    plt.grid(True, alpha=0.4)

plt.tight_layout()
# plt.savefig('./ScanOnlyC3/ScanOnlyC3_gaussian_fit.png', dpi=300, bbox_inches='tight')

# # 基于高斯拟合的最佳Weight值确定
# print("\n=== 基于高斯拟合的最佳Weight值确定 ===")

# # 找到最接近高斯均值的R21和R42对应的Weight值
# if len(R21_for_fit) > 2 and len(R42_for_fit) > 2:
#     # 找到最接近R21均值的点
#     closest_r21_idx = np.argmin(np.abs(R21_for_fit - r21_mean))
#     best_weight_r21 = weights_for_fit[closest_r21_idx]
    
#     # 找到最接近R42均值的点
#     closest_r42_idx = np.argmin(np.abs(R42_for_fit - r42_mean))
#     best_weight_r42 = weights_for_fit[closest_r42_idx]
    
#     # 综合考虑R21和R42的最佳Weight
#     combined_best_weight = (best_weight_r21 + best_weight_r42) / 2
    
#     print(f"\n基于R21高斯均值的最佳Weight: {best_weight_r21:.6f}")
#     print(f"基于R42高斯均值的最佳Weight: {best_weight_r42:.6f}")
#     print(f"综合最佳Weight: {combined_best_weight:.6f}")
    
#     # 绘制综合最佳点
#     plt.figure(figsize=(12, 6))
    
#     plt.plot(range_c3, R21, 'o-', label='R21', color='g')
#     plt.plot(range_c3, R42, 'o-', label='R42', color='b')
#     plt.plot(range_c3[outlier_indices], R21[outlier_indices], 'ro', alpha=0.6)
#     plt.plot(range_c3[outlier_indices], R42[outlier_indices], 'ro', alpha=0.6)
    
#     plt.axhline(y=r21_mean, color='g', linestyle='--', label=f'R21 Mean: {r21_mean:.4f}')
#     plt.axhline(y=r42_mean, color='b', linestyle='--', label=f'R42 Mean: {r42_mean:.4f}')
    
#     plt.axvline(x=best_weight_r21, color='g', linestyle=':', label=f'Best Weight (R21): {best_weight_r21:.4f}')
#     plt.axvline(x=best_weight_r42, color='b', linestyle=':', label=f'Best Weight (R42): {best_weight_r42:.4f}')
#     plt.axvline(x=combined_best_weight, color='k', linestyle='-', linewidth=2, label=f'Combined Best Weight: {combined_best_weight:.4f}')
    
#     plt.title('R21 & R42 vs Weight with Gaussian-Based Best Points')
#     plt.xlabel('Weight')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True, alpha=0.4)
    
#     plt.savefig('ScanOnlyC3_combined_best_points.png', dpi=300, bbox_inches='tight')

# ... existing code ...

# 打印详细的R42和R21值
print("\n=== R42和R21详细值（标记异常值）===")
print(f"Weight\tR42\t\tR21\t\t是否异常")
for i, weight in enumerate(range_c3):
    is_outlier = "true" if i in outlier_indices else "false"
    print(f"{weight:.4f}\t{R42[i]:.6f}\t{R21[i]:.6f}\t{is_outlier}")

# 新增：标记1σ之外点和标记1σ内R21对应的R42
print("\n=== 1σ范围标记分析 ===")

# 检查是否进行了高斯拟合
if 'r21_mean' in locals() and 'r21_std' in locals() and 'r42_mean' in locals() and 'r42_std' in locals():
    # 标记1σ之外的点
    r21_outside_1sigma = np.abs(R21 - r21_mean) > r21_std
    r42_outside_1sigma = np.abs(R42 - r42_mean) > r42_std
    
    # 标记1σ内的R21对应的R42
    r21_inside_1sigma = np.abs(R21 - r21_mean) <= r21_std
    
    print(f"R21 1σ范围: [{r21_mean - r21_std:.6f}, {r21_mean + r21_std:.6f}]")
    print(f"R42 1σ范围: [{r42_mean - r42_std:.6f}, {r42_mean + r42_std:.6f}]")
    
    print(f"\nR21在1σ之外的点数: {np.sum(r21_outside_1sigma)}")
    print(f"R42在1σ之外的点数: {np.sum(r42_outside_1sigma)}")
    print(f"R21在1σ之内的点数: {np.sum(r21_inside_1sigma)}")
    
    # 打印详细标记信息
    print(f"\n=== 详细标记结果 ===")
    print(f"Weight\tR42\t\tR21\t\tR21是否在1σ内\tR42是否在1σ内")
    for i, weight in enumerate(range_c3):
        r21_in_1sigma = "true" if r21_inside_1sigma[i] else "false"
        r42_in_1sigma = "true" if not r42_outside_1sigma[i] else "false"
        print(f"{weight:.4f}\t{R42[i]:.6f}\t{R21[i]:.6f}\t{r21_in_1sigma}\t\t{r42_in_1sigma}")
    
    # 标记1σ内R21对应的R42值
    print(f"\n=== 1σ内R21对应的R42值 ===")
    print(f"Weight\tR21\t\tR42")
    for i, weight in enumerate(range_c3):
        if r21_inside_1sigma[i]:
            print(f"{weight:.4f}\t{R21[i]:.6f}\t{R42[i]:.6f}")
    
    # 绘制1σ标记图
    plt.figure(figsize=(15, 10))
    
    # 子图1: R21和R42的1σ标记
    plt.subplot(2, 2, 1)
    plt.plot(range_c3, R21, 'o-', color='green', label='R21', alpha=0.7)
    plt.plot(range_c3, R42, 'o-', color='blue', label='R42', alpha=0.7)
    
    # 标记1σ之外的点
    plt.plot(range_c3[r21_outside_1sigma], R21[r21_outside_1sigma], 'ro', markersize=8, label='out R21 1σ')
    plt.plot(range_c3[r42_outside_1sigma], R42[r42_outside_1sigma], 'rs', markersize=8, label='out R42 1σ')
    
    # 标记1σ内的R21对应的R42
    plt.plot(range_c3[r21_inside_1sigma], R42[r21_inside_1sigma], 'g*', markersize=10, label='R42 in R21 1σ')
    
    plt.axhline(y=r21_mean, color='green', linestyle='--', label=f'R21 Mean: {r21_mean:.4f}')
    plt.axhline(y=r42_mean, color='blue', linestyle='--', label=f'R42 Mean: {r42_mean:.4f}')
    plt.fill_between(range_c3, r21_mean - r21_std, r21_mean + r21_std, color='green', alpha=0.1, label='R21 1σ')
    plt.fill_between(range_c3, r42_mean - r42_std, r42_mean + r42_std, color='blue', alpha=0.1, label='R42 1σ')
    
    plt.title('R21 & R42 with 1σ Range Marking')
    plt.xlabel('Weight')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    # 子图2: 1σ内R21对应的R42分布
    plt.subplot(2, 2, 2)
    r42_in_r21_1sigma = R42[r21_inside_1sigma]
    weights_in_r21_1sigma = range_c3[r21_inside_1sigma]
    
    if len(r42_in_r21_1sigma) > 0:
        plt.plot(weights_in_r21_1sigma, r42_in_r21_1sigma, 'g*', markersize=10, label='R42 in R21 1σ')
        plt.axhline(y=r42_mean, color='blue', linestyle='--', label=f'R42 Mean: {r42_mean:.4f}')
        plt.axhline(y=np.mean(r42_in_r21_1sigma), color='red', linestyle='--', 
                   label=f'R42 in R21 1σ Mean: {np.mean(r42_in_r21_1sigma):.4f}')
        plt.title('R42 Values Corresponding to R21 within 1σ')
        plt.xlabel('Weight')
        plt.ylabel('R42')
        plt.legend()
        plt.grid(True, alpha=0.4)
    
    # 子图3: R21的1σ分布
    plt.subplot(2, 2, 3)
    plt.plot(range_c3, R21, 'o-', color='green', label='R21', alpha=0.7)
    plt.plot(range_c3[r21_outside_1sigma], R21[r21_outside_1sigma], 'ro', markersize=8, label='out R21 1σ')
    plt.plot(range_c3[r21_inside_1sigma], R21[r21_inside_1sigma], 'go', markersize=6, label='R21 in 1σ')
    plt.axhline(y=r21_mean, color='k', linestyle='--', label=f'R21 Mean: {r21_mean:.4f}')
    plt.fill_between(range_c3, r21_mean - r21_std, r21_mean + r21_std, color='gray', alpha=0.2, label='R21 1σ')
    plt.title('R21 Distribution with 1σ Marking')
    plt.xlabel('Weight')
    plt.ylabel('R21')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    # 子图4: R42的1σ分布
    plt.subplot(2, 2, 4)
    plt.plot(range_c3, R42, 'o-', color='blue', label='R42', alpha=0.7)
    plt.plot(range_c3[r42_outside_1sigma], R42[r42_outside_1sigma], 'ro', markersize=8, label='out R42 1σ')
    plt.plot(range_c3[~r42_outside_1sigma], R42[~r42_outside_1sigma], 'bo', markersize=6, label='R42 in 1σ')
    plt.axhline(y=r42_mean, color='k', linestyle='--', label=f'R42 Mean: {r42_mean:.4f}')
    plt.fill_between(range_c3, r42_mean - r42_std, r42_mean + r42_std, color='gray', alpha=0.2, label='R42 1σ')
    plt.title('R42 Distribution with 1σ Marking')
    plt.xlabel('Weight')
    plt.ylabel('R42')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('./ScanOnlyC3/ScanOnlyC3_1sigma_marking.png', dpi=300, bbox_inches='tight')
    
    # 统计信息
    print(f"\n=== 1σ范围统计信息 ===")
    print(f"R21在1σ内的点对应的R42统计:")
    if len(r42_in_r21_1sigma) > 0:
        print(f"  数量: {len(r42_in_r21_1sigma)}")
        print(f"  均值: {np.mean(r42_in_r21_1sigma):.6f}")
        print(f"  标准差: {np.std(r42_in_r21_1sigma):.6f}")
        print(f"  最小值: {np.min(r42_in_r21_1sigma):.6f}")
        print(f"  最大值: {np.max(r42_in_r21_1sigma):.6f}")
        print(f"  范围: [{np.min(r42_in_r21_1sigma):.6f}, {np.max(r42_in_r21_1sigma):.6f}]")
    
    print(f"\nR21统计:")
    print(f"  1σ内点数: {np.sum(r21_inside_1sigma)}")
    print(f"  1σ外点数: {np.sum(r21_outside_1sigma)}")
    
    print(f"\nR42统计:")
    print(f"  1σ内点数: {np.sum(~r42_outside_1sigma)}")
    print(f"  1σ外点数: {np.sum(r42_outside_1sigma)}")
    
else:
    print("警告: 未找到高斯拟合结果，无法进行1σ标记分析")

# ... existing code ...

# 新增：标记最接近R21均值的点及其对应的R42
print("\n=== 标记最接近R21均值的点及其对应的R42 ===")

# 检查是否进行了高斯拟合
if 'r21_mean' in locals() and 'r21_std' in locals() and 'r42_mean' in locals() and 'r42_std' in locals():
    # 找到最接近R21均值的点（在所有数据点中寻找）
    closest_r21_idx = np.argmin(np.abs(R21 - r21_mean))
    closest_r21_value = R21[closest_r21_idx]
    closest_r21_weight = range_c3[closest_r21_idx]
    
    # 获取该点对应的R42值
    corresponding_r42_value = R42[closest_r21_idx]
    
    print(f"最接近R21均值的点:")
    print(f"  索引: {closest_r21_idx}")
    print(f"  Weight值: {closest_r21_weight:.6f}")
    print(f"  R21值: {closest_r21_value:.6f}")
    print(f"  对应的R42值: {corresponding_r42_value:.6f}")
    print(f"  R21均值: {r21_mean:.6f}")
    print(f"  与R21均值的绝对差值: {abs(closest_r21_value - r21_mean):.6f}")
    
    # 计算该点与R21均值的相对距离（以标准差为单位）
    distance_in_std = abs(closest_r21_value - r21_mean) / r21_std
    print(f"  与R21均值的距离（标准差单位）: {distance_in_std:.4f}σ")
    
    # 检查该点是否在1σ范围内
    if distance_in_std <= 1.0:
        print(f"  该点在R21的1σ范围内")
    else:
        print(f"  该点在R21的1σ范围外")
    
    # 绘制包含最接近R21均值点的标记图
    plt.figure(figsize=(15, 12))
    
    # 子图1: R21和R42的整体分布，标记最接近R21均值的点
    plt.subplot(2, 2, 1)
    plt.plot(range_c3, R21, 'o-', color='green', label='R21', alpha=0.7)
    plt.plot(range_c3, R42, 'o-', color='blue', label='R42', alpha=0.7)
    
    # 标记最接近R21均值的点（用紫色大星号标记）
    plt.plot(closest_r21_weight, closest_r21_value, 'm*', markersize=15, 
             label=f'Closest to R21 Mean (R21={closest_r21_value:.4f})')
    plt.plot(closest_r21_weight, corresponding_r42_value, 'c*', markersize=15, 
             label=f'Corresponding R42 (R42={corresponding_r42_value:.4f})')
    
    # 标记1σ之外的点
    plt.plot(range_c3[r21_outside_1sigma], R21[r21_outside_1sigma], 'ro', markersize=6, label='out R21 1σ', alpha=0.6)
    plt.plot(range_c3[r42_outside_1sigma], R42[r42_outside_1sigma], 'rs', markersize=6, label='out R42 1σ', alpha=0.6)
    
    plt.axhline(y=r21_mean, color='green', linestyle='--', label=f'R21 Mean: {r21_mean:.4f}')
    plt.axhline(y=r42_mean, color='blue', linestyle='--', label=f'R42 Mean: {r42_mean:.4f}')
    plt.fill_between(range_c3, r21_mean - r21_std, r21_mean + r21_std, color='green', alpha=0.1, label='R21 1σ')
    plt.fill_between(range_c3, r42_mean - r42_std, r42_mean + r42_std, color='blue', alpha=0.1, label='R42 1σ')
    
    plt.title('R21 & R42 with Closest-to-Mean Point Marking')
    plt.xlabel('Weight')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.4)
    
    # 子图2: 最接近R21均值的点及其邻域放大图
    plt.subplot(2, 2, 2)
    
    # 定义邻域范围（±0.5σ）
    neighborhood_std = 0.5
    lower_bound = r21_mean - neighborhood_std * r21_std
    upper_bound = r21_mean + neighborhood_std * r21_std
    
    # 找到邻域内的点
    neighborhood_mask = (R21 >= lower_bound) & (R21 <= upper_bound)
    neighborhood_weights = range_c3[neighborhood_mask]
    neighborhood_r21 = R21[neighborhood_mask]
    neighborhood_r42 = R42[neighborhood_mask]
    
    plt.plot(neighborhood_weights, neighborhood_r21, 'go', markersize=8, label='R21 in neighborhood', alpha=0.7)
    plt.plot(neighborhood_weights, neighborhood_r42, 'bo', markersize=8, label='R42 in neighborhood', alpha=0.7)
    
    # 标记最接近R21均值的点
    plt.plot(closest_r21_weight, closest_r21_value, 'm*', markersize=15, 
             label=f'Closest to R21 Mean')
    plt.plot(closest_r21_weight, corresponding_r42_value, 'c*', markersize=15, 
             label=f'Corresponding R42')
    
    plt.axhline(y=r21_mean, color='green', linestyle='--', label=f'R21 Mean')
    plt.axhline(y=r42_mean, color='blue', linestyle='--', label=f'R42 Mean')
    plt.axvline(x=closest_r21_weight, color='purple', linestyle=':', 
                label=f'Weight at closest point: {closest_r21_weight:.4f}')
    
    plt.title(f'Neighborhood Analysis (±{neighborhood_std}σ around R21 Mean)')
    plt.xlabel('Weight')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    # 子图3: R21值的分布直方图，标记最接近均值的点
    plt.subplot(2, 2, 3)
    n, bins, patches = plt.hist(R21, bins=20, density=True, alpha=0.6, color='green', label='R21 Histogram')
    
    # 高斯拟合曲线
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, r21_mean, r21_std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit')
    
    # 标记最接近R21均值的点
    plt.axvline(x=closest_r21_value, color='magenta', linestyle='-', linewidth=2, 
                label=f'Closest R21: {closest_r21_value:.4f}')
    plt.axvline(x=r21_mean, color='red', linestyle='--', linewidth=2, 
                label=f'R21 Mean: {r21_mean:.4f}')
    
    plt.title('R21 Distribution with Closest Point Marking')
    plt.xlabel('R21 Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    # 子图4: 最接近R21均值的点对应的R42值分析
    plt.subplot(2, 2, 4)
    
    # 计算所有R21值对应的R42值
    plt.scatter(R21, R42, c=range_c3, cmap='viridis', alpha=0.6, label='R21 vs R42')
    
    # 标记最接近R21均值的点
    plt.scatter([closest_r21_value], [corresponding_r42_value], 
                c='magenta', s=100, marker='*', label='Closest to R21 Mean')
    
    # 标记R21均值和R42均值
    plt.axvline(x=r21_mean, color='red', linestyle='--', label=f'R21 Mean')
    plt.axhline(y=r42_mean, color='blue', linestyle='--', label=f'R42 Mean')
    
    plt.xlabel('R21')
    plt.ylabel('R42')
    plt.title('R21 vs R42 Scatter Plot')
    plt.colorbar(label='Weight')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    plt.tight_layout()
    # plt.savefig('./ScanOnlyC3/ScanOnlyC3_closest_to_mean_marking.png', dpi=300, bbox_inches='tight')
    
    # 打印详细的分析结果
    print(f"\n=== 最接近R21均值点的详细分析 ===")
    print(f"该点的统计特征:")
    print(f"  Weight值: {closest_r21_weight:.6f}")
    print(f"  R21值: {closest_r21_value:.6f} (与均值差: {abs(closest_r21_value - r21_mean):.6f})")
    print(f"  对应的R42值: {corresponding_r42_value:.6f} (与均值差: {abs(corresponding_r42_value - r42_mean):.6f})")
    print(f"  该点R21与R21均值的距离: {distance_in_std:.4f}σ")
    
    # 分析该点对应的R42值相对于R42均值的距离
    r42_distance_in_std = abs(corresponding_r42_value - r42_mean) / r42_std
    print(f"  该点R42与R42均值的距离: {r42_distance_in_std:.4f}σ")
    
    if r42_distance_in_std <= 1.0:
        print(f"  该点对应的R42值在R42的1σ范围内")
    else:
        print(f"  该点对应的R42值在R42的1σ范围外")
    
    # 分析该点在整体数据中的位置
    r21_rank = np.sum(R21 <= closest_r21_value) / len(R21) * 100
    r42_rank = np.sum(R42 <= corresponding_r42_value) / len(R42) * 100
    print(f"  该点R21值在所有R21值中的百分位: {r21_rank:.1f}%")
    print(f"  该点R42值在所有R42值中的百分位: {r42_rank:.1f}%")
    
    # 分析邻域内的统计特征
    if len(neighborhood_r21) > 0:
        print(f"\n邻域分析 (±{neighborhood_std}σ范围内):")
        print(f"  邻域内点数: {len(neighborhood_r21)}")
        print(f"  邻域内R21均值: {np.mean(neighborhood_r21):.6f}")
        print(f"  邻域内R21标准差: {np.std(neighborhood_r21):.6f}")
        print(f"  邻域内R42均值: {np.mean(neighborhood_r42):.6f}")
        print(f"  邻域内R42标准差: {np.std(neighborhood_r42):.6f}")
        
        # 计算最接近点与邻域均值的差异
        r21_diff_from_neighborhood = abs(closest_r21_value - np.mean(neighborhood_r21))
        r42_diff_from_neighborhood = abs(corresponding_r42_value - np.mean(neighborhood_r42))
        print(f"  最接近点R21与邻域均值的差异: {r21_diff_from_neighborhood:.6f}")
        print(f"  最接近点R42与邻域均值的差异: {r42_diff_from_neighborhood:.6f}")
    
else:
    print("警告: 未找到高斯拟合结果，无法进行最接近均值点标记分析")

print("\n=== 分析完成 ===")

# ... existing code ...

# 新增：R21与R42分开绘图，添加二维轮廓图和一维分布图
print("\n=== R21与R42分开绘图及二维轮廓图分析 ===")

# 检查是否进行了高斯拟合
if 'r21_mean' in locals() and 'r21_std' in locals() and 'r42_mean' in locals() and 'r42_std' in locals():
    
    # 1. R21和R42分开的分布图
    plt.figure(figsize=(15, 6))
    
    # 子图1: R21分布图
    plt.subplot(1, 2, 1)
    plt.plot(range_c3, R21, 'o-', color='green', label='R21', alpha=0.7)
    
    # 标记1σ之外的点
    plt.plot(range_c3[r21_outside_1sigma], R21[r21_outside_1sigma], 'ro', markersize=8, label='out 1σ')
    
    # 标记1σ内的点
    plt.plot(range_c3[r21_inside_1sigma], R21[r21_inside_1sigma], 'go', markersize=6, label='in 1σ')
    
    plt.axhline(y=r21_mean, color='k', linestyle='--', linewidth=2, label=f'Mean: {r21_mean:.4f}')
    plt.fill_between(range_c3, r21_mean - r21_std, r21_mean + r21_std, color='gray', alpha=0.2, label='1σ Range')
    
    plt.title('R21 Distribution with 1σ Marking')
    plt.xlabel('Weight')
    plt.ylabel('R21')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    # 子图2: R42分布图
    plt.subplot(1, 2, 2)
    plt.plot(range_c3, R42, 'o-', color='blue', label='R42', alpha=0.7)
    
    # 标记1σ之外的点
    plt.plot(range_c3[r42_outside_1sigma], R42[r42_outside_1sigma], 'ro', markersize=8, label='out 1σ')
    
    # 标记1σ内的点
    plt.plot(range_c3[~r42_outside_1sigma], R42[~r42_outside_1sigma], 'bo', markersize=6, label='in 1σ')
    
    plt.axhline(y=r42_mean, color='k', linestyle='--', linewidth=2, label=f'Mean: {r42_mean:.4f}')
    plt.fill_between(range_c3, r42_mean - r42_std, r42_mean + r42_std, color='gray', alpha=0.2, label='1σ Range')
    
    plt.title('R42 Distribution with 1σ Marking')
    plt.xlabel('Weight')
    plt.ylabel('R42')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    plt.tight_layout()
    # plt.savefig('./ScanOnlyC3/ScanOnlyC3_separate_distributions.png', dpi=300, bbox_inches='tight')
    
    # 2. R21 1σ内点的二维轮廓图及一维分布图
    if len(r42_in_r21_1sigma) > 0:
        # 准备数据：R21 1σ内的R21和R42值
        r21_in_1sigma = R21[r21_inside_1sigma]
        weights_in_1sigma = range_c3[r21_inside_1sigma]
        
        # 创建二维轮廓图
        fig = plt.figure(figsize=(12, 10))
        
        # 定义网格布局
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
        
        # 主散点图（二维轮廓图）
        ax_main = fig.add_subplot(grid[1:4, 0:3])
        scatter = ax_main.scatter(r21_in_1sigma, r42_in_r21_1sigma, c=weights_in_1sigma, 
                                 cmap='viridis', alpha=0.7, s=50)
        
        # 核密度估计
        if len(r21_in_1sigma) > 1:
            # 创建网格点
            xmin, xmax = r21_in_1sigma.min(), r21_in_1sigma.max()
            ymin, ymax = r42_in_r21_1sigma.min(), r42_in_r21_1sigma.max()
            x = np.linspace(xmin, xmax, 100)
            y = np.linspace(ymin, ymax, 100)
            X, Y = np.meshgrid(x, y)
            positions = np.vstack([X.ravel(), Y.ravel()])
            
            # 计算核密度估计
            values = np.vstack([r21_in_1sigma, r42_in_r21_1sigma])
            kernel = gaussian_kde(values)
            Z = np.reshape(kernel(positions).T, X.shape)
            
            # 绘制轮廓线
            contour = ax_main.contour(X, Y, Z, levels=5, colors='black', alpha=0.6, linewidths=1)
            ax_main.clabel(contour, inline=True, fontsize=8)
            
            # 找到概率最大点
            max_prob_idx = np.argmax(Z)
            max_prob_x = X.ravel()[max_prob_idx]
            max_prob_y = Y.ravel()[max_prob_idx]
            
            # 标记概率最大点
            ax_main.plot(max_prob_x, max_prob_y, 'r*', markersize=15, 
                        label=f'Max Probability (R21={max_prob_x:.4f}, R42={max_prob_y:.4f})')
        
        # 标记均值点
        ax_main.axvline(x=r21_mean, color='red', linestyle='--', alpha=0.7, label='R21 Mean')
        ax_main.axhline(y=r42_mean, color='blue', linestyle='--', alpha=0.7, label='R42 Mean')
        
        ax_main.set_xlabel('R21 (within 1σ)')
        ax_main.set_ylabel('R42')
        ax_main.set_title('2D Contour Plot: R21 vs R42 (within R21 1σ)')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        # R21的一维分布图（上方）
        ax_top = fig.add_subplot(grid[0, 0:3], sharex=ax_main)
        ax_top.hist(r21_in_1sigma, bins=15, density=True, alpha=0.7, color='green')
        ax_top.axvline(x=r21_mean, color='red', linestyle='--', linewidth=2)
        if len(r21_in_1sigma) > 1:
            ax_top.axvline(x=max_prob_x, color='red', linestyle='-', linewidth=2, alpha=0.7)
        ax_top.set_ylabel('Density')
        ax_top.set_title('R21 Distribution (within 1σ)')
        ax_top.grid(True, alpha=0.3)
        
        # R42的一维分布图（右侧）
        ax_right = fig.add_subplot(grid[1:4, 3], sharey=ax_main)
        ax_right.hist(r42_in_r21_1sigma, bins=15, density=True, alpha=0.7, color='blue', orientation='horizontal')
        ax_right.axhline(y=r42_mean, color='blue', linestyle='--', linewidth=2)
        if len(r21_in_1sigma) > 1:
            ax_right.axhline(y=max_prob_y, color='blue', linestyle='-', linewidth=2, alpha=0.7)
        ax_right.set_xlabel('Density')
        ax_right.set_title('R42 Distribution')
        ax_right.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Weight')
        
        plt.tight_layout()
        plt.savefig('./ScanOnlyC3/ScanOnlyC3_2d_contour_with_marginals.png', dpi=300, bbox_inches='tight')
        
        # 打印二维轮廓图信息
        print(f"\n=== R21 1σ内点的二维轮廓图分析 ===")
        print(f"数据点数量: {len(r21_in_1sigma)}")
        if len(r21_in_1sigma) > 1:
            print(f"概率最大点: R21={max_prob_x:.6f}, R42={max_prob_y:.6f}")
        print(f"R21均值: {r21_mean:.6f}")
        print(f"R42均值: {r42_mean:.6f}")
    
    # 3. 标记R21均值对应的R42均值（而非最接近R21均值的点）
    print(f"\n=== R21均值对应的R42均值分析 ===")
    
    # 定义邻域范围（±0.1σ）
    neighborhood_std = 1
    lower_bound = r21_mean - neighborhood_std * r21_std
    upper_bound = r21_mean + neighborhood_std * r21_std
    
    # 找到邻域内的点
    neighborhood_mask = (R21 >= lower_bound) & (R21 <= upper_bound)
    neighborhood_weights = range_c3[neighborhood_mask]
    neighborhood_r21 = R21[neighborhood_mask]
    neighborhood_r42 = R42[neighborhood_mask]
    
    if len(neighborhood_r21) > 0:
        # 计算邻域内的R42均值
        r42_mean_near_r21_mean = np.mean(neighborhood_r42)
        r21_1sigma_mask = (R21 >= r21_mean - r21_std) & (R21 <= r21_mean + r21_std)
        r42_1sigma_mask = (R42 >= r42_mean - r42_std) & (R42 <= r42_mean + r42_std)
        combined_mask = r21_1sigma_mask & r42_1sigma_mask
        
        if np.sum(combined_mask) > 0:
            r42_mean_in_both_1sigma = np.mean(R42[combined_mask])
            print(f"R21 1σ内且R42 1σ内的点数: {np.sum(combined_mask)}")
            print(f"R21 1σ内且R42 1σ内的R42均值: {r42_mean_in_both_1sigma:.6f}")
        else:
            r42_mean_in_both_1sigma = None
            print("警告: 在R21 1σ内且R42 1σ内没有找到数据点")
        
        print(f"邻域范围: R21均值 ±{neighborhood_std}σ")
        print(f"邻域内点数: {len(neighborhood_r21)}")
        print(f"邻域内R21均值: {np.mean(neighborhood_r21):.6f}")
        print(f"邻域内R42均值: {r42_mean_near_r21_mean:.6f}")
        print(f"邻域内R21标准差: {np.std(neighborhood_r21):.6f}")
        print(f"邻域内R42标准差: {np.std(neighborhood_r42):.6f}")
        
        # 可视化邻域分析
        plt.figure(figsize=(12, 5))
        
        # 子图1: R21邻域分析
        plt.subplot(1, 2, 1)
        plt.plot(range_c3, R21, 'o-', color='green', label='R21', alpha=0.3)
        plt.plot(neighborhood_weights, neighborhood_r21, 'go', markersize=8, label=f'R21 in ±{neighborhood_std}σ')
        plt.axhline(y=r21_mean, color='red', linestyle='--', linewidth=2, label=f'R21 Mean')
        plt.axvline(x=np.mean(neighborhood_weights), color='purple', linestyle=':', 
                   label=f'Mean Weight: {np.mean(neighborhood_weights):.4f}')
        plt.fill_between(range_c3, lower_bound, upper_bound, color='green', alpha=0.1, 
                        label=f'R21 Mean ±{neighborhood_std}σ')
        plt.title(f'R21 Neighborhood Analysis (±{neighborhood_std}σ)')
        plt.xlabel('Weight')
        plt.ylabel('R21')
        plt.legend()
        plt.grid(True, alpha=0.4)
        
        # 子图2: R42对应分析
        plt.subplot(1, 2, 2)
        plt.plot(range_c3, R42, 'o-', color='blue', label='R42', alpha=0.3)
        plt.plot(neighborhood_weights, neighborhood_r42, 'bo', markersize=8, label=f'R42 in R21 ±{neighborhood_std}σ')
        plt.fill_between(range_c3, r42_mean - r42_std, r42_mean + r42_std, color='blue', alpha=0.1, label='R42 1σ Range')

        plt.axhline(y=r42_mean, color='blue', linestyle='--', linewidth=2, label=f'R42 Mean')
        plt.axhline(y=r42_mean_near_r21_mean, color='red', linestyle='-', linewidth=2, 
                   label=f'R42 Mean near R21 Mean: {r42_mean_near_r21_mean:.4f}')
        plt.axvline(x=np.mean(neighborhood_weights), color='purple', linestyle=':', 
                   label=f'Mean Weight: {np.mean(neighborhood_weights):.4f}')
        if r42_mean_in_both_1sigma is not None:
            plt.axhline(y=r42_mean_in_both_1sigma, color='orange', linestyle='-.', linewidth=2, 
                       label=f'R42 Mean (R21&R42 in 1σ): {r42_mean_in_both_1sigma:.4f}')
        
        plt.title(f'R42 Corresponding to R21 ±{neighborhood_std}σ')
        plt.xlabel('Weight')
        plt.ylabel('R42')
        plt.legend()
        plt.grid(True, alpha=0.4)
        
        plt.tight_layout()
        plt.savefig('./ScanOnlyC3/ScanOnlyC3_r42_mean_near_r21_mean.png', dpi=300, bbox_inches='tight')
        
    else:
        print("警告: 在R21均值附近没有找到足够的数据点")
    
else:
    print("警告: 未找到高斯拟合结果，无法进行分开绘图和二维轮廓图分析")

print("\n=== 新增功能分析完成 ===")

# ... existing code ...