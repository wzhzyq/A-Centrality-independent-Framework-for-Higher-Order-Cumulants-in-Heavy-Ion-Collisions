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
    Q1 = np.percentile(data, 20)
    Q3 = np.percentile(data, 80)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1 * IQR
    upper_bound = Q3 + 1 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers

# 定义C3值的范围
scan_weight_dir = './ScanMax'
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
# exit()
# 创建图形
fig, axes = plt.subplots(4, 4, figsize=(24, 10))
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
    path = f'./ScanMax/fit_results_{c3:.4f}.txt'
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
axes[12].plot(x, weight_last / weight_last_1, marker='o', linestyle='-', color=colors, 
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

axes[12].set_title('Weight (last) / Weight (last-1) vs Weight', fontsize=14)
axes[12].set_ylabel('Weight (last) / Weight (last-1)', fontsize=12)
axes[12].set_xlabel('Weight', fontsize=12)
axes[12].grid(True, alpha=0.4)
axes[12].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 调整布局，为图例留出空间
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# 保存和显示图形
plt.savefig('./ScanMax/ScanMax_tmp.png', dpi=300, bbox_inches='tight')
# plt.show()
R21 = C2_last_raw/C1_last_raw
R42 = C4_last_raw/C2_last_raw
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
        plt.savefig('./ScanMax/ScanMax_stability_analysis_filtered.png', dpi=300, bbox_inches='tight')

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
plt.savefig('./ScanMax/ScanMax_gaussian_fit.png', dpi=300, bbox_inches='tight')

# 基于高斯拟合的最佳Weight值确定
print("\n=== 基于高斯拟合的最佳Weight值确定 ===")

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
    
#     plt.savefig('ScanMax_combined_best_points.png', dpi=300, bbox_inches='tight')

# 打印详细的R42和R21值
print("\n=== R42和R21详细值（标记异常值）===")
print(f"Weight\tR42\t\tR21\t\t是否异常")
for i, weight in enumerate(range_c3):
    is_outlier = "是" if i in outlier_indices else "否"
    print(f"{weight:.4f}\t{R42[i]:.6f}\t{R21[i]:.6f}\t{is_outlier}")