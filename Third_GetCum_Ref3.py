import numpy as np
import matplotlib.pyplot as plt

def calculate_moments(y, x):
    y = y/np.sum(y)
    m1 = np.sum(y*x)
    m2 = np.sum(y*x**2)
    m3 = np.sum(y*x**3)
    m4 = np.sum(y*x**4)
    
    c1 = m1
    c2 = m2 - m1**2
    c3 = m3 - 3*m1*m2 + 2*m1**3
    c4 = m4 - 3*m2**2 - 4*m1*m3 + 12*m1**2*m2 - 6*m1**4
    
    return c1, c2, c3, c4

def calculate_CBWC_for_range(start_bin, end_bin):
    """计算指定区间内的CBWC矩"""
    # print(start_bin, end_bin)
    c1_values, c2_values, c3_values, c4_values = [], [], [], []
    entries = []
    x_array = []
    temp_c1, temp_c2, temp_c3, temp_c4 = [], [], [], []
    for i in range(start_bin, end_bin):
        try:
            data = np.loadtxt(f"EachBinDistributionFromRef3/Proton_Bin{i}.txt")
            x, y = data[:, 0], data[:, 1]
            entry = np.sum(y)
            c1, c2, c3, c4 = calculate_moments(y, x)
            x_array.append(x)
            c1_values.append(c1)
            c2_values.append(c2)
            c3_values.append(c3)
            c4_values.append(c4)
            temp_c1.append(c1)
            temp_c2.append(c2)
            temp_c3.append(c3)
            temp_c4.append(c4)
            entries.append(entry)
        except:
            continue

    entries = np.array(entries)
    Entry_total = np.sum(entries)
    
    c1_CBWC = np.sum(np.array(c1_values) * entries) / Entry_total
    c2_CBWC = np.sum(np.array(c2_values) * entries) / Entry_total
    c3_CBWC = np.sum(np.array(c3_values) * entries) / Entry_total
    c4_CBWC = np.sum(np.array(c4_values) * entries) / Entry_total
    
    return c1_CBWC, c2_CBWC, c3_CBWC, c4_CBWC
x_Cent = []
y_Cent = []
Err_Cent = []
# 主程序
from CentBin import bin_ref3
bin_ref3[:-1] = bin_ref3[:-1] + 1
if __name__ == "__main__":
    bin_labels = []
    for i in range(len(bin_ref3)-1):
        bin_labels.append(f"{bin_ref3[i]}-{bin_ref3[i+1]}")
    
    # 存储每个区间的结果
    total_c1, total_c2, total_c3, total_c4 = [], [], [], []
    cbwc_c1, cbwc_c2, cbwc_c3, cbwc_c4 = [], [], [], []
    
    # 对每个中心度区间计算
    for i in range(len(bin_ref3)-1):
        # 计算区间总分布的矩
        y_total = np.zeros_like(np.loadtxt(f"EachBinDistributionFromRef3/Proton_Bin{bin_ref3[i+1]}.txt")[:, 1])
        y_total_error = np.zeros_like(np.loadtxt(f"EachBinDistributionFromRef3/Proton_Bin{bin_ref3[i+1]}.txt")[:, 2])
        x = None
        
        for bin_num in range(bin_ref3[i+1], bin_ref3[i]):
            try:
                data = np.loadtxt(f"EachBinDistributionFromRef3/Proton_Bin{bin_num}.txt")
                if x is None:
                    x = data[:, 0]
                y_total += data[:, 1]
                y_total_error += data[:, 2]**2
            except:
                continue
        
        if np.sum(y_total) <= 0:
            print(f"警告：区间 {bin_ref3[i]}-{bin_ref3[i+1]} 的数据总和为0或负数")
            continue
        
        # 计算总分布的矩
        c1, c2, c3, c4 = calculate_moments(y_total, x)
        x_Cent.append(x)
        y_Cent.append(y_total)
        y_total_error = np.sqrt(y_total_error)
        Err_Cent.append(y_total_error)
        total_c1.append(c1)
        total_c2.append(c2)
        total_c3.append(c3)
        total_c4.append(c4)

        # with open(f"./RemoveLowEvents_ProDist/Proton_Total_Cent{bin_edge[i]}_{bin_edge[i+1]}.txt", "w") as f:
        #     for i in range(len(x)):
        #         f.write(f"{x[i]} {y_total[i]}\n")
        # 计算该区间的CBWC矩
        c1_cbwc, c2_cbwc, c3_cbwc, c4_cbwc = calculate_CBWC_for_range(bin_ref3[i+1], bin_ref3[i])
        cbwc_c1.append(c1_cbwc)
        cbwc_c2.append(c2_cbwc)
        cbwc_c3.append(c3_cbwc)
        cbwc_c4.append(c4_cbwc)

    # 计算比值
    ratio_c1 = np.array(total_c1) / np.array(cbwc_c1)
    ratio_c2 = np.array(total_c2) / np.array(cbwc_c2)
    ratio_c3 = np.array(total_c3) / np.array(cbwc_c3)
    ratio_c4 = np.array(total_c4) / np.array(cbwc_c4)

    n = len(total_c1)
    Trans_total_c1 = [0]*n
    Trans_total_c2 = [0]*n
    Trans_total_c3 = [0]*n
    Trans_total_c4 = [0]*n
    Trans_cbwc_c1 = [0]*n
    Trans_cbwc_c2 = [0]*n
    Trans_cbwc_c3 = [0]*n
    Trans_cbwc_c4 = [0]*n
    Trans_ratio_c1 = [0]*n
    Trans_ratio_c2 = [0]*n
    Trans_ratio_c3 = [0]*n
    Trans_ratio_c4 = [0]*n
    Trans_bin_labels = [0]*n

    for i in range(n):
        Trans_total_c1[i] = total_c1[n-1-i]
        Trans_total_c2[i] = total_c2[n-1-i]
        Trans_total_c3[i] = total_c3[n-1-i]
        Trans_total_c4[i] = total_c4[n-1-i]
        Trans_cbwc_c1[i] = cbwc_c1[n-1-i]
        Trans_cbwc_c2[i] = cbwc_c2[n-1-i]
        Trans_cbwc_c3[i] = cbwc_c3[n-1-i]
        Trans_cbwc_c4[i] = cbwc_c4[n-1-i]
        Trans_ratio_c1[i] = ratio_c1[n-1-i]
        Trans_ratio_c2[i] = ratio_c2[n-1-i]
        Trans_ratio_c3[i] = ratio_c3[n-1-i]
        Trans_ratio_c4[i] = ratio_c4[n-1-i]
        Trans_bin_labels[i] = bin_labels[n-1-i]
    # 绘图
    plt.figure(figsize=(18, 8))
    
    plt.subplot(221)
    plt.plot(Trans_bin_labels, Trans_ratio_c1, 'bo-', markersize=8)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.ylabel('C1 Total / C1 CBWC')
    plt.title('First Order Cumulant Ratio')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.subplot(222)
    plt.plot(Trans_bin_labels, Trans_ratio_c2, 'ro-', markersize=8)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.ylabel('C2 Total / C2 CBWC')
    plt.title('Second Order Cumulant Ratio')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.subplot(223)
    plt.plot(Trans_bin_labels, Trans_ratio_c3, 'go-', markersize=8)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.ylabel('C3 Total / C3 CBWC')
    plt.title('Third Order Cumulant Ratio')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.subplot(224)
    plt.plot(Trans_bin_labels, Trans_ratio_c4, 'mo-', markersize=8)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.ylabel('C4 Total / C4 CBWC')
    plt.title('Fourth Order Cumulant Ratio')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('./PaperFIG/Third_Cumulant_Ratio_CBWC_Total_FromRef3.pdf')
    # plt.show()

    plt.figure(figsize=(18, 8))
    plt.subplot(221)
    plt.plot(Trans_bin_labels, Trans_total_c1, 'bo--', markersize=17, label='w/o CBWC', fillstyle='none', alpha=1, linewidth=2)
    plt.plot(Trans_bin_labels, Trans_cbwc_c1, 'ro--', markersize=14, label='w/ CBWC', alpha=0.7, linewidth=2)
    plt.title('First Order Cumulant', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=14, framealpha=0.8, loc='best')
    
    plt.subplot(222)
    plt.plot(Trans_bin_labels, Trans_total_c2, 'bo--', markersize=17, label='w/o CBWC', fillstyle='none', alpha=1, linewidth=2)
    plt.plot(Trans_bin_labels, Trans_cbwc_c2, 'ro--', markersize=14, label='w/ CBWC', alpha=0.7, linewidth=2)
    plt.title('Second Order Cumulant', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=14, framealpha=0.8, loc='best')
    
    plt.subplot(223)
    plt.plot(Trans_bin_labels, Trans_total_c3, 'bo--', markersize=17, label='w/o CBWC', fillstyle='none', alpha=1, linewidth=2)
    plt.plot(Trans_bin_labels, Trans_cbwc_c3, 'ro--', markersize=14, label='w/ CBWC', alpha=0.7, linewidth=2)
    plt.title('Third Order Cumulant', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=14, framealpha=0.8, loc='best')
    
    plt.subplot(224)
    plt.plot(Trans_bin_labels, Trans_total_c4, 'bo--', markersize=17, label='w/o CBWC', fillstyle='none', alpha=1, linewidth=2)
    plt.plot(Trans_bin_labels, Trans_cbwc_c4, 'ro--', markersize=14, label='w/ CBWC', alpha=0.7, linewidth=2)
    plt.title('Fourth Order Cumulant', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=14, framealpha=0.8, loc='best')
    plt.ylim(-10,50)
    plt.tight_layout()
    plt.savefig('./PaperFIG/Third_Cumulant_Comparison_CBWC_Total_FromRef3.pdf')
    # plt.show()
    
    # 打印结果
    print("\nComparison for each centrality bin:")
    print("CBWC")
    for i in range(len(bin_ref3)-1):
        print(f"C1: Total = {Trans_total_c1[i]:.4f}, CBWC = {Trans_cbwc_c1[i]:.4f}, Ratio = {Trans_ratio_c1[i]:.4f}")
        print(f"C2: Total = {Trans_total_c2[i]:.4f}, CBWC = {Trans_cbwc_c2[i]:.4f}, Ratio = {Trans_ratio_c2[i]:.4f}")
        print(f"C3: Total = {Trans_total_c3[i]:.4f}, CBWC = {Trans_cbwc_c3[i]:.4f}, Ratio = {Trans_ratio_c3[i]:.4f}")
        print(f"C4: Total = {Trans_total_c4[i]:.4f}, CBWC = {Trans_cbwc_c4[i]:.4f}, Ratio = {Trans_ratio_c4[i]:.4f}")

    for i in range(len(bin_ref3)-1):
        with open(f"RemoveLowEvents_ProDist_FromRef3/Proton_Total_Cent{bin_ref3[i]}_{bin_ref3[i+1]}.txt", "w") as f:
            for j in range(len(x_Cent[i])):
                f.write(f"{x_Cent[i][j]} {y_Cent[i][j]} {Err_Cent[i][j]}\n")

import os

def save_results_to_txt(bin_labels, total_c1, total_c2, total_c3, total_c4, 
                       cbwc_c1, cbwc_c2, cbwc_c3, cbwc_c4):
    # 创建结果目录（如果不存在）
    os.makedirs('CBWC_Results_FromRef3', exist_ok=True)
    
    # 保存总结果
    with open('CBWC_Results_FromRef3/all_results.txt', 'w') as f:
        f.write("Centrality\tTotal_C1_Ref3\tTotal_C2_Ref3\tTotal_C3_Ref3\tTotal_C4_Ref3\tCBWC_C1_Ref3\tCBWC_C2_Ref3\tCBWC_C3_Ref3\tCBWC_C4_Ref3\n")
        for i in range(len(bin_labels)):
            f.write(f"{bin_labels[i]}\t{total_c1[i]:.6f}\t{total_c2[i]:.6f}\t{total_c3[i]:.6f}\t{total_c4[i]:.6f}\t")
            f.write(f"{cbwc_c1[i]:.6f}\t{cbwc_c2[i]:.6f}\t{cbwc_c3[i]:.6f}\t{cbwc_c4[i]:.6f}\n")
    
    # 分别保存每个矩的结果
    moments = {
        'C1': (total_c1, cbwc_c1),
        'C2': (total_c2, cbwc_c2),
        'C3': (total_c3, cbwc_c3),
        'C4': (total_c4, cbwc_c4)
    }
    
    for moment, (total, cbwc) in moments.items():
        with open(f'CBWC_Results_FromRef3/{moment}_results.txt', 'w') as f:
            f.write("Centrality\tTotal\tCBWC\tRatio\n")
            for i in range(len(bin_labels)):
                ratio = total[i] / cbwc[i] if cbwc[i] != 0 else float('inf')
                f.write(f"{bin_labels[i]}\t{total[i]:.6f}\t{cbwc[i]:.6f}\t{ratio:.6f}\n")

# 在计算完CBWC后调用保存函数
save_results_to_txt(
    Trans_bin_labels,
    Trans_total_c1,
    Trans_total_c2,
    Trans_total_c3,
    Trans_total_c4,
    Trans_cbwc_c1,
    Trans_cbwc_c2,
    Trans_cbwc_c3,
    Trans_cbwc_c4
)

print("\n结果已保存到 CBWC_Results_FromRef3 目录")