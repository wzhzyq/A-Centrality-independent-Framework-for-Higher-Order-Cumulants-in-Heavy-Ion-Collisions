import numpy as np
import matplotlib.pyplot as plt
import ROOT as rt
from Run import Get_Raw_Cumulants_From_CorrectedCumulants, NCENT
from FinalResults.fit_results_mcmc import *
from FinalResults.weights_mcmc import weights
from Efficiency import EFF_default as EFFICIENCY_ARRAY
EFF = EFFICIENCY_ARRAY[:NCENT][::-1]
def calkappa(C1,C2,C3,C4):
    k1 = C1
    k2 = C2 - C1
    k3 = C3 - 3*C2 + 2*C1
    k4 = C4 - 6*C3 + 11*C2 - 6*C1
    return k1,k2,k3,k4

def edgeworth_pdf(x, mu, c2, c3, c4):
    k_2 = c2
    k_3 = c3
    k_4 = c4
    skew = k_3 / (k_2**(3/2))
    kurt = k_4 / (k_2**2)
    z = (x - mu) / (k_2**(1/2))
    correction = (1 + skew / 6 * (z**3 - 3 * z) +
                    kurt / 24 * (z**4 - 6 * z**2 + 3) +
                    skew**2 / 72 * (z**6 - 15 * z**4 + 45 * z**2 - 15))
    gaussian = np.exp(-0.5 * z**2) / (2.50663 * k_2**(1/2))
    return gaussian * correction

def calculate_moments(y, x):
    # 确保 y 是有效的概率分布
    y_sum = np.sum(y)
    if y_sum <= 1e-9:
        return np.nan, np.nan, np.nan, np.nan # 或者根据需要处理
    y = y / y_sum

    # 计算矩
    m1 = np.sum(y * x)
    m2 = np.sum(y * x**2)
    m3 = np.sum(y * x**3)
    m4 = np.sum(y * x**4)

    # 计算累积量
    c1 = m1
    c2 = m2 - m1**2
    c3 = m3 - 3*m1*m2 + 2*m1**3
    c4 = m4 - 3*m2**2 - 4*m1*m3 + 12*m1**2*m2 - 6*m1**4

    k1 = c1
    k2 = c2 - c1
    k3 = c3 - 3*c2 + 2*c1
    k4 = c4 - 6*c3 + 11*c2 - 6*c1

    R21 = c2 / c1
    R31 = c3 / c2
    R42 = c4 / c2
    k21 = k2 / k1
    k31 = k3 / k1
    k41 = k4 / k1
    # 检查是否有 NaN 或 Inf 出现，这可能由非常小的 nevents 导致
    if not all(np.isfinite([c1, c2, c3, c4])):
         print(f"Warning: Non-finite cumulant encountered during calculation. m1={m1}, m2={m2}, m3={m3}, m4={m4}")
         return np.nan, np.nan, np.nan, np.nan

    return c1, c2, c3, c4, R21, R31, R42, k1, k2, k3, k4, k21, k31, k41

## 重建各个中心度下的分布
data = np.loadtxt('ProtonPDF.txt')
x = data[:, 0]
########################################################################
events = 29067975

C1, C2, C3, C4 = Get_Raw_Cumulants_From_CorrectedCumulants(C1_Fit_corr_mcmc_data, C2_Fit_corr_mcmc_data, C3_Fit_corr_mcmc_data, C4_Fit_corr_mcmc_data)
TotalEvents = events
# print(weights)
# exit()
################################# Distribution #################################
pdf_list = []
for i in range(len(C1)):
    mu = C1[i]
    pdf = edgeworth_pdf(x, mu, C2[i], C3[i], C4[i])
    pdf = np.maximum(pdf, 0)
    if np.sum(pdf) > 1e-9:
        pdf /= np.sum(pdf)
    pdf_list.append(pdf)
pdf_list = np.array(pdf_list)

################################# Bootstrap #################################
# 确定 TH1D 的 binning
n_bins = len(x)
x_min = x[0]
x_max = x[-1]
# 假设 x 是 bin 中心，计算 bin 宽度和边界
if n_bins > 1:
    bin_width = (x_max - x_min) / (n_bins - 1)
    hist_x_min = x_min - bin_width / 2.0
    hist_x_max = x_max + bin_width / 2.0
else:
    # 处理只有一个 bin 的情况
    bin_width = 1.0 # 或者其他合适的默认值
    hist_x_min = x_min - bin_width / 2.0
    hist_x_max = x_max + bin_width / 2.0

n_bootstrap = 500  # 重抽样次数
n_events_per_sample = np.zeros(len(C1), dtype=int) # 每个 bootstrap 样本生成的事件数，确保是整数
for i in range(len(C1)):
    n_events_per_sample[i] = int(weights[i] * TotalEvents)
print("事件数 per sample:", n_events_per_sample)

# 存储每个子分布 C1-C4 的标准差
all_std_devs_c1 = []
all_std_devs_c2 = []
all_std_devs_c3 = []
all_std_devs_c4 = []
all_std_devs_R21 = []
all_std_devs_R31 = []
all_std_devs_R42 = []
all_std_devs_k1 = []
all_std_devs_k2 = []
all_std_devs_k3 = []
all_std_devs_k4 = []
all_std_devs_k21 = []
all_std_devs_k31 = []
all_std_devs_k41 = []
for i in range(len(C1)): # 对所有 25 个分布进行处理 (!!! 目前只处理第一个)
    print(f"\nProcessing distribution {i} (mu ~ {C1[i]:.2f})...")
    # 1. 为当前子分布创建原始 TH1D
    hist_name = f"pdf_hist_{i}"
    hist_title = f"PDF for mu ~ {C1[i]:.2f}"
    original_hist = rt.TH1D(hist_name, hist_title, n_bins, hist_x_min, hist_x_max)

    # 填充原始 TH1D
    for j in range(n_bins):
        original_hist.SetBinContent(j + 1, pdf_list[i][j])

    # 检查原始直方图积分
    integral = original_hist.Integral()
    if integral <= 1e-9: # 使用一个小的阈值
        print(f"Warning: Histogram {hist_name} has non-positive integral ({integral}). Skipping bootstrap.")
        all_std_devs_c1.append(np.nan)
        all_std_devs_c2.append(np.nan)
        all_std_devs_c3.append(np.nan)
        all_std_devs_c4.append(np.nan)
        all_std_devs_R21.append(np.nan)
        all_std_devs_R31.append(np.nan)
        all_std_devs_R42.append(np.nan)
        all_std_devs_k1.append(np.nan)
        all_std_devs_k2.append(np.nan)
        all_std_devs_k3.append(np.nan)
        all_std_devs_k4.append(np.nan)
        all_std_devs_k21.append(np.nan)
        all_std_devs_k31.append(np.nan)
        all_std_devs_k41.append(np.nan)
        original_hist.Delete()
        continue
    
    # 存储当前子分布的 bootstrap 累积量
    current_bootstrap_c1s = []
    current_bootstrap_c2s = []
    current_bootstrap_c3s = []
    current_bootstrap_c4s = []
    current_bootstrap_R21s = []
    current_bootstrap_R31s = []
    current_bootstrap_R42s = []
    current_bootstrap_k1s = []
    current_bootstrap_k2s = []
    current_bootstrap_k3s = []
    current_bootstrap_k4s = []
    current_bootstrap_k21s = []
    current_bootstrap_k31s = []
    current_bootstrap_k41s = []
    nevents = n_events_per_sample[i]
    if nevents <= 0:
        print(f"Warning: Number of events for distribution {i} is {nevents}. Skipping bootstrap.")
        all_std_devs_c1.append(np.nan)
        all_std_devs_c2.append(np.nan)
        all_std_devs_c3.append(np.nan)
        all_std_devs_c4.append(np.nan)
        all_std_devs_R21.append(np.nan)
        all_std_devs_R31.append(np.nan)
        all_std_devs_R42.append(np.nan)
        all_std_devs_k1.append(np.nan)
        all_std_devs_k2.append(np.nan)
        all_std_devs_k3.append(np.nan)
        all_std_devs_k4.append(np.nan)
        all_std_devs_k21.append(np.nan)
        all_std_devs_k31.append(np.nan)
        all_std_devs_k41.append(np.nan)
        original_hist.Delete()
        continue

    print(f"Generating {n_bootstrap} bootstrap samples with {nevents} events each...")
    # 2. 进行 Bootstrap 重抽样
    for k in range(n_bootstrap):
        if (k + 1) % 10 == 0: # 每 10 次打印一次进度
            print(f"  Bootstrap sample {k+1}/{n_bootstrap}")
        bootstrap_hist_name = f"bootstrap_{i}_{k}"
        # 创建临时的 bootstrap 直方图
        bootstrap_hist = rt.TH1D(bootstrap_hist_name, bootstrap_hist_name, n_bins, hist_x_min, hist_x_max)

        # 从原始直方图代表的分布中抽样填充新的 bootstrap 直方图
        # FillRandom 需要整数事件数
        bootstrap_hist.FillRandom(original_hist, int(nevents))

        # 检查生成的事件数是否为 0
        if bootstrap_hist.GetEntries() == 0:
            # print(f"Warning: Bootstrap sample {k} for distribution {i} has 0 entries. Skipping cumulant calculation.")
            # 可以选择跳过这个样本或记录 NaN
            # 为了保持列表长度一致，我们可能仍然想计算，但 calculate_moments 应该能处理全零输入
            pass # 让 calculate_moments 处理

        # 提取 bin content (y_values)
        y_values = np.array([bootstrap_hist.GetBinContent(j) for j in range(1, n_bins + 1)])

        # 计算累积量
        c1, c2, c3, c4, R21, R31, R42, k1, k2, k3, k4, k21, k31, k41 = calculate_moments(y_values, x) # x 是 bin 中心
        eff = EFF[i]
        ep1 = 1/eff
        ep2 = 1/(eff**2)
        ep3 = 1/(eff**3)
        ep4 = 1/(eff**4)

        Corr_C1 = c1 * ep1
        Corr_C2 = c2 * ep2 + (ep1 - ep2) * c1
        Corr_C3 = c3 * ep3 + (ep2*3 - ep3*3) * c2 + (ep1 + ep3*2 - ep2*3) * c1
        Corr_C4 = c4 * ep4 + (ep3*6 - ep4*6) * c3 + (ep2*7 - ep3*18 + ep4*11) * c2 + (ep1 - ep2*7 + 12*ep3 - 6*ep4) * c1
        k1, k2, k3, k4 = calkappa(Corr_C1,Corr_C2,Corr_C3,Corr_C4)
        k21=k2/k1
        k31=k3/k1
        k41=k4/k1

        R21=Corr_C2/Corr_C1
        R31=Corr_C3/Corr_C1
        R42=Corr_C4/Corr_C2

        # 存储累积量 (即使是 NaN 也存储，以便后续分析)
        current_bootstrap_c1s.append(Corr_C1)
        current_bootstrap_c2s.append(Corr_C2)
        current_bootstrap_c3s.append(Corr_C3)
        current_bootstrap_c4s.append(Corr_C4)
        current_bootstrap_R21s.append(R21)
        current_bootstrap_R31s.append(R31)
        current_bootstrap_R42s.append(R42)
        current_bootstrap_k1s.append(k1)
        current_bootstrap_k2s.append(k2)
        current_bootstrap_k3s.append(k3)
        current_bootstrap_k4s.append(k4)
        current_bootstrap_k21s.append(k21)
        current_bootstrap_k31s.append(k31)
        current_bootstrap_k41s.append(k41)
        # 删除临时的 bootstrap 直方图以节省内存
        bootstrap_hist.Delete()

    # 清理 NaN 值，只保留有效的 bootstrap 结果用于计算标准差和绘图
    valid_c1s = np.array(current_bootstrap_c1s)[~np.isnan(current_bootstrap_c1s)]
    valid_c2s = np.array(current_bootstrap_c2s)[~np.isnan(current_bootstrap_c2s)]
    valid_c3s = np.array(current_bootstrap_c3s)[~np.isnan(current_bootstrap_c3s)]
    valid_c4s = np.array(current_bootstrap_c4s)[~np.isnan(current_bootstrap_c4s)]
    valid_R21s = np.array(current_bootstrap_R21s)[~np.isnan(current_bootstrap_R21s)]
    valid_R31s = np.array(current_bootstrap_R31s)[~np.isnan(current_bootstrap_R31s)]
    valid_R42s = np.array(current_bootstrap_R42s)[~np.isnan(current_bootstrap_R42s)]
    valid_k1s = np.array(current_bootstrap_k1s)[~np.isnan(current_bootstrap_k1s)]
    valid_k2s = np.array(current_bootstrap_k2s)[~np.isnan(current_bootstrap_k2s)]
    valid_k3s = np.array(current_bootstrap_k3s)[~np.isnan(current_bootstrap_k3s)]
    valid_k4s = np.array(current_bootstrap_k4s)[~np.isnan(current_bootstrap_k4s)]
    valid_k21s = np.array(current_bootstrap_k21s)[~np.isnan(current_bootstrap_k21s)]
    valid_k31s = np.array(current_bootstrap_k31s)[~np.isnan(current_bootstrap_k31s)]
    valid_k41s = np.array(current_bootstrap_k41s)[~np.isnan(current_bootstrap_k41s)]
    

    print(f"Finished bootstrap for distribution {i}. Valid samples: C1({len(valid_c1s)}), C2({len(valid_c2s)}), C3({len(valid_c3s)}), C4({len(valid_c4s)})")

    # 3. 计算累积量的标准差
    if len(valid_c1s) > 1: # 需要至少两个点来计算标准差
        std_dev_c1 = np.std(valid_c1s)
        std_dev_c2 = np.std(valid_c2s)
        std_dev_c3 = np.std(valid_c3s)
        std_dev_c4 = np.std(valid_c4s)
        std_dev_R21 = np.std(valid_R21s)
        std_dev_R31 = np.std(valid_R31s)
        std_dev_R42 = np.std(valid_R42s)
        std_dev_k1 = np.std(valid_k1s)
        std_dev_k2 = np.std(valid_k2s)
        std_dev_k3 = np.std(valid_k3s)
        std_dev_k4 = np.std(valid_k4s)
        std_dev_k21 = np.std(valid_k21s)
        std_dev_k31 = np.std(valid_k31s)
        std_dev_k41 = np.std(valid_k41s)
        print(f"  Std Devs: C1={std_dev_c1:.4g}, C2={std_dev_c2:.4g}, C3={std_dev_c3:.4g}, C4={std_dev_c4:.4g}, R21={std_dev_R21:.4g}, R31={std_dev_R31:.4g}, R42={std_dev_R42:.4g}, k1={std_dev_k1:.4g}, k2={std_dev_k2:.4g}, k3={std_dev_k3:.4g}, k4={std_dev_k4:.4g}, k21={std_dev_k21:.4g}, k31={std_dev_k31:.4g}, k41={std_dev_k41:.4g}")
    else:
        print(f"Warning: Not enough valid bootstrap samples (<2) for distribution {i} to calculate std dev.")
        std_dev_c1, std_dev_c2, std_dev_c3, std_dev_c4, std_dev_R21, std_dev_R31, std_dev_R42, std_dev_k1, std_dev_k2, std_dev_k3, std_dev_k4, std_dev_k21, std_dev_k31, std_dev_k41 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # 存储标准差
    all_std_devs_c1.append(std_dev_c1)
    all_std_devs_c2.append(std_dev_c2)
    all_std_devs_c3.append(std_dev_c3)
    all_std_devs_c4.append(std_dev_c4)
    all_std_devs_R21.append(std_dev_R21)
    all_std_devs_R31.append(std_dev_R31)
    all_std_devs_R42.append(std_dev_R42)
    all_std_devs_k1.append(std_dev_k1)
    all_std_devs_k2.append(std_dev_k2)
    all_std_devs_k3.append(std_dev_k3)
    all_std_devs_k4.append(std_dev_k4)
    all_std_devs_k21.append(std_dev_k21)
    all_std_devs_k31.append(std_dev_k31)
    all_std_devs_k41.append(std_dev_k41)
    # --- 新增：绘制 Bootstrap 样本累积量的分布 ---
    # if len(valid_c1s) > 0: # 确保有数据可绘图
    #     print(f"Plotting cumulant distributions for distribution {i}...")
    #     fig, axs = plt.subplots(2, 4, figsize=(12, 10))
    #     fig.suptitle(f'Bootstrap Cumulant Distributions for $\\mu$ ~ {C1[i]:.2f}, $N_{{events}}={nevents}$', fontsize=16)

    #     axs[0, 0].hist(valid_c1s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_c1:.3g}')
    #     axs[0, 0].set_title('$C_1$ Distribution', fontsize=14, fontweight='bold')
    #     axs[0, 0].set_xlabel('$C_1$ Value', fontsize=12, fontweight='bold')
    #     axs[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs[0, 0].legend(fontsize=12)
    #     axs[0, 0].grid(True)
    #     for spine in axs[0, 0].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs[0, 0].get_xticklabels() + axs[0, 0].get_yticklabels():
    #         label.set_fontweight('bold')


    #     axs[0, 1].hist(valid_c2s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_c2:.3g}')
    #     axs[0, 1].set_title('$C_2$ Distribution', fontsize=14, fontweight='bold')
    #     axs[0, 1].set_xlabel('$C_2$ Value', fontsize=12, fontweight='bold')
    #     axs[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs[0, 1].legend(fontsize=12)
    #     axs[0, 1].grid(True)
    #     for spine in axs[0, 1].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs[0, 1].get_xticklabels() + axs[0, 1].get_yticklabels():
    #         label.set_fontweight('bold')


    #     axs[0, 2].hist(valid_c3s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_c3:.3g}')
    #     axs[0, 2].set_title('$C_3$ Distribution', fontsize=14, fontweight='bold')
    #     axs[0, 2].set_xlabel('$C_3$ Value', fontsize=12, fontweight='bold')
    #     axs[0, 2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs[0, 2].legend(fontsize=12)
    #     axs[0, 2].grid(True)
    #     for spine in axs[0, 2].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs[0, 2].get_xticklabels() + axs[0, 2].get_yticklabels():
    #         label.set_fontweight('bold')


    #     axs[0, 3].hist(valid_c4s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_c4:.3g}')
    #     axs[0, 3].set_title('$C_4$ Distribution', fontsize=14, fontweight='bold')
    #     axs[0, 3].set_xlabel('$C_4$ Value', fontsize=12, fontweight='bold')
    #     axs[0, 3].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs[0, 3].legend(fontsize=12)
    #     axs[0, 3].grid(True)
    #     for spine in axs[0, 3].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs[0, 3].get_xticklabels() + axs[0, 3].get_yticklabels():
    #         label.set_fontweight('bold')

    #     axs[1, 0].hist(valid_R21s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_R21:.3g}')
    #     axs[1, 0].set_title('$R_{21}$ Distribution', fontsize=14, fontweight='bold')
    #     axs[1, 0].set_xlabel('$R_{21}$ Value', fontsize=12, fontweight='bold')
    #     axs[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs[1, 0].legend(fontsize=12)
    #     axs[1, 0].grid(True)
    #     for spine in axs[1, 0].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs[1, 0].get_xticklabels() + axs[1, 0].get_yticklabels():
    #         label.set_fontweight('bold')

    #     axs[1, 1].hist(valid_R31s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_R31:.3g}')
    #     axs[1, 1].set_title('$R_{32}$ Distribution', fontsize=14, fontweight='bold')
    #     axs[1, 1].set_xlabel('$R_{32}$ Value', fontsize=12, fontweight='bold')
    #     axs[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs[1, 1].legend(fontsize=12)
    #     axs[1, 1].grid(True)
    #     for spine in axs[1, 1].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs[1, 1].get_xticklabels() + axs[1, 1].get_yticklabels():
    #         label.set_fontweight('bold')

    #     axs[1, 2].hist(valid_R42s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_R42:.3g}')
    #     axs[1, 2].set_title('$R_{42}$ Distribution', fontsize=14, fontweight='bold')
    #     axs[1, 2].set_xlabel('$R_{42}$ Value', fontsize=12, fontweight='bold')
    #     axs[1, 2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs[1, 2].legend(fontsize=12)
    #     axs[1, 2].grid(True)
    #     for spine in axs[1, 2].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs[1, 2].get_xticklabels() + axs[1, 2].get_yticklabels():
    #         label.set_fontweight('bold')
            

    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应主标题
    #     plt.savefig(f"./StatErr_corr_fit/cumulant_dist_bootstrap_{i}.pdf") # 可以选择保存图像

    #     print(f"Plotting factorial cumulant distributions for distribution {i}...")
    #     fig1, axs1 = plt.subplots(2, 4, figsize=(12, 10))
    #     fig1.suptitle(f'Bootstrap Factorial Cumulant Distributions for $\\mu$ ~ {C1[i]:.2f}, $N_{{events}}={nevents}$', fontsize=16)

    #     axs1[0, 0].hist(valid_k1s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_k1:.3g}')
    #     axs1[0, 0].set_title('$\kappa_1$ Distribution', fontsize=14, fontweight='bold')
    #     axs1[0, 0].set_xlabel('$\kappa_1$ Value', fontsize=12, fontweight='bold')
    #     axs1[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs1[0, 0].legend(fontsize=12)
    #     axs1[0, 0].grid(True)
    #     for spine in axs1[0, 0].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs1[0, 0].get_xticklabels() + axs1[0, 0].get_yticklabels():
    #         label.set_fontweight('bold')


    #     axs1[0, 1].hist(valid_k2s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_k2:.3g}')
    #     axs1[0, 1].set_title('$\kappa_2$ Distribution', fontsize=14, fontweight='bold')
    #     axs1[0, 1].set_xlabel('$\kappa_2$ Value', fontsize=12, fontweight='bold')
    #     axs1[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs1[0, 1].legend(fontsize=12)
    #     axs1[0, 1].grid(True)
    #     for spine in axs1[0, 1].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs1[0, 1].get_xticklabels() + axs1[0, 1].get_yticklabels():
    #         label.set_fontweight('bold')


    #     axs1[0, 2].hist(valid_k3s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_k3:.3g}')
    #     axs1[0, 2].set_title('$\kappa_3$ Distribution', fontsize=14, fontweight='bold')
    #     axs1[0, 2].set_xlabel('$\kappa_3$ Value', fontsize=12, fontweight='bold')
    #     axs1[0, 2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs1[0, 2].legend(fontsize=12)
    #     axs1[0, 2].grid(True)
    #     for spine in axs1[0, 2].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs1[0, 2].get_xticklabels() + axs1[0, 2].get_yticklabels():
    #         label.set_fontweight('bold')


    #     axs1[0, 3].hist(valid_k4s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_k4:.3g}')
    #     axs1[0, 3].set_title('$\kappa_4$ Distribution', fontsize=14, fontweight='bold')
    #     axs1[0, 3].set_xlabel('$\kappa_4$ Value', fontsize=12, fontweight='bold')
    #     axs1[0, 3].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs1[0, 3].legend(fontsize=12)
    #     axs1[0, 3].grid(True)
    #     for spine in axs1[0, 3].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs1[0, 3].get_xticklabels() + axs1[0, 3].get_yticklabels():
    #         label.set_fontweight('bold')

    #     axs1[1, 0].hist(valid_k21s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_k21:.3g}')
    #     axs1[1, 0].set_title('$k_{21}$ Distribution', fontsize=14, fontweight='bold')
    #     axs1[1, 0].set_xlabel('$k_{21}$ Value', fontsize=12, fontweight='bold')
    #     axs1[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs1[1, 0].legend(fontsize=12)
    #     axs1[1, 0].grid(True)
    #     for spine in axs1[1, 0].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs1[1, 0].get_xticklabels() + axs1[1, 0].get_yticklabels():
    #         label.set_fontweight('bold')

    #     axs1[1, 1].hist(valid_k31s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_k31:.3g}')
    #     axs1[1, 1].set_title('$k_{31}$ Distribution', fontsize=14, fontweight='bold')
    #     axs1[1, 1].set_xlabel('$k_{31}$ Value', fontsize=12, fontweight='bold')
    #     axs1[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs1[1, 1].legend(fontsize=12)
    #     axs1[1, 1].grid(True)
    #     for spine in axs1[1, 1].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs1[1, 1].get_xticklabels() + axs1[1, 1].get_yticklabels():
    #         label.set_fontweight('bold')

    #     axs1[1, 2].hist(valid_k41s, bins=15, alpha=0.7, label=f'Std Dev: {std_dev_k41:.3g}')
    #     axs1[1, 2].set_title('$k_{41}$ Distribution', fontsize=14, fontweight='bold')
    #     axs1[1, 2].set_xlabel('$k_{41}$ Value', fontsize=12, fontweight='bold')
    #     axs1[1, 2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    #     axs1[1, 2].legend(fontsize=12)
    #     axs1[1, 2].grid(True)
    #     for spine in axs1[1, 2].spines.values():
    #         spine.set_linewidth(1.5) # 调整线宽
    #     for label in axs1[1, 2].get_xticklabels() + axs1[1, 2].get_yticklabels():
    #         label.set_fontweight('bold')
            

    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应主标题
    #     # plt.savefig(f"./StatErr_corr_fit/factorial_cumulant_dist_bootstrap_{i}.pdf") # 可以选择保存图像
    #     # plt.show() # 显示图像

    # else:
    #     print(f"Skipping plotting for distribution {i} due to lack of valid data.")
    # ------------------------------------------

    # 清理原始直方图
    original_hist.Delete()

print("\nBootstrap finished.")

# 打印或保存结果
print("\nEstimated Standard Deviations (Errors) from Bootstrap:")
print("Distribution Index | Std Dev C1 | Std Dev C2 | Std Dev C3 | Std Dev C4 | Std Dev R21 | Std Dev R31 | Std Dev R42 | Std Dev k1 | Std Dev k2 | Std Dev k3 | Std Dev k4 | Std Dev k21 | Std Dev k31 | Std Dev k41")
print("-" * 100)
# 填充可能缺失的值（如果循环范围不是 range(25)）
num_processed = len(all_std_devs_c1)
for i in range(len(C1)):
    if i < num_processed:
        print(f"{i:^18} | {all_std_devs_c1[i]:^10.4g} | {all_std_devs_c2[i]:^10.4g} | {all_std_devs_c3[i]:^10.4g} | {all_std_devs_c4[i]:^10.4g} | {all_std_devs_R21[i]:^10.4g} | {all_std_devs_R31[i]:^10.4g} | {all_std_devs_R42[i]:^10.4g} | {all_std_devs_k1[i]:^10.4g} | {all_std_devs_k2[i]:^10.4g} | {all_std_devs_k3[i]:^10.4g} | {all_std_devs_k4[i]:^10.4g} | {all_std_devs_k21[i]:^10.4g} | {all_std_devs_k31[i]:^10.4g} | {all_std_devs_k41[i]:^10.4g}")
    else:
        # 如果循环提前停止，打印占位符
        print(f"{i:^18} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10}")


# 可以将结果保存到文件
std_dev_results = np.array([all_std_devs_c1, all_std_devs_c2, all_std_devs_c3, all_std_devs_c4, all_std_devs_R21, all_std_devs_R31, all_std_devs_R42, all_std_devs_k1, all_std_devs_k2, all_std_devs_k3, all_std_devs_k4, all_std_devs_k21, all_std_devs_k31, all_std_devs_k41]).T
# Pad with NaNs if fewer than 25 distributions were processed
if num_processed < len(C1):
    padding = np.full((len(C1) - num_processed, 14), np.nan)
    std_dev_results = np.vstack((std_dev_results, padding))
np.savetxt("bootstrap_cumulant_stddevs_corr_fit_tmp.txt", std_dev_results, header="StdDev_C1 StdDev_C2 StdDev_C3 StdDev_C4 StdDev_R21 StdDev_R31 StdDev_R42 StdDev_k1 StdDev_k2 StdDev_k3 StdDev_k4 StdDev_k21 StdDev_k31 StdDev_k41", fmt='%10.4g', comments='')

# # 不再需要写入包含大量直方图的 ROOT 文件，除非有其他目的
# # output_file = rt.TFile("bootstrap_analysis.root", "RECREATE")
# # ... (如果需要保存其他内容，可以添加)
# # output_file.Close()