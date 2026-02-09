import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import os
import arviz as az
import pymc as pm
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from output_results import *
import tempfile
import sys
from CentBin import *
from C1_polynomial_fit import a, b, c, d, e
from GenPlot import GenPlot, cleanup
temp_dir = tempfile.mkdtemp()
os.environ["PYTENSOR_FLAGS"] = f"compiledir={temp_dir}"
from Efficiency import EFF_default as EFFICIENCY_ARRAY
#####################################################################
ScanMax = False
RunResults = True
ScanOnlyC3 = False
bin_ref3[:-1] = bin_ref3[:-1] + 1
NCENT = len(bin_ref3)-2
START_FIT = ...
MAX = ...
LAST_C3 = ...
ENERGY = ...
average_weight = ...
# LAST_C3 = float(sys.argv[1])
# MAX = int(sys.argv[1])

############################################ 加载数据 ##############################################
data = np.loadtxt('ProtonPDF.txt')
x = data[:, 0]
y = data[:, 1]
y_error = data[:, 2]
x_DE = x[:MAX]
y_DE = y[:MAX]
y_error_DE = y_error[:MAX]

x_edge = x
data_edge = np.loadtxt(f'RemoveLowEvents_ProDist_FromRef3/Proton_Total_Cent{bin_ref3[-2]}_{bin_ref3[-1]}.txt')[:100]
x_edge = data_edge[:, 0]
y_edge = data_edge[:, 1]/np.sum(data_edge[:, 1])
total_weight = average_weight * NCENT
y_edge = y_edge * (1 - total_weight)
x_edge = x

x_edge_DE = x_DE
y_edge_DE = y_edge[:MAX]

def calc_kappa(c1, c2, c3, c4):
    k1 = c1
    k2 = c2 - c1
    k3 = c3 - 3*c2 + 2*c1
    k4 = c4 - 6*c3 + 11*c2 - 6*c1
    return k1, k2, k3, k4

def GetCorrCum(C1_uncorr, C2_uncorr, C3_uncorr, C4_uncorr):
    eff = EFFICIENCY_ARRAY[:NCENT][::-1]
    ep1 = 1/eff
    ep2 = 1/(eff**2)
    ep3 = 1/(eff**3)
    ep4 = 1/(eff**4)
    Corr_C1 = C1_uncorr * ep1
    Corr_C2 = C2_uncorr * ep2 + (ep1 - ep2) * C1_uncorr
    Corr_C3 = C3_uncorr * ep3 + (ep2*3 - ep3*3) * C2_uncorr + (ep1 + ep3*2 - ep2*3) * C1_uncorr
    Corr_C4 = C4_uncorr * ep4 + (ep3*6 - ep4*6) * C3_uncorr + (ep2*7 - ep3*18 + ep4*11) * C2_uncorr + (ep1 - ep2*7 + 12*ep3 - 6*ep4) * C1_uncorr
    return Corr_C1, Corr_C2, Corr_C3, Corr_C4
def Get_Raw_Cumulants_From_CorrectedCumulants(c1_corr, c2_corr, c3_corr, c4_corr):
    eff = EFFICIENCY_ARRAY[:NCENT][::-1]
    ep1 = 1/eff
    ep2 = 1/(eff**2)
    ep3 = 1/(eff**3)
    ep4 = 1/(eff**4)
    c1_raw = c1_corr / ep1
    c2_raw = (c2_corr - (ep1 - ep2) * c1_raw) / ep2
    c3_raw = (c3_corr - (3*ep2 - 3*ep3) * c2_raw - (ep1 + 2*ep3 - 3*ep2) * c1_raw) / ep3
    c4_raw = (c4_corr - (6*ep3 - 6*ep4) * c3_raw - (7*ep2 - 18*ep3 + 11*ep4) * c2_raw - (ep1 - 7*ep2 + 12*ep3 - 6*ep4) * c1_raw) / ep4
    return c1_raw, c2_raw, c3_raw, c4_raw

Useless = 1
Total_C1_Ref3 = Total_C1_Ref3[Useless:]
Total_C2_Ref3 = Total_C2_Ref3[Useless:]
Total_C3_Ref3 = Total_C3_Ref3[Useless:]
Total_C4_Ref3 = Total_C4_Ref3[Useless:]
CBWC_C1_Ref3 = CBWC_C1_Ref3[Useless:]
CBWC_C2_Ref3 = CBWC_C2_Ref3[Useless:]
CBWC_C3_Ref3 = CBWC_C3_Ref3[Useless:]
CBWC_C4_Ref3 = CBWC_C4_Ref3[Useless:]
Total_C1_Npart = Total_C1_Ref3[Useless:]
Total_C2_Npart = Total_C2_Ref3[Useless:]
Total_C3_Npart = Total_C3_Ref3[Useless:]
Total_C4_Npart = Total_C4_Ref3[Useless:]
CBWC_C1_Npart = CBWC_C1_Ref3[Useless:]
CBWC_C2_Npart = CBWC_C2_Ref3[Useless:]
CBWC_C3_Npart = CBWC_C3_Ref3[Useless:]
CBWC_C4_Npart = CBWC_C4_Ref3[Useless:]
C4_C2_raw = CBWC_C4_Ref3 / CBWC_C2_Ref3

Total_C1_Ref3, Total_C2_Ref3, Total_C3_Ref3, Total_C4_Ref3 = GetCorrCum(Total_C1_Ref3, Total_C2_Ref3, Total_C3_Ref3, Total_C4_Ref3)
CBWC_C1_Ref3, CBWC_C2_Ref3, CBWC_C3_Ref3, CBWC_C4_Ref3 = GetCorrCum(CBWC_C1_Ref3, CBWC_C2_Ref3, CBWC_C3_Ref3, CBWC_C4_Ref3)
C3_C2 = Total_C3_Ref3 / Total_C2_Ref3
C4_C2 = CBWC_C4_Ref3 / CBWC_C2_Ref3
CBWC_K1, CBWC_K2, CBWC_K3, CBWC_K4 = calc_kappa(CBWC_C1_Ref3, CBWC_C2_Ref3, CBWC_C3_Ref3, CBWC_C4_Ref3)
CBWC_K21 = CBWC_K2 / CBWC_K1
CBWC_K31 = CBWC_K3 / CBWC_K1
CBWC_K41 = CBWC_K4 / CBWC_K1
# print(CBWC_K41)
# exit()
def edgeworth_pdf(x, mu, k2, k3, k4):
    skew = k3 / (k2**(3/2))
    kurt = k4 / (k2**2)
    z = (x - mu) / (k2**(1/2))
    correction = (1 + 
                     skew/6 * (z**3 - 3*z) +
                     kurt/24 * (z**4 - 6*z**2 + 3) +
                     skew**2/72 * (z**6 - 15*z**4 + 45*z**2 - 15)
                )
    gaussian = np.exp(-0.5 * z**2) / (2.50663 * k2**(1/2))
    return gaussian * correction

def edgeworth_pdf_vectorized(x, mu_array, k2_array, k3_array, k4_array):
    # 重塑数组以便广播
    x_reshaped = x.reshape(-1, 1)  # 形状为(n, 1)
    mu_reshaped = mu_array.reshape(1, -1)  # 形状为(1, m)
    k2_reshaped = k2_array.reshape(1, -1)  # 形状为(1, m)
    
    # 计算标准化变量z - 形状为(n, m)
    z = (x_reshaped - mu_reshaped) / np.sqrt(k2_reshaped)
    
    # 计算偏度和峰度 - 形状为(1, m)
    skew = k3_array.reshape(1, -1) / (k2_reshaped**(3/2))
    kurt = k4_array.reshape(1, -1) / (k2_reshaped**2)
    
    # 计算修正项 - 形状为(n, m)
    correction = (1 + 
                     skew/6 * (z**3 - 3*z) +
                     kurt/24 * (z**4 - 6*z**2 + 3) +
                     skew**2/72 * (z**6 - 15*z**4 + 45*z**2 - 15) 
                )
    
    # 计算高斯项 - 形状为(n, m)
    gaussian = np.exp(-0.5 * z**2) / (2.50663 * np.sqrt(k2_reshaped))
    
    # 返回最终PDF - 形状为(n, m)
    return gaussian * correction
# exit()
def huber_loss(residuals, delta=1.0):
    # abs_residuals = np.abs(residuals)
    squared_loss = 0.5 * residuals**2
    # linear_loss = delta * (abs_residuals - 0.5 * delta)
    # combin_value = np.zeros_like(residuals)
    # combin_value[:CUT] = squared_loss[:CUT]
    # combin_value[CUT:] = np.where(abs_residuals[CUT:] <= delta, squared_loss[CUT:], linear_loss[CUT:])
    return np.sum(squared_loss)

def objective_function(params):
    """
    使用高权重软约束确保斜率严格满足的损失函数
    """
    try:    
        # 参数解包
        a_c2, b_c2, c_c2 = params[0:3]
        a_c3, b_c3, c_c3, d_c3 = params[3:7]
        a_c4, b_c4, c_c4 = params[7:10]
        C1_a, C1_b, C1_c, C1_d, C1_e = params[10:15]
        weight_fluctuations = params[15:15+NCENT]
        
        # 计算优化后的C1分布
        i_values = np.arange(NCENT)  # 假设有NCENT个i值
        optimized_C1 = C1_a*(i_values**4) + C1_b*(i_values**3) + C1_c*(i_values**2) + C1_d*i_values**1 + C1_e
        
        # 标度因子
        weight_fluctuations = average_weight * np.array(weight_fluctuations)
        factorDE = np.array(weight_fluctuations)        
        # 使用参数化公式计算C2、C3/C2和C4/C2
        Cum2 = a_c2 + b_c2*optimized_C1 + c_c2*(optimized_C1**2)
        Cum3 = a_c3 + b_c3*optimized_C1 + c_c3*(optimized_C1**2) + d_c3*(optimized_C1**3)  # C3/C2
        R42 = a_c4 + b_c4*optimized_C1 + c_c4*(optimized_C1**2)  # C4/C2
        Cum4 = R42 * Cum2
        R31 = Cum3 / optimized_C1
        R21 = Cum2 / optimized_C1
        
        # 计算预测分布
        C1_raw, C2_raw, C3_raw, C4_raw = Get_Raw_Cumulants_From_CorrectedCumulants(optimized_C1, Cum2, Cum3, Cum4)
        sub_distributions = edgeworth_pdf_vectorized(x_DE, C1_raw, C2_raw, C3_raw, C4_raw)
        y_pred = np.sum(factorDE * sub_distributions, axis=1)

        k1, k2, k3, k4 = calc_kappa(optimized_C1, Cum2, Cum3, Cum4)
        k41 = k4[-1] / k1[-1]
        k41_tot = k4/k1
        delta_k41_tot = k41_tot - 0
        deviation = 0
        deviation += np.sum(abs(delta_k41_tot[:NCENT-8])) * 1e-1
        if delta_k41_tot[-1] < 0:
            deviation += abs(delta_k41_tot[-1]) * 1e1
        normalized_residuals_chi2 = (y_pred[START_FIT:] - y_DE[START_FIT:]) / y_error_DE[START_FIT:]

        delta = 1
        huber_data_loss = huber_loss(normalized_residuals_chi2, delta)

        chi2_data_normalized = huber_data_loss
        
        chi2_last_c3 = 0
        dev_th = 0.001
        last_c3 = C3_raw[-1]
        upper = LAST_C3 * (1 + dev_th)
        lower = LAST_C3 * (1 - dev_th)
        if last_c3 > upper or last_c3 < lower:
            chi2_last_c3 = abs(last_c3 - LAST_C3) * 1e1

        error_data = 0
        if C4_raw[-1]/C2_raw[-1] < np.min(C4_C2_raw): 
            error_data += abs(C4_raw[-1]/C2_raw[-1] - np.min(C4_C2_raw)) * 1e1
        error_data += abs(np.max(R42) - np.max(R31))
        if chi2_data_normalized * 2 < (MAX - START_FIT):
            error_data += abs(chi2_data_normalized * 2 - (MAX - START_FIT))
        weights_diff = np.sum(factorDE - average_weight)
        error_data += abs(weights_diff) * 1e3
        total = error_data  + deviation + chi2_data_normalized + chi2_last_c3
        return np.nan_to_num(total, nan=1e20)
    
    except Exception as e:
        return 1e20

bounds = [
    (-0.3, -0.1),                # a_c2
    (1.15, 1.2),               # b_c2
    (-0.01, -0.006),       # c_c2
    (-1, 0),               # a_c3
    (1., 1.6),        # b_c3
    (-0.03, -0.01),    # c_c3
    (-0.0001, 0.0001), # d_c3
    (1., 1.6),             # a_r42
    (-0.2, 0.2),         # b_r42
    (-0.0001, 0.0001),       # c_r42
    (0, 0.001),     # C1_a
    (-0.01, 0),      # C1_b
    (0, 0.1),            # C1_c
    (0, 1),               # C1_d
    (e, e + 0.4),                 # C1_e
]

for i in range(NCENT):
    bounds.append((0.95, 1.05)) 

if __name__ == "__main__":
    result = differential_evolution(
        objective_function,
        bounds,
        strategy='randtobest1exp',  # 使用最佳个体策略
        maxiter=5000,        # 最大迭代次数
        popsize=10,          # 增加种群大小以提高并行性能
        mutation=(0.5, 1.3), # 调整变异参数范围
        recombination=0.95,   # 降低重组率以增加多样性
        tol=1e-4,            # 收敛容差
        seed=42,             # 随机种子
        workers=10,          # 使用所有可用CPU核心
        disp=False,           # 显示优化进
        polish=True,         # 使用局部优化
        updating='deferred', # 延迟更新种群
        init='sobol',        # 使用Sobol序列初始化种群
        atol=1e-1,           # s绝对收敛容差
    )

    # 提取优化结果
    print(f"优化完成，最终目标函数值: {result.fun}")
    a_c2, b_c2, c_c2 = result.x[0:3]
    a_c3, b_c3, c_c3, d_c3 = result.x[3:7]
    a_c4, b_c4, c_c4 = result.x[7:10]
    C1_a, C1_b, C1_c, C1_d, C1_e = result.x[10:15]
    weight_fluctuations = result.x[15:15+NCENT]

    # print(f'C1_a = {C1_a}')
    # print(f'C1_b = {C1_b}') 
    # print(f'C1_c = {C1_c}')
    # print(f'C1_d = {C1_d}')
    # print(f'C1_e = {C1_e}')
    print(f'a_c2 = {a_c2}')
    print(f'b_c2 = {b_c2}') 
    print(f'c_c2 = {c_c2}')
    # print(f'a_c3 = {a_c3}')
    # print(f'b_c3 = {b_c3}')
    # print(f'c_c3 = {c_c3}')
    # print(f'd_c3 = {d_c3}')
    # print(f'a_c4 = {a_c4}')
    # print(f'b_c4 = {b_c4}')
    # print(f'c_c4 = {c_c4}')
    # 计算实际权重
    C1_corr = np.array([(C1_a*i**4 + C1_b*i**3 + C1_c*i**2 + C1_d*i + C1_e) for i in range(NCENT)])
    factorDE = np.array(weight_fluctuations) * average_weight
    print(f"最后两个权重比值: {factorDE[-1]/factorDE[-2]:.4f}")

    # 计算参数化的C2、C3、C4
    C2_values_corr = a_c2 + b_c2*C1_corr + c_c2*C1_corr**2
    C3_values_corr = a_c3 + b_c3*C1_corr + c_c3*C1_corr**2 + d_c3*C1_corr**3
    R42_corr = a_c4 + b_c4*C1_corr + c_c4*C1_corr**2  # C4/C2
    C4_values_corr = R42_corr * C2_values_corr
    
    C1_raw, C2_raw, C3_raw, C4_raw = Get_Raw_Cumulants_From_CorrectedCumulants(C1_corr, C2_values_corr, C3_values_corr, C4_values_corr)
    k1, k2, k3, k4 = calc_kappa(C1_corr, C2_values_corr, C3_values_corr, C4_values_corr)
    R41 = C4_raw / C1_raw
    slope = (R41[-1]-R41[0])/(C1_raw[-1]-C1_raw[0])

    print(f'R42_corr[-1] = {C4_values_corr[-1]/C2_values_corr[-1]:.4f}')
    print(f'R42_raw[-1] = {C4_raw[-1]/C2_raw[-1]:.4f}')
    print(f'C2_values_corr[-1]/C1_corr[-1] = {C2_values_corr[-1]/C1_corr[-1]:.4f}')
    print(f'C2_raw/C1_raw[-1] = {C2_raw[-1]/C1_raw[-1]:.4f}')
    print(f'k41 = {k4[-1]/k1[-1]:.4f}')

    if ScanMax == True:
        with open(f'./ScanMax/fit_results_{MAX:.4f}.txt', 'w') as f:
            f.write(f"import numpy as np\n")
            f.write(f"chi2 = {result.fun:.4f}\n")
            f.write(f'weights = np.array([{", ".join([f"{x:.4f}" for x in factorDE])}])\n')
            f.write(f"C1_Fit_corr_DE_data = np.array([{', '.join(map(str, C1_corr))}])\n")
            f.write(f"C2_Fit_corr_DE_data = np.array([{', '.join(map(str, C2_values_corr))}])\n") 
            f.write(f"C3_Fit_corr_DE_data = np.array([{', '.join(map(str, C3_values_corr))}])\n")
            f.write(f"C4_Fit_corr_DE_data = np.array([{', '.join(map(str, C4_values_corr))}])\n")
    if ScanOnlyC3 == True:
        with open(f'./ScanOnlyC3/fit_results_{LAST_C3:.4f}.txt', 'w') as f:
            f.write(f"import numpy as np\n")
            f.write(f"chi2 = {result.fun:.4f}\n")
            f.write(f'weights = np.array([{", ".join([f"{x:.4f}" for x in factorDE])}])\n')
            f.write(f"C1_Fit_corr_DE_data = np.array([{', '.join(map(str, C1_corr))}])\n")
            f.write(f"C2_Fit_corr_DE_data = np.array([{', '.join(map(str, C2_values_corr))}])\n") 
            f.write(f"C3_Fit_corr_DE_data = np.array([{', '.join(map(str, C3_values_corr))}])\n")
            f.write(f"C4_Fit_corr_DE_data = np.array([{', '.join(map(str, C4_values_corr))}])\n")
    if RunResults == True:
        with open('./FinalResults/weights.py', 'w') as f:
            f.write(f"import numpy as np\n")
            f.write(f'weights = np.array([{", ".join([f"{x:.4f}" for x in factorDE])}])\n')
        with open(f'./FinalResults/fit_results.py', 'w') as f:
            f.write(f"import numpy as np\n")
            f.write(f"chi2 = {result.fun:.4f}\n")
            f.write(f'C1_Fit_corr_DE_data = np.array([{", ".join([f"{x:.3f}" for x in C1_corr])}])\n')
            f.write(f'C2_Fit_corr_DE_data = np.array([{", ".join([f"{x:.3f}" for x in C2_values_corr])}])\n') 
            f.write(f'C3_Fit_corr_DE_data = np.array([{", ".join([f"{x:.3f}" for x in C3_values_corr])}])\n')
            f.write(f'C4_Fit_corr_DE_data = np.array([{", ".join([f"{x:.3f}" for x in C4_values_corr])}])\n')

    if ScanMax == True or ScanOnlyC3 == True:
        cleanup(temp_dir)
        exit()
    GenPlot(C1_corr, C2_values_corr, C3_values_corr, C4_values_corr, factorDE, "ResultsDE", ENERGY,
            CBWC_C1_Ref3, CBWC_C2_Ref3, CBWC_C3_Ref3, CBWC_C4_Ref3, Total_C1_Ref3, Total_C2_Ref3, Total_C3_Ref3, Total_C4_Ref3)
    
    C1_raw, C2_raw, C3_raw, C4_raw = Get_Raw_Cumulants_From_CorrectedCumulants(C1_corr, C2_values_corr, C3_values_corr, C4_values_corr)
    plt.figure(figsize=(10, 8))
    plt.yscale('log')
    plt.plot(x, y, 'ko', label=f'Raw data({ENERGY:.1f} GeV, Data, DE)', markersize=16, alpha=0.6)
    y_total = np.zeros(len(x))
    for i in range(len(C1_corr)):
        y_DE_i = factorDE[i]*edgeworth_pdf(x, C1_raw[i], C2_raw[i], C3_raw[i], C4_raw[i])
        plt.plot(x, y_DE_i, '--', label=f'', alpha=0.6, linewidth=2)
        y_total += y_DE_i
    plt.plot(x_edge_DE, y_edge_DE, marker='o', label=f'50-100% Distribution', alpha=0.6, markersize=14, fillstyle='none', markeredgewidth=2, markeredgecolor='brown', linewidth=0)
    chi2 = np.sum( ((y_DE[START_FIT:] - y_total[START_FIT:MAX]) / y_error_DE[START_FIT:]) **2 )
    plt.plot(x_edge, y_total + y_edge, 'r-', label=f'Reconstructed($\chi^2$ = {chi2:.4f}$)$', alpha=0.8, linewidth=3)
    plt.legend(fontsize=18)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.xlabel('$\mathbf{N_{proton}}$', fontsize=18)
    plt.ylabel('$\mathbf{Probablity}$', fontsize=18)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.xlim(0,MAX + 10)
    plt.ylim(1e-5, 3e-1)
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.savefig(f'./Results/ResultsDE_PDF.pdf')

    # cleanup(temp_dir)
    C1_a = result.x[10]
    C1_b = result.x[11]
    C1_c = result.x[12]
    C1_d = result.x[13]
    C1_e = result.x[14]
    a_c2 = result.x[0]
    b_c2 = result.x[1]
    c_c2 = result.x[2]
    a_c3 = result.x[3]
    b_c3 = result.x[4]
    c_c3 = result.x[5]
    d_c3 = result.x[6]
    a_c4 = result.x[7]
    b_c4 = result.x[8]
    c_c4 = result.x[9]
    C1_corr = np.array([(C1_a*i**4 + C1_b*i**3 + C1_c*i**2 + C1_d*i + C1_e) for i in range(NCENT)])
    C2_values_corr = a_c2 + b_c2*C1_corr + c_c2*C1_corr**2

    DE_a_C3 = a_c3
    DE_b_C3 = b_c3
    DE_c_C3 = c_c3
    DE_d_C3 = d_c3
    DE_a_R42 = a_c4
    DE_b_R42 = b_c4
    DE_c_R42 = c_c4
    DE_weights = np.ones(len(factorDE))*average_weight
    C1 = C1_corr
    # eff = np.ones(23)
    eff = EFFICIENCY_ARRAY[:NCENT][::-1]
    ep1 = 1/eff
    ep2 = 1/(eff**2)
    ep3 = 1/(eff**3)
    ep4 = 1/(eff**4)
    ep1 = ep1[:, None]
    ep2 = ep2[:, None]
    ep3 = ep3[:, None]
    ep4 = ep4[:, None]

    import pytensor.tensor as tt
    # from pymc import CompoundStep, Metropolis

    with pm.Model() as model:
        # Basic parameters - C2 parameterization
        a_c3_param = pm.Normal('a_c3', mu=DE_a_C3, sigma=abs(DE_a_C3)*0.05)
        b_c3_param = pm.Normal('b_c3', mu=DE_b_C3, sigma=abs(DE_b_C3)*0.05)
        c_c3_param = pm.Normal('c_c3', mu=DE_c_C3, sigma=abs(DE_c_C3)*0.05)
        d_c3_param = pm.Normal('d_c3', mu=DE_d_C3, sigma=abs(DE_d_C3)*0.05)

        # C4/C2参数
        a_c4OverC2 = pm.Normal('a_c4OverC2', mu=DE_a_R42, sigma=abs(DE_a_R42)*0.05)
        b_c4OverC2 = pm.Normal('b_c4OverC2', mu=DE_b_R42, sigma=abs(DE_b_R42)*0.05)
        c_c4OverC2 = pm.Normal('c_c4OverC2', mu=DE_c_R42, sigma=abs(DE_c_R42)*0.05)

        weights = pm.Normal('weights', 
                                    mu=DE_weights, 
                                    sigma=DE_weights*0.05, 
                                    shape=len(DE_weights))
        mu_matrix = pm.math.stack([C1]*len(x_DE)).T
        Cum2 = C2_values_corr[:, None]

        # Calculate C3 and C4 through ratio
        Cum3 = a_c3_param + b_c3_param * mu_matrix + c_c3_param * mu_matrix**2 + d_c3_param * mu_matrix**3
        R42 = a_c4OverC2 + b_c4OverC2 * mu_matrix + c_c4OverC2 * mu_matrix**2
        R32 = Cum3 / Cum2
        Cum4 = R42 * Cum2
        R31 = Cum3 / mu_matrix

        # 计算原始累积量
        c1_raw_matrix = mu_matrix / ep1
        c2_raw_matrix = (Cum2 - (ep1 - ep2) * c1_raw_matrix) / ep2
        c3_raw_matrix = (Cum3 - (3*ep2 - 3*ep3) * c2_raw_matrix - (ep1 + 2*ep3 - 3*ep2) * c1_raw_matrix) / ep3
        c4_raw_matrix = (Cum4 - (6*ep3 - 6*ep4) * c3_raw_matrix - (7*ep2 - 18*ep3 + 11*ep4) * c2_raw_matrix - (ep1 - 7*ep2 + 12*ep3 - 6*ep4) * c1_raw_matrix) / ep4
        
        # 计算累积量
        k4 = Cum4 - 6*Cum3 + 11*Cum2 - 6*mu_matrix
        
        # 计算k4/k1比率及其偏差
        delta_k41_tot_corr = k4[:, 0] / mu_matrix[:, 0]
        first = delta_k41_tot_corr[:NCENT-8]
        pen_first = pm.math.sum(abs(first))

        # 最后一个点必须位于 [0, 0.5] 之内，超出时按超出量的平方惩罚
        last_val = delta_k41_tot_corr[-1]
        last_pen = pm.math.switch(
            last_val < 0,
            abs(last_val),
            0.0
        )

        total_pen = -1e-1*pen_first - 1e1*last_pen
        pm.Potential('k41_soft_constraint', total_pen)
        
        # Vectorized Edgeworth expansion calculation
        z = (x_DE - c1_raw_matrix) / pm.math.sqrt(c2_raw_matrix)
        skew = c3_raw_matrix / pm.math.sqrt(c2_raw_matrix)**3
        kurt = c4_raw_matrix / pm.math.sqrt(c2_raw_matrix)**4
        
        # 计算Edgeworth展开修正
        correction = (1 + 
                    (skew/6)*(z**3 - 3*z) + 
                    (kurt/24)*(z**4 - 6*z**2 + 3) + 
                    (skew**2/72)*(z**6 - 15*z**4 + 45*z**2 - 15))

        # 计算高斯分布
        gaussian = pm.math.exp(-0.5 * z**2) / (2.50663 * pm.math.sqrt(c2_raw_matrix))

        # 计算模型预测值
        y_model = pm.math.sum(weights[:, None] * gaussian * correction, axis=0)
        # y_model += y_edge

        residuals = (y_model[START_FIT:] - y_DE[START_FIT:]) / y_error_DE[START_FIT:]

        squared_loss = 0.5 * residuals**2
        
        # 根据残差大小选择损失函数
        huber_losses = squared_loss
        
        # 将Huber损失作为负对数似然添加到模型中
        pm.Potential('huber_likelihood', -pm.math.sum(huber_losses))
        
        # C3 末端软约束（允许小范围偏差，超出部分按平方惩罚）
        dev_th = 0.001
        last_c3 = c3_raw_matrix[-1, 0]
        upper = LAST_C3 * (1 + dev_th)
        lower = LAST_C3 * (1 - dev_th)
        excess1 = pm.math.switch(last_c3 > upper, abs(last_c3-LAST_C3), pm.math.switch(last_c3 < lower, abs(LAST_C3 - last_c3), 0.0))
        # excess2 = pm.math.switch(pm.math.sum(huber_losses)*2 < (MAX - START_FIT), abs(pm.math.sum(huber_losses)*2 - (MAX - START_FIT)), 0.0)
        total_pen = excess1

        # 将总惩罚作为潜在项加入模型（系数可调，与 k41 的处理一致）
        pm.Potential('soft_constraints', -total_pen * 1e1)

        last_c4 = c4_raw_matrix[-1, 0]
        excess_c4 = pm.math.switch(last_c4 < pm.math.min(C4_C2_raw), abs(last_c4-pm.math.min(C4_C2_raw)), 0.0)
        error_data1 =  -abs(pm.math.max(R42[:, 0]) - pm.math.max(R31[:, 0])) - abs(pm.math.sum(weights - average_weight)) * 1e3
        pm.Potential('soft_constraints_c4', -excess_c4 * 1e1 + error_data1)
        
        # 采样设置
        trace = pm.sample(
            draws=2000,                # 增加采样数
            tune=4000,                 # 增加预热步数
            chains=4,                  # 增加链数
            cores=4,                   # 增加并行核数
            return_inferencedata=True,
            random_seed=42,
            compute_convergence_checks=False,
            progressbar=True,
            step=pm.NUTS(
                target_accept=0.8,
                max_treedepth=15
            )
        )

    print(az.summary(trace))
    # Draw parameter tracking diagram
    # ax = az.plot_trace(trace, var_names=['a'])
    # if hasattr(ax, 'flat'):  # 处理多子图情况
    #     for subplot in ax.flat:
    #         if subplot.get_xlabel():  # 只处理有x轴标签的子图（右侧分布图）
    #             subplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    # plt.savefig('./results/MCMC_a.pdf')

    # # 对其他变量做同样处理
    # ax = az.plot_trace(trace, var_names=['b'])
    # if hasattr(ax, 'flat'):
    #     for subplot in ax.flat:
    #         if subplot.get_xlabel():
    #             subplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    # plt.savefig('./results/MCMC_b.pdf')

    # ax = az.plot_trace(trace, var_names=['c'])
    # if hasattr(ax, 'flat'):
    #     for subplot in ax.flat:
    #         if subplot.get_xlabel():
    #             subplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.5f'))
    # plt.savefig('./results/MCMC_c.pdf')

    # # c3参数单独绘制
    # ax = az.plot_trace(trace, var_names=['a_c3OverC2'])
    # if hasattr(ax, 'flat'):
    #     for subplot in ax.flat:
    #         if subplot.get_xlabel():
    #             subplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    # plt.savefig('./results/MCMC_a_c3OverC2.pdf')

    # ax = az.plot_trace(trace, var_names=['b_c3OverC2'])
    # if hasattr(ax, 'flat'):
    #     for subplot in ax.flat:
    #         if subplot.get_xlabel():
    #             subplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    # plt.savefig('./results/MCMC_b_c3OverC2.pdf')

    # ax = az.plot_trace(trace, var_names=['c_c3OverC2'])
    # if hasattr(ax, 'flat'):
    #     for subplot in ax.flat:
    #         if subplot.get_xlabel():
    #             subplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.5f'))
    # plt.savefig('./results/MCMC_c_c3OverC2.pdf')

    # # c4参数单独绘制
    # ax = az.plot_trace(trace, var_names=['a_c4OverC2'])
    # if hasattr(ax, 'flat'):
    #     for subplot in ax.flat:
    #         if subplot.get_xlabel():
    #             subplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    # plt.savefig('./results/MCMC_a_c4OverC2.pdf')

    # ax = az.plot_trace(trace, var_names=['b_c4OverC2'])
    # if hasattr(ax, 'flat'):
    #     for subplot in ax.flat:
    #         if subplot.get_xlabel():
    #             subplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    # plt.savefig('./results/MCMC_b_c4OverC2.pdf')

    # ax = az.plot_trace(trace, var_names=['c_c4OverC2'])
    # if hasattr(ax, 'flat'):
    #     for subplot in ax.flat:
    #         if subplot.get_xlabel():
    #             subplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.5f'))
    # plt.savefig('./results/MCMC_c_c4OverC2.pdf')

    # # weights参数
    ax = az.plot_trace(trace, var_names=['weights'])
    if hasattr(ax, 'flat'):
        for subplot in ax.flat:
            if subplot.get_xlabel():
                subplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    plt.savefig('./Results/MCMC_weights.pdf')

    # Use the median
    import scipy.stats as stats
    # Add C1 parameters to the list
    var_names = ['a_c3', 'b_c3', 'c_c3', 'd_c3', 'a_c4OverC2', 'b_c4OverC2', 'c_c4OverC2'] 
    params = {}
    for var in var_names:
        # 获取所有链采样值
        samples = trace.posterior[var].values.flatten()
        
        # 使用核密度估计找到峰值
        kde = stats.gaussian_kde(samples)
        # Rename x and y in this scope to avoid shadowing global variables
        x_kde = np.linspace(np.min(samples), np.max(samples), 100) 
        y_kde = kde(x_kde)
        mode = x_kde[np.argmax(y_kde)]
        
        params[var] = mode
        print(f"{var} Peak: {mode:.6f}")

    print("\nFinal parameter estimation (peak):")
    print(params)

    # Recalculate C1 numerically using the estimated parameters
    # C1_numerical = np.array([(params['C1_a']*i**4 + params['C1_b']*i**3 + params['C1_c']*i**2 + params['C1_d']*i + params['C1_e']) for i in range(25)])
    weights = trace.posterior['weights'].mean(axis=(0, 1))
    print(weights)
    weights = np.array(weights)
    # weights = weights / np.sum(weights)
    print(np.sum(weights))


    # import corner
    # corner.corner(
    #     trace,
    #     var_names = ['a', 'b', 'c', 'a_c3OverC2', 'b_c3OverC2', 'c_c3OverC2', 'a_c4OverC2', 'b_c4OverC2', 'c_c4OverC2'],
    #     labels=[r"$a_{c_2}$", r"$b_{c_2}$", r"$c_{c_2}$", r"$a_{c_3/c_2}$", r"$b_{c_3/c_2}$", r"$c_{c_3/c_2}$", r"$a_{c_4/c_2}$", r"$b_{c_4/c_2}$", r"$c_{c_4/c_2}$"],
    #     quantiles=[0.16, 0.5, 0.84],
    #     title_quantiles=[0.16, 0.5, 0.84],
    #     show_titles=True,
    #     title_kwargs={"fontsize": 12},
    #     label_kwargs={"fontsize": 12},
    #     title_fmt='.2e',
    #     figsize=(12, 12),
    #     color='darkblue',
    #     hist_kwargs={'alpha': 0.6},
    #     levels=[0.1, 0.3, 0.5, 0.7, 0.9], 
    #     fill_contours=True,
    #     plot_datapoints=True,
    #     smooth=1.5,
    #     bins=50,
    #     max_n_ticks=4,
    #     truths=params,
    #     truth_color='red',
    #     tick_format='.2e'
    # )
    plt.figure(figsize=(18, 6))
    # C2 拟合
    plt.subplot(2, 3, 1)
    # cumulant_2 = np.array(params['a']) + np.array(params['b']) * np.array(C1) + np.array(params['c']) * np.array(C1)**2
    cumulant_2 = C2_values_corr
    plt.plot(C1, cumulant_2, 'ro--', label=f'Fit({ENERGY:.1f} GeV, UrQMD)', markersize=14, alpha=0.7)
    # plt.plot(CBWC_C1_Npart, CBWC_C2_Npart, 'go--', label='w/ CBWC($N_{part}$)', fillstyle='none', markersize=11, alpha=0.5, markeredgewidth=1.5)
    # plt.plot(Total_C1_Npart, Total_C2_Npart, 'ko--', label='w/o CBWC($N_{part}$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.plot(CBWC_C1_Ref3, CBWC_C2_Ref3, 'gs--', label='w/ CBWC($RefMult3$)', markersize=11, fillstyle='none', alpha=0.5, markeredgewidth=1.5)
    plt.plot(Total_C1_Ref3, Total_C2_Ref3, 'ks--', label='w/o CBWC($RefMult3$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.xlabel('${C_1}$')
    plt.text(0.1, 0.8, '$\mathbf{C_2}$', fontsize=22, fontweight='bold', transform=plt.gca().transAxes)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.ylim(np.min(Total_C2_Ref3)-20, np.max(Total_C2_Ref3)+10)
    plt.legend()
    # C3 拟合
    plt.subplot(2, 3, 2)
    cumulant_3 = (np.array(params['a_c3']) + np.array(params['b_c3']) * np.array(C1) + np.array(params['c_c3']) * np.array(C1)**2 + np.array(params['d_c3']) * np.array(C1)**3)
    plt.plot(C1, cumulant_3, 'ro--', label='MCMC Optimized', markersize=14, alpha=0.7)
    # plt.plot(CBWC_C1_Npart, CBWC_C3_Npart, 'go--', label='w/ CBWC($N_{part}$)', fillstyle='none', markersize=11, alpha=0.5, markeredgewidth=1.5)
    # plt.plot(Total_C1_Npart, Total_C3_Npart, 'ko--', label='w/o CBWC($N_{part}$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.plot(CBWC_C1_Ref3, CBWC_C3_Ref3, 'gs--', label='w/ CBWC($RefMult3$)', markersize=11, fillstyle='none', alpha=0.5, markeredgewidth=1.5)
    plt.plot(Total_C1_Ref3, Total_C3_Ref3, 'ks--', label='w/o CBWC($RefMult3$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.xlabel('${C_1}$')
    plt.text(0.1, 0.8, '$\mathbf{C_3}$', fontsize=22, fontweight='bold', transform=plt.gca().transAxes)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.ylim(np.min(Total_C3_Ref3)-18, np.max(Total_C3_Ref3)+10)



    # C4 拟合
    plt.subplot(2, 3, 3)
    cumulant_4 = (np.array(params['a_c4OverC2']) + np.array(params['b_c4OverC2']) * np.array(C1) + np.array(params['c_c4OverC2']) * np.array(C1)**2) * np.array(cumulant_2)
    plt.plot(C1, cumulant_4, 'ro--', label='MCMC Optimized', markersize=14, alpha=0.7)
    # plt.plot(CBWC_C1_Npart, CBWC_C4_Npart, 'go--', label='w/ CBWC($N_{part}$)', fillstyle='none', markersize=11, alpha=0.5, markeredgewidth=1.5)
    # plt.plot(Total_C1_Npart, Total_C4_Npart, 'ko--', label='w/o CBWC($N_{part}$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.plot(CBWC_C1_Ref3, CBWC_C4_Ref3, 'gs--', label='w/ CBWC($RefMult3$)', markersize=11, fillstyle='none', alpha=0.5, markeredgewidth=1.5)
    plt.plot(Total_C1_Ref3, Total_C4_Ref3, 'ks--', label='w/o CBWC($RefMult3$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.xlabel('${C_1}$')
    plt.text(0.1, 0.8, '$\mathbf{C_4}$', fontsize=22, fontweight='bold', transform=plt.gca().transAxes)
    plt.ylim(np.min(Total_C4_Ref3)-10, np.max(Total_C4_Ref3)+20)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.grid(True, alpha=0.7, linestyle='--')

    # C2/C1
    plt.subplot(2, 3, 4)
    plt.plot(C1, cumulant_2/C1, 'ro--', label='MCMC Optimized', markersize=14, alpha=0.7)
    # plt.plot(CBWC_C1_Npart, CBWC_C2_Npart/CBWC_C1_Npart, 'go--', label='w/ CBWC($N_{part}$)', fillstyle='none', markersize=11, alpha=0.5, markeredgewidth=1.5)
    # plt.plot(Total_C1_Npart, Total_C2_Npart/Total_C1_Npart, 'ko--', label='w/o CBWC($N_{part}$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.plot(CBWC_C1_Ref3, CBWC_C2_Ref3/CBWC_C1_Ref3, 'gs--', label='w/ CBWC($RefMult3$)', markersize=11, fillstyle='none', alpha=0.5, markeredgewidth=1.5)
    plt.plot(Total_C1_Ref3, Total_C2_Ref3/Total_C1_Ref3, 'ks--', label='w/o CBWC($RefMult3$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.xlabel('$\mathbf{C_1}$', fontsize=16, fontweight='bold')
    plt.text(0.1, 0.8, '$\mathbf{C_2/C_1}$', fontsize=22, fontweight='bold', transform=plt.gca().transAxes)
    plt.ylim(np.min(Total_C2_Ref3/Total_C1_Ref3)-0.6, np.max(Total_C2_Ref3/Total_C1_Ref3)+0.6)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.grid(True, alpha=0.7, linestyle='--')

    # C3/C2
    plt.subplot(2, 3, 5)
    plt.plot(C1, cumulant_3/cumulant_2, 'ro--', label='MCMC Optimized', markersize=14, alpha=0.7)
    # plt.plot(CBWC_C1_Npart, CBWC_C3_Npart/CBWC_C2_Npart, 'go--', label='w/ CBWC($N_{part}$)', fillstyle='none', markersize=11, alpha=0.5, markeredgewidth=1.5)
    # plt.plot(Total_C1_Npart, Total_C3_Npart/Total_C2_Npart, 'ko--', label='w/o CBWC($N_{part}$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.plot(CBWC_C1_Ref3, CBWC_C3_Ref3/CBWC_C2_Ref3, 'gs--', label='w/ CBWC($RefMult3$)', markersize=11, fillstyle='none', alpha=0.5, markeredgewidth=1.5)
    plt.plot(Total_C1_Ref3, Total_C3_Ref3/Total_C2_Ref3, 'ks--', label='w/o CBWC($RefMult3$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.xlabel('$\mathbf{C_1}$', fontsize=16, fontweight='bold')
    plt.text(0.1, 0.8, '$\mathbf{C_3/C_2}$', fontsize=22, fontweight='bold', transform=plt.gca().transAxes)
    plt.ylim(np.min(Total_C3_Ref3/Total_C2_Ref3)-0.3, np.max(Total_C3_Ref3/Total_C2_Ref3)+0.8)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.grid(True, alpha=0.7, linestyle='--')

    # C4/C2
    plt.subplot(2, 3, 6)
    plt.plot(C1, cumulant_4/cumulant_2, 'ro--', label='MCMC Optimized', markersize=14, alpha=0.7)
    # plt.plot(CBWC_C1_Npart, CBWC_C4_Npart/CBWC_C2_Npart, 'go--', label='w/ CBWC($N_{part}$)', fillstyle='none', markersize=11, alpha=0.5, markeredgewidth=1.5)
    # plt.plot(Total_C1_Npart, Total_C4_Npart/Total_C2_Npart, 'ko--', label='w/o CBWC($N_{part}$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.plot(CBWC_C1_Ref3, CBWC_C4_Ref3/CBWC_C2_Ref3, 'gs--', label='w/ CBWC($RefMult3$)', markersize=11, fillstyle='none', alpha=0.5, markeredgewidth=1.5)
    plt.plot(Total_C1_Ref3, Total_C4_Ref3/Total_C2_Ref3, 'ks--', label='w/o CBWC($RefMult3$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.xlabel('$\mathbf{C_1}$', fontsize=16, fontweight='bold')
    plt.text(0.1, 0.8, '$\mathbf{C_4/C_2}$', fontsize=22, fontweight='bold', transform=plt.gca().transAxes)
    plt.ylim(np.min(Total_C4_Ref3/Total_C2_Ref3)-1, np.max(Total_C4_Ref3/Total_C2_Ref3)+3)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.subplots_adjust(hspace=0.0)
    plt.savefig('./Results/CumulantsMCMC.pdf')


    C1_raw, C2_raw, C3_raw, C4_raw = Get_Raw_Cumulants_From_CorrectedCumulants(C1_corr, cumulant_2, cumulant_3, cumulant_4)
    plt.figure(figsize=(10, 8))
    plt.yscale('log')
    plt.plot(x, y, 'ko', label=f'Raw data({ENERGY:.1f} GeV, UrQMD)', markersize=16, alpha=0.6)
    y_total = np.zeros(len(x))
    for i in range(len(C1_corr)):
        y_DE_i = weights[i]*edgeworth_pdf(x, C1_raw[i], C2_raw[i], C3_raw[i], C4_raw[i])
        plt.plot(x, y_DE_i, '--', label=f'', alpha=0.6, linewidth=2)    
        y_total += y_DE_i
    plt.plot(x_edge_DE, y_edge_DE, marker='o', label=f'50-100% Distribution', alpha=0.6, markersize=14, fillstyle='none', markeredgewidth=2, markeredgecolor='brown', linewidth=0)
    chi2 = np.sum( ((y_DE[START_FIT:] - y_total[START_FIT:MAX]) / y_error_DE[START_FIT:]) **2 )
    plt.plot(x_edge, y_total + y_edge, 'r-', label=f'Reconstructed($\chi^2$ = {chi2:.4f}$)$', alpha=0.8, linewidth=3)
    plt.legend(fontsize=18)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.xlabel('$\mathbf{N_{proton}}$', fontsize=18)
    plt.ylabel('$\mathbf{Probablity}$', fontsize=18)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.xlim(0,MAX + 10)
    plt.ylim(1e-6, 3e-1)
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.savefig('./Results/MCMC_EdgeworthPDF.pdf')

    with open('./FinalResults/fit_results_mcmc.py', 'w') as f:
        f.write(f"import numpy as np\n\n")
        f.write(f'C1_Fit_corr_mcmc_data = np.array([{", ".join([f"{x:.3f}" for x in C1_corr])}])\n')
        f.write(f'C2_Fit_corr_mcmc_data = np.array([{", ".join([f"{x:.3f}" for x in cumulant_2])}])\n') 
        f.write(f'C3_Fit_corr_mcmc_data = np.array([{", ".join([f"{x:.3f}" for x in cumulant_3])}])\n')
        f.write(f'C4_Fit_corr_mcmc_data = np.array([{", ".join([f"{x:.3f}" for x in cumulant_4])}])\n')

    with open('./FinalResults/weights_mcmc.py', 'w') as f:
        f.write(f"import numpy as np\n\n")
        f.write(f'weights = np.array([{", ".join([f"{x:.4f}" for x in weights])}])\n')

    cleanup(temp_dir)