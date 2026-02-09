import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 读取数据，跳过中心度列
data = np.loadtxt('CBWC_Results_FromRef3/all_results.txt', skiprows=1, usecols=(1,2,3,4,5,6,7,8))

# def Get_Raw_Cumulants_From_CorrectedCumulants(c1_corr, c2_corr, c3_corr, c4_corr):
#     eff = np.array([0.722839, 0.719768, 0.716613, 0.711829, 0.708493, 0.711356, 0.710682, 0.706527, 0.704337, 0.703265, 0.701041, 0.693923, 0.69711, 0.685838, 0.686175, 0.696915, 0.683854, 0.681256, 0.690743, 0.673624, 0.66149, 0.670618, 0.672559])
#     # eff = np.ones(23)
#     eff = eff[:20][::-1]
#     ep1 = 1/eff
#     ep2 = 1/(eff**2)
#     ep3 = 1/(eff**3)
#     ep4 = 1/(eff**4)
#     c1_raw = c1_corr / ep1
#     c2_raw = (c2_corr - (ep1 - ep2) * c1_raw) / ep2
#     c3_raw = (c3_corr - (3*ep2 - 3*ep3) * c2_raw - (ep1 + 2*ep3 - 3*ep2) * c1_raw) / ep3
#     c4_raw = (c4_corr - (6*ep3 - 6*ep4) * c3_raw - (7*ep2 - 18*ep3 + 11*ep4) * c2_raw - (ep1 - 7*ep2 + 12*ep3 - 6*ep4) * c1_raw) / ep4
#     return c1_raw, c2_raw, c3_raw, c4_raw

# def GetCorrCum(C1_uncorr, C2_uncorr, C3_uncorr, C4_uncorr):
#     eff = np.array([0.722839, 0.719768, 0.716613, 0.711829, 0.708493, 0.711356, 0.710682, 0.706527, 0.704337, 0.703265, 0.701041, 0.693923, 0.69711, 0.685838, 0.686175, 0.696915, 0.683854, 0.681256, 0.690743, 0.673624, 0.66149, 0.670618, 0.672559])
#     eff = eff[:20][::-1]
#     ep1 = 1/eff
#     ep2 = 1/(eff**2)
#     ep3 = 1/(eff**3)
#     ep4 = 1/(eff**4)
#     Corr_C1 = C1_uncorr * ep1
#     Corr_C2 = C2_uncorr * ep2 + (ep1 - ep2) * C1_uncorr
#     Corr_C3 = C3_uncorr * ep3 + (ep2*3 - ep3*3) * C2_uncorr + (ep1 + ep3*2 - ep2*3) * C1_uncorr
#     Corr_C4 = C4_uncorr * ep4 + (ep3*6 - ep4*6) * C3_uncorr + (ep2*7 - ep3*18 + ep4*11) * C2_uncorr + (ep1 - ep2*7 + 12*ep3 - 6*ep4) * C1_uncorr
#     return Corr_C1, Corr_C2, Corr_C3, Corr_C4

# # 提取total数据
# # total_c1 = data[:, 0]
# total_c2 = data[:, 1][1:]
# total_c3 = data[:, 2][1:]
# total_c4 = data[:, 3][1:]

# # 提取CBWC数据
# # cbwc_c1 = data[:, 4]  # CBWC_C1
# cbwc_c2 = data[:, 5][1:]  # CBWC_C2
# cbwc_c3 = data[:, 6][1:]  # CBWC_C3
# cbwc_c4 = data[:, 7][1:]  # CBWC_C4
# C1 = np.array([1.706009, 2.016164, 2.340801, 2.672210, 3.079910, 3.501734, 3.991376, 4.577641, 5.170628, 5.773534, 6.453423, 7.230661, 8.079252, 9.033314, 10.054685, 11.172378, 12.366073, 13.804772, 15.400283, 17.139706])
# cbwc_c1 = C1
# cbwc_c1, cbwc_c2, cbwc_c3, cbwc_c4 = GetCorrCum(cbwc_c1, cbwc_c2, cbwc_c3, cbwc_c4)
# total_c1, total_c2, total_c3, total_c4 = GetCorrCum(C1, total_c2, total_c3, total_c4)
# # total_c1 = cbwc_c1
# # total_c2 = cbwc_c2
# # total_c3 = cbwc_c3
# # total_c4 = cbwc_c4
# print(total_c3[-1], max(total_c3))
# print(','.join(str(x) for x in cbwc_c1))
# exit()

# 定义拟合函数
def fit_func0(x, a, b):
    return a + b*x
def fit_func(x, a, b, c):
    return a  + b * x + c*x**2
def fit_func2(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3
def fit_func3(x, a, b, c, d, e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4
def fit_func4(x, a, b, c):
    return a + b*x + c*x**1.5

# print(total_c3[-1], max(total_c3))
# exit()
# 创建图形
# plt.figure(figsize=(15, 10))
# # C1 = np.array([0.768415, 0.868449, 1.027279, 1.246344, 1.482928, 1.725878, 2.042025, 2.369142, 2.706297, 3.118013, 3.543978, 4.042858, 4.637110, 5.233363, 5.843194, 6.535456, 7.321136, 8.185110, 9.147204, 10.175915, 11.310304, 12.522497, 13.976743, 15.580159, 17.314582])
# # C1 = np.array([2.764298,3.297053,3.849795,4.520912,5.327302,6.147643,7.100291,8.074384,9.176033,10.541130,11.931250,13.446866,15.237842,17.038094,19.093597,21.533042,24.098234,27.003818,30.417728,34.090562])
# total_c1 = cbwc_c1
# # cbwc_c1 = C1
# # 拟合和绘制C2 vs C1
# plt.subplot(341)
# selected_indices = [0, 1, 2, -1]
# selected_x = total_c1[selected_indices]
# selected_y = total_c2[selected_indices]
# popt_c2, pcov_c2 = curve_fit(fit_func, selected_x, selected_y)
# fit_y_c2 = fit_func(total_c1, *popt_c2)
# plt.plot(total_c1, total_c2, 'bo--', markeredgecolor='blue', fillstyle='none', markersize=10, alpha=0.9, markeredgewidth=1.5, label='w/o CBWC')
# # plt.plot([total_c1[4], total_c1[-1]], [total_c2[4], total_c2[-1]], 'r-', linewidth=2, label='')
# plt.plot(cbwc_c1, cbwc_c2, 'ro--', alpha=0.5, linewidth=3, markersize=10, label='w/ CBWC')
# plt.plot(cbwc_c1, fit_y_c2, 'r--', alpha=0.8, linewidth=3)

# param_variations = []
# for scale in [0.9, 1.1]:
#     varied_params = popt_c2 * scale
#     varied_fit = fit_func(total_c1, *varied_params)
#     param_variations.append(varied_fit)
#     plt.plot(total_c1, varied_fit, 'g--', alpha=0.3)
#     print(f"Parameters for scale {scale}:")
#     print(f"a = {varied_params[0]:.3f}")
#     print(f"b = {varied_params[1]:.3f}") 
#     print(f"c = {varied_params[2]:.3f}")

# # Fill area between parameter variations
# plt.fill_between(total_c1, param_variations[0], param_variations[1], color='green', alpha=0.1)

# plt.xlabel('C1')
# plt.ylabel('C2')
# plt.title('C2 vs C1')
# plt.grid(True, alpha=0.3)
# plt.legend()

# plt.subplot(342)
# # Select first two points and last point for fitting
# selected_indices = [0, 1, 2, -1]
# selected_x = total_c1[selected_indices]
# selected_y = total_c3[selected_indices]
# selected_y[-1] = 22.29

# # Adjust initial parameter guesses to control quadratic term
# # Set bounds to ensure quadratic term > -0.01
# bounds = ([-np.inf, -np.inf, -0.2], [np.inf, np.inf, np.inf])
# p0 = [1.0, 1.0, 0.0]  # Initial guess with moderate quadratic coefficient
# popt_c3, pcov_c3 = curve_fit(fit_func, selected_x, selected_y, p0=p0, bounds=bounds)
# fit_y_c3 = fit_func(total_c1, *popt_c3)

# # Plot original data points
# plt.plot(total_c1, total_c3, 'bo--', markeredgecolor='blue', fillstyle='none', markersize=10, alpha=0.9, markeredgewidth=1.5, label='w/o CBWC')
# plt.plot(cbwc_c1, cbwc_c3, 'ro--', alpha=0.5, linewidth=3, markersize=10, label='w/ CBWC')

# # Plot best fit
# plt.plot(cbwc_c1, fit_y_c3, 'r--', alpha=0.9, linewidth=4)

# # Plot parameter variations with adjusted scales
# param_variations = []
# for scale in [0.8, 1.2]:  # More moderate scaling factors
#     varied_params = popt_c3.copy()
#     varied_params[0] *= scale  # Scale a parameter
#     varied_params[1] *= scale  # Scale b parameter 
#     varied_params[2] *= scale  # Scale c parameter with minimum bound
#     varied_fit = fit_func(total_c1, *varied_params)
#     param_variations.append(varied_fit)
#     plt.plot(total_c1, varied_fit, 'g--', alpha=0.3)
#     print(f"Parameters for scale {scale}:")
#     print(f"a = {varied_params[0]:.3f}")
#     print(f"b = {varied_params[1]:.3f}") 
#     print(f"c = {varied_params[2]:.3f}")

# # Fill area between parameter variations
# plt.fill_between(total_c1, param_variations[0], param_variations[1], color='green', alpha=0.1)

# plt.xlabel('C1')
# plt.ylabel('C3')
# plt.title('C3 vs C1')
# plt.grid(True, alpha=0.3)
# plt.legend()

# plt.subplot(343)
# popt_c4, pcov_c4 = curve_fit(fit_func3, total_c1, total_c4)
# fit_y_c4 = fit_func3(total_c1, *popt_c4)
# plt.plot(total_c1, total_c4, 'bo--', markeredgecolor='blue', fillstyle='none', markersize=10, alpha=0.9, markeredgewidth=1.5, label='w/o CBWC')
# plt.plot(cbwc_c1, cbwc_c4, 'ro--', alpha=0.5, linewidth=3, markersize=10, label='w/ CBWC')
# plt.plot(cbwc_c1, fit_y_c4, 'r--', alpha=0.8, linewidth=3)
# plt.xlabel('C1')
# plt.ylabel('C4')
# plt.title('C4 vs C1')
# plt.grid(True, alpha=0.3)
# # plt.ylim(4, 20)
# plt.legend()

# plt.subplot(344)
# popt_c2_c1, pcov_c2_c1 = curve_fit(fit_func, total_c1, total_c2/total_c1)
# fit_y_c2_c1 = fit_func(total_c1, *popt_c2_c1)
# plt.plot(total_c1, total_c2/total_c1, 'bo--', markeredgecolor='blue', fillstyle='none', markersize=10, alpha=0.9, markeredgewidth=1.5, label='w/o CBWC')
# # plt.plot([total_c1[5], total_c1[-1]], [total_c2[5]/total_c1[5], total_c2[-1]/total_c1[-1]], 'r--', linewidth=3, alpha=0.8, label='')
# plt.plot(cbwc_c1, cbwc_c2/cbwc_c1, 'ro--', alpha=0.5, linewidth=3, markersize=10, label='w/ CBWC')
# plt.plot(cbwc_c1, fit_y_c2_c1, 'r--', alpha=0.8, linewidth=3)
# plt.xlabel('C1')
# plt.ylabel('C2/C1')
# plt.title('C2/C1 VS C1')
# plt.grid(True, alpha=0.3)
# plt.legend()
# # plt.ylim(0.6, 1.3)

# plt.subplot(345)
# popt_c3_c2, pcov_c3_c2 = curve_fit(fit_func2, total_c1, total_c3/total_c2)
# fit_y_c3_c2 = fit_func2(total_c1, *popt_c3_c2)
# plt.plot(total_c1, total_c3/total_c2, 'bo--', markeredgecolor='blue', fillstyle='none', markersize=10, alpha=0.9, markeredgewidth=1.5, label='w/o CBWC')
# # plt.plot([total_c1[0], total_c1[-1]], [total_c3[0]/total_c2[0], total_c3[-1]/total_c2[-1]], 'r--', linewidth=3, alpha=0.8, label='')
# plt.plot(cbwc_c1, cbwc_c3/cbwc_c2, 'ro--', alpha=0.5, linewidth=3, markersize=10, label='w/ CBWC')
# plt.xlabel('C1')
# plt.ylabel('C3/C2')
# plt.title('C3/C2 VS C1')
# plt.grid(True, alpha=0.3)
# plt.legend()

# plt.subplot(346)
# popt_c3_c1, pcov_c3_c1 = curve_fit(fit_func, total_c1, total_c3/total_c1)
# fit_y_c3_c1 = fit_func(total_c1, *popt_c3_c1)
# plt.plot(cbwc_c1, cbwc_c3/cbwc_c1, 'ro--', alpha=0.5, linewidth=3, markersize=10, label='w/ CBWC')
# plt.plot(total_c1, total_c3/total_c1, 'bo--', markeredgecolor='blue', fillstyle='none', markersize=10, alpha=0.9, markeredgewidth=1.5, label='w/o CBWC')
# # plt.plot([total_c1[0], total_c1[-1]], [total_c3[0]/total_c1[0], total_c3[-1]/total_c1[-1]], 'r--', linewidth=2, alpha=0.6, label='')
# plt.plot(total_c1, fit_y_c3_c1, 'r-', alpha=0.8, label=f'Fit: {popt_c3_c1[0]:.2e} + {popt_c3_c1[1]:.2f}x + {popt_c3_c1[2]:.2f}x^2')
# plt.xlabel('C1')
# plt.ylabel('C3/C1')
# plt.title('C3/C1 VS C1')
# plt.grid(True, alpha=0.3)
# plt.legend()

# plt.subplot(347)
# popt_c4_c1, pcov_c4_c1 = curve_fit(fit_func, total_c1, total_c4/total_c1)
# fit_y_c4_c1 = fit_func(total_c1, *popt_c4_c1)
# plt.plot(total_c1, total_c4/total_c1, 'bo--', markeredgecolor='blue', fillstyle='none', markersize=10, alpha=0.9, markeredgewidth=1.5, label='')
# plt.plot(cbwc_c1, cbwc_c4/cbwc_c1, 'ro--', alpha=0.5, linewidth=3, markersize=10, label='w/ CBWC')
# # plt.plot([total_c1[0], total_c1[-1]], [total_c4[0]/total_c1[0], total_c4[-1]/total_c1[-1]], 'r--', linewidth=2, alpha=0.6, label='')
# plt.plot(total_c1, fit_y_c4_c1, 'r-', alpha=0.8, label=f'Fit: {popt_c4_c1[0]:.2e} + {popt_c4_c1[1]:.2f}x + {popt_c4_c1[2]:.2f}x^2')
# plt.xlabel('C1')
# plt.ylabel('C4/C1')
# plt.title('C4/C1 VS C1')
# plt.grid(True, alpha=0.3)
# plt.legend()

# plt.subplot(348)
# popt_c4_c2, pcov_c4_c2 = curve_fit(fit_func, total_c1, total_c4/total_c2)
# fit_y_c4_c2 = fit_func(total_c1, *popt_c4_c2)
# plt.plot(total_c1, total_c4/total_c2, 'bo--', markeredgecolor='blue', fillstyle='none', markersize=10, alpha=0.9, markeredgewidth=1.5, label='w/o CBWC')
# plt.plot(cbwc_c1, cbwc_c4/cbwc_c2, 'ro--', alpha=0.5, linewidth=3, markersize=10, label='w/ CBWC')
# # plt.plot([total_c1[0], total_c1[-1]], [total_c4[0]/total_c2[0], total_c4[-1]/total_c2[-1]], 'r--', linewidth=3, alpha=0.8, label='')
# plt.plot(total_c1, fit_y_c4_c2, 'r-', alpha=0.8, label=f'Fit: {popt_c4_c2[0]:.2e} + {popt_c4_c2[1]:.2f}x + {popt_c4_c2[2]:.2f}x^2')
# plt.xlabel('C1')
# plt.ylabel('C4/C2')
# plt.title('C4/C2 VS C1')
# plt.grid(True, alpha=0.3)
# plt.legend()
# # C4 / C3
# # plt.subplot(449)
# # popt_c4_c3, pcov_c4_c3 = curve_fit(fit_func0, total_c1, total_c4/total_c3)
# # fit_y_c4_c3 = fit_func0(total_c1, *popt_c4_c3)
# # plt.scatter(cbwc_c1, cbwc_c4/cbwc_c3, c='red', alpha=0.6, label='CBWC')
# # plt.scatter(total_c1, total_c4/total_c3, c='blue', alpha=0.6, label='Total')
# # plt.plot(total_c1, fit_y_c4_c3, 'r-', alpha=0.8, label=f'Fit: {popt_c4_c3[0]:.2e} + {popt_c4_c3[1]:.2f}x')
# # plt.xlabel('C1 (CBWC)')
# # plt.ylabel('C4 / C3')
# # plt.title('C4 / C3')
# # plt.grid(True, alpha=0.3)
# # plt.legend()


# # C4/C2 VS C2/C1
# plt.subplot(3,4,9)
# popt_c4_c2_c1, pcov_c4_c2_c1 = curve_fit(fit_func, total_c1/total_c2, total_c4/total_c2)
# fit_y_c4_c2_c1 = fit_func(total_c1/total_c2, *popt_c4_c2_c1)
# # plt.scatter(cbwc_c1/cbwc_c2, cbwc_c4/cbwc_c2, c='red', alpha=0.6, label='w/ CBWC')
# plt.plot(total_c1/total_c2, total_c4/total_c2, 'bo--', markeredgecolor='blue', fillstyle='none', markersize=10, alpha=0.9, markeredgewidth=1.5, label='w/o CBWC')
# # plt.plot([total_c1[10]/total_c2[10], total_c1[-1]/total_c2[-1]], [total_c4[10]/total_c2[10], total_c4[-1]/total_c2[-1]], 'r--', linewidth=3, alpha=0.8, label='')
# plt.plot(cbwc_c1/cbwc_c2, cbwc_c4/cbwc_c2, 'ro--', alpha=0.5, linewidth=3, markersize=10, label='w/ CBWC')
# plt.plot(total_c1/total_c2, fit_y_c4_c2_c1, 'r-', alpha=0.8, label=f'Fit: {popt_c4_c2_c1[0]:.2e} + {popt_c4_c2_c1[1]:.2f}x + {popt_c4_c2_c1[2]:.2f}x^2')
# plt.xlabel('C1/C2')
# plt.ylabel('C4/C2')
# plt.title('C4/C2 VS C1/C2')
# plt.grid(True, alpha=0.3)
# plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# plt.legend()

# # C3/C2 VS C1/C2
# plt.subplot(3,4,10)
# popt_c3_c2_c1, pcov_c3_c2_c1 = curve_fit(fit_func, total_c1/total_c2, total_c3/total_c2)
# fit_y_c3_c2_c1 = fit_func(total_c1/total_c2, *popt_c3_c2_c1)
# plt.plot(cbwc_c1/cbwc_c2, cbwc_c3/cbwc_c2, 'ro--', alpha=0.5, linewidth=3, markersize=10, label='w/ CBWC')
# plt.plot(total_c1/total_c2, total_c3/total_c2, 'bo--', markeredgecolor='blue', fillstyle='none', markersize=10, alpha=0.9, markeredgewidth=1.5, label='w/o CBWC')
# # plt.plot([total_c1[10]/total_c2[10], total_c1[-1]/total_c2[-1]], [total_c3[10]/total_c2[10], total_c3[-1]/total_c2[-1]], 'r--', linewidth=3, alpha=0.8, label='')
# plt.plot(total_c1/total_c2, fit_y_c3_c2_c1, 'r-', alpha=0.8, label=f'Fit: {popt_c3_c2_c1[0]:.2e} + {popt_c3_c2_c1[1]:.2f}x + {popt_c3_c2_c1[2]:.2f}x^2')
# plt.xlabel('C1/C2')
# plt.ylabel('C3/C2')
# plt.title('C3/C2 VS C1/C2')
# plt.grid(True, alpha=0.3)
# plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# plt.legend()

# # C4/C2 VS C4/C1    
# plt.subplot(3,4,11)
# popt_c4_c2_c1, pcov_c4_c2_c1 = curve_fit(fit_func, total_c4/total_c2, total_c4/total_c1)
# fit_y_c4_c2_c1 = fit_func(total_c4/total_c2, *popt_c4_c2_c1)
# plt.plot(cbwc_c4/cbwc_c1, cbwc_c4/cbwc_c2, 'ro--', alpha=0.5, linewidth=3, markersize=10, label='w/ CBWC')
# plt.plot(total_c4/total_c1, total_c4/total_c2, 'bo--', markeredgecolor='blue', fillstyle='none', markersize=10, alpha=0.9, markeredgewidth=1.5, label='w/o CBWC')
# # plt.plot([total_c4[5]/total_c1[5], total_c4[-1]/total_c1[-1]], [total_c4[5]/total_c2[5], total_c4[-1]/total_c2[-1]], 'r--', linewidth=3, alpha=0.8, label='')
# plt.plot(total_c4/total_c2, fit_y_c4_c2_c1, 'r-', alpha=0.8, label=f'Fit: {popt_c4_c2_c1[0]:.2e} + {popt_c4_c2_c1[1]:.2f}x + {popt_c4_c2_c1[2]:.2f}x^2')
# plt.xlabel('C4/C1', fontsize=10)
# plt.ylabel('C4/C2', fontsize=10)
# plt.title('C4/C2 VS C4/C1')
# plt.grid(True, alpha=0.3)
# plt.legend()

# # C3/C2 VS C3/C1
# plt.subplot(3,4,12)
# popt_c3_c2_c1, pcov_c3_c2_c1 = curve_fit(fit_func, total_c3/total_c2, total_c3/total_c1)
# fit_y_c3_c2_c1 = fit_func(total_c3/total_c2, *popt_c3_c2_c1)
# plt.plot(cbwc_c3/cbwc_c1, cbwc_c3/cbwc_c2, 'ro--', alpha=0.5, linewidth=3, markersize=10, label='w/ CBWC')
# plt.plot(total_c3/total_c1, total_c3/total_c2, 'bo--', markeredgecolor='blue', fillstyle='none', markersize=10, alpha=0.9, markeredgewidth=1.5, label='w/o CBWC')
# # plt.plot([total_c3[5]/total_c1[5], total_c3[-1]/total_c1[-1]], [total_c3[5]/total_c2[5], total_c3[-1]/total_c2[-1]], 'r--', linewidth=3, alpha=0.8, label='')
# plt.plot(total_c3/total_c2, fit_y_c3_c2_c1, 'r-', alpha=0.8, label=f'Fit: {popt_c3_c2_c1[0]:.2e} + {popt_c3_c2_c1[1]:.2f}x + {popt_c3_c2_c1[2]:.2f}x^2')
# plt.xlabel('C3/C1')
# plt.ylabel('C3/C2')
# plt.title('C3/C2 VS C3/C1')
# plt.grid(True, alpha=0.3)
# plt.legend()


# # C4/C2 VS C3/C1
# # plt.subplot(4,4,14)
# # popt_c4_c2_c3, pcov_c4_c2_c3 = curve_fit(fit_func, total_c3/total_c1, total_c4/total_c1)
# # fit_y_c4_c2_c3 = fit_func(total_c3/total_c1, *popt_c4_c2_c3)
# # plt.scatter(cbwc_c3/cbwc_c1, cbwc_c4/cbwc_c1, c='red', alpha=0.6, label='CBWC')
# # plt.scatter(total_c3/total_c1, total_c4/total_c1, c='blue', alpha=0.6, label='Total')
# # plt.plot(total_c3/total_c1, fit_y_c4_c2_c3, 'r-', alpha=0.8, label=f'Fit: {popt_c4_c2_c3[0]:.2e} + {popt_c4_c2_c3[1]:.2f}x + {popt_c4_c2_c3[2]:.2f}x^2')
# # plt.xlabel('C3/C1 (CBWC)')
# # plt.ylabel('C4/C1')
# # plt.title('C4/C2 VS C3/C1')
# # plt.grid(True, alpha=0.3)
# # plt.legend()


# plt.tight_layout()
# plt.savefig('./PaperFIG/Forth_MomentsRelation_with_fits_Ref3_WithCBWC.pdf')
# plt.show()

# # exit()



# # 打印拟合参数和相关系数
# print("\n拟合参数 (ax² + bx + c):")
# print(f"C2: {popt_c2[0]:.6f} + {popt_c2[1]:.6f}*C1 + {popt_c2[2]:.6f}*C1^2")
# print(f"C3: {popt_c3[0]:.6f} + {popt_c3[1]:.6f}*C1 + {popt_c3[2]:.6f}*C1^2")
# print(f"C4: {popt_c4[0]:.6f} + {popt_c4[1]:.6f}*C1 + {popt_c4[2]:.6f}*C1^2")

# print(f"C2/C1: {popt_c2_c1[0]:.6f} + {popt_c2_c1[1]:.6f}*C1 + {popt_c2_c1[2]:.6f}*C1^2")
# print(f"C3/C2: {popt_c3_c2[0]:.6f} + {popt_c3_c2[1]:.6f}*C1 + {popt_c3_c2[2]:.6f}*C1^2")
# print(f"C4/C2: {popt_c4_c2[0]:.6f} + {popt_c4_c2[1]:.6f}*C1 + {popt_c4_c2[2]:.6f}*C1^2")


# # exit()
# # 计算R²
# def r_squared(y_true, y_pred):
#     ss_res = np.sum((y_true - y_pred) ** 2)
#     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#     return 1 - (ss_res / ss_tot)

# r2_c2 = r_squared(cbwc_c2, fit_y_c2)
# r2_c3 = r_squared(cbwc_c3, fit_y_c3)
# r2_c4 = r_squared(cbwc_c4, fit_y_c4)

# print("\nR² 值:")
# print(f"C2: {r2_c2:.6f}")
# print(f"C3: {r2_c3:.6f}")
# print(f"C4: {r2_c4:.6f}")

# 保存结果到文件
# with open('CBWC_Fit_Results.txt', 'w') as f:
#     print("\n拟合参数 (ax² + bx + c):")
#     f.write(f"C2: {popt_c2[0]:.6f} + {popt_c2[1]:.6f}*C1 + {popt_c2[2]:.6f}*C1^2\n")
#     f.write(f"C3: {popt_c3[0]:.6f} + {popt_c3[1]:.6f}*C1 + {popt_c3[2]:.6f}*C1^2\n")
#     f.write(f"C4: {popt_c4[0]:.6f} + {popt_c4[1]:.6f}*C1 + {popt_c4[2]:.6f}*C1^2 + {popt_c4[3]:.6f}*C1^3\n")

#     f.write(f"C2/C1: {popt_c2_c1[0]:.6f} + {popt_c2_c1[1]:.6f}*C1 + {popt_c2_c1[2]:.6f}*C1^2\n")
#     f.write(f"C3/C2: {popt_c3_c2[0]:.6f} + {popt_c3_c2[1]:.6f}*C1 + {popt_c3_c2[2]:.6f}*C1^2\n")
#     f.write(f"C4/C2: {popt_c4_c2[0]:.6f} + {popt_c4_c2[1]:.6f}*C1 + {popt_c4_c2[2]:.6f}*C1^2\n")
from CentBin import bin_ref3
bin_ref3[:-1] = bin_ref3[:-1] + 1
plt.figure(figsize=(10, 6))
# 读取第一个文件来获取x值
first_data = np.loadtxt(f"RemoveLowEvents_ProDist_FromRef3/Proton_Total_Cent{bin_ref3[0]}_{bin_ref3[1]}.txt")
x = first_data[:, 0]
y_total = np.zeros_like(x)
y_total_error = np.zeros_like(x)
scale = []
# 直接累加每个文件的y值
for i in range(len(bin_ref3)-1):
    data = np.loadtxt(f"RemoveLowEvents_ProDist_FromRef3/Proton_Total_Cent{bin_ref3[i]}_{bin_ref3[i+1]}.txt")
    y = data[:, 1]  
    y_error = data[:, 2] / np.sum(y)
    # y = y/np.sum(y)  # 归一化
    y_total += y
    y_total_error += (y_error)**2
    scale.append(np.sum(y))
    plt.errorbar(x, y, yerr=y_error, fmt='--', alpha=0.6, linewidth=1, label=f'')
y_total_error = np.sqrt(y_total_error)
plt.errorbar(x, y_total, yerr=y_total_error, fmt='bo-', fillstyle='none', linewidth=2, label='', capsize=3, markersize=10)
y_total = y_total/np.sum(y_total)
scale = np.array(scale) / np.sum(scale)
# 绘制总和
plt.xlabel('Proton Number', fontsize=14, fontweight='bold')
plt.ylabel('Counts', fontsize=14, fontweight='bold')
plt.title('Proton Distribution in each centrality from RefMult3', fontsize=16, fontweight='bold', pad=15)
plt.yscale('log')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, framealpha=0.8, loc='best')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0, 55)
plt.ylim(100,2e7)
plt.tight_layout()
plt.savefig('./PaperFIG/Forth_ProtonDistribution_FromRef3.pdf', dpi=300, bbox_inches='tight')
# plt.show()
for i in range(len(bin_ref3)-1):
    print(f'scale{i} =  {scale[i]:.4f}')

print(np.sum(y_total))
mean = np.mean(scale[:20])
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(bin_ref3)-1), scale, 'ro--', markersize=10)
plt.plot(np.arange(len(bin_ref3)-1), np.ones(len(bin_ref3)-1)*mean, 'b--', linewidth=2, label=f'Average = {mean:.4f}, \nWeght(0-2.5%)/Average = {scale[0]/mean:.4f}')
plt.xlabel('Centrality Bin Index', fontsize=14, fontweight='bold')
plt.ylabel('Scale Factor', fontsize=14, fontweight='bold')  
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=16, framealpha=0.8, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.ylim(scale[0]*0.5, scale[0]*1.5)
plt.savefig('./PaperFIG/Forth_ScaleFactor_FromRef3.pdf', dpi=300, bbox_inches='tight')
# plt.show()