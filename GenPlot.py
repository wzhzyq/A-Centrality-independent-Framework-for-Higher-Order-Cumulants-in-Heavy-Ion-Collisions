import matplotlib.pyplot as plt
import numpy as np
def GenPlot(C1_corr, C2_corr, C3_corr, C4_corr, weights, output_dir, ENERGY, 
            CBWC_C1_Ref3, CBWC_C2_Ref3, CBWC_C3_Ref3, CBWC_C4_Ref3, Total_C1_Ref3, Total_C2_Ref3, Total_C3_Ref3, Total_C4_Ref3):
    plt.figure(figsize=(18, 6))
    # C2 拟合
    plt.subplot(2, 3, 1)
    plt.plot(C1_corr, C2_corr, 'ro--', label=f'Fit({ENERGY:.1f}GeV, Data, DE)', markersize=14, alpha=0.7)
    # plt.errorbar(CBWC_C1_Npart, CBWC_C2_Npart, yerr=0, fmt='go--', label='w/ CBWC($N_{part}$)', fillstyle='none', markersize=11, alpha=0.5, markeredgewidth=1.5)
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
    plt.ylim(np.min(Total_C2_Ref3)-5, np.max(Total_C2_Ref3)+10)
    plt.legend()
    # C3 拟合
    plt.subplot(2, 3, 2)
    plt.plot(C1_corr, C3_corr, 'ro--', label='DE Optimized', markersize=14, alpha=0.7)
    # plt.errorbar(CBWC_C1_Npart, CBWC_C3_Npart, 0, fmt='go--', label='w/ CBWC($N_{part}$)', fillstyle='none', markersize=11, alpha=0.5, markeredgewidth=1.5)
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
    plt.ylim(np.min(Total_C3_Ref3)-5, np.max(Total_C3_Ref3)+10)

    # C4 拟合
    plt.subplot(2, 3, 3)
    plt.plot(C1_corr, C4_corr, 'ro--', label='DE Optimized', markersize=14, alpha=0.7)
    # plt.errorbar(CBWC_C1_Npart, CBWC_C4_Npart, 0, fmt='go--', label='w/ CBWC($N_{part}$)', fillstyle='none', markersize=11, alpha=0.5, markeredgewidth=1.5)
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
    plt.plot(C1_corr, C2_corr/C1_corr, 'ro--', label='DE Optimized', markersize=14, alpha=0.7)
    # plt.errorbar(CBWC_C1_Npart, Total_C2_Npart/Total_C1_Npart, 0, fmt='go--', label='w/ CBWC($N_{part}$)', fillstyle='none', markersize=11, alpha=0.5, markeredgewidth=1.5)
    # plt.plot(Total_C1_Npart, Total_C2_Npart/Total_C1_Npart, 'ko--', label='w/o CBWC($N_{part}$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.plot(CBWC_C1_Ref3, CBWC_C2_Ref3/CBWC_C1_Ref3, 'gs--', label='w/ CBWC($RefMult3$)', markersize=11, fillstyle='none', alpha=0.5, markeredgewidth=1.5)
    plt.plot(Total_C1_Ref3, Total_C2_Ref3/Total_C1_Ref3, 'ks--', label='w/o CBWC($RefMult3$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.xlabel('$\mathbf{C_1}$', fontsize=16, fontweight='bold')
    plt.text(0.1, 0.8, '$\mathbf{C_2/C_1}$', fontsize=22, fontweight='bold', transform=plt.gca().transAxes)
    plt.ylim(np.min(Total_C2_Ref3/Total_C1_Ref3)-0.2, np.max(Total_C2_Ref3/Total_C1_Ref3)+0.2)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.grid(True, alpha=0.7, linestyle='--')

    # C3/C2
    plt.subplot(2, 3, 5)
    plt.plot(C1_corr, C3_corr/C1_corr, 'ro--', label='DE Optimized', markersize=14, alpha=0.7) 
    # plt.errorbar(CBWC_C1_Npart, CBWC_C3_Npart/CBWC_C1_Npart, 0, fmt='go--', label='w/ CBWC($N_{part}$)', fillstyle='none', markersize=11, alpha=0.5, markeredgewidth=1.5)
    # plt.plot(Total_C1_Npart, Total_C3_Npart/Total_C1_Npart, 'ko--', label='w/o CBWC($N_{part}$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.plot(CBWC_C1_Ref3, CBWC_C3_Ref3/CBWC_C1_Ref3, 'gs--', label='w/ CBWC($RefMult3$)', markersize=11, fillstyle='none', alpha=0.5, markeredgewidth=1.5)
    plt.plot(Total_C1_Ref3, Total_C3_Ref3/Total_C1_Ref3, 'ks--', label='w/o CBWC($RefMult3$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.xlabel('$\mathbf{C_1}$', fontsize=16, fontweight='bold')
    plt.text(0.1, 0.8, '$\mathbf{C_3/C_1}$', fontsize=22, fontweight='bold', transform=plt.gca().transAxes)
    plt.ylim(np.min(Total_C3_Ref3/Total_C2_Ref3)-0.2, np.max(Total_C3_Ref3/Total_C2_Ref3)+0.2)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.grid(True, alpha=0.7, linestyle='--')

    # C4/C2
    plt.subplot(2, 3, 6)
    plt.plot(C1_corr, C4_corr/C2_corr, 'ro--', label='DE Optimized', markersize=14, alpha=0.7)
    # plt.errorbar(CBWC_C1_Npart, CBWC_C4_Npart/CBWC_C2_Npart, 0, fmt='go--', label='w/ CBWC($N_{part}$)', fillstyle='none', markersize=11, alpha=0.5, markeredgewidth=1.5)
    # plt.plot(Total_C1_Npart, Total_C4_Npart/Total_C2_Npart, 'ko--', label='w/o CBWC($N_{part}$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.plot(CBWC_C1_Ref3, CBWC_C4_Ref3/CBWC_C2_Ref3, 'gs--', label='w/ CBWC($RefMult3$)', markersize=11, fillstyle='none', alpha=0.5, markeredgewidth=1.5)
    plt.plot(Total_C1_Ref3, Total_C4_Ref3/Total_C2_Ref3, 'ks--', label='w/o CBWC($RefMult3$)', markersize=15, alpha=0.6, fillstyle='none', markerfacecolor='none', markeredgewidth=1.2)
    plt.xlabel('$\mathbf{C_1}$', fontsize=16, fontweight='bold')
    plt.text(0.1, 0.8, '$\mathbf{C_4/C_2}$', fontsize=22, fontweight='bold', transform=plt.gca().transAxes)
    plt.ylim(np.min(Total_C4_Ref3/Total_C2_Ref3)-0.8, np.max(Total_C4_Ref3/Total_C2_Ref3)+1.0)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.subplots_adjust(hspace=0.0)

    # plt.tight_layout()
    plt.savefig(f'./Results/{output_dir}.pdf')

def cleanup(temp_dir):
    import atexit
    import shutil
    def cleanup():
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    atexit.register(cleanup)
    plt.close('all')
