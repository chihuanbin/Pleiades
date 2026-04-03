import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)  # 可复现性

# ==========================================
# 1. 数据加载与清洗（含距离修正）
# ==========================================
def load_and_clean_data(filepath):
    print(f">>> Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep=',', engine='python', skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(axis=1, how='all')
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    numeric_cols = ['ra', 'dec', 'parallax', 'parallax_error', 'pmra', 'pmra_error',
                    'pmdec', 'pmdec_error', 'ruwe', 'M1', 'q', 'Pb']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    required = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'M1', 'Pb']
    df = df.dropna(subset=required)
    df = df[df['parallax'] > 0]

    # 贝叶斯修正视差 (Liu+2025 附录)
    pi_c = 7.41          # 集群平均视差 (mas)
    sigma_pi_c = 0.17    # 内在弥散 (mas)
    w_obs = df['parallax'].values
    sigma_obs = df.get('parallax_error', 0.1 * np.ones_like(w_obs))
    w_adj = (pi_c / sigma_pi_c**2 + w_obs / sigma_obs**2) / (1/sigma_pi_c**2 + 1/sigma_obs**2)
    df['parallax_adj'] = w_adj
    df['dist_pc'] = 1000.0 / w_adj

    # 系统质量
    df['System_Mass'] = df['M1']
    binary_idx = (df['Pb'] >= 0.5) & (df['q'].notna())
    df.loc[binary_idx, 'System_Mass'] = df['M1'] * (1 + df['q'])

    # 纯净样本定义
    df['Sample'] = 'intermediate'
    df.loc[(df['Pb'] < 0.2) & (df['ruwe'] < 1.4), 'Sample'] = 'single'
    df.loc[df['Pb'] > 0.7, 'Sample'] = 'binary'

    return df

# ==========================================
# 2. 动力学解算（使用修正距离）
# ==========================================
def calculate_dynamics(df):
    print(">>> Calculating Cluster Dynamics with distance correction...")
    single_mask = df['Sample'] == 'single'
    c_ra = df.loc[single_mask, 'ra'].median()
    c_dec = df.loc[single_mask, 'dec'].median()
    c_pmra = df.loc[single_mask, 'pmra'].median()
    c_pmdec = df.loc[single_mask, 'pmdec'].median()

    center_coord = SkyCoord(ra=c_ra*u.deg, dec=c_dec*u.deg)
    star_coords = SkyCoord(ra=df['ra'].values*u.deg, dec=df['dec'].values*u.deg)

    # 投影半径（个体距离）
    sep_deg = center_coord.separation(star_coords).degree
    df['R_pc'] = np.tan(np.radians(sep_deg)) * df['dist_pc']

    # 相对坐标
    cos_dec = np.cos(np.radians(c_dec))
    df['X_pc'] = (df['ra'] - c_ra) * cos_dec * (df['dist_pc'] * np.pi / 180)
    df['Y_pc'] = (df['dec'] - c_dec) * (df['dist_pc'] * np.pi / 180)

    k = 4.74047
    df['v_east'] = k * (df['pmra'] - c_pmra) / df['parallax_adj']
    df['v_north'] = k * (df['pmdec'] - c_pmdec) / df['parallax_adj']
    df['v_tan'] = np.sqrt(df['v_east']**2 + df['v_north']**2)

    pa_rad = center_coord.position_angle(star_coords).radian
    sin_pa, cos_pa = np.sin(pa_rad), np.cos(pa_rad)
    df['v_rad'] = df['v_east'] * sin_pa + df['v_north'] * cos_pa
    df['v_tan_rot'] = df['v_east'] * cos_pa - df['v_north'] * sin_pa

    return df

# ==========================================
# 3. 辅助函数：bootstrap 标准差
# ==========================================
def bootstrap_std(data, n_bootstrap=1000):
    if len(data) < 3:
        return np.nan, np.nan
    orig_std = np.std(data, ddof=1)
    boot_stds = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stds.append(np.std(sample, ddof=1))
    return orig_std, np.std(boot_stds)

# ==========================================
# 4. 独立图版绘制函数
# ==========================================

def plot_panel_A(df, save_path):
    """Panel A: 质量匹配后的径向分布 CDF（无标题）"""
    single = df[df['Sample'] == 'single'].copy()
    binary = df[df['Sample'] == 'binary'].copy()

    mass_min, mass_max = binary['System_Mass'].min(), binary['System_Mass'].max()
    single_in_range = single[(single['System_Mass'] >= mass_min) &
                             (single['System_Mass'] <= mass_max)]
    if len(single_in_range) < len(binary):
        single_matched = single_in_range.sample(n=len(binary), replace=True)
    else:
        single_matched = single_in_range.sample(n=len(binary), replace=False)

    r_single = np.sort(single_matched['R_pc'])
    r_binary = np.sort(binary['R_pc'])
    y_single = np.arange(1, len(r_single)+1) / len(r_single)
    y_binary = np.arange(1, len(r_binary)+1) / len(r_binary)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r_single, y_single, color='#2C7BB6', lw=2.5, label='Singles (mass-matched)')
    ax.plot(r_binary, y_binary, color='#D7191C', lw=2.5, label='Binaries')

    ks_stat, p_val = stats.ks_2samp(r_single, r_binary)
    ax.text(0.05, 0.9, f'KS test: p = {p_val:.2e}', transform=ax.transAxes,
            fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    ax.set_xscale('log')
    ax.set_xlabel('Projected radius $R$ (pc)', fontsize=14)
    ax.set_ylabel('Cumulative fraction', fontsize=14)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> Panel A saved as '{save_path}'")

def plot_panel_B(df, save_path):
    """Panel B: 速度弥散 vs 系统质量（折线风格 + bootstrap误差）"""
    core_mask = df['R_pc'] < 2.0
    single = df[(df['Sample'] == 'single') & core_mask].copy()
    binary = df[(df['Sample'] == 'binary') & core_mask].copy()

    mass_bins = np.linspace(0.2, 1.4, 6)
    bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    fig, ax = plt.subplots(figsize=(8, 6))

    def compute_binned_stats(subset, label, color, marker):
        subset['mass_bin'] = pd.cut(subset['System_Mass'], mass_bins, right=False)
        x_vals, y_vals, y_errs = [], [], []
        for i, interval in enumerate(subset['mass_bin'].cat.categories):
            in_bin = subset[subset['mass_bin'] == interval]
            if len(in_bin) < 5:
                continue
            std_val, std_err = bootstrap_std(in_bin['v_tan'].values)
            x_vals.append(bin_centers[i])
            y_vals.append(std_val)
            y_errs.append(std_err)
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        y_errs = np.array(y_errs)
        # 使用带连线的 errorbar (fmt='-o' 等)
        ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt=f'-{marker}', color=color,
                    label=label, capsize=4, markersize=8, elinewidth=1.5)

    compute_binned_stats(single, 'Singles', '#2C7BB6', 'o')
    compute_binned_stats(binary, 'Binaries', '#D7191C', 's')

    # 加权幂律拟合（同前）
    single_binned = []
    single['mass_bin'] = pd.cut(single['System_Mass'], mass_bins, right=False)
    for i, interval in enumerate(single['mass_bin'].cat.categories):
        in_bin = single[single['mass_bin'] == interval]
        if len(in_bin) < 5:
            continue
        std_val, std_err = bootstrap_std(in_bin['v_tan'].values)
        single_binned.append({'mass': bin_centers[i], 'std': std_val, 'err': std_err})
    if len(single_binned) >= 3:
        x = np.array([d['mass'] for d in single_binned])
        y = np.array([d['std'] for d in single_binned])
        w = 1.0 / np.array([d['err']**2 for d in single_binned])
        logx = np.log10(x)
        logy = np.log10(y)
        w_log = w * (y**2)
        a, b = np.polyfit(logx, logy, 1, w=w_log)
        eta = -a * 2
        x_fit = np.linspace(0.3, 1.3, 100)
        logy_fit = a * np.log10(x_fit) + b
        y_fit = 10**logy_fit
        ax.plot(x_fit, y_fit, '--', color='#2C7BB6', lw=2,
                label=rf'Fit: $\eta={eta:.2f}$')

    ax.set_xlabel(r'System mass $M_{\mathrm{sys}}$ ($M_\odot$)', fontsize=14)
    ax.set_ylabel(r'Velocity dispersion $\sigma_v$ (km/s)', fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(0.2, 1.4)
    ax.set_ylim(0.2, 0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> Panel B saved as '{save_path}'")

def plot_panel_C(df, save_path):
    """Panel C: 各向异性剖面（折线风格 + bootstrap误差）"""
    r_bins = np.linspace(0, 5, 6)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    fig, ax = plt.subplots(figsize=(8, 6))

    def bootstrap_beta(data, n_bootstrap=500):
        if len(data) < 6:
            return np.nan, np.nan
        s_r = np.std(data['v_rad'], ddof=1)
        s_t = np.std(data['v_tan_rot'], ddof=1)
        beta = 1.0 - (s_t**2 / (s_r**2 + 1e-10))
        betas = []
        for _ in range(n_bootstrap):
            boot = data.sample(n=len(data), replace=True)
            s_r_b = np.std(boot['v_rad'], ddof=1)
            s_t_b = np.std(boot['v_tan_rot'], ddof=1)
            beta_b = 1.0 - (s_t_b**2 / (s_r_b**2 + 1e-10))
            betas.append(beta_b)
        return beta, np.std(betas)

    def compute_anisotropy(subset, label, color, marker):
        subset['r_bin'] = pd.cut(subset['R_pc'], r_bins, right=False)
        x_vals, y_vals, y_errs = [], [], []
        for i, interval in enumerate(subset['r_bin'].cat.categories):
            in_bin = subset[subset['r_bin'] == interval]
            if len(in_bin) < 6:
                continue
            beta, beta_err = bootstrap_beta(in_bin)
            x_vals.append(r_centers[i])
            y_vals.append(beta)
            y_errs.append(beta_err)
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        y_errs = np.array(y_errs)
        # 使用带连线的 errorbar (fmt='-o')
        ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt=f'-{marker}', color=color,
                    label=label, capsize=4, markersize=8, elinewidth=1.5)

    single = df[df['Sample'] == 'single']
    binary = df[df['Sample'] == 'binary']
    compute_anisotropy(single, 'Singles', '#2C7BB6', 'o')
    compute_anisotropy(binary, 'Binaries', '#D7191C', 's')

    ax.axhline(0, color='k', ls='--', alpha=0.7)
    ax.axvspan(0, 2.0, color='gray', alpha=0.15, label='Core region')
    ax.set_xlabel('Radius $R$ (pc)', fontsize=14)
    ax.set_ylabel(r'Anisotropy $\beta = 1 - \sigma_t^2 / \sigma_r^2$', fontsize=14)
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> Panel C saved as '{save_path}'")

def plot_panel_D(df, save_path):
    """Panel D: 双星矢量流图（含速度标尺）"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 背景星
    ax.scatter(df['X_pc'], df['Y_pc'], s=5, c='lightgray', alpha=0.3, edgecolor='none')

    binary = df[df['Sample'] == 'binary'].copy()
    core = binary[binary['R_pc'] < 2.0]
    halo = binary[binary['R_pc'] >= 2.0]

    # 随机抽样（保持图面清晰）
    if len(core) > 50:
        core = core.sample(n=50, random_state=42)
    if len(halo) > 50:
        halo = halo.sample(n=50, random_state=42)

    scale = 15
    q_core = ax.quiver(core['X_pc'], core['Y_pc'],
                       core['v_east'], core['v_north'],
                       color='purple', scale=scale, width=0.005,
                       alpha=0.8, label='Core binaries', pivot='mid')
    q_halo = ax.quiver(halo['X_pc'], halo['Y_pc'],
                       halo['v_east'], halo['v_north'],
                       color='red', scale=scale, width=0.005,
                       alpha=0.6, label='Halo binaries', pivot='mid')

    # 速度标尺
    ax.quiver(-5, -5, 5, 0, color='black', scale=scale, width=0.008,
              headwidth=3, headlength=4, label='5 km/s')
    ax.text(-5, -5.6, '5 km/s', ha='center', va='top', fontsize=10)

    circle = plt.Circle((0, 0), 2.0, color='gray', fill=False, ls='--', lw=1.5, label='R=2 pc')
    ax.add_patch(circle)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_xlabel('East–West position (pc)', fontsize=14)
    ax.set_ylabel('North–South position (pc)', fontsize=14)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=False, edgecolor='black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> Panel D saved as '{save_path}'")

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    file_path = 'pleiades_merged.csv'
    df = load_and_clean_data(file_path)
    if df is not None:
        df = calculate_dynamics(df)

        plot_panel_A(df, 'PanelA_CDF.png')
        plot_panel_B(df, 'PanelB_Heating.png')
        plot_panel_C(df, 'PanelC_Anisotropy.png')
        plot_panel_D(df, 'PanelD_VectorMap.png')

        print(">>> All panels generated successfully.")