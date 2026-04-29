import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord

# ======================================
# 全局画图风格
# ======================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'savefig.dpi': 300
})

# ======================================
# 常数
# ======================================
DIST_PC = 136.2
K_CONV = 4.74047 * (DIST_PC / 1000.0)   # mas/yr -> km/s


# ======================================
# 1. 数据读取与预处理
# ======================================
def load_and_preprocess(all_csv, binary_txt,
                        cluster_center_ra=56.75, cluster_center_dec=24.11):
    print("[Step 1] Loading and preprocessing base data...")
    df = pd.read_csv(all_csv)

    required_cols = ['source_id', 'ra', 'dec', 'parallax', 'pmra', 'pmdec']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # ---- 读 photometric binary list ----
    with open(binary_txt, 'r') as f:
        lines = f.readlines()

    binary_ids = []
    for l in lines:
        parts = l.split()
        if len(parts) == 0:
            continue
        if parts[0].isdigit() and len(parts[0]) >= 18:
            binary_ids.append(parts[0])
    binary_ids = set(binary_ids)

    df = df.dropna(subset=required_cols).copy()
    df['source_id_str'] = df['source_id'].astype(str)
    df['is_phot_binary'] = df['source_id_str'].isin(binary_ids)

    if 'ruwe' not in df.columns:
        df['ruwe'] = 1.0
    df['ruwe'] = df['ruwe'].fillna(1.0)

    if 'pmra_error' in df.columns:
        df['pmra_error'] = df['pmra_error'].fillna(df['pmra_error'].median())
    else:
        df['pmra_error'] = 0.05 / K_CONV

    if 'pmdec_error' in df.columns:
        df['pmdec_error'] = df['pmdec_error'].fillna(df['pmdec_error'].median())
    else:
        df['pmdec_error'] = 0.05 / K_CONV

    # ---- r_proj ----
    center = SkyCoord(ra=cluster_center_ra * u.deg,
                      dec=cluster_center_dec * u.deg,
                      frame='icrs')
    sc = SkyCoord(ra=df['ra'].values * u.deg,
                  dec=df['dec'].values * u.deg,
                  frame='icrs')
    df['r_proj'] = sc.separation(center).rad * DIST_PC

    print(f"      Base sample size: {len(df)}")
    print(f"      Photometric binary fraction: {df['is_phot_binary'].mean():.2%}")
    print(f"      RUWE>1.4 fraction: {(df['ruwe'] > 1.4).mean():.2%}")
    print(f"      Radius range: {df['r_proj'].min():.2f} -- {df['r_proj'].max():.2f} pc")

    return df


# ======================================
# 2. binary 定义模式
# ======================================
def assign_binary_definition(df, mode='combined'):
    """
    根据不同模式生成 is_binary 列，并重新做单星去心速度。
    模式:
      - photometric
      - ruwe
      - combined
      - overlap
    """
    out = df.copy()

    if mode == 'photometric':
        out['is_binary'] = out['is_phot_binary']
    elif mode == 'ruwe':
        out['is_binary'] = (out['ruwe'] > 1.4)
    elif mode == 'combined':
        out['is_binary'] = out['is_phot_binary'] | (out['ruwe'] > 1.4)
    elif mode == 'overlap':
        out['is_binary'] = out['is_phot_binary'] & (out['ruwe'] > 1.4)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ---- 用该模式下的 single stars 去心 ----
    single_mask = ~out['is_binary']
    if single_mask.sum() < 10:
        pmra_med = out['pmra'].median()
        pmdec_med = out['pmdec'].median()
    else:
        pmra_med = out.loc[single_mask, 'pmra'].median()
        pmdec_med = out.loc[single_mask, 'pmdec'].median()

    out['v_ra_res'] = K_CONV * (out['pmra'] - pmra_med)
    out['v_dec_res'] = K_CONV * (out['pmdec'] - pmdec_med)
    out['v_ra_err'] = K_CONV * out['pmra_error']
    out['v_dec_err'] = K_CONV * out['pmdec_error']

    return out


# ======================================
# 3. 速度方差估计
# ======================================
def robust_sigma2_2d(vx, vy, errx=None, erry=None):
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)

    mask = np.isfinite(vx) & np.isfinite(vy)
    vx = vx[mask]
    vy = vy[mask]

    if len(vx) < 3:
        return np.nan

    varx = np.var(vx, ddof=1)
    vary = np.var(vy, ddof=1)

    if errx is not None and erry is not None:
        errx = np.asarray(errx, dtype=float)[mask]
        erry = np.asarray(erry, dtype=float)[mask]
        varx -= np.nanmean(errx**2)
        vary -= np.nanmean(erry**2)

    varx = max(varx, 1e-8)
    vary = max(vary, 1e-8)

    return 0.5 * (varx + vary)


# ======================================
# 4. 单bin测量
# ======================================
def compute_delta_once(df_sub):
    singles = df_sub[~df_sub['is_binary']]
    binaries = df_sub[df_sub['is_binary']]

    n_s = len(singles)
    n_b = len(binaries)
    n_t = len(df_sub)

    if n_s < 3 or n_b < 3:
        return None

    sig2_s = robust_sigma2_2d(
        singles['v_ra_res'], singles['v_dec_res'],
        singles['v_ra_err'], singles['v_dec_err']
    )
    sig2_b = robust_sigma2_2d(
        binaries['v_ra_res'], binaries['v_dec_res'],
        binaries['v_ra_err'], binaries['v_dec_err']
    )

    if not np.isfinite(sig2_s) or not np.isfinite(sig2_b) or sig2_s <= 0:
        return None

    f_b = n_b / n_t
    delta_kin = (sig2_b - sig2_s) / sig2_s
    sdyn_eff = delta_kin / f_b if f_b > 0.02 else np.nan

    return {
        'sig2_single': sig2_s,
        'sig2_binary': sig2_b,
        'f_b': f_b,
        'Delta_kin': delta_kin,
        'Sdyn_eff': sdyn_eff,
        'N_all': n_t,
        'N_single': n_s,
        'N_binary': n_b
    }


# ======================================
# 5. bootstrap + permutation
# ======================================
def infer_delta_significance(bin_df, n_boot=2000, n_perm=2000,
                             min_single=5, min_binary=5, random_state=42):
    rng = np.random.default_rng(random_state)

    obs = compute_delta_once(bin_df)
    if obs is None:
        return None

    if obs['N_single'] < min_single or obs['N_binary'] < min_binary:
        return None

    singles = bin_df[~bin_df['is_binary']].copy()
    binaries = bin_df[bin_df['is_binary']].copy()

    n_s = len(singles)
    n_b = len(binaries)

    s_idx = np.arange(n_s)
    b_idx = np.arange(n_b)

    boot_delta = []
    boot_eff = []
    boot_sig2s = []
    boot_sig2b = []
    boot_fb = []

    for _ in range(n_boot):
        s_pick = rng.choice(s_idx, size=n_s, replace=True)
        b_pick = rng.choice(b_idx, size=n_b, replace=True)

        s_bs = singles.iloc[s_pick]
        b_bs = binaries.iloc[b_pick]
        all_bs = pd.concat([s_bs, b_bs], ignore_index=True)

        tmp = compute_delta_once(all_bs)
        if tmp is not None and np.isfinite(tmp['Delta_kin']):
            boot_delta.append(tmp['Delta_kin'])
            boot_sig2s.append(tmp['sig2_single'])
            boot_sig2b.append(tmp['sig2_binary'])
            boot_fb.append(tmp['f_b'])
            if np.isfinite(tmp['Sdyn_eff']):
                boot_eff.append(tmp['Sdyn_eff'])

    if len(boot_delta) < 50:
        return None

    boot_delta = np.array(boot_delta)
    boot_sig2s = np.array(boot_sig2s)
    boot_sig2b = np.array(boot_sig2b)
    boot_fb = np.array(boot_fb)
    boot_eff = np.array(boot_eff) if len(boot_eff) > 0 else np.array([])

    d16, d50, d84 = np.percentile(boot_delta, [16, 50, 84])
    d025, d975 = np.percentile(boot_delta, [2.5, 97.5])
    p_boot_gt0 = np.mean(boot_delta > 0)
    d_err = np.std(boot_delta, ddof=1)

    perm_delta = []
    values = bin_df.copy()
    n_total = len(values)
    n_binary = obs['N_binary']

    for _ in range(n_perm):
        shuffled = values.copy()
        rand_labels = np.zeros(n_total, dtype=bool)
        rand_labels[rng.choice(np.arange(n_total), size=n_binary, replace=False)] = True
        shuffled['is_binary'] = rand_labels

        tmp = compute_delta_once(shuffled)
        if tmp is not None and np.isfinite(tmp['Delta_kin']):
            perm_delta.append(tmp['Delta_kin'])

    if len(perm_delta) < 50:
        return None

    perm_delta = np.array(perm_delta)

    p_perm_one = (np.sum(perm_delta >= obs['Delta_kin']) + 1) / (len(perm_delta) + 1)
    p_perm_two = (np.sum(np.abs(perm_delta) >= abs(obs['Delta_kin'])) + 1) / (len(perm_delta) + 1)

    perm_std = np.std(perm_delta, ddof=1)
    z_like = ((obs['Delta_kin'] - np.mean(perm_delta)) / perm_std) if perm_std > 0 else np.nan

    if p_perm_one < 0.001:
        sig_flag = '***'
    elif p_perm_one < 0.01:
        sig_flag = '**'
    elif p_perm_one < 0.05:
        sig_flag = '*'
    else:
        sig_flag = 'ns'

    return {
        'sig2_single': obs['sig2_single'],
        'sig2_single_err': np.std(boot_sig2s, ddof=1),
        'sig2_binary': obs['sig2_binary'],
        'sig2_binary_err': np.std(boot_sig2b, ddof=1),
        'f_b': obs['f_b'],
        'f_b_err': np.std(boot_fb, ddof=1),
        'Delta_kin': obs['Delta_kin'],
        'Delta_kin_err': d_err,
        'Delta_kin_med': d50,
        'Delta_kin_lo68': d16,
        'Delta_kin_hi68': d84,
        'Delta_kin_lo95': d025,
        'Delta_kin_hi95': d975,
        'P_boot_gt0': p_boot_gt0,
        'Sdyn_eff': obs['Sdyn_eff'],
        'Sdyn_eff_err': np.std(boot_eff, ddof=1) if len(boot_eff) > 10 else np.nan,
        'p_perm_one': p_perm_one,
        'p_perm_two': p_perm_two,
        'z_like': z_like,
        'sig_flag': sig_flag,
        'N_all': obs['N_all'],
        'N_single': obs['N_single'],
        'N_binary': obs['N_binary']
    }


# ======================================
# 6. 径向剖面
# ======================================
def compute_radial_heating_profiles_significance(df,
                                                 r_min=0.2,
                                                 r_max=10.0,
                                                 bins_num=8,
                                                 n_boot=2000,
                                                 n_perm=2000,
                                                 min_per_bin=12,
                                                 min_single=5,
                                                 min_binary=5):
    bins = np.logspace(np.log10(r_min), np.log10(r_max), bins_num + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])

    rows = []

    for i in range(bins_num):
        r1, r2 = bins[i], bins[i+1]
        mask = (df['r_proj'] >= r1) & (df['r_proj'] < r2)
        sub = df.loc[mask].copy()

        if len(sub) < min_per_bin:
            continue

        out = infer_delta_significance(
            sub,
            n_boot=n_boot,
            n_perm=n_perm,
            min_single=min_single,
            min_binary=min_binary,
            random_state=1000 + i
        )

        if out is None:
            continue

        rows.append([
            r_centers[i], r1, r2,
            out['Delta_kin'], out['Delta_kin_err'],
            out['Delta_kin_med'], out['Delta_kin_lo68'], out['Delta_kin_hi68'],
            out['Delta_kin_lo95'], out['Delta_kin_hi95'],
            out['P_boot_gt0'],
            out['Sdyn_eff'], out['Sdyn_eff_err'],
            out['f_b'], out['f_b_err'],
            out['sig2_single'], out['sig2_single_err'],
            out['sig2_binary'], out['sig2_binary_err'],
            out['p_perm_one'], out['p_perm_two'],
            out['z_like'], out['sig_flag'],
            out['N_all'], out['N_single'], out['N_binary']
        ])

    cols = [
        'R', 'R_in', 'R_out',
        'Delta_kin', 'Delta_kin_err',
        'Delta_kin_med', 'Delta_kin_lo68', 'Delta_kin_hi68',
        'Delta_kin_lo95', 'Delta_kin_hi95',
        'P_boot_gt0',
        'Sdyn_eff', 'Sdyn_eff_err',
        'f_b', 'f_b_err',
        'sig2_single', 'sig2_single_err',
        'sig2_binary', 'sig2_binary_err',
        'p_perm_one', 'p_perm_two',
        'z_like', 'sig_flag',
        'N_all', 'N_single', 'N_binary'
    ]

    return pd.DataFrame(rows, columns=cols)


# ======================================
# 7. 单模式绘图
# ======================================
def plot_heating_profiles_significance(res_df, mode_label='combined',
                                       savefile='heating_profile.pdf'):
    if len(res_df) == 0:
        print(f"[Warning] Empty result for mode={mode_label}")
        return

    fig, axes = plt.subplots(4, 1, figsize=(8, 14), sharex=True)
    plt.subplots_adjust(hspace=0.08)

    ax1, ax2, ax3, ax4 = axes
    R = res_df['R'].values

    # panel 1
    ax1.errorbar(
        R, res_df['Delta_kin'], yerr=res_df['Delta_kin_err'],
        fmt='o', color='crimson', markersize=7, capsize=4,
        elinewidth=1.4, markeredgecolor='black'
    )
    ax1.plot(R, res_df['Delta_kin'], color='crimson', alpha=0.35, lw=2)
    ax1.axhline(0, color='black', ls='--', alpha=0.6)
    ax1.axvspan(0, 1.5, color='gold', alpha=0.10)
    ax1.set_ylabel(r'$\Delta_{\rm kin}$', fontsize=14)
    ax1.set_title(f'Pleiades Binary Heating: {mode_label}', fontsize=15, pad=10)

    yscale = np.nanmax(np.abs(res_df['Delta_kin'])) if len(res_df) > 0 else 1.0
    if not np.isfinite(yscale) or yscale == 0:
        yscale = 1.0

    for _, row in res_df.iterrows():
        y = row['Delta_kin']
        dy = row['Delta_kin_err'] if np.isfinite(row['Delta_kin_err']) else 0.0
        ax1.text(row['R'], y + dy + 0.05 * yscale, row['sig_flag'],
                 ha='center', va='bottom', fontsize=11)

    # panel 2
    ax2.plot(R, res_df['P_boot_gt0'], '-o', color='purple', lw=2,
             markersize=6, markeredgecolor='black')
    ax2.axhline(0.5, color='gray', ls='--', alpha=0.6)
    ax2.axhline(0.95, color='green', ls=':', alpha=0.7)
    ax2.axhline(0.99, color='darkgreen', ls=':', alpha=0.7)
    ax2.axvspan(0, 1.5, color='gold', alpha=0.10)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel(r'$P_{\rm boot}(\Delta>0)$', fontsize=13)

    # panel 3
    ax3.errorbar(
        R, res_df['Sdyn_eff'], yerr=res_df['Sdyn_eff_err'],
        fmt='s', color='darkorange', markersize=6, capsize=4,
        elinewidth=1.4, markeredgecolor='black'
    )
    ax3.plot(R, res_df['Sdyn_eff'], color='darkorange', alpha=0.35, lw=2)
    ax3.axhline(0, color='black', ls='--', alpha=0.6)
    ax3.axvspan(0, 1.5, color='gold', alpha=0.10)
    ax3.set_ylabel(r'$S_{\rm dyn}^{\star}$', fontsize=14)

    # panel 4
    ax4.errorbar(
        R, res_df['f_b'], yerr=res_df['f_b_err'],
        fmt='^', color='navy', markersize=6, capsize=4,
        elinewidth=1.4, markeredgecolor='black'
    )
    ax4.plot(R, res_df['f_b'], color='navy', alpha=0.35, lw=2)
    ax4.axvspan(0, 1.5, color='gold', alpha=0.10)
    ax4.set_ylabel(r'$f_b$', fontsize=14)
    ax4.set_xlabel(r'Projected Radius $R$ [pc]', fontsize=14)

    for ax in axes:
        ax.minorticks_on()
        ax.grid(which='major', linestyle=':', alpha=0.45)

    plt.savefig(savefile, bbox_inches='tight')
    plt.close()


# ======================================
# 8. 单模式运行
# ======================================
def run_single_mode(base_df, mode='combined',
                    outdir='robustness_outputs',
                    r_min=0.2, r_max=10.0, bins_num=8,
                    n_boot=2000, n_perm=2000,
                    min_per_bin=12, min_single=5, min_binary=5):
    print(f"\n[Mode] Running mode = {mode}")

    df_mode = assign_binary_definition(base_df, mode=mode)

    print(f"      Binary fraction = {df_mode['is_binary'].mean():.2%}")
    print(f"      N_binary = {df_mode['is_binary'].sum()} / N_total = {len(df_mode)}")

    res_df = compute_radial_heating_profiles_significance(
        df_mode,
        r_min=r_min,
        r_max=r_max,
        bins_num=bins_num,
        n_boot=n_boot,
        n_perm=n_perm,
        min_per_bin=min_per_bin,
        min_single=min_single,
        min_binary=min_binary
    )

    table_file = os.path.join(outdir, f'Heating_Profile_{mode}.csv')
    fig_file = os.path.join(outdir, f'Heating_Profile_{mode}.pdf')

    res_df.to_csv(table_file, index=False)
    plot_heating_profiles_significance(res_df, mode_label=mode, savefile=fig_file)

    # 汇总指标
    if len(res_df) > 0:
        summary = {
            'mode': mode,
            'N_total': len(df_mode),
            'N_binary_total': int(df_mode['is_binary'].sum()),
            'binary_fraction_total': df_mode['is_binary'].mean(),
            'n_valid_bins': len(res_df),
            'n_sig_p005': int(np.sum(res_df['p_perm_one'] < 0.05)),
            'n_sig_p001': int(np.sum(res_df['p_perm_one'] < 0.01)),
            'n_boot95': int(np.sum(res_df['P_boot_gt0'] > 0.95)),
            'n_boot99': int(np.sum(res_df['P_boot_gt0'] > 0.99)),
            'mean_delta': np.nanmean(res_df['Delta_kin']),
            'max_delta': np.nanmax(res_df['Delta_kin']),
            'max_zlike': np.nanmax(res_df['z_like'])
        }
    else:
        summary = {
            'mode': mode,
            'N_total': len(df_mode),
            'N_binary_total': int(df_mode['is_binary'].sum()),
            'binary_fraction_total': df_mode['is_binary'].mean(),
            'n_valid_bins': 0,
            'n_sig_p005': 0,
            'n_sig_p001': 0,
            'n_boot95': 0,
            'n_boot99': 0,
            'mean_delta': np.nan,
            'max_delta': np.nan,
            'max_zlike': np.nan
        }

    return df_mode, res_df, summary


# ======================================
# 9. 总览比较图
# ======================================
def plot_robustness_comparison(results_dict,
                               savefile='robustness_outputs/Heating_Profile_Comparison.pdf'):
    """
    对比四种 binary 定义下的 Delta_kin(R)
    """
    plt.figure(figsize=(8, 6))

    color_map = {
        'photometric': 'royalblue',
        'ruwe': 'darkorange',
        'combined': 'crimson',
        'overlap': 'seagreen'
    }

    marker_map = {
        'photometric': 'o',
        'ruwe': 's',
        'combined': '^',
        'overlap': 'D'
    }

    for mode, res_df in results_dict.items():
        if len(res_df) == 0:
            continue

        plt.errorbar(
            res_df['R'], res_df['Delta_kin'],
            yerr=res_df['Delta_kin_err'],
            fmt=marker_map.get(mode, 'o'),
            color=color_map.get(mode, 'black'),
            capsize=3, lw=1.2, markersize=5,
            label=mode
        )
        plt.plot(res_df['R'], res_df['Delta_kin'],
                 color=color_map.get(mode, 'black'), alpha=0.4)

    plt.axhline(0, color='black', ls='--', alpha=0.6)
    plt.axvspan(0, 1.5, color='gold', alpha=0.10)
    plt.xscale('log')
    plt.xlabel(r'Projected Radius $R$ [pc]')
    plt.ylabel(r'$\Delta_{\rm kin}$')
    plt.title('Robustness of Binary Heating Signal to Binary Definition')
    plt.legend(frameon=False)
    plt.grid(ls=':', alpha=0.4)
    plt.savefig(savefile, bbox_inches='tight')
    plt.close()


# ======================================
# 10. 汇总表
# ======================================
def save_summary_table(summary_list,
                       filename='robustness_outputs/Heating_Robustness_Summary.csv'):
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(filename, index=False)
    return summary_df


# ======================================
# 11. 主稳健性批量运行
# ======================================
def run_robustness_suite(all_csv='Pleiades_GAIA_ALL.csv',
                         binary_txt='member.txt',
                         outdir='robustness_outputs',
                         modes=('photometric', 'ruwe', 'combined', 'overlap'),
                         r_min=0.2, r_max=10.0, bins_num=8,
                         n_boot=2000, n_perm=2000,
                         min_per_bin=12, min_single=5, min_binary=5):
    os.makedirs(outdir, exist_ok=True)

    base_df = load_and_preprocess(all_csv, binary_txt)

    results_dict = {}
    summary_list = []

    for mode in modes:
        df_mode, res_df, summary = run_single_mode(
            base_df,
            mode=mode,
            outdir=outdir,
            r_min=r_min,
            r_max=r_max,
            bins_num=bins_num,
            n_boot=n_boot,
            n_perm=n_perm,
            min_per_bin=min_per_bin,
            min_single=min_single,
            min_binary=min_binary
        )
        results_dict[mode] = res_df
        summary_list.append(summary)

    summary_df = save_summary_table(
        summary_list,
        filename=os.path.join(outdir, 'Heating_Robustness_Summary.csv')
    )

    plot_robustness_comparison(
        results_dict,
        savefile=os.path.join(outdir, 'Heating_Profile_Comparison.pdf')
    )

    print("\n===== Robustness Summary =====")
    print(summary_df)

    return results_dict, summary_df


# ======================================
# 12. main
# ======================================
if __name__ == "__main__":
    results_dict, summary_df = run_robustness_suite(
        all_csv='Pleiades_GAIA_ALL.csv',
        binary_txt='member.txt',
        outdir='robustness_outputs',
        modes=('photometric', 'ruwe', 'combined', 'overlap'),
        r_min=0.2,
        r_max=10.0,
        bins_num=8,
        n_boot=2000,
        n_perm=2000,
        min_per_bin=12,
        min_single=5,
        min_binary=5
    )

    print("\n[Done] Robustness suite completed.")
