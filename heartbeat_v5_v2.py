import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy import stats
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# Global configuration
# ============================================================

FILE_PATH = "pleiades_merged.csv"

OUTDIR = "figures_revised"
os.makedirs(OUTDIR, exist_ok=True)

# Adopted cluster parallax and intrinsic dispersion
PI_C_MAS = 7.41
SIGMA_PI_C_MAS = 0.17

# Fiducial limits
RMAX_MAIN_PC = 8.0
CORE_RADIUS_PC = 2.0

# Repeated mass matching
N_MATCH = 1000
N_NEIGHBOR_MASS = 10   # nearest-neighbor mass matching pool size

# Plot colors
COLOR_SINGLE = "#2C7BB6"
COLOR_BINARY = "#D7191C"
COLOR_GRAY = "lightgray"


# ============================================================
# 1. Data loading and cleaning
# ============================================================

def load_and_clean_data(filepath):
    """
    Load table, clean essential columns, apply Bayesian-like parallax shrinkage,
    define system mass and sample labels.
    """

    print(f">>> Loading data from {filepath} ...")

    try:
        df = pd.read_csv(filepath, sep=",", engine="python", skipinitialspace=True)
    except Exception as e:
        raise RuntimeError(f"Cannot read file {filepath}: {e}")

    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")

    numeric_cols = [
        "ra", "dec", "parallax", "parallax_error",
        "pmra", "pmra_error", "pmdec", "pmdec_error",
        "ruwe", "M1", "q", "Pb"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["ra", "dec", "parallax", "pmra", "pmdec", "M1", "Pb"]
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()
    df = df[df["parallax"] > 0].copy()

    # If parallax_error unavailable or invalid, adopt conservative fallback
    if "parallax_error" not in df.columns:
        df["parallax_error"] = 0.1
    df["parallax_error"] = pd.to_numeric(df["parallax_error"], errors="coerce")
    df["parallax_error"] = df["parallax_error"].fillna(0.1)
    df.loc[df["parallax_error"] <= 0, "parallax_error"] = 0.1

    # Bayesian-like parallax shrinkage to cluster prior
    w_obs = df["parallax"].values
    sigma_obs = df["parallax_error"].values

    w_adj = (
        PI_C_MAS / SIGMA_PI_C_MAS**2 + w_obs / sigma_obs**2
    ) / (
        1.0 / SIGMA_PI_C_MAS**2 + 1.0 / sigma_obs**2
    )

    df["parallax_adj"] = w_adj
    df["dist_pc"] = 1000.0 / df["parallax_adj"]

    # System mass
    df["System_Mass"] = df["M1"].astype(float)

    if "q" not in df.columns:
        df["q"] = np.nan

    binary_mass_mask = (df["Pb"] >= 0.5) & (df["q"].notna())
    df.loc[binary_mass_mask, "System_Mass"] = (
        df.loc[binary_mass_mask, "M1"] *
        (1.0 + df.loc[binary_mass_mask, "q"])
    )

    # Clean sample definition
    if "ruwe" not in df.columns:
        df["ruwe"] = np.nan

    df["Sample"] = "intermediate"
    df.loc[(df["Pb"] < 0.2) & (df["ruwe"] < 1.4), "Sample"] = "single"
    df.loc[df["Pb"] > 0.7, "Sample"] = "binary"

    # Remove pathological system masses
    df = df[np.isfinite(df["System_Mass"])].copy()
    df = df[df["System_Mass"] > 0].copy()

    print(f">>> Total clean sources: {len(df)}")
    print(df["Sample"].value_counts())

    return df


# ============================================================
# 2. Exact spherical geometry and dynamics
# ============================================================

def calculate_dynamics(df):
    """
    Compute exact projected positions, 3D cluster-centric radius,
    and projected proper-motion velocities.
    """

    print(">>> Calculating exact spherical geometry and projected dynamics ...")

    df = df.copy()

    single_mask = df["Sample"] == "single"

    if single_mask.sum() < 10:
        raise RuntimeError("Too few clean singles to define cluster center robustly.")

    # Adopt cluster center and systemic proper motion from clean singles
    c_ra = df.loc[single_mask, "ra"].median()
    c_dec = df.loc[single_mask, "dec"].median()
    c_pmra = df.loc[single_mask, "pmra"].median()
    c_pmdec = df.loc[single_mask, "pmdec"].median()
    c_dist = np.nanmedian(df.loc[single_mask, "dist_pc"])

    print(f">>> Adopted center: RA={c_ra:.5f} deg, Dec={c_dec:.5f} deg")
    print(f">>> Adopted PM: pmra={c_pmra:.5f} mas/yr, pmdec={c_pmdec:.5f} mas/yr")
    print(f">>> Adopted distance: d0={c_dist:.2f} pc")

    center_coord = SkyCoord(ra=c_ra * u.deg, dec=c_dec * u.deg)
    star_coords = SkyCoord(ra=df["ra"].values * u.deg,
                           dec=df["dec"].values * u.deg)

    # Exact angular separation and position angle
    sep = center_coord.separation(star_coords)
    pa = center_coord.position_angle(star_coords)

    rho_rad = sep.radian
    pa_rad = pa.radian

    df["rho_deg"] = sep.degree
    df["PA_rad"] = pa_rad

    # Exact tangent-plane projected coordinates using individual distance
    df["R_pc"] = df["dist_pc"].values * np.tan(rho_rad)
    df["X_pc"] = df["R_pc"].values * np.sin(pa_rad)  # East
    df["Y_pc"] = df["R_pc"].values * np.cos(pa_rad)  # North

    # Exact 3D cluster-centric radius
    d_i = df["dist_pc"].values
    d_0 = c_dist

    df["r3d_pc"] = np.sqrt(
        d_i**2 + d_0**2 - 2.0 * d_i * d_0 * np.cos(rho_rad)
    )

    # Also compute local Cartesian 3D components if needed
    ra = np.deg2rad(df["ra"].values)
    dec = np.deg2rad(df["dec"].values)
    ra0 = np.deg2rad(c_ra)
    dec0 = np.deg2rad(c_dec)

    n_x = np.cos(dec) * np.cos(ra)
    n_y = np.cos(dec) * np.sin(ra)
    n_z = np.sin(dec)

    n0_x = np.cos(dec0) * np.cos(ra0)
    n0_y = np.cos(dec0) * np.sin(ra0)
    n0_z = np.sin(dec0)

    xi = d_i * n_x
    yi = d_i * n_y
    zi = d_i * n_z

    x0 = d_0 * n0_x
    y0 = d_0 * n0_y
    z0 = d_0 * n0_z

    dx = xi - x0
    dy = yi - y0
    dz = zi - z0

    # Local basis at cluster center
    eE = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    eN = np.array([
        -np.cos(ra0) * np.sin(dec0),
        -np.sin(ra0) * np.sin(dec0),
         np.cos(dec0)
    ])
    eZ = np.array([n0_x, n0_y, n0_z])

    df["X3D_pc"] = dx * eE[0] + dy * eE[1] + dz * eE[2]
    df["Y3D_pc"] = dx * eN[0] + dy * eN[1] + dz * eN[2]
    df["Z3D_pc"] = dx * eZ[0] + dy * eZ[1] + dz * eZ[2]

    # Proper-motion velocities
    # Gaia pmra is already mu_alpha* = mu_alpha cos(delta)
    k = 4.74047
    df["v_east"] = k * (df["pmra"] - c_pmra) / df["parallax_adj"]
    df["v_north"] = k * (df["pmdec"] - c_pmdec) / df["parallax_adj"]
    df["v_2d"] = np.sqrt(df["v_east"]**2 + df["v_north"]**2)

    # Radial/tangential decomposition on the plane of sky
    sin_pa = np.sin(pa_rad)
    cos_pa = np.cos(pa_rad)

    df["v_rad"] = df["v_east"] * sin_pa + df["v_north"] * cos_pa
    df["v_tan_rot"] = df["v_east"] * cos_pa - df["v_north"] * sin_pa

    # Fiducial inner sample flag
    df["in_Rmax"] = df["R_pc"] <= RMAX_MAIN_PC

    return df, {
        "ra0": c_ra,
        "dec0": c_dec,
        "pmra0": c_pmra,
        "pmdec0": c_pmdec,
        "dist0": c_dist
    }


# ============================================================
# 3. Statistical helper functions
# ============================================================

def bootstrap_std(data, n_bootstrap=1000, random_state=None):
    data = np.asarray(data)
    data = data[np.isfinite(data)]

    if len(data) < 3:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    orig_std = np.std(data, ddof=1)

    boot_stds = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stds[i] = np.std(sample, ddof=1)

    return orig_std, np.std(boot_stds, ddof=1)


def safe_ad_test(x, y, n_resamples=999, random_state=42):
    """
    Anderson-Darling k-sample test.
    Prefer permutation method if scipy supports it.
    Return significance_level-like value.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan

    try:
        # Newer scipy
        method = stats.PermutationMethod(
            n_resamples=n_resamples,
            random_state=random_state
        )
        res = stats.anderson_ksamp([x, y], method=method)
        return res.statistic, res.pvalue
    except Exception:
        try:
            # Older scipy fallback
            res = stats.anderson_ksamp([x, y])
            return res.statistic, res.significance_level / 100.0
        except Exception:
            return np.nan, np.nan


def nearest_neighbor_mass_match(single, binary, mass_col="System_Mass",
                                n_neighbor=10, replace=False,
                                rng=None):
    """
    For each binary source, draw one single star from nearest neighbors in system mass.
    If replace=False, try not to reuse singles. If impossible, allow fallback.
    """

    if rng is None:
        rng = np.random.default_rng()

    single = single.copy().reset_index(drop=True)
    binary = binary.copy().reset_index(drop=True)

    single_m = single[mass_col].values
    binary_m = binary[mass_col].values

    chosen_indices = []
    used = set()

    for mb in binary_m:
        order = np.argsort(np.abs(single_m - mb))

        if replace:
            pool = order[:min(n_neighbor, len(order))]
            idx = rng.choice(pool)
            chosen_indices.append(idx)
        else:
            # Prefer unused nearest neighbors
            pool = [idx for idx in order[:max(n_neighbor, 1) * 5] if idx not in used]

            if len(pool) == 0:
                # fallback to replacement if all good candidates used
                pool = order[:min(n_neighbor, len(order))]

            pool = pool[:min(n_neighbor, len(pool))]
            idx = int(rng.choice(pool))
            chosen_indices.append(idx)
            used.add(idx)

    matched = single.iloc[chosen_indices].copy()
    return matched


def empirical_cdf_on_grid(values, grid):
    values = np.sort(np.asarray(values))
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    return np.searchsorted(values, grid, side="right") / len(values)


def repeated_mass_matching_tests(df, radius_col="R_pc",
                                 n_match=N_MATCH,
                                 n_neighbor=N_NEIGHBOR_MASS,
                                 random_seed=42):
    """
    Repeated nearest-neighbor mass matching.
    Return stats table and CDF envelope.
    """

    rng = np.random.default_rng(random_seed)

    use = df[df["in_Rmax"]].copy()

    single = use[use["Sample"] == "single"].copy()
    binary = use[use["Sample"] == "binary"].copy()

    single = single[np.isfinite(single["System_Mass"]) & np.isfinite(single[radius_col])].copy()
    binary = binary[np.isfinite(binary["System_Mass"]) & np.isfinite(binary[radius_col])].copy()

    if len(single) == 0 or len(binary) == 0:
        raise RuntimeError("No single or binary stars available for matching.")

    replace = len(single) < len(binary)

    print(f">>> Repeated mass matching using {radius_col}")
    print(f"    N_single={len(single)}, N_binary={len(binary)}, replace={replace}")

    # Grid for CDF envelope
    all_r = np.concatenate([single[radius_col].values, binary[radius_col].values])
    rmin = np.nanpercentile(all_r[all_r > 0], 0.5)
    rmax = np.nanpercentile(all_r, 99.5)
    grid = np.logspace(np.log10(max(rmin, 1e-3)), np.log10(rmax), 300)

    binary_r = binary[radius_col].values
    binary_cdf_grid = empirical_cdf_on_grid(binary_r, grid)

    cdf_matched = []
    rows = []

    for i in range(n_match):
        matched = nearest_neighbor_mass_match(
            single, binary,
            mass_col="System_Mass",
            n_neighbor=n_neighbor,
            replace=replace,
            rng=rng
        )

        r_s = matched[radius_col].values
        r_b = binary[radius_col].values

        m_s = matched["System_Mass"].values
        m_b = binary["System_Mass"].values

        ks_stat, ks_p = stats.ks_2samp(r_s, r_b)
        ad_stat, ad_p = safe_ad_test(r_s, r_b, n_resamples=999,
                                     random_state=random_seed + i)

        wd_r = wasserstein_distance(r_s, r_b)
        wd_m = wasserstein_distance(m_s, m_b)

        rows.append({
            "iter": i,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "ad_stat": ad_stat,
            "ad_p_or_sig": ad_p,
            "wd_radius": wd_r,
            "wd_mass": wd_m,
            "n_single_matched": len(matched),
            "n_binary": len(binary)
        })

        cdf_matched.append(empirical_cdf_on_grid(r_s, grid))

    stats_df = pd.DataFrame(rows)

    cdf_arr = np.vstack(cdf_matched)
    cdf_med = np.nanmedian(cdf_arr, axis=0)
    cdf_p16 = np.nanpercentile(cdf_arr, 16, axis=0)
    cdf_p84 = np.nanpercentile(cdf_arr, 84, axis=0)

    summary = {
        "N_single": len(single),
        "N_binary": len(binary),
        "ks_p_median": np.nanmedian(stats_df["ks_p"]),
        "ks_p_p16": np.nanpercentile(stats_df["ks_p"], 16),
        "ks_p_p84": np.nanpercentile(stats_df["ks_p"], 84),
        "ad_p_median": np.nanmedian(stats_df["ad_p_or_sig"]),
        "ad_p_p16": np.nanpercentile(stats_df["ad_p_or_sig"], 16),
        "ad_p_p84": np.nanpercentile(stats_df["ad_p_or_sig"], 84),
        "wd_radius_median": np.nanmedian(stats_df["wd_radius"]),
        "wd_radius_p16": np.nanpercentile(stats_df["wd_radius"], 16),
        "wd_radius_p84": np.nanpercentile(stats_df["wd_radius"], 84),
        "wd_mass_median": np.nanmedian(stats_df["wd_mass"]),
        "wd_mass_p16": np.nanpercentile(stats_df["wd_mass"], 16),
        "wd_mass_p84": np.nanpercentile(stats_df["wd_mass"], 84),
    }

    cdf_info = {
        "grid": grid,
        "binary_cdf": binary_cdf_grid,
        "matched_median": cdf_med,
        "matched_p16": cdf_p16,
        "matched_p84": cdf_p84,
        "binary_values": binary_r
    }

    return stats_df, summary, cdf_info

def caliper_mass_match(single, binary, mass_col="System_Mass",
                       caliper=0.05, replace=False, rng=None):
    """
    Strict mass matching with common mass support and a mass caliper.

    caliper is in solar masses.
    """

    if rng is None:
        rng = np.random.default_rng()

    single = single.copy().reset_index(drop=True)
    binary = binary.copy().reset_index(drop=True)

    # common support
    m_min = max(single[mass_col].min(), binary[mass_col].min())
    m_max = min(single[mass_col].max(), binary[mass_col].max())

    single = single[(single[mass_col] >= m_min) & (single[mass_col] <= m_max)].copy()
    binary = binary[(binary[mass_col] >= m_min) & (binary[mass_col] <= m_max)].copy()

    single = single.reset_index(drop=True)
    binary = binary.reset_index(drop=True)

    used = set()
    chosen_single_indices = []
    chosen_binary_indices = []

    single_m = single[mass_col].values

    # shuffle binary order to avoid systematic bias
    binary_order = rng.permutation(len(binary))

    for ib in binary_order:
        mb = binary.loc[ib, mass_col]

        diff = np.abs(single_m - mb)
        candidates = np.where(diff <= caliper)[0]

        if len(candidates) == 0:
            continue

        if not replace:
            candidates = np.array([idx for idx in candidates if idx not in used])
            if len(candidates) == 0:
                continue

        # choose closest few candidates randomly
        cand_diff = diff[candidates]
        order = np.argsort(cand_diff)
        candidates = candidates[order]
        pool = candidates[:min(5, len(candidates))]

        chosen = int(rng.choice(pool))

        chosen_single_indices.append(chosen)
        chosen_binary_indices.append(ib)

        if not replace:
            used.add(chosen)

    matched_single = single.iloc[chosen_single_indices].copy()
    matched_binary = binary.iloc[chosen_binary_indices].copy()

    return matched_single, matched_binary

# ============================================================
# 4. Plotting functions
# ============================================================

def plot_panel_A_massmatched_cdf(cdf_info, summary, save_path,
                                 xlabel="Projected radius $R$ (pc)"):
    """
    Panel A: repeated mass-matched radial CDF.
    """

    grid = cdf_info["grid"]
    binary_cdf = cdf_info["binary_cdf"]
    med = cdf_info["matched_median"]
    p16 = cdf_info["matched_p16"]
    p84 = cdf_info["matched_p84"]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.fill_between(grid, p16, p84, color=COLOR_SINGLE, alpha=0.25,
                    label="Singles matched, 16--84%")
    ax.plot(grid, med, color=COLOR_SINGLE, lw=2.5,
            label="Singles matched, median")
    ax.plot(grid, binary_cdf, color=COLOR_BINARY, lw=2.5,
            label="Binaries")

    text = (
        rf"KS $p_{{\rm med}}={summary['ks_p_median']:.2f}$"
        "\n"
        rf"AD $p_{{\rm med}}={summary['ad_p_median']:.2f}$"
        "\n"
        rf"$W_R={summary['wd_radius_median']:.2f}$ pc"
    )

    ax.text(0.05, 0.82, text, transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))

    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Cumulative fraction", fontsize=14)
    ax.set_xlim(max(np.nanmin(grid), 0.03), np.nanmax(grid))
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Panel A saved: {save_path}")


def plot_mass_distribution_matching(df, save_path):
    """
    Appendix-like plot: mass distribution before and after one representative matching.
    """

    use = df[df["in_Rmax"]].copy()
    single = use[use["Sample"] == "single"].copy()
    binary = use[use["Sample"] == "binary"].copy()

    rng = np.random.default_rng(123)
    matched = nearest_neighbor_mass_match(
        single, binary,
        mass_col="System_Mass",
        n_neighbor=N_NEIGHBOR_MASS,
        replace=(len(single) < len(binary)),
        rng=rng
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    bins = np.linspace(
        min(use["System_Mass"].quantile(0.005), binary["System_Mass"].min()),
        max(use["System_Mass"].quantile(0.995), binary["System_Mass"].max()),
        35
    )

    ax.hist(single["System_Mass"], bins=bins, histtype="stepfilled",
            color="gray", alpha=0.25, density=True, label="All singles")
    ax.hist(binary["System_Mass"], bins=bins, histtype="step",
            color=COLOR_BINARY, lw=2.5, density=True, label="Binaries")
    ax.hist(matched["System_Mass"], bins=bins, histtype="step",
            color=COLOR_SINGLE, lw=2.5, density=True,
            label="Matched singles")

    ax.set_xlabel(r"System mass $M_{\rm sys}$ ($M_\odot$)", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Mass distribution plot saved: {save_path}")


def plot_panel_B_heating(df, save_path):
    """
    Panel B: velocity dispersion vs system mass in core.
    """

    core_mask = df["R_pc"] < CORE_RADIUS_PC

    single = df[(df["Sample"] == "single") & core_mask].copy()
    binary = df[(df["Sample"] == "binary") & core_mask].copy()

    mass_bins = np.linspace(0.2, 1.6, 7)
    bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    fig, ax = plt.subplots(figsize=(8, 6))

    def compute_binned_stats(subset, label, color, marker):
        subset = subset.copy()
        subset["mass_bin"] = pd.cut(subset["System_Mass"], mass_bins, right=False)

        x_vals, y_vals, y_errs, n_vals = [], [], [], []

        for i, interval in enumerate(subset["mass_bin"].cat.categories):
            in_bin = subset[subset["mass_bin"] == interval]
            if len(in_bin) < 5:
                continue

            std_val, std_err = bootstrap_std(
                in_bin["v_2d"].values,
                n_bootstrap=1000,
                random_state=1000 + i
            )

            x_vals.append(bin_centers[i])
            y_vals.append(std_val)
            y_errs.append(std_err)
            n_vals.append(len(in_bin))

        if len(x_vals) > 0:
            ax.errorbar(
                x_vals, y_vals, yerr=y_errs,
                fmt=f"-{marker}", color=color,
                label=label, capsize=4,
                markersize=8, elinewidth=1.5, lw=2
            )

        return np.array(x_vals), np.array(y_vals), np.array(y_errs), np.array(n_vals)

    x_s, y_s, e_s, n_s = compute_binned_stats(single, "Singles", COLOR_SINGLE, "o")
    x_b, y_b, e_b, n_b = compute_binned_stats(binary, "Binaries", COLOR_BINARY, "s")

    # Power-law fit for singles: sigma ~ M^{-eta/2}
    # Add bootstrap uncertainty for eta
    def fit_eta_from_sample(sample_df, random_state_base=5000):
        """
        Given a single-star sample, bin it by mass, compute sigma_2D in each bin,
        and fit sigma_2D ~ M^{-eta/2}.
        Returns eta, slope a, intercept b, x, y, e.
        """
        sample_df = sample_df.copy()
        sample_df["mass_bin"] = pd.cut(sample_df["System_Mass"], mass_bins, right=False)

        x_vals, y_vals, e_vals = [], [], []

        for i, interval in enumerate(sample_df["mass_bin"].cat.categories):
            in_bin = sample_df[sample_df["mass_bin"] == interval]
            if len(in_bin) < 5:
                continue

            std_val, std_err = bootstrap_std(
                in_bin["v_2d"].values,
                n_bootstrap=500,
                random_state=random_state_base + i
            )

            if (
                np.isfinite(std_val) and np.isfinite(std_err)
                and std_val > 0 and std_err > 0
            ):
                x_vals.append(bin_centers[i])
                y_vals.append(std_val)
                e_vals.append(std_err)

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        e_vals = np.array(e_vals)

        valid = (
            np.isfinite(x_vals)
            & np.isfinite(y_vals)
            & np.isfinite(e_vals)
            & (y_vals > 0)
            & (e_vals > 0)
        )

        if valid.sum() < 3:
            return np.nan, np.nan, np.nan, x_vals, y_vals, e_vals

        x = x_vals[valid]
        y = y_vals[valid]
        e = e_vals[valid]

        logx = np.log10(x)
        logy = np.log10(y)

        # error propagation: sigma_logy = sigma_y / (y ln 10)
        sigma_logy = e / (y * np.log(10))
        valid_log = np.isfinite(sigma_logy) & (sigma_logy > 0)

        if valid_log.sum() < 3:
            return np.nan, np.nan, np.nan, x, y, e

        logx = logx[valid_log]
        logy = logy[valid_log]
        sigma_logy = sigma_logy[valid_log]

        # np.polyfit uses weights as 1/sigma, not 1/sigma^2
        weights = 1.0 / sigma_logy

        a, b = np.polyfit(logx, logy, 1, w=weights)

        # sigma ~ M^{-eta/2}
        eta = -2.0 * a

        return eta, a, b, x, y, e


    # Fiducial eta from original single sample
    eta, a, b, _, _, _ = fit_eta_from_sample(single, random_state_base=6000)

    # Bootstrap eta uncertainty by resampling single stars
    eta_boot = []
    n_eta_bootstrap = 1000
    rng = np.random.default_rng(12345)

    for j in range(n_eta_bootstrap):
        boot_indices = rng.choice(single.index.values, size=len(single), replace=True)
        boot_single = single.loc[boot_indices].copy()

        eta_j, _, _, _, _, _ = fit_eta_from_sample(
            boot_single,
            random_state_base=7000 + j * 20
        )

        if np.isfinite(eta_j):
            eta_boot.append(eta_j)

    eta_boot = np.array(eta_boot)

    if np.isfinite(eta) and len(eta_boot) >= 30:
        eta_err = np.std(eta_boot, ddof=1)

        x_fit = np.linspace(np.nanmin(x_s), np.nanmax(x_s), 200)
        y_fit = 10 ** (a * np.log10(x_fit) + b)

        ax.plot(
            x_fit, y_fit, "--",
            color=COLOR_SINGLE, lw=2,
            label=rf"Singles fit: $\eta={eta:.2f}\pm{eta_err:.2f}$"
        )

        print(f">>> Singles equipartition eta = {eta:.3f} ± {eta_err:.3f}")
        print(f">>> Number of valid eta bootstrap samples = {len(eta_boot)} / {n_eta_bootstrap}")

    elif np.isfinite(eta):
        x_fit = np.linspace(np.nanmin(x_s), np.nanmax(x_s), 200)
        y_fit = 10 ** (a * np.log10(x_fit) + b)

        ax.plot(
            x_fit, y_fit, "--",
            color=COLOR_SINGLE, lw=2,
            label=rf"Singles fit: $\eta={eta:.2f}$"
        )

        print(">>> Warning: eta uncertainty could not be robustly estimated.")


    ax.set_xlabel(r"System mass $M_{\rm sys}$ ($M_\odot$)", fontsize=14)
    ax.set_ylabel(r"2D velocity dispersion $\sigma_{\rm 2D}$ (km s$^{-1}$)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim(mass_bins[0], mass_bins[-1])

    # Auto y-limit
    all_y = np.concatenate([y_s[np.isfinite(y_s)], y_b[np.isfinite(y_b)]])
    if len(all_y) > 0:
        ax.set_ylim(0, max(0.8, np.nanmax(all_y) * 1.35))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Panel B saved: {save_path}")


def plot_panel_C_anisotropy(df, save_path):
    """
    Panel C: anisotropy profile beta = 1 - sigma_t^2 / sigma_r^2.
    """

    r_bins = np.linspace(0, 5, 6)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    fig, ax = plt.subplots(figsize=(8, 6))

    def bootstrap_beta(data, n_bootstrap=1000, random_state=None):
        if len(data) < 6:
            return np.nan, np.nan

        rng = np.random.default_rng(random_state)

        vr = data["v_rad"].values
        vt = data["v_tan_rot"].values

        vr = vr[np.isfinite(vr)]
        vt = vt[np.isfinite(vt)]

        if len(vr) < 6 or len(vt) < 6:
            return np.nan, np.nan

        s_r = np.std(vr, ddof=1)
        s_t = np.std(vt, ddof=1)

        if s_r <= 0:
            return np.nan, np.nan

        beta = 1.0 - (s_t**2 / s_r**2)

        betas = []
        idx = np.arange(len(data))

        for _ in range(n_bootstrap):
            boot_idx = rng.choice(idx, size=len(idx), replace=True)
            boot = data.iloc[boot_idx]

            s_r_b = np.std(boot["v_rad"].values, ddof=1)
            s_t_b = np.std(boot["v_tan_rot"].values, ddof=1)

            if s_r_b > 0:
                betas.append(1.0 - (s_t_b**2 / s_r_b**2))

        if len(betas) < 10:
            return beta, np.nan

        return beta, np.std(betas, ddof=1)

    def compute_anisotropy(subset, label, color, marker):
        subset = subset.copy()
        subset["r_bin"] = pd.cut(subset["R_pc"], r_bins, right=False)

        x_vals, y_vals, y_errs = [], [], []

        for i, interval in enumerate(subset["r_bin"].cat.categories):
            in_bin = subset[subset["r_bin"] == interval].copy()

            if len(in_bin) < 8:
                continue

            beta, beta_err = bootstrap_beta(
                in_bin,
                n_bootstrap=1000,
                random_state=2000 + i
            )

            if not np.isfinite(beta):
                continue

            x_vals.append(r_centers[i])
            y_vals.append(beta)
            y_errs.append(beta_err)

        if len(x_vals) > 0:
            ax.errorbar(
                x_vals, y_vals, yerr=y_errs,
                fmt=f"-{marker}", color=color,
                label=label, capsize=4,
                markersize=8, elinewidth=1.5, lw=2
            )

    single = df[df["Sample"] == "single"].copy()
    binary = df[df["Sample"] == "binary"].copy()

    compute_anisotropy(single, "Singles", COLOR_SINGLE, "o")
    compute_anisotropy(binary, "Binaries", COLOR_BINARY, "s")

    ax.axhline(0, color="k", ls="--", alpha=0.7)
    ax.axvspan(0, CORE_RADIUS_PC, color="gray", alpha=0.15,
               label=rf"Core $R<{CORE_RADIUS_PC:.0f}$ pc")

    ax.set_xlabel("Projected radius $R$ (pc)", fontsize=14)
    ax.set_ylabel(r"Anisotropy $\beta = 1 - \sigma_t^2/\sigma_r^2$", fontsize=14)
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Panel C saved: {save_path}")


def plot_panel_D_vector_map(df, save_path):
    """
    Panel D: binary velocity vector map.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    use = df[df["R_pc"] <= RMAX_MAIN_PC].copy()

    ax.scatter(
        use["X_pc"], use["Y_pc"],
        s=5, c=COLOR_GRAY, alpha=0.35,
        edgecolor="none", label="All sources"
    )

    binary = use[use["Sample"] == "binary"].copy()

    core = binary[binary["R_pc"] < CORE_RADIUS_PC].copy()
    halo = binary[binary["R_pc"] >= CORE_RADIUS_PC].copy()

    if len(core) > 60:
        core = core.sample(n=60, random_state=42)
    if len(halo) > 60:
        halo = halo.sample(n=60, random_state=42)

    scale = 15

    if len(core) > 0:
        ax.quiver(
            core["X_pc"], core["Y_pc"],
            core["v_east"], core["v_north"],
            color="purple", scale=scale, width=0.005,
            alpha=0.85, label="Core binaries", pivot="mid"
        )

    if len(halo) > 0:
        ax.quiver(
            halo["X_pc"], halo["Y_pc"],
            halo["v_east"], halo["v_north"],
            color=COLOR_BINARY, scale=scale, width=0.005,
            alpha=0.65, label="Outer binaries", pivot="mid"
        )

    # Velocity scale arrow
    ax.quiver(
        -0.80 * RMAX_MAIN_PC, -0.80 * RMAX_MAIN_PC,
        5, 0,
        color="black", scale=scale, width=0.008,
        headwidth=3, headlength=4, pivot="mid"
    )
    ax.text(
        -0.80 * RMAX_MAIN_PC, -0.88 * RMAX_MAIN_PC,
        "5 km s$^{-1}$", ha="center", va="top", fontsize=10
    )

    circle = plt.Circle(
        (0, 0), CORE_RADIUS_PC,
        color="gray", fill=False, ls="--", lw=1.5,
        label=rf"$R={CORE_RADIUS_PC:.0f}$ pc"
    )
    ax.add_patch(circle)

    lim = min(RMAX_MAIN_PC, 8.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_xlabel("East offset $X$ (pc)", fontsize=14)
    ax.set_ylabel("North offset $Y$ (pc)", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper right", fontsize=10,
              frameon=True, fancybox=False, edgecolor="black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Panel D saved: {save_path}")


def plot_projected_vs_3d_radius(df, save_path):
    """
    Diagnostic plot: projected radius vs 3D cluster-centric radius.
    """

    use = df[df["in_Rmax"]].copy()

    fig, ax = plt.subplots(figsize=(6.5, 6))

    for sample, color, label, alpha in [
        ("single", COLOR_SINGLE, "Singles", 0.45),
        ("binary", COLOR_BINARY, "Binaries", 0.65),
        ("intermediate", "gray", "Intermediate", 0.25)
    ]:
        sub = use[use["Sample"] == sample]
        if len(sub) == 0:
            continue

        ax.scatter(
            sub["R_pc"], sub["r3d_pc"],
            s=12, color=color, alpha=alpha,
            edgecolor="none", label=label
        )

    maxv = np.nanpercentile(use[["R_pc", "r3d_pc"]].values, 99)
    ax.plot([0, maxv], [0, maxv], "k--", lw=1.2, alpha=0.7)

    ax.set_xlabel("Projected radius $R$ (pc)", fontsize=13)
    ax.set_ylabel(r"3D radius $r_{\rm 3D}$ (pc)", fontsize=13)
    ax.set_xlim(0, maxv)
    ax.set_ylim(0, maxv * 1.1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Projected-vs-3D radius plot saved: {save_path}")


# ============================================================
# 5. Summary writer
# ============================================================

def write_summary(summary_R, summary_r3d, center_info, save_path):
    lines = []

    lines.append("Revised Pleiades binary-rich kinematics analysis summary")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Adopted cluster center and systemic motion")
    lines.append(f"RA0       = {center_info['ra0']:.6f} deg")
    lines.append(f"Dec0      = {center_info['dec0']:.6f} deg")
    lines.append(f"pmra0     = {center_info['pmra0']:.6f} mas/yr")
    lines.append(f"pmdec0    = {center_info['pmdec0']:.6f} mas/yr")
    lines.append(f"d0        = {center_info['dist0']:.3f} pc")
    lines.append("")

    def block(name, s):
        lines.append(name)
        lines.append("-" * 70)
        lines.append(f"N_single              = {s['N_single']}")
        lines.append(f"N_binary              = {s['N_binary']}")
        lines.append(f"KS p median           = {s['ks_p_median']:.4f}")
        lines.append(f"KS p 16--84           = {s['ks_p_p16']:.4f} -- {s['ks_p_p84']:.4f}")
        lines.append(f"AD p/sig median       = {s['ad_p_median']:.4f}")
        lines.append(f"AD p/sig 16--84       = {s['ad_p_p16']:.4f} -- {s['ad_p_p84']:.4f}")
        lines.append(f"W radius median       = {s['wd_radius_median']:.4f}")
        lines.append(f"W radius 16--84       = {s['wd_radius_p16']:.4f} -- {s['wd_radius_p84']:.4f}")
        lines.append(f"W mass median         = {s['wd_mass_median']:.4f}")
        lines.append(f"W mass 16--84         = {s['wd_mass_p16']:.4f} -- {s['wd_mass_p84']:.4f}")
        lines.append("")

    block("Projected radius R mass-matching test", summary_R)
    block("3D radius r3D mass-matching test", summary_r3d)

    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    print(f">>> Summary saved: {save_path}")

def diagnostic_radius_and_los(df):
    use = df[df["in_Rmax"]].copy()
    single = use[use["Sample"] == "single"].copy()
    binary = use[use["Sample"] == "binary"].copy()

    cols = [
        "System_Mass", "R_pc", "r3d_pc", "X3D_pc", "Y3D_pc", "Z3D_pc",
        "parallax", "parallax_adj", "parallax_error", "ruwe"
    ]

    print("\n>>> Diagnostic medians and percentiles")
    for name, sub in [("single", single), ("binary", binary)]:
        print(f"\n{name.upper()} N={len(sub)}")
        for col in cols:
            if col in sub.columns:
                arr = sub[col].replace([np.inf, -np.inf], np.nan).dropna().values
                if len(arr) == 0:
                    continue
                p16, p50, p84 = np.percentile(arr, [16, 50, 84])
                print(f"{col:15s}: {p50:8.4f}  [{p16:8.4f}, {p84:8.4f}]")

    # Direct tests
    for col in ["R_pc", "r3d_pc", "Z3D_pc", "parallax", "parallax_adj", "ruwe", "System_Mass"]:
        if col in use.columns:
            xs = single[col].replace([np.inf, -np.inf], np.nan).dropna().values
            xb = binary[col].replace([np.inf, -np.inf], np.nan).dropna().values
            if len(xs) > 5 and len(xb) > 5:
                ks = stats.ks_2samp(xs, xb)
                print(f"\n{col}: KS stat={ks.statistic:.4f}, p={ks.pvalue:.4e}")
                print(f"median single={np.median(xs):.4f}, median binary={np.median(xb):.4f}")


def repeated_caliper_matching_tests(df, radius_col="R_pc",
                                    n_match=1000,
                                    caliper=0.05,
                                    random_seed=42):

    rng = np.random.default_rng(random_seed)

    use = df[df["in_Rmax"]].copy()

    single = use[use["Sample"] == "single"].copy()
    binary = use[use["Sample"] == "binary"].copy()

    single = single[np.isfinite(single["System_Mass"]) & np.isfinite(single[radius_col])]
    binary = binary[np.isfinite(binary["System_Mass"]) & np.isfinite(binary[radius_col])]

    rows = []

    for i in range(n_match):
        ms, mb = caliper_mass_match(
            single, binary,
            mass_col="System_Mass",
            caliper=caliper,
            replace=False,
            rng=rng
        )

        if len(ms) < 20 or len(mb) < 20:
            continue

        r_s = ms[radius_col].values
        r_b = mb[radius_col].values

        m_s = ms["System_Mass"].values
        m_b = mb["System_Mass"].values

        ks_r = stats.ks_2samp(r_s, r_b)
        ks_m = stats.ks_2samp(m_s, m_b)

        ad_stat, ad_p = safe_ad_test(r_s, r_b, n_resamples=999,
                                     random_state=random_seed + i)

        rows.append({
            "iter": i,
            "n_pair": len(ms),
            "ks_radius_stat": ks_r.statistic,
            "ks_radius_p": ks_r.pvalue,
            "ad_radius_stat": ad_stat,
            "ad_radius_p": ad_p,
            "wd_radius": wasserstein_distance(r_s, r_b),
            "ks_mass_stat": ks_m.statistic,
            "ks_mass_p": ks_m.pvalue,
            "wd_mass": wasserstein_distance(m_s, m_b),
            "median_single_radius": np.median(r_s),
            "median_binary_radius": np.median(r_b),
            "median_single_mass": np.median(m_s),
            "median_binary_mass": np.median(m_b)
        })

    return pd.DataFrame(rows)

def repeated_strict_matched_diagnostics(df, n_match=500, caliper=0.05,
                                        random_seed=999, save_path=None):
    rng = np.random.default_rng(random_seed)

    use = df[df["in_Rmax"]].copy()
    single = use[use["Sample"] == "single"].copy()
    binary = use[use["Sample"] == "binary"].copy()

    rows = []

    for i in range(n_match):
        ms, mb = caliper_mass_match(
            single,
            binary,
            mass_col="System_Mass",
            caliper=caliper,
            replace=False,
            rng=rng
        )

        if len(ms) < 20:
            continue

        row = {"iter": i, "n_pair": len(ms)}

        for col in [
            "System_Mass", "R_pc", "r3d_pc", "Z3D_pc",
            "parallax", "parallax_adj", "dist_pc", "ruwe"
        ]:
            if col not in ms.columns or col not in mb.columns:
                continue

            xs = ms[col].replace([np.inf, -np.inf], np.nan).dropna().values
            xb = mb[col].replace([np.inf, -np.inf], np.nan).dropna().values

            if len(xs) < 5 or len(xb) < 5:
                continue

            ks = stats.ks_2samp(xs, xb)

            row[f"{col}_med_single"] = np.median(xs)
            row[f"{col}_med_binary"] = np.median(xb)
            row[f"{col}_delta_med_BminusS"] = np.median(xb) - np.median(xs)
            row[f"{col}_ks_p"] = ks.pvalue
            row[f"{col}_wd"] = wasserstein_distance(xs, xb)

        rows.append(row)

    diag = pd.DataFrame(rows)

    if save_path is not None:
        diag.to_csv(save_path, index=False)

    print("\n>>> Strict matched diagnostics, median over realizations")
    cols_to_print = [
        "System_Mass", "R_pc", "r3d_pc", "Z3D_pc",
        "parallax", "parallax_adj", "dist_pc", "ruwe"
    ]

    for col in cols_to_print:
        key_delta = f"{col}_delta_med_BminusS"
        key_p = f"{col}_ks_p"
        key_wd = f"{col}_wd"

        if key_delta in diag.columns:
            print(
                f"{col:14s} "
                f"DeltaMed(B-S)={diag[key_delta].median(): .5f}, "
                f"KS p={diag[key_p].median():.4e}, "
                f"WD={diag[key_wd].median():.5f}"
            )

    return diag

# ============================================================
# 6. Main program
# ============================================================

if __name__ == "__main__":

    df = load_and_clean_data(FILE_PATH)
    df, center_info = calculate_dynamics(df)

    # Restrict fiducial analysis to inner projected region
    df_main = df[df["R_pc"] <= RMAX_MAIN_PC].copy()

    print("")
    print(">>> Fiducial inner sample")
    print(f"    R <= {RMAX_MAIN_PC:.1f} pc")
    print(df_main["Sample"].value_counts())
    print("")

    # Save processed catalog
    processed_path = os.path.join(OUTDIR, "pleiades_processed_revised.csv")
    df.to_csv(processed_path, index=False)
    print(f">>> Processed catalog saved: {processed_path}")

    # --------------------------------------------------------
    # Repeated mass matching for projected radius R
    # --------------------------------------------------------
    stats_R, summary_R, cdf_R = repeated_mass_matching_tests(
        df,
        radius_col="R_pc",
        n_match=N_MATCH,
        n_neighbor=N_NEIGHBOR_MASS,
        random_seed=42
    )

    stats_R_path = os.path.join(OUTDIR, "mass_matching_R_stats.csv")
    stats_R.to_csv(stats_R_path, index=False)
    print(f">>> R mass-matching stats saved: {stats_R_path}")

    plot_panel_A_massmatched_cdf(
        cdf_R,
        summary_R,
        os.path.join(OUTDIR, "PanelA_CDF_massmatched_R.png"),
        xlabel="Projected radius $R$ (pc)"
    )

    # --------------------------------------------------------
    # Repeated mass matching for 3D radius r3D
    # --------------------------------------------------------
    stats_r3d, summary_r3d, cdf_r3d = repeated_mass_matching_tests(
        df,
        radius_col="r3d_pc",
        n_match=N_MATCH,
        n_neighbor=N_NEIGHBOR_MASS,
        random_seed=4242
    )

    stats_r3d_path = os.path.join(OUTDIR, "mass_matching_r3d_stats.csv")
    stats_r3d.to_csv(stats_r3d_path, index=False)
    print(f">>> r3D mass-matching stats saved: {stats_r3d_path}")

    plot_panel_A_massmatched_cdf(
        cdf_r3d,
        summary_r3d,
        os.path.join(OUTDIR, "PanelA_CDF_massmatched_r3D.png"),
        xlabel=r"3D cluster-centric radius $r_{\rm 3D}$ (pc)"
    )

    # --------------------------------------------------------
    # Additional plots
    # --------------------------------------------------------
    plot_mass_distribution_matching(
        df,
        os.path.join(OUTDIR, "Appendix_mass_distribution_matching.png")
    )

    plot_panel_B_heating(
        df_main,
        os.path.join(OUTDIR, "PanelB_Heating.png")
    )

    plot_panel_C_anisotropy(
        df_main,
        os.path.join(OUTDIR, "PanelC_Anisotropy.png")
    )

    plot_panel_D_vector_map(
        df_main,
        os.path.join(OUTDIR, "PanelD_VectorMap.png")
    )

    plot_projected_vs_3d_radius(
        df,
        os.path.join(OUTDIR, "Appendix_projected_vs_3D_radius.png")
    )

    # --------------------------------------------------------
    # Write summary
    # --------------------------------------------------------
    write_summary(
        summary_R,
        summary_r3d,
        center_info,
        os.path.join(OUTDIR, "analysis_summary.txt")
    )
    diagnostic_radius_and_los(df)
    strict_R = repeated_caliper_matching_tests(
    df, radius_col="R_pc", n_match=1000, caliper=0.05, random_seed=123
    )

    strict_r3d = repeated_caliper_matching_tests(
        df, radius_col="r3d_pc", n_match=1000, caliper=0.05, random_seed=456
    )

    strict_R.to_csv(os.path.join(OUTDIR, "strict_caliper_matching_R.csv"), index=False)
    strict_r3d.to_csv(os.path.join(OUTDIR, "strict_caliper_matching_r3d.csv"), index=False)

    print("\nSTRICT R")
    print(strict_R[["n_pair", "ks_radius_p", "ad_radius_p", "wd_radius", "ks_mass_p", "wd_mass",
                    "median_single_radius", "median_binary_radius"]].median())

    print("\nSTRICT r3D")
    print(strict_r3d[["n_pair", "ks_radius_p", "ad_radius_p", "wd_radius", "ks_mass_p", "wd_mass",
                    "median_single_radius", "median_binary_radius"]].median())

    print("")
    print(">>> All revised analysis products generated successfully.")
    print(f">>> Output directory: {OUTDIR}")
    df_ruwe = df[df["ruwe"] < 1.4].copy()

    strict_R_ruwe = repeated_caliper_matching_tests(
        df_ruwe,
        radius_col="R_pc",
        n_match=1000,
        caliper=0.05,
        random_seed=1234
    )

    strict_r3d_ruwe = repeated_caliper_matching_tests(
        df_ruwe,
        radius_col="r3d_pc",
        n_match=1000,
        caliper=0.05,
        random_seed=5678
    )

    print("\nSTRICT R, RUWE < 1.4")
    print(strict_R_ruwe[[
        "n_pair", "ks_radius_p", "ad_radius_p", "wd_radius",
        "ks_mass_p", "wd_mass",
        "median_single_radius", "median_binary_radius"
    ]].median())

    print("\nSTRICT r3D, RUWE < 1.4")
    print(strict_r3d_ruwe[[
        "n_pair", "ks_radius_p", "ad_radius_p", "wd_radius",
        "ks_mass_p", "wd_mass",
        "median_single_radius", "median_binary_radius"
    ]].median())

    strict_R_ruwe.to_csv(
        os.path.join(OUTDIR, "strict_caliper_matching_R_ruwe14.csv"),
        index=False
    )

    strict_r3d_ruwe.to_csv(
        os.path.join(OUTDIR, "strict_caliper_matching_r3d_ruwe14.csv"),
        index=False
    )
    diag_strict = repeated_strict_matched_diagnostics(
    df,
    n_match=500,
    caliper=0.05,
    random_seed=999,
    save_path=os.path.join(OUTDIR, "strict_matched_diagnostics.csv")
)
