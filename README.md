# Pleiades Binary Dynamics from Gaia DR3

This repository contains data products and analysis scripts for exploring the spatial and kinematic properties of unresolved binaries in the **Pleiades** using **Gaia DR3** member stars and multiband binary-model fitting results.

The current workflow combines:

- a Gaia DR3 member catalog of the Pleiades,
- a catalog of binary probabilities and fitted binary parameters,
- a Python analysis script that derives projected positions, tangential velocities, and several diagnostic figures.

---

## Repository contents

### 1. `Pleiades_GAIA_ALL.csv`
Gaia DR3 catalog of Pleiades member stars.

This file contains the Gaia astrometric and photometric measurements for candidate or confirmed Pleiades members, including for example:

- sky position (`ra`, `dec`)
- parallax and uncertainty
- proper motion (`pmra`, `pmdec`) and uncertainties
- RUWE
- Gaia photometry
- radial velocity information where available
- stellar parameter estimates from Gaia DR3

In the current analysis pipeline, this file provides the **astrometric baseline sample**.

---

### 2. `member.txt`
Catalog of binary-model fitting results for Pleiades members.

This file corresponds to the table described in:

**Liu, R., Shao, Z., Li, L.**  
*Photometric Determination of Unresolved Main-sequence Binaries in the Pleiades: Binary Fraction and Mass Ratio Distribution*

It includes, for stars with membership probability \( P_k \ge 0.5 \), the following quantities:

- `Gaia` : Gaia DR3 source identifier
- `M1` : best-fit primary mass
- `q` : best-fit mass ratio \( q = M_2/M_1 \)
- quantiles for `M1` and `q`
- `Pb` : probability of being a binary
- `Flag` : exclusion flag(s)

This file is the **binary classification and stellar-mass input catalog**.

> See the header in `member.txt` for the original byte-by-byte documentation and field definitions.

---

### 3. `heartbeat_v4.py`
Main analysis script.

This Python script:

1. loads the merged input catalog,
2. performs basic cleaning,
3. applies a parallax correction / shrinkage estimate,
4. computes projected positions and tangential velocities,
5. divides the sample into `single`, `binary`, and `intermediate`,
6. produces four analysis panels:

- **Panel A** — radial cumulative distribution function (CDF)
- **Panel B** — velocity dispersion versus system mass
- **Panel C** — anisotropy profile
- **Panel D** — binary velocity vector map

Output figures are written as PNG files.

---

## Scientific purpose

The repository is designed to test whether unresolved binaries in the Pleiades show measurable differences in:

- **radial segregation**
- **kinematic heating**
- **velocity anisotropy**
- **projected vector-flow structure**

relative to likely single stars.

This is intended as a data-driven dynamical exploration using Gaia astrometry plus photometrically inferred binary information.

---

## Expected workflow

The script assumes that the Gaia member catalog and the binary-fitting catalog have already been merged into a working CSV file named:

```bash
pleiades_merged.csv
