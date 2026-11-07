# Experimental EBSD → Neper Statistical Calibration

## Scope

This workflow calibrates periodic two-dimensional Neper tessellations to experimental EBSD grain statistics. It does not run MOOSE, crystal plasticity, constitutive calibration, or mechanical inverse design.

The four sections are analyzed independently. No unique three-dimensional reconstruction or oBD/pBD versus 00/90 coordinate equivalence is claimed without the original acquisition metadata.

## Data provenance and cleaning

Raw private CSV files are read through `SPP2489_EBSD_DATA_ROOT`; absolute private paths and raw records are not included in the public repository. Source filenames and SHA-256 hashes are retained locally.

The cleaned population removes invalid/duplicate records and grains below the explicit 2.8 μm equivalent-diameter threshold. It does not silently remove the upper grain-size or aspect-ratio tails.

The private-source MTEX script explicitly selects `fcc` grains before applying the 2.8 μm threshold. The exported CSVs contain only numeric phase label `0`; the workflow preserves that label because no explicit numeric phase-index mapping was found. Separate DRAGen preprocessing applies `b ≥ 0.5 μm` for geometry/meshing, which is documented but not substituted for the experimental-statistics filter.

The CSV `alpha` field is constant by section (0° or 90°), and the Al1 export code explicitly assigns it from the filename. It is section metadata, not a measured per-grain principal-axis direction. The direction discrepancy is therefore reported diagnostically with zero calibration weight; a true direction distribution cannot be recovered from these four tables.

Because the direction is unavailable, `r_x/r_y` and its reciprocal cannot be distinguished from AR alone. Calibration therefore constrains the reported ratio to ≥1 as an unsigned major/minor elongation convention; it must not be interpreted as a measured preference for the global x direction.

| Dataset | Population | Grains | Mean area (μm²) | Mean ECD (μm) | CV ECD | Mean AR | Spearman size–AR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Al1_oBD | raw_valid_geometry | 2437 | 416.0497 | 14.2419 | 1.2698 | 2.5847 | 0.1312 |
| Al1_oBD | cleaned | 1888 | 536.1440 | 17.8240 | 1.0721 | 2.6723 | 0.0122 |
| Al1_pBD | raw_valid_geometry | 848 | 817.4607 | 17.6616 | 1.5295 | 4.0819 | 0.3789 |
| Al1_pBD | cleaned | 659 | 1051.0315 | 22.1764 | 1.3129 | 4.5137 | 0.2617 |
| Al0_00 | raw_valid_geometry | 2632 | 259.4873 | 12.1737 | 1.1090 | 2.5778 | 0.0908 |
| Al0_00 | cleaned | 2114 | 322.2792 | 14.6736 | 0.9519 | 2.6485 | 0.0545 |
| Al0_90 | raw_valid_geometry | 1519 | 657.1549 | 17.8561 | 1.2749 | 3.9940 | 0.2310 |
| Al0_90 | cleaned | 1366 | 730.3065 | 19.6036 | 1.1919 | 4.1552 | 0.1565 |

## Statistical model

ECD candidates are lognormal, Weibull, gamma, and a two-component lognormal mixture, ranked by BIC and checked against empirical CDF and q95/q99 tail error. Aspect ratio is fitted as the positive excess `AR − 1`, preserving `AR ≥ 1`.

Texture is represented by complete Bunge rotations using the same crystal-to-sample convention as SPP2489. The pipeline uses all 24 proper cubic symmetry rotations and empirical SO(3) sampling; Euler components are never sampled independently. Texture statistics remain phase-separated.

| Dataset | ECD model selected by BIC | ECD BIC | ECD CDF sup. error | AR−1 model selected by BIC | AR−1 BIC |
|---|---|---:|---:|---|---:|
| Al1_oBD | two_component_lognormal_mixture | 14022.569 | 0.0293 | two_component_lognormal_mixture | 5469.945 |
| Al1_pBD | two_component_lognormal_mixture | 5171.724 | 0.0238 | gamma | 2962.495 |
| Al0_00 | two_component_lognormal_mixture | 14810.161 | 0.0289 | two_component_lognormal_mixture | 6218.176 |
| Al0_90 | two_component_lognormal_mixture | 10419.530 | 0.0457 | gamma | 5794.049 |

BIC is not treated as sufficient by itself: empirical CDF, q95/q99 tail error, and the achieved Neper distributions are retained for review.

## Neper calibration and independent validation

Neper uses paired cell-wise `diameq:file(...)` and `1-circularity:file(...)` targets so wide tails and size--shape dependence are not forced into independent one-parameter distributions. The circularity target is a documented fitted-ellipse proxy because the CSVs do not contain measured grain perimeter. A deterministic candidate matrix calibrates ECD spread and global aspect ratio; an explicit input copula parameter compensates the Laguerre generator's intrinsic size--shape bias without changing either target marginal. Five disjoint validation seeds quantify realization variability.

All final tessellations use the same 60 s native Neper morphology-optimization budget. The outer process timeout is only a fault guard; it is not used as a surrogate convergence result.

| Dataset | Recommended side (μm) | Cells | Spread | x/y ratio | Input size–shape ρ | Evaluations | Validation objective mean ± SD (max) | Experimental / synthetic size–AR ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Al1_oBD | 400 | 298 | 0.950000 | 1.256606 | 0.750 | 4 | 0.145781 ± 0.010145 (0.161879) | 0.0122 / -0.0982 |
| Al1_pBD | 400 | 152 | 0.950000 | 2.450000 | 1.000 | 4 | 0.169229 ± 0.009218 (0.179748) | 0.2617 / 0.0382 |
| Al0_00 | 400 | 496 | 0.950000 | 1.260000 | 0.750 | 4 | 0.141754 ± 0.002657 (0.143817) | 0.0545 / -0.1473 |
| Al0_90 | 400 | 219 | 0.950000 | 2.250000 | 1.000 | 4 | 0.155121 ± 0.012848 (0.176873) | 0.1565 / 0.0017 |

The discrepancy combines normalized ECD and AR Wasserstein distances, size–AR rank-correlation error, and a cubic-IPF texture histogram distance. The section-label `alpha` direction term has zero weight. The complete component values and weights are stored in the validation JSON.

The final output correlations are substantially closer to experiment than the uncorrected generator, but they are not exact. In particular, Al1_pBD retains a lower output size–AR correlation than the experiment. This is reported as a capability limit of the present Laguerre + global-`aspratio` parameterization, not hidden by selecting a favorable realization.

## RVE-size representativeness

| Dataset | Side (μm) | Grains | Objective mean ± SD | Worst realization | Maximum area error |
|---|---:|---:|---:|---:|---:|
| Al1_oBD | 200 | 75 | 0.198491 ± 0.011008 | 0.211868 | 9.095e-16 |
| Al1_oBD | 400 | 298 | 0.145781 ± 0.010145 | 0.161879 | 9.095e-16 |
| Al1_pBD | 200 | 38 | 0.286008 ± 0.021370 | 0.313420 | 1.273e-15 |
| Al1_pBD | 400 | 152 | 0.169229 ± 0.009218 | 0.179748 | 1.091e-15 |
| Al1_pBD | 200 | — | failed (seed 8304) | — | — |
| Al0_00 | 200 | 124 | 0.173952 ± 0.014719 | 0.192240 | 1.455e-15 |
| Al0_00 | 400 | 496 | 0.141754 ± 0.002657 | 0.143817 | 7.276e-16 |
| Al0_90 | 200 | 55 | 0.226765 ± 0.011890 | 0.243224 | 1.637e-15 |
| Al0_90 | 400 | 219 | 0.155121 ± 0.012848 | 0.176873 | 7.276e-16 |

The RVE assessment uses the ensemble, not the single closest seed. A size is not declared representative solely because its mean objective is low; the change in ensemble spread and tail coverage is also inspected.

All four datasets recommend a 400 μm periodic RVE. The 200 μm Al1_pBD ensemble contains one invalid-polygon failure (seed 8304), while its 400 μm ensemble completes 5/5 realizations. The attempted 800 μm study was stopped after exceeding the declared 30-minute wall-time cap and is not reported as completed.

## Reproduction

```bash
export SPP2489_EBSD_DATA_ROOT=/read-only/path/to/damic_spp2489
python scripts/experimental_microstructure_calibration/run_calibration.py --statistics-only --require-reference-count
# Run the reviewed Neper candidate/validation commands documented in the JSON history.
python scripts/experimental_microstructure_calibration/finalize_calibration_results.py
python scripts/experimental_microstructure_calibration/export_public_results.py
```

All generated experimental derivatives remain under `results/experimental_microstructure_calibration/`, which is intentionally excluded from public Git tracking pending data-sharing approval.
