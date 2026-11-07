# SPP2489 Experimental EBSD–Neper Calibration V2

## Scope and data governance

This study improves the existing periodic Neper calibration and tests its mechanical significance with the existing MOOSE FCC crystal-plasticity model. No constitutive equations or private EBSD records were modified. Raw and cleaned per-grain experimental tables remain outside the public deliverable.

The phase column is still a numeric label without independently verified metallurgical phase metadata. Any MOOSE calculation below is therefore a single-phase FCC numerical benchmark, not a validated prediction for the experimental alloy.

## Reproducibility and statistical protocol

- 500 grain-level bootstrap replicates and 500 matched-size experimental   sample pairs were evaluated per dataset.
- Grain-level bootstrap quantifies finite-grain sampling only; a single EBSD   field cannot estimate field-to-field or specimen-to-specimen variability.
- Geometry calibration excludes ODF discrepancy. Number- and area-weighted   ODF distances are validated separately.
- Four real Neper controls were searched jointly: diameter spread scale,   directional ratio, shape-proxy scale, and input size–shape rank coupling.
- Parameter selection used calibration seeds only. Reported validation seeds   were withheld from the corresponding selection decision.
- Al0 production searches and all new validation runs stop at a fixed 60,000   Neper iterations. The Al1 parameter searches predate this correction and   are retained with that provenance; their selected parameters are checked   separately using new fixed-iteration confirmation seeds.
- The fixed-iteration rule replaces the legacy 60 s wall-clock stop, which   produced load-dependent differences. An independent duplicate test gave   identical tessellation and grain-table SHA-256 values.

## Independent statistical validation

| Dataset | Valid baseline | Valid v2 | J baseline | J v2 | Change | Mean ECD error baseline→v2 | Mean AR error baseline→v2 | |Δρ| baseline→v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Al1_oBD | 5 | 5 | 0.1600 | 0.0741 | 53.69% | 8.44%→4.07% | 20.67%→3.63% | 0.098→0.029 |
| Al1_pBD | 5 | 5 | 0.1621 | 0.1621 | 0.00% | 12.91%→12.91% | 9.57%→9.57% | 0.235→0.235 |
| Al0_00 | 4 | 4 | 0.1695 | 0.1139 | 32.80% | 8.02%→0.75% | 20.35%→15.14% | 0.201→0.134 |
| Al0_90 | 5 | 5 | 0.1299 | 0.1083 | 16.65% | 11.90%→3.10% | 4.12%→6.28% | 0.182→0.250 |

Al1_pBD requires special interpretation. The initial four-parameter winner overfit its screening seed and completed only three of five first holdouts. A new robust calibration set and a second independent holdout selected the original baseline parameters. Consequently, no robust pBD parameter improvement is claimed; small A/B differences for identical parameters in the legacy time-stopped runs are numerical non-determinism, not calibration gain.

Robust pBD selected candidate: `0`; reason: The first selected candidate failed 2/5 initial holdouts; those seeds were retired from independent validation before this robust refinement.

## Selected four-parameter controls

| Dataset | diameter spread scale | directional ratio | shape-proxy scale | input rank coupling |
|---|---:|---:|---:|---:|
| Al1_oBD | 1.047670 | 1.770556 | 0.933989 | 0.841938 |
| Al1_pBD | 0.950000 | 2.450000 | 1.000000 | 1.000000 |
| Al0_00 | 1.184686 | 2.102194 | 0.923010 | 0.910606 |
| Al0_90 | 1.160823 | 2.729768 | 0.500989 | 0.460177 |

Every selected point is linked to its candidate ID, calibration seeds, holdout seeds, objective components, failure records, and selection rule in `calibration_history.csv`, `optimized_parameters.json`, and `independent_validation.json`. Al1_pBD deliberately retains the baseline point after robust reselection.

## Fixed-iteration confirmation of legacy Al1 selections

| Dataset | Valid baseline | Valid selected | J baseline | J selected |
|---|---:|---:|---:|---:|
| Al1_oBD | 5 | 5 | 0.1600 | 0.0741 |
| Al1_pBD | 5 | 5 | 0.1621 | 0.1621 |

These seeds confirm the already selected parameters; they were not used to reselect or tune them. Al1_pBD baseline and selected parameters are identical, so its rows intentionally reuse the exact same generated geometries and cannot imply a calibration gain.

## Why mean ECD was systematically high

At fixed area A and grain count N, the tessellation fixes E[d²] = 4A/(πN), not E[d]. The baseline synthetic diameter distributions were too narrow. For a population coefficient of variation CV, E[d] = sqrt(E[d²]/(1+CV²)); therefore an underestimated CV necessarily raises the mean diameter. The screening sensitivities consistently identify the diameter log-spread scale as the dominant control. Joint calibration nearly removed the mean bias for Al0_00 and materially reduced it for Al1_oBD and Al0_90.

## Aspect ratio and joint-statistics limitations

The ellipse-derived EBSD AR target is not equivalent to final Laguerre polygon AR. Global `aspratio`, a circularity proxy, and input rank coupling cannot independently control mean AR, the full AR distribution, and size-conditioned AR. Al0_00 demonstrates the limitation most clearly: ECD metrics improve strongly while conditional-AR error increases. Al0_90 also trades improved ECD for worse mean AR and size–AR rank correlation. These are expression-capability limits, not finite-sample effects alone.

The `alpha` CSV column was not used as a grain-wise principal direction; it is a fixed 0°/90° section label in the available data.

## ODF sampling

Whole Bunge triplets are sampled on SO(3) with cubic-symmetry-aware descriptors; Euler components are never sampled independently. Across all four datasets, number-weighted ODF discrepancies lie inside the matched-size experimental–experimental distribution. Area-weighted ODF variability is larger because a few coarse grains dominate. The observed ODF errors are therefore principally finite-grain orientation-sampling uncertainty, not a geometry-calibration failure.

## Meshability and geometry-level correction

The uncorrected broad-tail Laguerre tessellations contain acute triple-junction geometry. Neper 5.0 post-regularization segfaulted for these periodic, cell-wise-targeted tessellations. A conservative optional planar-graph correction was therefore used only for the mechanical pilot: shared vertices below an activation threshold are moved, boundary vertices stay fixed, and every trial must preserve convex positive-area polygons, shared edges, topology, IDs, domain area, and periodic pairs.

| Family | TRI6 | min quality | max aspect | periodic error (µm) | area L1 change | area correlation |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 15738 | 0.1018 | 17.24 | 8.282e-11 | 3.22% | 0.998985 |
| improved | 15756 | 0.1229 | 14.38 | 3.220e-10 | 2.25% | 0.999212 |

## FCC crystal-plasticity pilot

Both paired Al1_oBD cases completed with the unchanged A4 FCC model, the same material parameters, loading, 8 MPI ranks, and h=6 µm mesh policy.

This is the requested full-statistics Case A pilot, not strict single-descriptor causal isolation. Both geometries draw the same number-weighted orientation set (identical number-weighted ODF discrepancy for seed 10303), while altered grain areas change the area-weighted ODF discrepancy from 0.4499 to 0.4663. Consequently, morphology and area-weighted texture effects remain partly confounded in this pilot.

| Family | wall (s) | increments | Newton iters | failed attempts | proof (MPa) | flow (MPa) | elemental Q95 slip |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 732.83 | 57 | 210 | 7 | 288.001505 | 327.158944 | 0.08561631 |
| improved | 900.40 | 61 | 229 | 9 | 290.728906 | 328.752865 | 0.08356290 |

Paired relative differences:

- proof stress: 0.95%
- flow stress: 0.49%
- area-weighted elemental-field Q95 accumulated slip: 2.40%
- maximum elemental accumulated slip: 6.54%

In this one paired realization the two macroscopic targets move by less than 1%, whereas the local slip-tail measures move by roughly 2.4% (Q95) and 6.5% (maximum). This is evidence that local plasticity is more sensitive than the homogenized response, but it is not yet a significance claim because realization, mesh, time, and raw-QP uncertainties have not all been bounded.

The h=6 µm pilot still under-resolves the extreme smallest-grain tail; therefore it is a feasibility/sensitivity calculation, not a mesh-converged experimental prediction. The reported Q95 is an element-area-weighted quantile of the elemental Exodus field, not a raw integration-point quantile; a true JxW-weighted quadrature-point export remains required for the final local-tail verification. Any mechanical difference at or below mesh/time/realization uncertainty must be treated as insignificant.

## Answers to the ten research questions

1. Mean ECD bias is primarily the consequence of an under-dispersed    diameter distribution under fixed area and N, not an incorrect area.
2. Four-parameter calibration removes most of that bias for three datasets;    it is not robustly improved for Al1_pBD.
3. Low-AR mean bias improves for Al1_oBD but remains large for Al0_00; the    current proxy is not fully expressive.
4. Size–AR correlation is not reproduced stably. The input copula coupling    does not map one-to-one to final polygon correlation.
5. ECD residuals are mainly optimization/finite-N; persistent conditional-AR    residuals are mainly parameterization limits; uncertainty tables separate    both from finite-sample noise.
6. ODF discrepancy is largely explained by finite-grain sampling, especially    for area-weighted descriptors.
7. Al1_oBD and Al0_90 improvements survive independent seeds; Al0_00 improves    with one invalid realization per family; Al1_pBD does not show a robust    parameter improvement.
8. The completed single paired FEM pilot changes proof and flow stress by    less than 1%. Without paired mesh/time/realization intervals this is not    evidence of a significant macroscopic effect.
9. The elemental-field Q95 and maximum slip change more than the macro    targets in the pilot, supporting greater local sensitivity, but the    raw-QP and uncertainty qualifications above remain mandatory.
10. Recommended future Composition–Microstructure GP outputs are diameter     spread/CV and tail quantiles, AR marginal descriptors, a separate     size-conditioned AR descriptor, and symmetry-aware ODF descriptors with     uncertainty. A single mean size, mean AR, or input copula coefficient is     insufficient.

## Verification status

- Experimental-calibration Python tests: 15 passed.
- Two-parameter inverse-design Python tests: 15 passed in the   `spp2489-inverse` environment.
- Texture–morphology regression tests: 11 passed.
- MOOSE TestHarness: 8 passed, 0 failed.
- Both 298-grain, 8-rank FCC-CP pilots converged and passed the periodic   constraints, P22/shear controls, Hill–Mandel residual, positive   dissipation, slip resistance, finite-state, and det(Fp) checks.
- The pilot has not completed paired coarse/fine, time-refinement, or   realization-ensemble mechanics. Those missing uncertainty bounds are   why no statistical-significance claim is made.

## Reproduction

```bash
python scripts/experimental_microstructure_calibration/bootstrap_ebsd_statistics.py --help
python scripts/experimental_microstructure_calibration/optimize_neper_joint_statistics.py --help
python scripts/experimental_microstructure_calibration/confirm_fixed_iteration_validation.py --help
python scripts/experimental_microstructure_calibration/build_mechanical_pilot.py --help
python scripts/experimental_microstructure_calibration/finalize_calibration_v2.py
python scripts/experimental_microstructure_calibration/write_calibration_v2_report.py
```

The local experimental data root must remain private. Public results contain aggregate statistics, synthetic figures, code, and this report only.
