# SPP2489 Morphology--Texture FCC Crystal-Plasticity Study

## Executive result

This study tested whether a directional grain-morphology parameter provides a
resolved third design variable, in addition to the validated texture parameters
`(eta, theta)`, for the two-dimensional plane-strain, 20-grain FCC
crystal-plasticity RVE.

The answer for the present model and RVE size is **no at the macroscopic level**.
Grain morphology changes local accumulated-slip patterns and changes the
macroscopic response of an individual geometry realization by about 0.3%, but
the paired morphology contrast reverses sign across deterministic geometry
seeds. The realization-to-realization variation is substantially larger than
the five-level morphology range. Consequently, the predeclared scientific gate
rejected the three-variable Bayesian optimization. No synthetic BO data,
`best_design.json`, optimization history, or animation was generated.

This is a negative scientific result, not a solver failure. All 17 new MOOSE
FEM calculations converged and passed the physical checks.

## Immutable model and loading

- Application baseline: commit `be198402` plus this incremental workflow.
- Constitutive model: existing A4 finite-strain FCC crystal plasticity; no
  material-model changes were made.
- Geometry: 200 x 200 micrometre periodic RVE, 20 grains.
- Mesh: conforming Gmsh TRI6, fully bonded grain boundaries.
- Texture: `eta = 0.7729026253348713`,
  `theta = 4.242186925858952 degrees`.
- Loading: `F11 = 1 + 0.02 t`, `P22 = 0`, zero in-plane macro shear,
  `F33 = 1`, `0 <= t <= 1`.
- Time integration: the existing `adaptive_strict` protocol; selected cases
  were repeated with `adaptive_refined`.
- Parallel execution: 8 MPI ranks.
- Material parameters, tolerances, slip systems, hardening law, and Grain-ID to
  orientation mapping were fixed.

The proof stress is the existing 0.2% offset definition under the prescribed
plane-strain mixed macroscopic constraints. It is not a standard three-dimensional
uniaxial experimental yield stress.

## Native Neper morphology family

The installed Neper 5.0.0 supports the verified command form
`-morpho voronoi,aspratio(a,1)` together with `-periodicity all`. The generator
uses a fixed seed and records the complete command, return code, log, and file
hashes. The implementation follows the
[Neper tessellation documentation](https://neper.info/doc/neper_t.html).

For seed 2026, `a=1` reproduces the previous tessellation, coarse mesh, and fine
mesh byte-for-byte. Their SHA256 hashes match the existing two-parameter
inverse-design baseline. This provides an exact continuity point rather than a
new random baseline.

| Input a | Achieved directional a | Area CV | Coarse elements | Minimum quality | Maximum aspect ratio |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 0.712154 | 0.399810 | 3260 | 0.05441 | 31.83 |
| 0.85 | 0.877418 | 0.397070 | 3274 | 0.14548 | 11.89 |
| 1.00 | 1.034526 | 0.390092 | 3280 | 0.10101 | 17.77 |
| 1.25 | 1.240719 | 0.374549 | 3292 | 0.26225 | 7.13 |
| 1.50 | 1.432087 | 0.354224 | 3274 | 0.03181 | 56.07 |

The two extreme geometries contain acute triple-junction corners. They were not
silently replaced by a morphology-optimized tessellation because that procedure
was discontinuous near `a=1`. Instead, the quality risk is recorded explicitly
and both extremes were repeated on fine meshes from the identical continuous
tessellation.

Periodic boundary-crossing grains are measured in the unwrapped Neper torus
representation. Grain area, perimeter, central second moments, directional
extents, principal lengths, principal direction, circularity, and area-weighted
texture descriptors are retained in the compact result files.

## Homogeneous single-crystal control

All 20 grain blocks were assigned the same crystal orientation and constitutive
parameters, and only the morphology partition was changed.

| Input a | Proof stress (MPa) | Flow stress (MPa) | Local P11 range (MPa) |
|---:|---:|---:|---:|
| 0.70 | 404.026464891921 | 475.211101839340 | 8.98e-9 |
| 1.00 | 404.026464891927 | 475.211101839440 | 1.91e-8 |
| 1.50 | 404.026464891908 | 475.211101840840 | 5.82e-9 |

The relative proof and flow spreads are `4.57e-14` and `3.16e-12`. Thus the
grain partition, periodic mapping, material blocks, and fully bonded interfaces
do not create an artificial morphology effect.

## Five-level morphology sensitivity

| Input a | Achieved a | Proof stress (MPa) | 2% flow stress (MPa) | Mean accumulated slip | Maximum accumulated slip | Wall time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 0.712154 | 358.920022 | 415.490977 | 0.0585334 | 0.151899 | 117.83 |
| 0.85 | 0.877418 | 358.106243 | 414.956822 | 0.0584559 | 0.147477 | 103.21 |
| 1.00 | 1.034526 | 358.328052 | 414.865112 | 0.0585031 | 0.140346 | 102.34 |
| 1.25 | 1.240719 | 359.241188 | 414.774882 | 0.0585159 | 0.139737 | 119.82 |
| 1.50 | 1.432087 | 359.086416 | 414.247071 | 0.0584242 | 0.164604 | 101.20 |

The seed-2026 morphology ranges are:

- proof stress: `1.134946 MPa`, or `0.3167%` of the `a=1` value;
- flow stress: `1.243906 MPa`, or `0.2998%` of the `a=1` value;
- mean accumulated slip: `1.09206e-4`.

Local accumulated-slip distributions show visible redistribution even though
the homogenized response curves nearly overlap.

## Numerical uncertainty

Fine meshes were generated from the same tessellations for `a=0.70`, `1.00`,
and `1.50`. The two extremes were also repeated with the refined adaptive-time
protocol.

| a | Proof coarse-to-fine change (MPa) | Flow coarse-to-fine change (MPa) |
|---:|---:|---:|
| 0.70 | -0.01454 | -0.07433 |
| 1.00 | +0.14750 | -0.09253 |
| 1.50 | -0.09142 | -0.11037 |

| a | Proof refined-time change (MPa) | Flow refined-time change (MPa) |
|---:|---:|---:|
| 0.70 | +0.05456 | +0.08047 |
| 1.50 | +0.30363 | +0.08557 |

The conservative numerical envelopes are therefore `0.303629 MPa` for proof
stress and `0.110368 MPa` for flow stress.

## Geometry-realization robustness and the decision gate

Additional deterministic geometry seeds were evaluated at `a=1` and `a=1.5`
with the same orientation list and Grain-ID mapping.

| Geometry seed | Proof at a=1 (MPa) | Proof contrast a=1.5-a=1 (MPa) | Flow at a=1 (MPa) | Flow contrast a=1.5-a=1 (MPa) |
|---:|---:|---:|---:|---:|
| 2026 | 358.328052 | +0.758364 | 414.865112 | -0.618041 |
| 2027 | 364.053757 | -0.885005 | 420.721820 | -3.470392 |
| 2028 | 358.729443 | +0.882987 | 412.149153 | +1.027605 |

At `a=1`, the realization spans are `5.725705 MPa` for proof stress and
`8.572668 MPa` for flow stress. Both paired morphology contrasts reverse sign.
Changing morphology also changes the Grain-ID area fractions, and hence the
area-weighted texture descriptor; the compact verification data quantify this
confounder rather than assigning every stress difference to aspect ratio.

The gate was fixed before the realization results were read:

1. the five-level range must exceed twice the numerical envelope;
2. all three paired `a=1.5-a=1` contrasts must have the same sign;
3. the smallest paired effect must exceed twice the numerical envelope.

Neither proof nor flow stress passes because condition 2 fails. Therefore the
three-variable `(eta, theta, a)` Bayesian optimization was not executed.

## Physical and numerical checks

Every executed FEM result passed:

- all 20 grain blocks and stable Grain-ID/orientation mapping;
- positive TRI6 areas, 40000 square-micrometre total area, conforming shared
  grain-boundary edges, and matching x/y periodic nodes;
- prescribed zero macro shear and `P22` constraint;
- periodic displacement fluctuations;
- Hill--Mandel residual tolerance;
- positive slip resistance, non-negative integrated plastic dissipation, finite
  constitutive state, and plastic-volume tolerance;
- readable Exodus fields and global equilibrium residual check.

The study comprises 17 new real FEM evaluations: 3 homogeneous controls, 5
five-level sensitivity runs, 3 fine-mesh runs, 2 time-refined runs, and 4
additional-realization runs. Their measured aggregate wall time is about
`2111 s` (`35.2 min`) on the local MPI-8 configuration, excluding mesh
generation and plotting.

## Figures and stored results

The compact results and six real-data figures are under
`results/cp_inverse_design_texture_morphology/`:

1. `fig01_grain_morphology_design_space.png`;
2. `fig02_morphology_sensitivity.png`;
3. `fig03_plastic_slip_fields.png`;
4. `fig05_forward_response_comparison.png`;
5. `fig07_coarse_fine_time_verification.png`;
6. `fig08_morphology_texture_property_summary.png`.

The requested three-parameter BO design-space figure, target-versus-optimized
microstructure, and BO GIF/MP4 were intentionally omitted because no
three-parameter optimization occurred. Creating them would falsely present GP
or interpolated data as executed FEM.

Large reproducible meshes, Exodus files, CSV histories, and solver logs remain
Git-ignored. The committed CSV/JSON summaries retain run directories, numerical
values, checks, mesh quality, wall times, and decision provenance.

## Reproduction

```bash
python scripts/cp_inverse_design_texture_morphology/generate_morphology.py
python scripts/cp_inverse_design_texture_morphology/run_homogeneous_control.py
python scripts/cp_inverse_design_texture_morphology/run_morphology_sensitivity.py
python scripts/cp_inverse_design_texture_morphology/verify_texture_morphology.py
conda run --no-capture-output -n spp2489-inverse \
  python scripts/cp_inverse_design_texture_morphology/plot_morphology_sensitivity.py
```

## Limitations and next steps

This is a 2D plane-strain RVE with only 20 grains and a local rate-dependent
crystal-plasticity law. It does not contain Hall--Petch strengthening,
strain-gradient terms, or explicit grain-boundary constitutive physics. The
current negative macroscopic result does not prove that morphology is
irrelevant in larger 3D RVEs or real materials.

Before treating morphology as an inverse-design variable, the next study should
use larger multi-realization ensembles with matched area-weighted texture, then
3D periodic EBSD-derived or statistically equivalent microstructures. If grain
size or boundary strengthening is the research target, a size-dependent or
grain-boundary-aware crystal-plasticity model is required rather than tuning the
existing benchmark parameters.
