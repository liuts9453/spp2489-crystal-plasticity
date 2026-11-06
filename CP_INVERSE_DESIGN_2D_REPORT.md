# SPP2489 2D Texture Inverse Design with a Neper RVE

## Executive result

This report contains only executed MOOSE results. The optimization used a fixed Neper 20-grain periodic geometry, second-order triangular finite elements, the existing A4 fully implicit FCC crystal-plasticity integrator, and two texture variables `(eta, theta)`. No random-search comparison was performed.

- Public target: `P0.2 = 360.534480 MPa`, `Pflow(2%) = 416.762444 MPa`.
- Best observed design: `eta = 0.772902625`, `theta = +4.2421869 deg`.
- Independent coarse errors: proof `0.611988%`, flow `0.455255%`.
- BO used `28` real candidate FEM evaluations (`9` initial, `19` sequential), with `0` failed terminal evaluations.
- Candidate FEM wall time accumulated by BO: `3273.14 s`; GP/acquisition time: `5.71 s`.
- Target + candidate BO + independent/fine/time verification: `34` real FEM calls and `4212.38 s` summed wall time (validation Cases A/B are separate).

## 1. Software and immutable physical model

- Application branch: `feat/cp-inverse-design-2d-neper`; base commit recorded in runs: `95b3322c665c9ecc8e0c18099c9ed8983d122c6c`.
- MOOSE commit: `11853d1623555a9e444ffb0725a4dc16f64fb255`.
- Python `3.12.16`, PyTorch `2.15.0+cpu`, BoTorch `0.18.1`, GPyTorch `1.15.2`.
- Neper version: `5.0.0`; Gmsh `4.12.1`.
- Preserved model: multiplicative finite strain, Green--Lagrange elastic strain, 12 FCC `{111}<110>` systems, power-law kinetics, Kalidindi hardening, backward Euler, PK2 constitutive interface and analytic algorithmic tangent.
- A4 path: 10 local unknowns (6 PK2 components + 4 slip-plane resistances), grouped mode, LU reuse, optimized tensor/Jacobian evaluation, scalar specialization and previous-state predictor.

## 2. Real Neper periodic geometry and meshes

The fixed seed-2026 `.tess` has SHA-256 `9dbe4c2feb2df9bd07abc939e67b4f12cf4983e357e7d3597ce1597cc7e7e047`. Neper generated both the tessellation and raw periodic mesh/provenance files. The converter clips exact periodic Neper cell polygons into the 200 x 200 um fundamental square, preserves original grain IDs, constructs shared conforming interfaces, and creates matching named sidesets.

| Mesh | TRI6 elements | Nodes | QPs | Min quality | Max aspect | Periodic coordinate error (um) |
|---|---:|---:|---:|---:|---:|---:|
| Coarse | 3280 | 6709 | 22960 | 0.101009 | 17.7741 | 1.000e-10 |
| Fine | 6798 | 13813 | 47586 | 0.146568 | 12.3926 | 9.990e-11 |

Both meshes use the exact same tessellation and pass: 20 blocks, area 40000 um^2, all nodes used, positive/manifold elements, shared grain interfaces, four correct boundary sets and one-to-one periodic coordinate pairing. The minimum quality near one acute internal Voronoi vertex is reported rather than hidden; it persists under independent cut-origin studies.

## 3. Texture parameterization

The design uses `Q_i = SLERP(Q_i_random, Rz(theta) Q0_preferred, eta)` with shortest-quaternion sign handling. Formal bounds are `0.1 <= eta <= 0.9` and `-45 <= theta <= 45 deg`; endpoint generation at eta=0 and eta=1 remains available for validation. Elastic tensor and slip systems consume the same MOOSE `crysrot` property. Orthogonality, determinant, quaternion norm and Euler roundtrip tests pass below 1e-10.

## 4. Neper-grid mechanics benchmarks

| Case | P0.2 (MPa) | Pflow 2% (MPa) | Steps | Failed attempts | Newton iters | Wall (s) | Max |P22| (MPa) | HM residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: all same orientation | 401.783303 | 473.502975 | 42 | 0 | 128 | 51.86 | 7.606e-06 | 1.003e-12 |
| B: fixed random orientations | 336.110310 | 383.634466 | 57 | 8 | 412 | 135.53 | 3.577e-07 | 2.008e-09 |

Case A is uniform to a final local P11 range of `1.970e-08 MPa`; its periodic fluctuation mismatch is `1.477e-21`. Case B has a local P11 range of `148.634` to `793.239 MPa` and grain-dependent slip, as expected.

The pre-existing structured-RVE same-orientation reference gives P11(2%) = `473.502974833950 MPa`; the Neper Case A value differs by only `1.290e-11` relatively. This is the quantitative homogeneous-single-crystal regression.

The internal translation gauge is placed on the mesh node closest to (100,100) um. Pinning a periodic corner was rejected because the corner can be a periodic slave and leave a rigid translation. A global relative tolerance of 1e-10 is used so the automatically scaled scalar P22 equation does not terminate with a visible stress residual.

## 5. Proof-stress extraction

The initial t=0 CSV output row is not used directly because TIMESTEP_END AuxKernels have not yet initialized F there; the exact undeformed state F=I, P=0 is inserted. K0 is least-squares fitted through 0.1% nominal strain and checked against the narrower 0.05% window. The first positive crossing with `P = K0*(strain-0.002)` is linearly interpolated. All candidates use the same rule, loading rate and synchronized 2% endpoint.

## 6. Hidden target and Bayesian optimization

The hidden design, disclosed only after optimization, was `eta=0.84`, `theta=-8.5 deg`. The public target file contained only properties and numerical configuration. The objective was the equal-weight sum of squared relative errors in proof and 2% flow stress. A `SingleTaskGP` modeled `-J`; candidates came from `LogExpectedImprovement`. The nine deterministic DOE points span eta in {0.15, 0.50, 0.85} and theta in {-30, 0, 30} deg.

The hidden reference was selected from a post-hoc two-property GP scan of the fixed pilot DOE to avoid a trivial target near an initial response; the target properties were then replaced by a real MOOSE evaluation. In the executed data its nearest initial point is call `9` at a maximum relative property distance of `4.1743%`; its distance from initial call 1 is `12.4444%`.

| Call | Stage | eta | theta (deg) | P0.2 | Pflow | eY (%) | eF (%) | J | Wall (s) | Status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | initial | 0.1500000 | -30.00000 | 319.4700 | 364.8991 | 11.3899 | 12.4444 | 1.423e-02 | 137.75 | converged |
| 2 | initial | 0.1500000 | +0.00000 | 323.4947 | 371.6279 | 10.2736 | 10.8298 | 1.114e-02 | 122.61 | converged |
| 3 | initial | 0.1500000 | +30.00000 | 318.9043 | 365.9290 | 11.5468 | 12.1972 | 1.411e-02 | 105.44 | converged |
| 4 | initial | 0.5000000 | -30.00000 | 295.0440 | 340.8110 | 18.1648 | 18.2242 | 3.310e-02 | 116.36 | converged |
| 5 | initial | 0.5000000 | +0.00000 | 301.6728 | 345.4329 | 16.3262 | 17.1152 | 2.797e-02 | 111.47 | converged |
| 6 | initial | 0.5000000 | +30.00000 | 311.3809 | 355.8710 | 13.6335 | 14.6106 | 1.997e-02 | 137.27 | converged |
| 7 | initial | 0.8500000 | -30.00000 | 290.5845 | 335.1535 | 19.4018 | 19.5816 | 3.799e-02 | 113.80 | converged |
| 8 | initial | 0.8500000 | +0.00000 | 388.0381 | 449.2858 | 7.6286 | 7.8038 | 5.955e-03 | 111.12 | converged |
| 9 | initial | 0.8500000 | +30.00000 | 351.3383 | 399.3657 | 2.5507 | 4.1743 | 1.197e-03 | 125.80 | converged |
| 10 | sequential | 0.9000000 | +45.00000 | 321.6232 | 362.7171 | 10.7927 | 12.9679 | 1.423e-02 | 98.61 | converged |
| 11 | sequential | 0.8402335 | +17.47330 | 380.4302 | 435.9386 | 5.5184 | 4.6012 | 2.581e-03 | 118.10 | converged |
| 12 | sequential | 0.7662808 | +30.02850 | 340.7288 | 391.8902 | 5.4934 | 5.9680 | 3.290e-03 | 116.59 | converged |
| 13 | sequential | 0.9000000 | +23.70038 | 380.0309 | 432.6688 | 5.4076 | 3.8166 | 2.190e-03 | 129.39 | converged |
| 14 | sequential | 0.8426297 | +26.55262 | 357.3713 | 407.4225 | 0.8774 | 2.2411 | 2.896e-04 | 127.42 | converged |
| 15 | sequential | 0.1000000 | -45.00000 | 328.7938 | 377.9774 | 8.8038 | 9.3063 | 8.206e-03 | 114.40 | converged |
| 16 | sequential | 0.1000000 | +45.00000 | 328.6606 | 376.0357 | 8.8407 | 9.7722 | 8.683e-03 | 113.06 | converged |
| 17 | sequential | 0.2952671 | +45.00000 | 295.2555 | 338.7598 | 18.1062 | 18.7163 | 3.391e-02 | 111.49 | converged |
| 18 | sequential | 0.2559074 | -45.00000 | 321.7327 | 368.2889 | 10.7623 | 11.6310 | 1.256e-02 | 116.83 | converged |
| 19 | sequential | 0.1000000 | +13.58156 | 330.2083 | 378.5719 | 8.4115 | 9.1636 | 7.736e-03 | 115.07 | converged |
| 20 | sequential | 0.7252064 | +18.02775 | 340.8369 | 395.0301 | 5.4634 | 5.2146 | 2.852e-03 | 114.86 | converged |
| 21 | sequential | 0.8225072 | +25.02136 | 357.0164 | 408.5703 | 0.9758 | 1.9657 | 2.408e-04 | 114.51 | converged |
| 22 | sequential | 0.7440641 | -0.19890 | 345.0236 | 399.1614 | 4.3022 | 4.2233 | 1.817e-03 | 104.82 | converged |
| 23 | sequential | 0.7529971 | -7.63987 | 332.4938 | 384.0282 | 7.7775 | 7.8544 | 6.109e-03 | 130.91 | converged |
| 24 | sequential | 0.2413157 | +12.85861 | 300.8846 | 344.9308 | 16.5448 | 17.2356 | 2.854e-02 | 121.16 | converged |
| 25 | sequential | 0.1000000 | -10.25796 | 319.0968 | 365.0256 | 11.4934 | 12.4140 | 1.431e-02 | 113.11 | converged |
| 26 | sequential | 0.5855331 | +45.00000 | 324.6372 | 377.2474 | 9.9567 | 9.4814 | 9.452e-03 | 104.01 | converged |
| 27 | sequential | 0.6851744 | +1.96828 | 328.6405 | 378.6846 | 8.8463 | 9.1366 | 8.087e-03 | 122.54 | converged |
| 28 | sequential | 0.7729026 | +4.24219 | 358.3281 | 414.8651 | 0.6120 | 0.4553 | 2.909e-05 | 104.65 | converged |

No random-search data or comparison was generated.

## 7. Independent and numerical verification

| Fidelity | Target P0.2 | Candidate P0.2 | eY | Target Pflow | Candidate Pflow | eF | Candidate wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Coarse independent | 360.534480 | 358.328052 | 0.61199% | 416.762444 | 414.865112 | 0.45526% | 105.65 |
| Fine same-mesh | 360.446076 | 358.475551 | 0.54669% | 416.712792 | 414.772577 | 0.46560% | 226.55 |

Target coarse-to-fine changes: proof `0.02453%`, flow `0.01192%`. Candidate changes: proof `0.04115%`, flow `0.02231%`.

Adaptive-refined time integration (halved dtmax) gives default-to-refined relative changes:

- Target: proof `0.03850%`, flow `0.01932%`.
- Candidate: proof `0.10986%`, flow `0.02127%`.

Every reported candidate passed end time/synchronization, P22, zero macro shear, plastic activity, finite-state, positive resistance, det(Fp), nonnegative dissipation, Hill--Mandel, periodicity, 20-block and Exodus-field checks. Failed/retried adaptive attempts remain recorded separately.

## 8. Files and reproduction

```bash
conda run -n spp2489-neper python scripts/cp_inverse_design_2d/generate_neper_rve.py
conda run -n spp2489-inverse python scripts/cp_inverse_design_2d/optimize_texture_2d.py
conda run -n spp2489-inverse python scripts/cp_inverse_design_2d/verify_high_fidelity.py --eta <best> --theta <best>
conda run -n spp2489-inverse python scripts/cp_inverse_design_2d/plot_optimization.py
```

Raw FEM runs are under `results/cp_inverse_design_2d/runs/`; target, verification, copied Exodus and figures have dedicated subdirectories. Cache keys include both design variables, mesh and manifest hashes, material/load/time/solver settings, MPI ranks and code commits.

## 9. Limitations and next steps

- These are demonstrative CP constants, not a calibrated SS316L dataset or experimental target.
- The inverse map need not be unique; success means recovering both target properties, not hidden parameters.
- The fixed Voronoi geometry contains one acute vertex; its measured quality and coarse/fine influence are reported.
- Future work may add morphology descriptors, experimental calibration and uncertainty, but none were mixed into this demo.
