# FCC crystal-plasticity local-integrator optimization report

## 1. Scope and conclusion

This work optimizes `ComputeLagrangianFCCCrystalPlasticityStress` without changing
its constitutive model or its backward-Euler discrete equations. The retained
model is multiplicative finite-strain FCC crystal plasticity with Green--Lagrange
elastic strain, 12 `{111}<110>` systems, power-law slip, Kalidindi-type hardening,
and a Total-Lagrangian PK2 interface with an analytic algorithmic tangent.

The production configuration (A4) reduces the local Newton problem from 18 to 10
unknowns, reuses one LU factorization for all six tangent directions, evaluates
the local elastic strain through a cached trial tensor, and specializes only the
scalar powers for which the specialization is algebraically exact. The original
18-variable implementation remains selectable with `local_solver_mode=legacy`.

On the 64x64, 20-grain, 4-rank benchmark, the local constitutive wall-equivalent
time decreased from 152.87 s to 72.32 s (2.11x) and end-to-end wall time decreased
from 240.06 s to 154.69 s (1.55x). Final `P11` differed by 2.57e-14 relative.

## 2. Original discrete update

For a substep of size `dt`, the implementation uses

```text
Fe = F Fp^{-1}
Ee = 1/2 (Fe^T Fe - I)
Se = C : Ee
tau_a = Se : P_a
gamma_dot_a = gamma_0 |tau_a/g_a|^(1/m) sign(tau_a)
A = I - dt sum_a gamma_dot_a P_a
Fp^{-1}_{n+1} = Fp^{-1}_n A
```

with 12 rotated Schmid tensors `P_a`. The hardening equations are

```text
g_a - g_a^n - dt sum_b q_ab h_b(g_b) |gamma_dot_b| = 0,
q_ab = 1.0 (same {111} plane), 1.4 (different plane).
```

The reference PK2 stress returned to MOOSE is

```text
S = Jp Fp^{-1} Se Fp^{-T},    Jp = det(Fp).
```

The legacy coupled local vector is `[Se(6), g_1,...,g_12]`. Local Newton,
backtracking, and power-of-two constitutive substepping are unchanged.

## 3. Exact 18-to-10 reduction

Let `G_r={3r,3r+1,3r+2}`, `r=0,...,3`, denote the three systems on plane `r`.
Within a plane, the three rows of `q_ab` are identical. If their old resistances
are equal, their backward-Euler residual equations are identical. Thus the
subspace

```text
g_a = g_r,  a in G_r
```

is exactly invariant under the existing discrete update; this is not an
approximation or a modified hardening law. Define

```text
H_s = sum_{b in G_s} h_b(g_s) |gamma_dot_b|,
R_r = g_r - g_r^n - dt sum_s q_rs H_s.
```

The reduced vector is `[Se(6), g_0,...,g_3]`. All 12 resolved shears and slip
rates remain distinct. The exact hardening Jacobian is

```text
dR_r/dx_q = delta_(q,6+r)
             - dt sum_s q_rs sum_(b in G_s)
               [h'_b delta_(q,6+s) |gamma_dot_b|
                + h_b sign(gamma_dot_b) dgamma_dot_b/dx_q].
```

For the flow rule,

```text
d gamma_dot_a / d tau_a = (1/m) |gamma_dot_a|/|tau_a|,
d gamma_dot_a / d g_r   = -(1/m) gamma_dot_a/g_r,  a in G_r.
```

The stress residual is differentiated with

```text
dA = -dt sum_a P_a dgamma_dot_a,
dEe = 1/2 (dA^T Ce_trial A + A^T Ce_trial dA),
dR_Se = dSe - C:dEe.
```

The material still stores and outputs 12 resistances by expanding each plane
value to its three systems. `grouped_resistance_tolerance` checks every history
before using the reduced layout. `grouped` mode throws a diagnostic on an
incompatible history; `auto` falls back to the legacy solver. The present model
constructs the required plane-symmetric interaction matrix internally, so a more
general hardening interaction cannot be silently grouped.

## 4. Tangent and hot-loop changes

### LU reuse

The implicit-function tangent advances all six Green--Lagrange strain
sensitivities through every accepted constitutive substep. At each substep, the
same converged local Jacobian is copied once, factorized by the first
`DenseMatrix::lu_solve`, and reused for the remaining five right-hand sides.
Sensitivities of `Fp^{-1}`, all 12 expanded resistances, `Jp`, and the reference
PK2 transformation are propagated exactly. Setting
`reuse_tangent_factorization=false` restores the six-factorization path.

### Tensor evaluation

For each local substep,

```text
Ce_trial = Fp_n^{-T} F^T F Fp_n^{-1}
Ee_{n+1} = 1/2 (A^T Ce_trial A - I)
```

is computed instead of rebuilding `Fe=F Fp^{-1}` during every Newton residual
and Jacobian evaluation. `Ce_trial` is formed once, `A` is cached, and the exact
`dA` expression above is used. The legacy tensor route remains selectable with
`optimized_tensor_evaluation=false`.

### Scalar specialization

When `1/m` is an integer in `[1,64]`, exponentiation-by-squaring replaces
general `pow`; otherwise the general function is used. When the hardening
exponent is exactly one, the hardening and derivative are evaluated linearly.
All positivity, finite-value, saturation, and slip-increment safeguards remain.
The noninteger test (`m=0.13`, hardening exponent 1.3) verifies the fallback.

### Predictor

The default remains the previous converged intermediate stress. An optional
bounded elastic blend was implemented for measurement only. It increased
failures and runtime in the tested RVE and is therefore rejected as a production
default.

## 5. Instrumentation

Low-overhead monotonic-clock counters separately measure residual evaluation,
Jacobian assembly, dense local solves, line-search-only residual evaluation, and
consistent tangent evaluation. Counters also record material calls, local Newton
iterations, accepted internal substeps, tangent calls, and legacy/grouped calls.

The component timers are exclusive of each other; `cp_update_seconds` is the
inclusive material total. MPI values are sums over ranks. Tables below divide
rank-summed component times by four to give a wall-equivalent number; this is an
approximation when rank imbalance exists, but A0--A5 use the same partition and
the speedup ratio is meaningful.

## 6. Numerical verification

### Full-state history comparison

Legacy and optimized runs use identical prescribed time increments. Every
stress component, 12 slip rates, 12 resistances, all 9 `Fp` components,
accumulated slip, and plastic dissipation is compared.

| case | rows | worst scaled difference | final P11 legacy / A4 (MPa) | min dissipation |
|---|---:|---:|---:|---:|
| orientation (0,0,0) | 21 | 3.95e-10 (near-zero P33) | 238.39132568695 / 238.39132568702 | 3.50e-6 |
| orientation (37,52,11) | 21 | 6.55e-11 | 354.85032366553 / 354.85032366555 | 5.43e-2 |
| general scalar fallback | 21 | 6.49e-11 | 428.32305663456 / 428.32305663454 | 1.13e-1 |

The maximum within-plane resistance spread was zero in every comparison.

### Tangent and TestHarness

- Centered finite-difference `dS/dE`, plastic state with four internal substeps:
  maximum relative error **5.0942e-9** (target 1e-6).
- PETSc global Jacobian tests pass for both legacy and grouped paths, including a
  state that activates four constitutive substeps.
- TestHarness result: **8 passed, 0 skipped, 0 failed**.

### Polycrystal and physical checks

Optimized random-orientation RVEs at 16x16, 32x32, and 64x64 complete to 2%
macroscopic tension. Same-orientation RVEs at 16x16 and 32x32 recover identical
responses (`P11=473.30414122011 MPa`) and uniform fields to the verification
tolerance. For the optimized 64x64 random RVE:

- `P22 = 1.397e-8 MPa`; prescribed macroscopic shears pass.
- maximum periodic fluctuation mismatch: **4.22e-15**.
- maximum Hill--Mandel relative power residual: **3.1932e-9**.
- `det(Fp)` range: `[1.00000000027, 1.00007813635]`, not worse than A0.
- plastic dissipation: **7.56454**, positive.
- required Exodus fields are present and the file is readable.

## 7. Performance results

Hardware: AMD Ryzen 7 5800X (8 cores/16 threads), four MPI ranks, one thread per
rank; GCC 15.3; MOOSE commit `11853d1623`. Runs were sequential on the same
machine with identical mesh, partition, tolerances, and adaptive-step settings.

### Incremental 32x32 RVE benchmark

| config | change | local dim. | wall (s) | CP wall-eq. (s) | tangent (s) | failures / total global Newton | final P11 (MPa) |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 | 18-variable reference | 18 | 44.63 | 26.55 | 3.925 | 6 / 286 | 392.69091019379 |
| A1 | grouped solver | 10 | 33.09 | 15.66 | 1.827 | 6 / 286 | 392.69091019379 |
| A2 | A1 + LU reuse | 10 | 33.03 | 15.19 | 1.117 | 6 / 286 | 392.69091019379 |
| A3 | A2 + tensor route | 10 | 32.29 | 14.82 | 1.108 | 6 / 286 | 392.69091019379 |
| A4 | A3 + scalar route | 10 | 30.36 | 12.69 | 1.098 | 6 / 286 | 392.69091019380 |
| A5 | A4 + elastic blend | 10 | 72.63 | 35.11 | 2.689 | 18 / 654 | 392.83011258322 |

A4 gives 2.09x local and 1.47x end-to-end speedup at 32x32. A0--A4 have the same
accepted time history and agree to about 2.5e-14 relative in `P11`. A5 changes
the adaptive retry/accepted-step history and is both slower and discretely
different; it is retained only as an opt-in experiment.

### 64x64 production endpoints

| metric | A0 legacy | A4 optimized | ratio A0/A4 |
|---|---:|---:|---:|
| total wall (s) | 240.06 | 154.69 | **1.552x** |
| local CP wall-equivalent (s) | 152.867 | 72.325 | **2.114x** |
| residual (s) | 19.294 | 10.039 | 1.922x |
| Jacobian assembly (s) | 59.983 | 29.503 | 2.033x |
| local dense solve (s) | 26.372 | 10.493 | 2.513x |
| line-search residual (s) | 13.367 | 7.028 | 1.902x |
| consistent tangent (s) | 24.312 | 6.731 | **3.612x** |
| peak RSS (KiB) | 261308 | 262092 | 0.997x |
| local Newton iterations | 58,839,539 | 58,836,337 | -- |
| accepted internal substeps | 17,076,862 | 17,076,862 | -- |
| global failures / total Newton | 11 / 442 | 11 / 442 | identical |
| final P11 (MPa) | 389.11109699657 | 389.11109699656 | rel. error 2.57e-14 |

The measured A0 endpoint reproduces the preserved benchmark's 442 global Newton
iterations and `P11=389.111097 MPa`; wall time differs from the older 365.80 s
measurement because the executable/environment and machine state are not assumed
to be identical across dates.

### Fixed prescribed increments

With `ConstantDT`, 200 equal increments, unchanged convergence tolerances, and a
higher iteration ceiling to prevent automatic cutback, both A0 and A4 complete
with zero failed attempts and 595 global Newton iterations. A0/A4 final
`P11=401.47172087675 MPa` exactly at CSV precision. Local times are 13.315 s and
6.459 s (2.06x); walls are 33.11 s and 26.18 s (1.26x). This result is reported
separately from the adaptive history because rate-dependent backward Euler is
time-step-history dependent.

### Single-integration-point kernel

A 1000-step single-HEX8 run (56,048 material update calls) gives:

| metric | A0 (s) | A4 (s) | speedup |
|---|---:|---:|---:|
| inclusive local CP | 1.5835 | 0.7749 | **2.043x** |
| residual | 0.2154 | 0.1178 | 1.829x |
| Jacobian | 0.6139 | 0.3143 | 1.953x |
| dense solve | 0.2896 | 0.1115 | 2.597x |
| tangent | 0.2293 | 0.0591 | 3.877x |

Final `P11` is 238.49558317525/238.49558317523 MPa. End-to-end single-element
wall is dominated by MOOSE execution/output and is therefore not used as the
constitutive-kernel speedup.

## 8. Reproduction

```bash
conda run --no-capture-output -n moose make -j8
conda run --no-capture-output -n moose ./run_tests -j4

conda run --no-capture-output -n moose \
  python scripts/crystal_plasticity/verify_cp_optimization.py

conda run --no-capture-output -n moose \
  python scripts/crystal_plasticity/benchmark_cp_optimizations.py \
  --configs A0 A1 A2 A3 A4 A5 --meshes 32 --steps 20 --ranks 4

conda run --no-capture-output -n moose \
  python scripts/crystal_plasticity/benchmark_cp_optimizations.py \
  --configs A0 A4 --meshes 64 --steps 20 --ranks 4

conda run --no-capture-output -n moose \
  python scripts/crystal_plasticity/benchmark_cp_optimizations.py \
  --configs A0 A4 --meshes 16 --steps 200 --ranks 4 --mode fixed
```

Per-run CSV, Exodus, logs, perf-graph data, and summary JSON/CSV are generated in
`results/crystal_plasticity/optimization/` and are not committed.

## 9. Limitations and remaining opportunities

- The exact grouping is valid only for equal within-plane histories and the
  current plane-symmetric interaction matrix. General per-system hardening must
  use the legacy solver.
- A5 shows that a seemingly stronger predictor can be counterproductive for the
  steep power law; predictor changes should remain opt-in and benchmark-driven.
- A4 still spends most local time in Jacobian assembly, residual/line-search
  evaluation, and dense solves. Further work could use fixed-size stack kernels
  or batched small-matrix routines, provided the analytic residual and tangent
  remain unchanged.
- The 1.55x end-to-end ceiling is lower than the 2.11x constitutive speedup
  because global assembly, PETSc solves, mesh operations, and output are outside
  this optimization.
