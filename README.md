# SPP2489 MOOSE application

SPP2489 is a MOOSE application for Total-Lagrangian finite-strain computational
homogenization. The repository contains periodic 20-grain RVE examples and a
direct-PK2, fully implicit FCC crystal-plasticity material with an analytic
algorithmic consistent tangent.

The crystal-plasticity model retains multiplicative kinematics, Green--Lagrange
elastic strain, twelve FCC `{111}<110>` systems, power-law slip, Kalidindi-type
hardening, and backward Euler integration. Its optimized local solver can use
the exact four-plane hardening reduction (10 unknowns instead of 18), reuse one
LU factorization for the six tangent right-hand sides, and enable algebraically
equivalent tensor and scalar hot-loop specializations. The 18-variable path is
selectable as a regression reference.

## Build and test

The commands below assume the local `moose` Conda environment and a MOOSE source
tree configured by `MOOSE_DIR`/the application Makefile.

```bash
conda run --no-capture-output -n moose make -j8
conda run --no-capture-output -n moose ./run_tests -j4
```

Run the focused numerical-equivalence checks and incremental performance suite:

```bash
conda run --no-capture-output -n moose \
  python scripts/crystal_plasticity/verify_cp_optimization.py

conda run --no-capture-output -n moose \
  python scripts/crystal_plasticity/benchmark_cp_optimizations.py \
  --configs A0 A1 A2 A3 A4 A5 --meshes 16 --steps 20 --ranks 4
```

Run the complete finite-strain crystal-plasticity inverse-design demo (real
MOOSE FEM, tolerance-driven BoTorch optimization, random baseline, independent
32x32 rerun, and same-mesh 64x64 verification):

```bash
conda run --no-capture-output -n spp2489-inverse \
  python scripts/cp_inverse_design/optimize_cp_texture.py \
  --mesh 32 --max-bo-evals 12 --initial-samples 6 \
  --target-tolerance 0.01 --seed 2026 --reuse-cache \
  --validate-high-fidelity --mpi-ranks 8
```

Generated Exodus, CSV, timing logs, and benchmark summaries are written below
`results/crystal_plasticity/` and `results/cp_inverse_design/`; these large,
reproducible artifacts are intentionally ignored by Git.

Run the two-parameter `(eta, theta)` workflow on a fixed, real Neper periodic
TRI6 polycrystal (deterministic DOE, sequential BO, independent coarse rerun,
same-tessellation fine mesh, time refinement, figures, and report):

```bash
conda run --no-capture-output -n spp2489-inverse \
  python scripts/cp_inverse_design_2d/optimize_texture_2d.py \
  --max-bo-evals 20 --target-tolerance 0.01 --seed 2026 --mpi-ranks 8
```

Neper itself is isolated in the `spp2489-neper` environment. Generate or audit
the fixed meshes with `scripts/cp_inverse_design_2d/generate_neper_rve.py`.

Build the PI-presentation package from the 28 cached real-FEM evaluations
without rerunning MOOSE or Bayesian optimization:

```bash
conda run --no-capture-output -n spp2489-inverse \
  python scripts/cp_inverse_design_2d/presentation/build_all_presentation_visuals.py
```

This writes three static figures (PNG/PDF/SVG), a 28-frame GIF, a 1080p H.264
MP4, a provenance manifest, and a validation report to
`results/cp_inverse_design_2d/presentation/`.

## Project structure

- `include/materials/`, `src/materials/`: finite-strain constitutive models,
  including `ComputeLagrangianFCCCrystalPlasticityStress`.
- `include/postprocessors/`, `src/postprocessors/`: homogenized quantities and
  low-overhead crystal-plasticity timing counters.
- `examples/crystal_plasticity/`: single-crystal and periodic 20-grain RVE
  inputs, orientations, slip systems, and kernel benchmark driver.
- `scripts/crystal_plasticity/`: material generation, validation, plotting, and
  A0--A5 performance automation.
- `scripts/cp_inverse_design/`: SO(3) texture generation, isolated CP forward
  solves, dual-property extraction, Bayesian optimization, validation, plots,
  animation, and focused Python regressions.
- `scripts/cp_inverse_design_2d/`: fixed-seed Neper tessellation/mesh conversion,
  periodic-mesh validation, two-parameter SO(3) texture generation, proof/flow
  extraction, real-FEM Bayesian optimization, high-fidelity verification, and
  publication plotting. This workflow intentionally has no random-search
  comparison. Its `presentation/` package creates the cached-data PI figures
  and synchronized optimization animation.
- `results/cp_inverse_design_2d/`: reproducible raw runs, public target,
  compact optimization history, verification artifacts, copied Exodus files,
  and figures (large raw files remain ignored by Git).
- `test/tests/crystal_plasticity/`: TestHarness response, full-state history,
  substepping, and global Jacobian regressions.

## Reports

- [Original FCC benchmark](CRYSTAL_PLASTICITY_BENCHMARK_REPORT.md)
- [FCC local-integrator optimization](CRYSTAL_PLASTICITY_OPTIMIZATION_REPORT.md)
- [Finite-strain CP inverse design](CP_INVERSE_DESIGN_REPORT.md)
- [CP inverse design with a discriminating target](CP_INVERSE_DESIGN_REPORT_CHALLENGE_03.md)
- [Two-parameter CP inverse design on a Neper RVE](CP_INVERSE_DESIGN_2D_REPORT.md)
- [PI presentation visualization report](results/cp_inverse_design_2d/presentation/PRESENTATION_VISUALIZATION_REPORT.md)

The optimization report records the exact reduction and derivatives, validation
data, benchmark commands, measured speedups, and remaining bottlenecks.
