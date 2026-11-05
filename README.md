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

Generated Exodus, CSV, timing logs, and benchmark summaries are written below
`results/crystal_plasticity/` and are intentionally ignored by Git.

## Project structure

- `include/materials/`, `src/materials/`: finite-strain constitutive models,
  including `ComputeLagrangianFCCCrystalPlasticityStress`.
- `include/postprocessors/`, `src/postprocessors/`: homogenized quantities and
  low-overhead crystal-plasticity timing counters.
- `examples/crystal_plasticity/`: single-crystal and periodic 20-grain RVE
  inputs, orientations, slip systems, and kernel benchmark driver.
- `scripts/crystal_plasticity/`: material generation, validation, plotting, and
  A0--A5 performance automation.
- `test/tests/crystal_plasticity/`: TestHarness response, full-state history,
  substepping, and global Jacobian regressions.

## Reports

- [Original FCC benchmark](CRYSTAL_PLASTICITY_BENCHMARK_REPORT.md)
- [FCC local-integrator optimization](CRYSTAL_PLASTICITY_OPTIMIZATION_REPORT.md)

The optimization report records the exact reduction and derivatives, validation
data, benchmark commands, measured speedups, and remaining bottlenecks.
