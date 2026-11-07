# Experimental EBSD → periodic Neper calibration

This module reads the four private SPP2489 EBSD grain CSVs, audits and cleans them, fits morphology distributions, constructs cubic-symmetry-aware texture descriptors, calibrates periodic Neper tessellations, and validates independent realizations and RVE sizes.

Raw and derived experimental data are deliberately excluded from public Git history. Configure a local read-only source:

```bash
export SPP2489_EBSD_DATA_ROOT=/path/to/damic_spp2489
python scripts/experimental_microstructure_calibration/run_calibration.py
```

Independent Neper processes are controlled with `SPP2489_NEPER_JOBS`
(default 4). Candidate and validation generation limits can be configured with
`SPP2489_NEPER_CALIBRATION_TIMEOUT_S` (default 360 s) and
`SPP2489_NEPER_VALIDATION_TIMEOUT_S` (default 1800 s); a timeout is reported as
a failure and never replaced by a synthetic response. The native Neper
morphology-optimization budget is controlled independently with
`SPP2489_NEPER_MORPHOLOGY_TIME_S` (default 90 s); external timeouts are only
process guards.

Use `--statistics-only` to stop before Neper calibration. Use `--dataset Al1_oBD` to process one dataset during development. The default cleaning threshold is an explicit 2.8 μm equivalent diameter; no upper-tail trimming is performed.

The generator uses paired empirical `diameq:file(...)` and cell-wise
`1-circularity:file(...)` targets.  The latter is an explicitly documented
ellipse-circularity proxy derived from the fitted-ellipse aspect ratio because
the supplied CSVs do not contain a measured grain perimeter.  This pairing
transmits the experimental size--shape dependence into Neper. An explicit rank
copula can reorder the paired targets without changing their marginals to
compensate the intrinsic Laguerre size--shape bias; achieved polygon AR and its
rank correlation are always measured independently. Complete Bunge triplets are
assigned after geometry generation, and the three Euler components are never
sampled independently. Phase labels are retained in the grain mapping; current
single-phase FCC MOOSE input must later select only a verified FCC phase.

The supplied `alpha` column is not treated as a per-grain principal-axis
direction.  It is constant within each section, and the Al1 converter explicitly
assigns 0°/90° from the section filename.  The value is retained as provenance,
but its discrepancy has zero weight in Neper morphology calibration.  A true
grain-axis direction distribution requires the original ellipse-orientation
output or the segmented EBSD map.

Consequently, when direction is unavailable, calibration restricts the Neper
ratio to `r_x/r_y >= 1` as an unsigned major/minor-axis convention.  The global
x/y assignment is not claimed to be experimentally identified.

`audit_coordinate_metadata.py` records candidate evidence for sample/build/scan coordinate conventions from the private checkout. Its matches require manual scientific review and never cause the four sections to be merged automatically.

After the complete workflow and figure review, create the Git-safe aggregate
deliverable with:

```bash
python scripts/experimental_microstructure_calibration/finalize_calibration_results.py
python scripts/experimental_microstructure_calibration/export_public_results.py
```

The exporter uses an explicit whitelist. It excludes raw/cleaned per-grain
tables, private hashes and paths, and synthetic tessellations that contain
sampled experimental orientations.

## V2 joint calibration and mechanical sensitivity

The v2 extension retains the historical result directory and writes all new
private runs below `results/experimental_microstructure_calibration_v2/`.
Geometry optimization uses a fixed Neper iteration count rather than a
wall-clock stopping criterion, so results do not depend on concurrent machine
load. Geometry objectives contain ECD, polygon AR, and size-conditioned AR
terms; number- and area-weighted ODF errors are validated separately.

```bash
python scripts/experimental_microstructure_calibration/bootstrap_ebsd_statistics.py --help
python scripts/experimental_microstructure_calibration/optimize_neper_joint_statistics.py --help
python scripts/experimental_microstructure_calibration/confirm_fixed_iteration_validation.py
python scripts/experimental_microstructure_calibration/build_mechanical_pilot.py --help
python scripts/experimental_microstructure_calibration/finalize_calibration_v2.py
python scripts/experimental_microstructure_calibration/write_calibration_v2_report.py
python scripts/experimental_microstructure_calibration/export_public_v2_results.py
```

The mechanical pilot is deliberately separate from calibration. It converts a
selected real periodic tessellation to a conforming TRI6 mesh, validates all
periodic pairs and grain blocks, then invokes the existing A4 FCC-CP model.
Invalid meshes or failed FEM solves are recorded and are never replaced by a
synthetic property response. The public v2 exporter again uses a whitelist;
raw EBSD rows, sampled orientations, tessellations, meshes, Exodus files, and
absolute private paths are excluded.
