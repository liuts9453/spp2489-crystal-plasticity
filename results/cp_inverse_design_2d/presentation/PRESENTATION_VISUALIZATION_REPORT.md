# SPP2489 Presentation Visualization Report

## Outcome

This incremental visualization build reused the executed Neper--MOOSE--FCC crystal-plasticity Bayesian-optimization cache. It did not call MOOSE, rerun Bayesian optimization, alter the constitutive model, or synthesize missing stress--strain curves.

- Audited FEM records: `28/28`.
- Recovered full solution CSV files: `28/28`.
- Recovered candidate Exodus files: `28/28`.
- Best-so-far update calls: `[1, 2, 8, 9, 14, 21, 28]`.
- Target: proof `360.534480 MPa`, flow `416.762444 MPa`.
- Optimized: eta `0.7729026253`, theta `+4.24218693 deg`, proof `358.328052 MPa`, flow `414.865112 MPa`.
- Independent coarse errors: proof `0.612%`, flow `0.455%`.
- Same-tessellation fine errors: proof `0.547%`, flow `0.466%`.

## Deliverables

- **Figure A:** `figA_bo_design_landscape.png/.pdf/.svg`
- **Figure B:** `figB_final_macroscopic_response.png/.pdf/.svg`
- **Figure C:** `figC_target_optimized_microstructure.png/.pdf/.svg`
- **Animation:** `bo_inverse_design_history.gif and .mp4`

Recommended PI sequence: Figure B (property match), Figure A (search logic), Figure C (microstructure/local physics), then the synchronized BO animation.

## Data provenance

The compact optimization history is `results/cp_inverse_design_2d/optimization_history.csv`. Absolute paths stored in the CSV were not trusted blindly: the audit checked the current path, relocated the results suffix into the present worktree, and checked sibling SPP2489 worktrees by the run/cache identifier. Every recovered curve was re-read from `solution.csv`; only the known uninitialized MOOSE t=0 AuxKernel row was replaced by the exact undeformed state, using the already validated extraction routine.

Figure A uses a final post-hoc `SingleTaskGP` fit to `-J`. Its background and property target contours are labeled surrogate predictions; only the 28 markers are real FEM evaluations. The connecting line denotes evaluation order, not a continuous material path.

Figure B uses the real hidden-reference coarse curve and the independent optimized coarse curve without smoothing. Its inset reads the executed coarse/fine verification values. P0.2 is explicitly identified as the 0.2% offset proof stress under the prescribed plane-strain loading condition.

Figure C regenerates textures from the executed `(eta, theta)` parameters on one fixed Neper geometry and reads final accumulated slip directly from the target/optimized Exodus files. The fixed cubic IPF-X key evaluates sample X in the crystal frame as `Q^T e_X`, reduces it to the cubic fundamental sector, and maps [001]/[101]/[111] to red/green/blue. No per-panel or per-frame orientation normalization is used. The two slip panels share one numerical scale.

## Animation semantics

The animation contains `28` actual evaluation frames and has no interpolated design states. Calls 1--2 show a neutral parameter plane. For every frame `k >= 3`, the objective posterior is refitted using only calls `1..k`; future observations are never used. Current and best-so-far stress--strain histories are both real FEM curves. The GIF is 1280x720; the MP4 is 1920x1080 H.264 with yuv420p pixel format. The final design is held for three seconds.

## Validation

- PASS — `calls_are_exactly_1_through_28`
- PASS — `stages_are_9_DOE_plus_19_BO`
- PASS — `best_so_far_uses_prefix_only`
- PASS — `all_current_curves_are_real_FEM`
- PASS — `all_candidate_Exodus_files_are_recovered`
- PASS — `current_texture_and_curve_share_eta_theta`
- PASS — `fixed_Neper_geometry_for_all_frames`
- PASS — `mp4_duration_matches_frame_schedule`
- PASS — `target_properties_match_existing_JSON`
- PASS — `optimized_design_matches_best_design_JSON`
- PASS — `surrogate_is_labeled_as_prediction`
- PASS — `no_mechanics_or_original_BO_sources_changed`
- PASS — `no_MOOSE_or_BO_reexecution`

## Reproduction

```bash
conda run --no-capture-output -n spp2489-inverse \
  python scripts/cp_inverse_design_2d/presentation/build_all_presentation_visuals.py
```

This command performs read-only FEM data recovery and visualization generation. It does not invoke `spp2489-opt`.
