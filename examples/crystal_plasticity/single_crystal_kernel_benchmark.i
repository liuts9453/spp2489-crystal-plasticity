# Single-integration-point timing driver for the FCC constitutive kernel.
!include single_crystal_hex8.i

[Postprocessors]
  [cp_update_seconds]
    type = CrystalPlasticityTiming
    metric = seconds
  []
  [cp_residual_seconds]
    type = CrystalPlasticityTiming
    metric = residual_seconds
  []
  [cp_jacobian_seconds]
    type = CrystalPlasticityTiming
    metric = jacobian_seconds
  []
  [cp_linear_solve_seconds]
    type = CrystalPlasticityTiming
    metric = linear_solve_seconds
  []
  [cp_line_search_seconds]
    type = CrystalPlasticityTiming
    metric = line_search_seconds
  []
  [cp_tangent_seconds]
    type = CrystalPlasticityTiming
    metric = tangent_seconds
  []
  [cp_update_calls]
    type = CrystalPlasticityTiming
    metric = calls
  []
  [cp_local_iterations_total]
    type = CrystalPlasticityTiming
    metric = local_newton_iterations
  []
  [cp_substeps_total]
    type = CrystalPlasticityTiming
    metric = substeps
  []
  [cp_line_search_evaluations]
    type = CrystalPlasticityTiming
    metric = line_search_evaluations
  []
  [cp_tangent_calls]
    type = CrystalPlasticityTiming
    metric = tangent_calls
  []
[]
