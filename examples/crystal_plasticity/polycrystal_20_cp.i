# Periodic 20-grain plane-strain FCC crystal-plasticity RVE.
# Override nx, ny, steps, and run_name on the command line for the benchmark matrix.

nx = 16
ny = 16
steps = 20
H11 = 0.02
cp_xm = 0.10
run_name = cp_16_s20

!include ../polycrystal_20_centers.i
!include cp_20_random_materials.i

[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Mesh]
  [base]
    type = GeneratedMeshGenerator
    dim = 2
    nx = ${nx}
    ny = ${ny}
    xmin = 0
    xmax = 200
    ymin = 0
    ymax = 200
    elem_type = QUAD4
  []
  [grains]
    type = PeriodicVoronoiMeshGenerator
    input = base
    centers = '${voronoi_centers}'
    origin = '0 0 0'
    width = 200
    height = 200
    grain_id_name = grain_id
  []
  [pin]
    type = ExtraNodesetGenerator
    input = grains
    new_boundary = pin
    coord = '0 0 0'
  []
[]

[Physics/SolidMechanics/QuasiStatic/all]
  new_system = true
  formulation = TOTAL
  strain = FINITE
  add_variables = true
  volumetric_locking_correction = false
  constraint_types = 'strain strain none strain stress none none none none'
  targets = 'target_H11 zero zero zero'
  generate_output = 'pk1_stress_xx pk1_stress_xy pk1_stress_yx pk1_stress_yy
                     cauchy_stress_xx cauchy_stress_xy cauchy_stress_yx cauchy_stress_yy
                     deformation_gradient_xx deformation_gradient_xy
                     deformation_gradient_yx deformation_gradient_yy'
  additional_generate_output = 'vonmises_cauchy_stress'
[]

[Functions]
  [target_H11]
    type = ParsedFunction
    expression = '${H11}*t'
  []
  [zero]
    type = ConstantFunction
    value = 0
  []
[]

[BCs]
  [Periodic]
    [disp_x]
      variable = disp_x
      auto_direction = 'x y'
    []
    [disp_y]
      variable = disp_y
      auto_direction = 'x y'
    []
  []
  [pin_x]
    type = DirichletBC
    boundary = pin
    variable = disp_x
    value = 0
  []
  [pin_y]
    type = DirichletBC
    boundary = pin
    variable = disp_y
    value = 0
  []
[]

[AuxVariables]
  [grain_id]
    family = MONOMIAL
    order = CONSTANT
  []
  [euler_angle_1]
    family = MONOMIAL
    order = CONSTANT
  []
  [euler_angle_2]
    family = MONOMIAL
    order = CONSTANT
  []
  [euler_angle_3]
    family = MONOMIAL
    order = CONSTANT
  []
  [F11]
    family = MONOMIAL
    order = CONSTANT
  []
  [F22]
    family = MONOMIAL
    order = CONSTANT
  []
  [P11]
    family = MONOMIAL
    order = CONSTANT
  []
  [P22]
    family = MONOMIAL
    order = CONSTANT
  []
  [von_mises]
    family = MONOMIAL
    order = CONSTANT
  []
  [accumulated_slip]
    family = MONOMIAL
    order = CONSTANT
  []
  [total_slip_rate]
    family = MONOMIAL
    order = CONSTANT
  []
  [active_slip_systems]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_resistance_mean]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_resistance_min]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_resistance_max]
    family = MONOMIAL
    order = CONSTANT
  []
  [det_fp]
    family = MONOMIAL
    order = CONSTANT
  []
  [plastic_dissipation_rate]
    family = MONOMIAL
    order = CONSTANT
  []
  [local_power_density]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_11]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_12]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_21]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_22]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_33]
    family = MONOMIAL
    order = CONSTANT
  []
  [total_displacement_x]
  []
  [total_displacement_y]
  []
  [total_displacement_magnitude]
  []
  [slip_rate_00]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_01]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_02]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_03]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_04]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_05]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_06]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_07]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_08]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_09]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_10]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_11]
    family = MONOMIAL
    order = CONSTANT
  []
[]

[AuxKernels]
  [grain_id]
    type = ExtraElementIDAux
    variable = grain_id
    extra_id_name = grain_id
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [euler_angle_1]
    type = GrainPropertyAux
    variable = euler_angle_1
    values = '${orientation_phi1}'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [euler_angle_2]
    type = GrainPropertyAux
    variable = euler_angle_2
    values = '${orientation_Phi}'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [euler_angle_3]
    type = GrainPropertyAux
    variable = euler_angle_3
    values = '${orientation_phi2}'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [F11]
    type = ParsedAux
    variable = F11
    coupled_variables = deformation_gradient_xx
    expression = deformation_gradient_xx
    execute_on = TIMESTEP_END
  []
  [F22]
    type = ParsedAux
    variable = F22
    coupled_variables = deformation_gradient_yy
    expression = deformation_gradient_yy
    execute_on = TIMESTEP_END
  []
  [P11]
    type = ParsedAux
    variable = P11
    coupled_variables = pk1_stress_xx
    expression = pk1_stress_xx
    execute_on = TIMESTEP_END
  []
  [P22]
    type = ParsedAux
    variable = P22
    coupled_variables = pk1_stress_yy
    expression = pk1_stress_yy
    execute_on = TIMESTEP_END
  []
  [von_mises]
    type = ParsedAux
    variable = von_mises
    coupled_variables = vonmises_cauchy_stress
    expression = vonmises_cauchy_stress
    execute_on = TIMESTEP_END
  []
  [accumulated_slip]
    type = MaterialRealAux
    variable = accumulated_slip
    property = accumulated_slip
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [total_slip_rate]
    type = MaterialRealAux
    variable = total_slip_rate
    property = total_slip_rate
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [active_slip_systems]
    type = MaterialRealAux
    variable = active_slip_systems
    property = active_slip_systems
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_resistance_mean]
    type = MaterialRealAux
    variable = slip_resistance_mean
    property = slip_resistance_mean
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_resistance_min]
    type = MaterialRealAux
    variable = slip_resistance_min
    property = slip_resistance_min
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_resistance_max]
    type = MaterialRealAux
    variable = slip_resistance_max
    property = slip_resistance_max
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [det_fp]
    type = MaterialRealAux
    variable = det_fp
    property = det_plastic_deformation_gradient
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [plastic_dissipation_rate]
    type = MaterialRealAux
    variable = plastic_dissipation_rate
    property = plastic_dissipation_rate
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [local_power_density]
    type = MaterialRealAux
    variable = local_power_density
    property = local_power_density
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [fp_11]
    type = RankTwoAux
    variable = fp_11
    rank_two_tensor = plastic_deformation_gradient
    index_i = 0
    index_j = 0
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [fp_12]
    type = RankTwoAux
    variable = fp_12
    rank_two_tensor = plastic_deformation_gradient
    index_i = 0
    index_j = 1
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [fp_21]
    type = RankTwoAux
    variable = fp_21
    rank_two_tensor = plastic_deformation_gradient
    index_i = 1
    index_j = 0
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [fp_22]
    type = RankTwoAux
    variable = fp_22
    rank_two_tensor = plastic_deformation_gradient
    index_i = 1
    index_j = 1
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [fp_33]
    type = RankTwoAux
    variable = fp_33
    rank_two_tensor = plastic_deformation_gradient
    index_i = 2
    index_j = 2
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [total_displacement_x]
    type = HomogenizedTotalDisplacementAux
    variable = total_displacement_x
    macro_gradient = hvar
    displacements = 'disp_x disp_y'
    component = x
    execute_on = TIMESTEP_END
  []
  [total_displacement_y]
    type = HomogenizedTotalDisplacementAux
    variable = total_displacement_y
    macro_gradient = hvar
    displacements = 'disp_x disp_y'
    component = y
    execute_on = TIMESTEP_END
  []
  [total_displacement_magnitude]
    type = HomogenizedTotalDisplacementAux
    variable = total_displacement_magnitude
    macro_gradient = hvar
    displacements = 'disp_x disp_y'
    component = magnitude
    execute_on = TIMESTEP_END
  []
  [slip_rate_00]
    type = MaterialStdVectorAux
    variable = slip_rate_00
    property = slip_increment
    index = 0
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_01]
    type = MaterialStdVectorAux
    variable = slip_rate_01
    property = slip_increment
    index = 1
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_02]
    type = MaterialStdVectorAux
    variable = slip_rate_02
    property = slip_increment
    index = 2
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_03]
    type = MaterialStdVectorAux
    variable = slip_rate_03
    property = slip_increment
    index = 3
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_04]
    type = MaterialStdVectorAux
    variable = slip_rate_04
    property = slip_increment
    index = 4
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_05]
    type = MaterialStdVectorAux
    variable = slip_rate_05
    property = slip_increment
    index = 5
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_06]
    type = MaterialStdVectorAux
    variable = slip_rate_06
    property = slip_increment
    index = 6
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_07]
    type = MaterialStdVectorAux
    variable = slip_rate_07
    property = slip_increment
    index = 7
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_08]
    type = MaterialStdVectorAux
    variable = slip_rate_08
    property = slip_increment
    index = 8
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_09]
    type = MaterialStdVectorAux
    variable = slip_rate_09
    property = slip_increment
    index = 9
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_10]
    type = MaterialStdVectorAux
    variable = slip_rate_10
    property = slip_increment
    index = 10
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_11]
    type = MaterialStdVectorAux
    variable = slip_rate_11
    property = slip_increment
    index = 11
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Postprocessors]
  [benchmark_elapsed_seconds]
    type = BenchmarkElapsedTime
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [step]
    type = NumTimeSteps
    execute_on = TIMESTEP_END
  []
  [nonlinear_iterations]
    type = NumNonlinearIterations
    execute_on = TIMESTEP_END
  []
  [linear_iterations]
    type = NumLinearIterations
    execute_on = TIMESTEP_END
  []
  [F11_bar]
    type = ElementAverageValue
    variable = F11
    execute_on = TIMESTEP_END
  []
  [F12_bar]
    type = ElementAverageValue
    variable = deformation_gradient_xy
    execute_on = TIMESTEP_END
  []
  [F21_bar]
    type = ElementAverageValue
    variable = deformation_gradient_yx
    execute_on = TIMESTEP_END
  []
  [F22_bar]
    type = ElementAverageValue
    variable = F22
    execute_on = TIMESTEP_END
  []
  [P11_bar]
    type = ElementAverageValue
    variable = P11
    execute_on = TIMESTEP_END
  []
  [P12_bar]
    type = ElementAverageValue
    variable = pk1_stress_xy
    execute_on = TIMESTEP_END
  []
  [P21_bar]
    type = ElementAverageValue
    variable = pk1_stress_yx
    execute_on = TIMESTEP_END
  []
  [P22_bar]
    type = ElementAverageValue
    variable = P22
    execute_on = TIMESTEP_END
  []
  [accumulated_slip_bar]
    type = ElementAverageMaterialProperty
    mat_prop = accumulated_slip
    execute_on = TIMESTEP_END
  []
  [accumulated_slip_max]
    type = ElementExtremeValue
    variable = accumulated_slip
    value_type = max
    execute_on = TIMESTEP_END
  []
  [slip_resistance_mean_bar]
    type = ElementAverageMaterialProperty
    mat_prop = slip_resistance_mean
    execute_on = TIMESTEP_END
  []
  [det_fp_bar]
    type = ElementAverageMaterialProperty
    mat_prop = det_plastic_deformation_gradient
    execute_on = TIMESTEP_END
  []
  [det_fp_min]
    type = ElementExtremeValue
    variable = det_fp
    value_type = min
    execute_on = TIMESTEP_END
  []
  [det_fp_max]
    type = ElementExtremeValue
    variable = det_fp
    value_type = max
    execute_on = TIMESTEP_END
  []
  [plastic_dissipation_bar]
    type = ElementAverageMaterialProperty
    mat_prop = plastic_dissipation_rate
    execute_on = TIMESTEP_END
  []
  [local_power_bar]
    type = ElementAverageMaterialProperty
    mat_prop = local_power_density
    execute_on = TIMESTEP_END
  []
  [cp_update_seconds]
    type = CrystalPlasticityTiming
    metric = seconds
    execute_on = TIMESTEP_END
  []
  [cp_update_calls]
    type = CrystalPlasticityTiming
    metric = calls
    execute_on = TIMESTEP_END
  []
  [cp_local_iterations_bar]
    type = ElementAverageMaterialProperty
    mat_prop = cp_local_newton_iterations
    execute_on = TIMESTEP_END
  []
  [cp_local_substeps_max]
    type = ElementExtremeMaterialProperty
    mat_prop = cp_local_substeps
    value_type = max
    execute_on = TIMESTEP_END
  []
  [cp_local_residual_max]
    type = ElementExtremeMaterialProperty
    mat_prop = cp_local_residual_norm
    value_type = max
    execute_on = TIMESTEP_END
  []
  [fluctuation_l2_x]
    type = ElementL2Norm
    variable = disp_x
    execute_on = TIMESTEP_END
  []
  [fluctuation_l2_y]
    type = ElementL2Norm
    variable = disp_y
    execute_on = TIMESTEP_END
  []
[]

[Preconditioning/smp]
  type = SMP
  full = true
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  line_search = none
  automatic_scaling = true
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  nl_max_its = 25
  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-10
  l_tol = 1e-8
  start_time = 0
  dtmax = '${fparse 1.0 / steps}'
  end_time = 1
  [TimeStepper]
    type = IterationAdaptiveDT
    dt = '${fparse 0.25 / steps}'
    optimal_iterations = 8
    iteration_window = 2
    growth_factor = 1.5
    cutback_factor = 0.5
  []
[]

[Outputs]
  execute_on = 'INITIAL TIMESTEP_END'
  perf_graph = true
  perf_graph_json_file = results/crystal_plasticity/${run_name}_perf_graph.json
  [csv]
    type = CSV
    file_base = results/crystal_plasticity/${run_name}
  []
  [exodus]
    type = Exodus
    file_base = results/crystal_plasticity/${run_name}
  []
[]
