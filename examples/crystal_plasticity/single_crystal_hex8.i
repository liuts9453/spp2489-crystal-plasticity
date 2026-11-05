# One-element 3-D FCC crystal-plasticity verification using the new Total Lagrangian system.

phi1 = 0
Phi = 0
phi2 = 0
steps = 20
max_strain = 0.02

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
  type = GeneratedMesh
  dim = 3
  nx = 1
  ny = 1
  nz = 1
  elem_type = HEX8
[]

[Physics/SolidMechanics/QuasiStatic/all]
  new_system = true
  formulation = TOTAL
  strain = FINITE
  add_variables = true
  volumetric_locking_correction = false
  generate_output = 'pk1_stress_xx pk1_stress_yy pk1_stress_zz
                     cauchy_stress_xx cauchy_stress_yy cauchy_stress_zz
                     deformation_gradient_xx deformation_gradient_yy deformation_gradient_zz'
  additional_generate_output = 'vonmises_cauchy_stress'
[]

[Functions/load]
  type = ParsedFunction
  expression = '${max_strain}*t'
[]

[BCs]
  [fix_x]
    type = DirichletBC
    variable = disp_x
    boundary = left
    value = 0
  []
  [fix_y]
    type = DirichletBC
    variable = disp_y
    boundary = bottom
    value = 0
  []
  [fix_z]
    type = DirichletBC
    variable = disp_z
    boundary = back
    value = 0
  []
  [load_x]
    type = FunctionDirichletBC
    variable = disp_x
    boundary = right
    function = load
  []
[]

[Materials]
  [elasticity]
    type = ComputeElasticityTensorCP
    C_ijkl = '200000 120000 120000 200000 120000 200000 100000 100000 100000'
    fill_method = symmetric9
    euler_angle_1 = ${phi1}
    euler_angle_2 = ${Phi}
    euler_angle_3 = ${phi2}
  []
  [analytic_cp]
    type = ComputeLagrangianFCCCrystalPlasticityStress
    slip_system_file = fcc_12_slip_systems.txt
    reference_slip_rate = 0.001
    rate_sensitivity = 0.05
    initial_slip_resistance = 80
    hardening_modulus = 400
    saturation_resistance = 160
    hardening_exponent = 1.0
    latent_hardening_ratio = 1.4
    slip_increment_tolerance = 0.02
    local_absolute_tolerance = 1e-10
    local_relative_tolerance = 1e-9
    local_max_iterations = 60
  []
  [diagnostics]
    type = CrystalPlasticityDiagnostics
  []
[]

[AuxVariables]
  [accumulated_slip]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_resistance_mean]
    family = MONOMIAL
    order = CONSTANT
  []
  [det_fp]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_xx]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_yy]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_zz]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_0]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_1]
    family = MONOMIAL
    order = CONSTANT
  []
[]

[AuxKernels]
  [accumulated_slip]
    type = MaterialRealAux
    variable = accumulated_slip
    property = accumulated_slip
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_resistance_mean]
    type = MaterialRealAux
    variable = slip_resistance_mean
    property = slip_resistance_mean
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [det_fp]
    type = MaterialRealAux
    variable = det_fp
    property = det_plastic_deformation_gradient
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [fp_xx]
    type = RankTwoAux
    variable = fp_xx
    rank_two_tensor = plastic_deformation_gradient
    index_i = 0
    index_j = 0
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [fp_yy]
    type = RankTwoAux
    variable = fp_yy
    rank_two_tensor = plastic_deformation_gradient
    index_i = 1
    index_j = 1
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [fp_zz]
    type = RankTwoAux
    variable = fp_zz
    rank_two_tensor = plastic_deformation_gradient
    index_i = 2
    index_j = 2
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_0]
    type = MaterialStdVectorAux
    variable = slip_rate_0
    property = slip_increment
    index = 0
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip_rate_1]
    type = MaterialStdVectorAux
    variable = slip_rate_1
    property = slip_increment
    index = 1
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Postprocessors]
  [F11]
    type = ElementAverageValue
    variable = deformation_gradient_xx
  []
  [P11]
    type = ElementAverageValue
    variable = pk1_stress_xx
  []
  [P22]
    type = ElementAverageValue
    variable = pk1_stress_yy
  []
  [P33]
    type = ElementAverageValue
    variable = pk1_stress_zz
  []
  [accumulated_slip]
    type = ElementAverageValue
    variable = accumulated_slip
  []
  [slip_resistance_mean]
    type = ElementAverageValue
    variable = slip_resistance_mean
  []
  [det_fp]
    type = ElementAverageValue
    variable = det_fp
  []
  [fp_xx]
    type = ElementAverageValue
    variable = fp_xx
  []
  [fp_yy]
    type = ElementAverageValue
    variable = fp_yy
  []
  [fp_zz]
    type = ElementAverageValue
    variable = fp_zz
  []
  [nonlinear_iterations]
    type = NumNonlinearIterations
  []
  [linear_iterations]
    type = NumLinearIterations
  []
  [consistent_tangent_relative_error]
    type = ElementAverageMaterialProperty
    mat_prop = cp_consistent_tangent_relative_error
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
  nl_max_its = 30
  nl_rel_tol = 1e-9
  nl_abs_tol = 1e-10
  l_tol = 1e-8
  start_time = 0
  dt = '${fparse 1.0 / steps}'
  end_time = 1
[]

[Outputs]
  execute_on = 'INITIAL TIMESTEP_END'
  perf_graph = true
  [csv]
    type = CSV
    file_base = results/crystal_plasticity/single_crystal_o1
  []
  [exodus]
    type = Exodus
    file_base = results/crystal_plasticity/single_crystal_o1
  []
[]
