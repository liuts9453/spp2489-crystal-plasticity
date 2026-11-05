# Common 20-grain periodic RVE mechanics. Case files define orientation arrays and materials.

nx = 64
ny = 64
H11 = 0.005

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

[Physics]
  [SolidMechanics]
    [QuasiStatic]
      [all]
        new_system = true
        formulation = TOTAL
        strain = FINITE
        add_variables = true
        volumetric_locking_correction = false
        constraint_types = 'strain strain none strain stress none none none none'
        targets = 'target_H11 zero zero zero'
        generate_output = 'pk1_stress_xx pk1_stress_xy pk1_stress_yx pk1_stress_yy
                           pk2_stress_xx pk2_stress_xy pk2_stress_yx pk2_stress_yy
                           cauchy_stress_xx cauchy_stress_xy cauchy_stress_yx cauchy_stress_yy
                           deformation_gradient_xx deformation_gradient_xy
                           deformation_gradient_yx deformation_gradient_yy'
        additional_generate_output = 'vonmises_cauchy_stress'
      []
    []
  []
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
  [strain_energy_density]
    family = MONOMIAL
    order = CONSTANT
  []
  [J]
    family = MONOMIAL
    order = CONSTANT
  []
  [total_displacement_x]
  []
  [total_displacement_y]
  []
  [total_displacement_magnitude]
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
  [strain_energy_density]
    type = MaterialRealAux
    variable = strain_energy_density
    property = strain_energy_density
    execute_on = TIMESTEP_END
  []
  [J]
    type = ParsedAux
    variable = J
    coupled_variables = 'deformation_gradient_xx deformation_gradient_xy
                         deformation_gradient_yx deformation_gradient_yy'
    expression = 'deformation_gradient_xx*deformation_gradient_yy -
                  deformation_gradient_xy*deformation_gradient_yx'
    execute_on = TIMESTEP_END
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
[]

[Postprocessors]
  [step]
    type = NumTimeSteps
    execute_on = TIMESTEP_END
  []
  [nonlinear_iterations]
    type = NumNonlinearIterations
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
  [J_bar]
    type = ElementAverageValue
    variable = J
    execute_on = TIMESTEP_END
  []
  [energy_bar]
    type = ElementAverageValue
    variable = strain_energy_density
    execute_on = TIMESTEP_END
  []
  [P11_min]
    type = ElementExtremeValue
    variable = P11
    value_type = min
    execute_on = TIMESTEP_END
  []
  [P11_max]
    type = ElementExtremeValue
    variable = P11
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

[Preconditioning]
  [smp]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  line_search = none
  automatic_scaling = true
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  nl_max_its = 30
  nl_rel_tol = 1e-10
  nl_abs_tol = 1e-11
  l_tol = 1e-11
  start_time = 0
  dt = 0.1
  end_time = 1
[]

[Outputs]
  execute_on = 'INITIAL TIMESTEP_END'
  [csv]
    type = CSV
    file_base = results/polycrystal/csv/common
  []
  [exodus]
    type = Exodus
    file_base = results/polycrystal/exodus/common
  []
[]
