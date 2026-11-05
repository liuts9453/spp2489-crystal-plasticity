!include ../../../examples/crystal_plasticity/single_crystal_hex8.i

# Regression-only scalarization of every slip rate/resistance and the off-diagonal Fp entries.
[AuxVariables]
  [slip_rate_full_00]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_full_01]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_full_02]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_full_03]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_full_04]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_full_05]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_full_06]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_full_07]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_full_08]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_full_09]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_full_10]
    family = MONOMIAL
    order = CONSTANT
  []
  [slip_rate_full_11]
    family = MONOMIAL
    order = CONSTANT
  []
  [resistance_full_00]
    family = MONOMIAL
    order = CONSTANT
  []
  [resistance_full_01]
    family = MONOMIAL
    order = CONSTANT
  []
  [resistance_full_02]
    family = MONOMIAL
    order = CONSTANT
  []
  [resistance_full_03]
    family = MONOMIAL
    order = CONSTANT
  []
  [resistance_full_04]
    family = MONOMIAL
    order = CONSTANT
  []
  [resistance_full_05]
    family = MONOMIAL
    order = CONSTANT
  []
  [resistance_full_06]
    family = MONOMIAL
    order = CONSTANT
  []
  [resistance_full_07]
    family = MONOMIAL
    order = CONSTANT
  []
  [resistance_full_08]
    family = MONOMIAL
    order = CONSTANT
  []
  [resistance_full_09]
    family = MONOMIAL
    order = CONSTANT
  []
  [resistance_full_10]
    family = MONOMIAL
    order = CONSTANT
  []
  [resistance_full_11]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_xy]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_xz]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_yx]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_yz]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_zx]
    family = MONOMIAL
    order = CONSTANT
  []
  [fp_zy]
    family = MONOMIAL
    order = CONSTANT
  []
[]

[AuxKernels]
  [slip_rate_full_00]
    type = MaterialStdVectorAux
    variable = slip_rate_full_00
    property = slip_increment
    index = 0
  []
  [slip_rate_full_01]
    type = MaterialStdVectorAux
    variable = slip_rate_full_01
    property = slip_increment
    index = 1
  []
  [slip_rate_full_02]
    type = MaterialStdVectorAux
    variable = slip_rate_full_02
    property = slip_increment
    index = 2
  []
  [slip_rate_full_03]
    type = MaterialStdVectorAux
    variable = slip_rate_full_03
    property = slip_increment
    index = 3
  []
  [slip_rate_full_04]
    type = MaterialStdVectorAux
    variable = slip_rate_full_04
    property = slip_increment
    index = 4
  []
  [slip_rate_full_05]
    type = MaterialStdVectorAux
    variable = slip_rate_full_05
    property = slip_increment
    index = 5
  []
  [slip_rate_full_06]
    type = MaterialStdVectorAux
    variable = slip_rate_full_06
    property = slip_increment
    index = 6
  []
  [slip_rate_full_07]
    type = MaterialStdVectorAux
    variable = slip_rate_full_07
    property = slip_increment
    index = 7
  []
  [slip_rate_full_08]
    type = MaterialStdVectorAux
    variable = slip_rate_full_08
    property = slip_increment
    index = 8
  []
  [slip_rate_full_09]
    type = MaterialStdVectorAux
    variable = slip_rate_full_09
    property = slip_increment
    index = 9
  []
  [slip_rate_full_10]
    type = MaterialStdVectorAux
    variable = slip_rate_full_10
    property = slip_increment
    index = 10
  []
  [slip_rate_full_11]
    type = MaterialStdVectorAux
    variable = slip_rate_full_11
    property = slip_increment
    index = 11
  []
  [resistance_full_00]
    type = MaterialStdVectorAux
    variable = resistance_full_00
    property = slip_resistance
    index = 0
  []
  [resistance_full_01]
    type = MaterialStdVectorAux
    variable = resistance_full_01
    property = slip_resistance
    index = 1
  []
  [resistance_full_02]
    type = MaterialStdVectorAux
    variable = resistance_full_02
    property = slip_resistance
    index = 2
  []
  [resistance_full_03]
    type = MaterialStdVectorAux
    variable = resistance_full_03
    property = slip_resistance
    index = 3
  []
  [resistance_full_04]
    type = MaterialStdVectorAux
    variable = resistance_full_04
    property = slip_resistance
    index = 4
  []
  [resistance_full_05]
    type = MaterialStdVectorAux
    variable = resistance_full_05
    property = slip_resistance
    index = 5
  []
  [resistance_full_06]
    type = MaterialStdVectorAux
    variable = resistance_full_06
    property = slip_resistance
    index = 6
  []
  [resistance_full_07]
    type = MaterialStdVectorAux
    variable = resistance_full_07
    property = slip_resistance
    index = 7
  []
  [resistance_full_08]
    type = MaterialStdVectorAux
    variable = resistance_full_08
    property = slip_resistance
    index = 8
  []
  [resistance_full_09]
    type = MaterialStdVectorAux
    variable = resistance_full_09
    property = slip_resistance
    index = 9
  []
  [resistance_full_10]
    type = MaterialStdVectorAux
    variable = resistance_full_10
    property = slip_resistance
    index = 10
  []
  [resistance_full_11]
    type = MaterialStdVectorAux
    variable = resistance_full_11
    property = slip_resistance
    index = 11
  []
  [fp_xy]
    type = RankTwoAux
    variable = fp_xy
    rank_two_tensor = plastic_deformation_gradient
    index_i = 0
    index_j = 1
  []
  [fp_xz]
    type = RankTwoAux
    variable = fp_xz
    rank_two_tensor = plastic_deformation_gradient
    index_i = 0
    index_j = 2
  []
  [fp_yx]
    type = RankTwoAux
    variable = fp_yx
    rank_two_tensor = plastic_deformation_gradient
    index_i = 1
    index_j = 0
  []
  [fp_yz]
    type = RankTwoAux
    variable = fp_yz
    rank_two_tensor = plastic_deformation_gradient
    index_i = 1
    index_j = 2
  []
  [fp_zx]
    type = RankTwoAux
    variable = fp_zx
    rank_two_tensor = plastic_deformation_gradient
    index_i = 2
    index_j = 0
  []
  [fp_zy]
    type = RankTwoAux
    variable = fp_zy
    rank_two_tensor = plastic_deformation_gradient
    index_i = 2
    index_j = 1
  []
[]

[Postprocessors]
  [plastic_dissipation]
    type = ElementAverageMaterialProperty
    mat_prop = plastic_dissipation_rate
  []
  [local_substeps]
    type = ElementAverageMaterialProperty
    mat_prop = cp_local_substeps
  []
  [local_iterations]
    type = ElementAverageMaterialProperty
    mat_prop = cp_local_newton_iterations
  []
  [slip_rate_full_00]
    type = ElementAverageValue
    variable = slip_rate_full_00
  []
  [slip_rate_full_01]
    type = ElementAverageValue
    variable = slip_rate_full_01
  []
  [slip_rate_full_02]
    type = ElementAverageValue
    variable = slip_rate_full_02
  []
  [slip_rate_full_03]
    type = ElementAverageValue
    variable = slip_rate_full_03
  []
  [slip_rate_full_04]
    type = ElementAverageValue
    variable = slip_rate_full_04
  []
  [slip_rate_full_05]
    type = ElementAverageValue
    variable = slip_rate_full_05
  []
  [slip_rate_full_06]
    type = ElementAverageValue
    variable = slip_rate_full_06
  []
  [slip_rate_full_07]
    type = ElementAverageValue
    variable = slip_rate_full_07
  []
  [slip_rate_full_08]
    type = ElementAverageValue
    variable = slip_rate_full_08
  []
  [slip_rate_full_09]
    type = ElementAverageValue
    variable = slip_rate_full_09
  []
  [slip_rate_full_10]
    type = ElementAverageValue
    variable = slip_rate_full_10
  []
  [slip_rate_full_11]
    type = ElementAverageValue
    variable = slip_rate_full_11
  []
  [resistance_full_00]
    type = ElementAverageValue
    variable = resistance_full_00
  []
  [resistance_full_01]
    type = ElementAverageValue
    variable = resistance_full_01
  []
  [resistance_full_02]
    type = ElementAverageValue
    variable = resistance_full_02
  []
  [resistance_full_03]
    type = ElementAverageValue
    variable = resistance_full_03
  []
  [resistance_full_04]
    type = ElementAverageValue
    variable = resistance_full_04
  []
  [resistance_full_05]
    type = ElementAverageValue
    variable = resistance_full_05
  []
  [resistance_full_06]
    type = ElementAverageValue
    variable = resistance_full_06
  []
  [resistance_full_07]
    type = ElementAverageValue
    variable = resistance_full_07
  []
  [resistance_full_08]
    type = ElementAverageValue
    variable = resistance_full_08
  []
  [resistance_full_09]
    type = ElementAverageValue
    variable = resistance_full_09
  []
  [resistance_full_10]
    type = ElementAverageValue
    variable = resistance_full_10
  []
  [resistance_full_11]
    type = ElementAverageValue
    variable = resistance_full_11
  []
  [fp_xy]
    type = ElementAverageValue
    variable = fp_xy
  []
  [fp_xz]
    type = ElementAverageValue
    variable = fp_xz
  []
  [fp_yx]
    type = ElementAverageValue
    variable = fp_yx
  []
  [fp_yz]
    type = ElementAverageValue
    variable = fp_yz
  []
  [fp_zx]
    type = ElementAverageValue
    variable = fp_zx
  []
  [fp_zy]
    type = ElementAverageValue
    variable = fp_zy
  []
[]
