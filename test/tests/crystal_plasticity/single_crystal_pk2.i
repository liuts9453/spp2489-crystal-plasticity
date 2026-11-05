!include ../../../examples/crystal_plasticity/single_crystal_hex8.i

[Materials/analytic_cp]
  verify_analytic_tangent = true
[]

[Outputs/csv]
  file_base := single_crystal_pk2_out
[]
[Outputs/exodus]
  enable = false
[]
