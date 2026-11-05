# Case C: fixed-seed Haar-uniform three-dimensional crystal orientations.

!include polycrystal_20_centers.i
!include polycrystal_20_random_materials.i
!include polycrystal_20_common.i

[Outputs/csv]
  file_base := results/polycrystal/csv/case_c_random_orientation_64
[]
[Outputs/exodus]
  file_base := results/polycrystal/exodus/case_c_random_orientation_64
[]
