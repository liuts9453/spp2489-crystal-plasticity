#pragma once

#include "Material.h"

/** Lightweight scalar diagnostics for the direct PK2 FCC crystal-plasticity properties. */
class CrystalPlasticityDiagnostics : public Material
{
public:
  static InputParameters validParams();
  CrystalPlasticityDiagnostics(const InputParameters & parameters);

protected:
  virtual void initQpStatefulProperties() override;
  virtual void computeQpProperties() override;

  const MaterialProperty<RankTwoTensor> & _pk1_stress;
  const MaterialProperty<RankTwoTensor> & _deformation_gradient;
  const MaterialProperty<RankTwoTensor> & _deformation_gradient_old;
  const MaterialProperty<std::vector<Real>> & _slip_rate;
  const MaterialProperty<std::vector<Real>> & _slip_resistance;
  const MaterialProperty<std::vector<Real>> & _resolved_shear;
  const MaterialProperty<RankTwoTensor> & _fp;
  const MaterialProperty<Real> & _slip_increment_sum;

  MaterialProperty<Real> & _accumulated_slip;
  const MaterialProperty<Real> & _accumulated_slip_old;
  MaterialProperty<Real> & _total_slip_rate;
  MaterialProperty<Real> & _active_slip_systems;
  MaterialProperty<Real> & _slip_resistance_mean;
  MaterialProperty<Real> & _slip_resistance_min;
  MaterialProperty<Real> & _slip_resistance_max;
  MaterialProperty<Real> & _det_fp;
  MaterialProperty<Real> & _plastic_dissipation_rate;
  MaterialProperty<Real> & _local_power_density;
};
