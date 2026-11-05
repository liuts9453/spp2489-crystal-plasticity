#pragma once

#include "ComputeLagrangianStressPK2.h"

/**
 * St. Venant-Kirchhoff hyperelasticity with a general reference-frame elasticity tensor.
 *
 * The model uses S = C:E and Psi = 0.5 E:S. ComputeLagrangianStressPK2 supplies the
 * finite-strain PK2-to-PK1 transformation and the consistent global Jacobian.
 */
class ComputeCubicHyperelasticStress : public ComputeLagrangianStressPK2
{
public:
  static InputParameters validParams();
  ComputeCubicHyperelasticStress(const InputParameters & parameters);

protected:
  virtual void computeQpPK2Stress() override;

  const MaterialProperty<RankFourTensor> & _elasticity_tensor;
  MaterialProperty<Real> & _strain_energy_density;
};
