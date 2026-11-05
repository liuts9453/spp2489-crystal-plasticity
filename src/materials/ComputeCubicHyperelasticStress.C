#include "ComputeCubicHyperelasticStress.h"

registerMooseObject("Spp2489App", ComputeCubicHyperelasticStress);

InputParameters
ComputeCubicHyperelasticStress::validParams()
{
  InputParameters params = ComputeLagrangianStressPK2::validParams();
  params.addClassDescription(
      "Computes finite-strain anisotropic St. Venant-Kirchhoff hyperelastic stress.");
  params.addParam<MaterialPropertyName>(
      "elasticity_tensor", "elasticity_tensor", "Reference-frame elasticity tensor property.");
  return params;
}

ComputeCubicHyperelasticStress::ComputeCubicHyperelasticStress(
    const InputParameters & parameters)
  : ComputeLagrangianStressPK2(parameters),
    _elasticity_tensor(getMaterialProperty<RankFourTensor>(
        getParam<MaterialPropertyName>("elasticity_tensor"))),
    _strain_energy_density(
        declareProperty<Real>(_base_name + "strain_energy_density"))
{
}

void
ComputeCubicHyperelasticStress::computeQpPK2Stress()
{
  _C[_qp] = _elasticity_tensor[_qp];
  _S[_qp] = _C[_qp] * _E[_qp];
  _strain_energy_density[_qp] = 0.5 * _E[_qp].doubleContraction(_S[_qp]);
}
