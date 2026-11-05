#include "CrystalPlasticityDiagnostics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

registerMooseObject("Spp2489App", CrystalPlasticityDiagnostics);

InputParameters
CrystalPlasticityDiagnostics::validParams()
{
  InputParameters params = Material::validParams();
  params.addClassDescription(
      "Computes accumulated slip, resistance statistics, plastic-volume error, and dissipation "
      "from the direct PK2 crystal-plasticity material properties.");
  params.addParam<MaterialPropertyName>(
      "pk1_stress", "pk1_stress", "PK1 stress from the Total Lagrangian PK2 base class.");
  params.addParam<MaterialPropertyName>(
      "deformation_gradient", "deformation_gradient", "Total deformation gradient.");
  params.addParam<MaterialPropertyName>("slip_rate", "slip_increment", "Slip-rate vector.");
  params.addParam<MaterialPropertyName>(
      "slip_resistance", "slip_resistance", "Slip-resistance vector.");
  params.addParam<MaterialPropertyName>(
      "resolved_shear", "applied_shear_stress", "Resolved shear-stress vector.");
  params.addParam<MaterialPropertyName>(
      "plastic_deformation_gradient", "plastic_deformation_gradient", "Plastic F tensor.");
  params.addParam<MaterialPropertyName>("accumulated_slip_increment",
                                        "cp_accumulated_slip_increment",
                                        "Substep-integrated sum of absolute slip increments.");
  params.addParam<Real>(
      "active_slip_rate_tolerance", 1e-10, "Rate threshold used to count active systems.");
  return params;
}

CrystalPlasticityDiagnostics::CrystalPlasticityDiagnostics(const InputParameters & parameters)
  : Material(parameters),
    _pk1_stress(
        getMaterialProperty<RankTwoTensor>(getParam<MaterialPropertyName>("pk1_stress"))),
    _deformation_gradient(getMaterialProperty<RankTwoTensor>(
        getParam<MaterialPropertyName>("deformation_gradient"))),
    _deformation_gradient_old(getMaterialPropertyOld<RankTwoTensor>(
        getParam<MaterialPropertyName>("deformation_gradient"))),
    _slip_rate(
        getMaterialProperty<std::vector<Real>>(getParam<MaterialPropertyName>("slip_rate"))),
    _slip_resistance(getMaterialProperty<std::vector<Real>>(
        getParam<MaterialPropertyName>("slip_resistance"))),
    _resolved_shear(getMaterialProperty<std::vector<Real>>(
        getParam<MaterialPropertyName>("resolved_shear"))),
    _fp(getMaterialProperty<RankTwoTensor>(
        getParam<MaterialPropertyName>("plastic_deformation_gradient"))),
    _slip_increment_sum(getMaterialProperty<Real>(
        getParam<MaterialPropertyName>("accumulated_slip_increment"))),
    _accumulated_slip(declareProperty<Real>("accumulated_slip")),
    _accumulated_slip_old(getMaterialPropertyOld<Real>("accumulated_slip")),
    _total_slip_rate(declareProperty<Real>("total_slip_rate")),
    _active_slip_systems(declareProperty<Real>("active_slip_systems")),
    _slip_resistance_mean(declareProperty<Real>("slip_resistance_mean")),
    _slip_resistance_min(declareProperty<Real>("slip_resistance_min")),
    _slip_resistance_max(declareProperty<Real>("slip_resistance_max")),
    _det_fp(declareProperty<Real>("det_plastic_deformation_gradient")),
    _plastic_dissipation_rate(declareProperty<Real>("plastic_dissipation_rate")),
    _local_power_density(declareProperty<Real>("local_power_density"))
{
}

void
CrystalPlasticityDiagnostics::initQpStatefulProperties()
{
  _accumulated_slip[_qp] = 0.0;
}

void
CrystalPlasticityDiagnostics::computeQpProperties()
{
  const Real active_tol = getParam<Real>("active_slip_rate_tolerance");
  Real rate_sum = 0.0;
  Real dissipation = 0.0;
  unsigned int active = 0;
  for (std::size_t i = 0; i < _slip_rate[_qp].size(); ++i)
  {
    rate_sum += std::abs(_slip_rate[_qp][i]);
    active += std::abs(_slip_rate[_qp][i]) > active_tol;
    if (i < _resolved_shear[_qp].size())
      dissipation += _resolved_shear[_qp][i] * _slip_rate[_qp][i];
  }

  _total_slip_rate[_qp] = rate_sum;
  _active_slip_systems[_qp] = active;
  _accumulated_slip[_qp] = _accumulated_slip_old[_qp] + _slip_increment_sum[_qp];
  _det_fp[_qp] = _fp[_qp].det();
  _plastic_dissipation_rate[_qp] = dissipation;
  _local_power_density[_qp] =
      _dt > 0.0
          ? _pk1_stress[_qp].doubleContraction(
                (_deformation_gradient[_qp] - _deformation_gradient_old[_qp]) / _dt)
          : 0.0;

  if (_slip_resistance[_qp].empty())
  {
    _slip_resistance_mean[_qp] = 0.0;
    _slip_resistance_min[_qp] = 0.0;
    _slip_resistance_max[_qp] = 0.0;
  }
  else
  {
    const auto mm = std::minmax_element(_slip_resistance[_qp].begin(),
                                        _slip_resistance[_qp].end());
    _slip_resistance_min[_qp] = *mm.first;
    _slip_resistance_max[_qp] = *mm.second;
    _slip_resistance_mean[_qp] =
        std::accumulate(_slip_resistance[_qp].begin(), _slip_resistance[_qp].end(), 0.0) /
        _slip_resistance[_qp].size();
  }
}
