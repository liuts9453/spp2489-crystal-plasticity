#include "ComputeLagrangianFCCCrystalPlasticityStress.h"

#include "DelimitedFileReader.h"
#include "MooseException.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

registerMooseObject("Spp2489App", ComputeLagrangianFCCCrystalPlasticityStress);

std::atomic<long long> ComputeLagrangianFCCCrystalPlasticityStress::_elapsed_nanoseconds{0};
std::atomic<unsigned long long> ComputeLagrangianFCCCrystalPlasticityStress::_calls{0};

namespace
{
constexpr std::array<std::array<unsigned int, 2>, 6> voigt = {
    {{{0, 0}}, {{1, 1}}, {{2, 2}}, {{1, 2}}, {{0, 2}}, {{0, 1}}}};
}

InputParameters
ComputeLagrangianFCCCrystalPlasticityStress::validParams()
{
  InputParameters params = ComputeLagrangianStressPK2::validParams();
  params.addClassDescription(
      "Fully coupled backward-Euler FCC crystal plasticity with an analytic algorithmic PK2 "
      "tangent for the new Total-Lagrangian mechanics system.");
  params.addRequiredParam<FileName>("slip_system_file", "Twelve rows: plane normal then direction.");
  params.addParam<MaterialPropertyName>("elasticity_tensor", "elasticity_tensor", "Rotated C tensor.");
  params.addParam<MaterialPropertyName>("crystal_rotation", "crysrot", "Active crystal rotation.");
  params.addParam<Real>("reference_slip_rate", 0.001, "Reference slip rate in 1/s.");
  params.addRangeCheckedParam<Real>("rate_sensitivity", 0.05, "rate_sensitivity > 0", "Power-law m.");
  params.addRangeCheckedParam<Real>("initial_slip_resistance", 80.0, "initial_slip_resistance > 0", "Initial g in MPa.");
  params.addParam<Real>("hardening_modulus", 400.0, "Kalidindi hardening modulus in MPa.");
  params.addRangeCheckedParam<Real>("saturation_resistance", 160.0, "saturation_resistance > 0", "Saturation g in MPa.");
  params.addRangeCheckedParam<Real>("hardening_exponent", 1.0, "hardening_exponent > 0", "Hardening exponent.");
  params.addParam<Real>("latent_hardening_ratio", 1.4, "Latent/self hardening ratio.");
  params.addRangeCheckedParam<Real>("slip_increment_tolerance", 0.02, "slip_increment_tolerance > 0", "Maximum |gamma_dot| dt.");
  params.addRangeCheckedParam<Real>("local_absolute_tolerance", 1e-10, "local_absolute_tolerance > 0", "Scaled local absolute tolerance.");
  params.addRangeCheckedParam<Real>("local_relative_tolerance", 1e-9, "local_relative_tolerance > 0", "Scaled local relative tolerance.");
  params.addRangeCheckedParam<unsigned int>("local_max_iterations", 60, "local_max_iterations > 0", "Maximum coupled local Newton iterations.");
  params.addRangeCheckedParam<unsigned int>("line_search_max_iterations", 12, "line_search_max_iterations > 0", "Maximum local backtracking cuts.");
  params.addRangeCheckedParam<unsigned int>("maximum_substeps", 64, "maximum_substeps > 0", "Maximum power-of-two internal substeps.");
  params.addParam<bool>("verify_analytic_tangent", false, "Run a centered finite-difference check of dS/dE (verification only).");
  params.addRangeCheckedParam<Real>("tangent_perturbation", 1e-7, "tangent_perturbation > 0", "Green-Lagrange strain perturbation used only by the tangent check.");
  return params;
}

ComputeLagrangianFCCCrystalPlasticityStress::ComputeLagrangianFCCCrystalPlasticityStress(
    const InputParameters & parameters)
  : ComputeLagrangianStressPK2(parameters),
    _elasticity_tensor(getMaterialProperty<RankFourTensor>(getParam<MaterialPropertyName>("elasticity_tensor"))),
    _crystal_rotation(getMaterialProperty<RankTwoTensor>(getParam<MaterialPropertyName>("crystal_rotation"))),
    _F_old(getMaterialPropertyOld<RankTwoTensor>(_base_name + "deformation_gradient")),
    _Fp(declareProperty<RankTwoTensor>("plastic_deformation_gradient")),
    _Fp_old(getMaterialPropertyOld<RankTwoTensor>("plastic_deformation_gradient")),
    _intermediate_pk2(declareProperty<RankTwoTensor>("intermediate_pk2_stress")),
    _intermediate_pk2_old(getMaterialPropertyOld<RankTwoTensor>("intermediate_pk2_stress")),
    _slip_resistance(declareProperty<std::vector<Real>>("slip_resistance")),
    _slip_resistance_old(getMaterialPropertyOld<std::vector<Real>>("slip_resistance")),
    _slip_rate(declareProperty<std::vector<Real>>("slip_increment")),
    _resolved_shear(declareProperty<std::vector<Real>>("applied_shear_stress")),
    _flow_direction(declareProperty<std::vector<RankTwoTensor>>("flow_direction")),
    _local_newton_iterations(declareProperty<Real>("cp_local_newton_iterations")),
    _local_residual_norm(declareProperty<Real>("cp_local_residual_norm")),
    _consistent_tangent_relative_error(
        declareProperty<Real>("cp_consistent_tangent_relative_error")),
    _local_substeps(declareProperty<Real>("cp_local_substeps")),
    _accumulated_slip_increment(declareProperty<Real>("cp_accumulated_slip_increment")),
    _reference_slip_rate(getParam<Real>("reference_slip_rate")),
    _rate_sensitivity(getParam<Real>("rate_sensitivity")),
    _initial_resistance(getParam<Real>("initial_slip_resistance")),
    _hardening_modulus(getParam<Real>("hardening_modulus")),
    _saturation_resistance(getParam<Real>("saturation_resistance")),
    _hardening_exponent(getParam<Real>("hardening_exponent")),
    _latent_hardening(getParam<Real>("latent_hardening_ratio")),
    _slip_increment_tolerance(getParam<Real>("slip_increment_tolerance")),
    _local_absolute_tolerance(getParam<Real>("local_absolute_tolerance")),
    _local_relative_tolerance(getParam<Real>("local_relative_tolerance")),
    _local_max_iterations(getParam<unsigned int>("local_max_iterations")),
    _line_search_max_iterations(getParam<unsigned int>("line_search_max_iterations")),
    _maximum_substeps(getParam<unsigned int>("maximum_substeps")),
    _verify_analytic_tangent(getParam<bool>("verify_analytic_tangent")),
    _tangent_perturbation(getParam<Real>("tangent_perturbation"))
{
  MooseUtils::DelimitedFileReader reader(getParam<FileName>("slip_system_file"));
  reader.setFormatFlag(MooseUtils::DelimitedFileReader::FormatFlag::ROWS);
  reader.read();
  if (reader.getData().size() != N_SLIP)
    paramError("slip_system_file", "FCC benchmark requires exactly 12 slip-system rows.");
  for (unsigned int a = 0; a < N_SLIP; ++a)
  {
    if (reader.getData(a).size() != 6)
      paramError("slip_system_file", "Each row must contain 3 plane and 3 direction entries.");
    for (unsigned int i = 0; i < 3; ++i)
    {
      _crystal_plane_normal[a](i) = reader.getData(a)[i];
      _crystal_slip_direction[a](i) = reader.getData(a)[i + 3];
    }
    _crystal_plane_normal[a] /= _crystal_plane_normal[a].norm();
    _crystal_slip_direction[a] /= _crystal_slip_direction[a].norm();
    if (std::abs(_crystal_plane_normal[a] * _crystal_slip_direction[a]) > 1e-12)
      paramError("slip_system_file", "Slip direction and plane normal must be orthogonal.");
  }
}

void
ComputeLagrangianFCCCrystalPlasticityStress::initQpStatefulProperties()
{
  ComputeLagrangianStressPK2::initQpStatefulProperties();
  _Fp[_qp] = RankTwoTensor::Identity();
  _intermediate_pk2[_qp].zero();
  _slip_resistance[_qp].assign(N_SLIP, _initial_resistance);
  _slip_rate[_qp].assign(N_SLIP, 0.0);
  _resolved_shear[_qp].assign(N_SLIP, 0.0);
  _flow_direction[_qp].assign(N_SLIP, RankTwoTensor());
  _local_newton_iterations[_qp] = 0.0;
  _local_residual_norm[_qp] = 0.0;
  _consistent_tangent_relative_error[_qp] = 0.0;
  _local_substeps[_qp] = 1.0;
  _accumulated_slip_increment[_qp] = 0.0;
}

void
ComputeLagrangianFCCCrystalPlasticityStress::updateFlowDirections()
{
  _flow_direction[_qp].resize(N_SLIP);
  for (unsigned int a = 0; a < N_SLIP; ++a)
  {
    const RealVectorValue direction = _crystal_rotation[_qp] * _crystal_slip_direction[a];
    const RealVectorValue normal = _crystal_rotation[_qp] * _crystal_plane_normal[a];
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        _flow_direction[_qp][a](i, j) = direction(i) * normal(j);
  }
}

RankTwoTensor
ComputeLagrangianFCCCrystalPlasticityStress::stressBasis(unsigned int component) const
{
  RankTwoTensor basis;
  basis.zero();
  const auto i = voigt[component][0];
  const auto j = voigt[component][1];
  basis(i, j) = 1.0;
  basis(j, i) = 1.0;
  return basis;
}

RankTwoTensor
ComputeLagrangianFCCCrystalPlasticityStress::stressTensor(
    const libMesh::DenseVector<Real> & x) const
{
  RankTwoTensor stress;
  stress.zero();
  for (unsigned int p = 0; p < N_STRESS; ++p)
    stress += stressBasis(p) * x(p);
  return stress;
}

Real
ComputeLagrangianFCCCrystalPlasticityStress::scaledResidualNorm(
    const libMesh::DenseVector<Real> & residual, const libMesh::DenseVector<Real> & x) const
{
  Real norm = 0.0;
  for (unsigned int p = 0; p < N_STRESS; ++p)
    norm = std::max(norm, std::abs(residual(p)) / std::max(1.0, std::abs(x(p))));
  for (unsigned int a = 0; a < N_SLIP; ++a)
    norm = std::max(norm,
                    std::abs(residual(N_STRESS + a)) /
                        std::max(1.0, std::abs(x(N_STRESS + a))));
  return norm;
}

bool
ComputeLagrangianFCCCrystalPlasticityStress::evaluateLocalSystem(
    const libMesh::DenseVector<Real> & x,
    const RankTwoTensor & F,
    const RankTwoTensor & Fp_inv_old,
    const std::array<Real, N_SLIP> & resistance_old,
    Real local_dt,
    libMesh::DenseVector<Real> & residual,
    libMesh::DenseMatrix<Real> * jacobian,
    LocalState & state) const
{
  state.Se = stressTensor(x);
  const Real power = 1.0 / _rate_sensitivity;
  for (unsigned int a = 0; a < N_SLIP; ++a)
  {
    state.resistance[a] = x(N_STRESS + a);
    if (!(state.resistance[a] > 0.0) || !std::isfinite(state.resistance[a]))
      return false;
    state.tau[a] = state.Se.doubleContraction(_flow_direction[_qp][a]);
    const Real ratio = std::abs(state.tau[a]) / state.resistance[a];
    state.slip_rate[a] = _reference_slip_rate * std::pow(ratio, power);
    if (state.tau[a] < 0.0)
      state.slip_rate[a] *= -1.0;
    if (!std::isfinite(state.slip_rate[a]) ||
        std::abs(state.slip_rate[a]) * local_dt > _slip_increment_tolerance)
      return false;

    const Real z = 1.0 - state.resistance[a] / _saturation_resistance;
    state.hardening[a] =
        _hardening_modulus * std::copysign(std::pow(std::abs(z), _hardening_exponent), z);
    if (std::abs(z) < 1e-14 && _hardening_exponent < 1.0)
      return false;
    state.hardening_derivative[a] =
        -_hardening_modulus * _hardening_exponent / _saturation_resistance *
        std::pow(std::max(std::abs(z), 1e-30), _hardening_exponent - 1.0);

    const Real dslip_dtau = std::abs(state.tau[a]) > 1e-30
                                ? power * std::abs(state.slip_rate[a]) / std::abs(state.tau[a])
                                : 0.0;
    const Real dslip_dg = -power * state.slip_rate[a] / state.resistance[a];
    for (unsigned int q = 0; q < N_LOCAL; ++q)
    {
      const Real dtau = q < N_STRESS
                            ? stressBasis(q).doubleContraction(_flow_direction[_qp][a])
                            : 0.0;
      state.slip_derivative[a][q] = dslip_dtau * dtau;
      if (q == N_STRESS + a)
        state.slip_derivative[a][q] += dslip_dg;
    }
  }

  RankTwoTensor plastic_increment = RankTwoTensor::Identity();
  for (unsigned int a = 0; a < N_SLIP; ++a)
    plastic_increment -= _flow_direction[_qp][a] * (local_dt * state.slip_rate[a]);
  state.Fp_inv = Fp_inv_old * plastic_increment;
  state.Fe = F * state.Fp_inv;
  const RankTwoTensor Ee =
      0.5 * (state.Fe.transpose() * state.Fe - RankTwoTensor::Identity());
  const RankTwoTensor predicted_Se = _elasticity_tensor[_qp] * Ee;

  residual.resize(N_LOCAL);
  for (unsigned int p = 0; p < N_STRESS; ++p)
  {
    const auto i = voigt[p][0];
    const auto j = voigt[p][1];
    residual(p) = state.Se(i, j) - predicted_Se(i, j);
  }
  for (unsigned int a = 0; a < N_SLIP; ++a)
  {
    Real hardening_increment = 0.0;
    for (unsigned int b = 0; b < N_SLIP; ++b)
    {
      const Real interaction = (a / 3 == b / 3) ? 1.0 : _latent_hardening;
      hardening_increment +=
          interaction * state.hardening[b] * std::abs(state.slip_rate[b]);
    }
    residual(N_STRESS + a) =
        state.resistance[a] - resistance_old[a] - local_dt * hardening_increment;
  }

  if (!jacobian)
    return true;
  jacobian->resize(N_LOCAL, N_LOCAL);
  jacobian->zero();
  for (unsigned int q = 0; q < N_LOCAL; ++q)
  {
    RankTwoTensor dFp_inv;
    dFp_inv.zero();
    for (unsigned int a = 0; a < N_SLIP; ++a)
      dFp_inv -= Fp_inv_old * _flow_direction[_qp][a] *
                 (local_dt * state.slip_derivative[a][q]);
    const RankTwoTensor dFe = F * dFp_inv;
    const RankTwoTensor dEe =
        0.5 * (state.Fe.transpose() * dFe + dFe.transpose() * state.Fe);
    const RankTwoTensor dPredicted = _elasticity_tensor[_qp] * dEe;
    for (unsigned int p = 0; p < N_STRESS; ++p)
    {
      const auto i = voigt[p][0];
      const auto j = voigt[p][1];
      (*jacobian)(p, q) = (p == q ? 1.0 : 0.0) - dPredicted(i, j);
    }
    for (unsigned int a = 0; a < N_SLIP; ++a)
    {
      Real derivative = q == N_STRESS + a ? 1.0 : 0.0;
      for (unsigned int b = 0; b < N_SLIP; ++b)
      {
        const Real interaction = (a / 3 == b / 3) ? 1.0 : _latent_hardening;
        const Real sign_rate = state.slip_rate[b] < 0.0 ? -1.0 : 1.0;
        const Real dh = q == N_STRESS + b ? state.hardening_derivative[b] : 0.0;
        derivative -= local_dt * interaction *
                      (dh * std::abs(state.slip_rate[b]) +
                       state.hardening[b] * sign_rate * state.slip_derivative[b][q]);
      }
      (*jacobian)(N_STRESS + a, q) = derivative;
    }
  }
  return true;
}

bool
ComputeLagrangianFCCCrystalPlasticityStress::solveLocalSystem(
    const RankTwoTensor & F,
    const RankTwoTensor & Fp_inv_old,
    const RankTwoTensor & Se_old,
    const std::array<Real, N_SLIP> & resistance_old,
    Real local_dt,
    libMesh::DenseVector<Real> & x,
    libMesh::DenseMatrix<Real> & jacobian,
    LocalState & state,
    unsigned int & iterations) const
{
  x.resize(N_LOCAL);
  // The previous converged stress is a robust starting point for the steep FCC power law.  A
  // purely elastic trial stress can overshoot the admissible slip increment before Newton has a
  // chance to return to the coupled solution (especially for m=0.05).
  const RankTwoTensor Se_trial = Se_old;
  for (unsigned int p = 0; p < N_STRESS; ++p)
    x(p) = Se_trial(voigt[p][0], voigt[p][1]);
  for (unsigned int a = 0; a < N_SLIP; ++a)
    x(N_STRESS + a) = resistance_old[a];

  libMesh::DenseVector<Real> residual(N_LOCAL);
  Real initial_norm = 0.0;
  for (iterations = 0; iterations < _local_max_iterations; ++iterations)
  {
    if (!evaluateLocalSystem(
            x, F, Fp_inv_old, resistance_old, local_dt, residual, &jacobian, state))
      return false;
    const Real norm = scaledResidualNorm(residual, x);
    if (iterations == 0)
      initial_norm = norm;
    if (norm <= _local_absolute_tolerance ||
        norm <= _local_relative_tolerance * std::max(initial_norm, _local_absolute_tolerance))
      return true;

    libMesh::DenseVector<Real> rhs(N_LOCAL), increment(N_LOCAL);
    for (unsigned int q = 0; q < N_LOCAL; ++q)
      rhs(q) = -residual(q);
    libMesh::DenseMatrix<Real> solve_matrix(jacobian);
    solve_matrix.lu_solve(rhs, increment);

    bool accepted = false;
    Real step = 1.0;
    for (unsigned int ls = 0; ls < _line_search_max_iterations; ++ls)
    {
      libMesh::DenseVector<Real> candidate(x);
      for (unsigned int q = 0; q < N_LOCAL; ++q)
        candidate(q) += step * increment(q);
      libMesh::DenseVector<Real> candidate_residual(N_LOCAL);
      LocalState candidate_state;
      if (evaluateLocalSystem(candidate,
                              F,
                              Fp_inv_old,
                              resistance_old,
                              local_dt,
                              candidate_residual,
                              nullptr,
                              candidate_state) &&
          scaledResidualNorm(candidate_residual, candidate) < norm)
      {
        x = candidate;
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted)
      return false;
  }
  return false;
}

bool
ComputeLagrangianFCCCrystalPlasticityStress::integrateIncrement(
    const RankTwoTensor & F_old,
    const RankTwoTensor & F,
    const RankTwoTensor & Fp_inv_old,
    const RankTwoTensor & Se_old,
    const std::array<Real, N_SLIP> & resistance_old,
    std::vector<SubstepRecord> & records,
    unsigned int & total_iterations,
    Real & accumulated_slip_increment) const
{
  for (unsigned int number_substeps = 1; number_substeps <= _maximum_substeps;
       number_substeps *= 2)
  {
    records.clear();
    total_iterations = 0;
    accumulated_slip_increment = 0.0;
    RankTwoTensor previous_Fp_inv = Fp_inv_old;
    RankTwoTensor previous_Se = Se_old;
    std::array<Real, N_SLIP> previous_resistance = resistance_old;
    bool success = true;
    for (unsigned int substep = 1; substep <= number_substeps; ++substep)
    {
      SubstepRecord record;
      record.fraction = static_cast<Real>(substep) / number_substeps;
      record.dt = _dt / number_substeps;
      record.F = F_old + record.fraction * (F - F_old);
      record.Fp_inv_old = previous_Fp_inv;
      record.resistance_old = previous_resistance;
      unsigned int iterations = 0;
      if (!solveLocalSystem(record.F,
                            previous_Fp_inv,
                            previous_Se,
                            previous_resistance,
                            record.dt,
                            record.x,
                            record.jacobian,
                            record.state,
                            iterations))
      {
        success = false;
        break;
      }
      total_iterations += iterations;
      for (const auto rate : record.state.slip_rate)
        accumulated_slip_increment += record.dt * std::abs(rate);
      previous_Fp_inv = record.state.Fp_inv;
      previous_Se = record.state.Se;
      previous_resistance = record.state.resistance;
      records.push_back(record);
    }
    if (success)
      return true;
    if (number_substeps > _maximum_substeps / 2)
      break;
  }
  records.clear();
  return false;
}

void
ComputeLagrangianFCCCrystalPlasticityStress::computeConsistentTangent(
    const std::vector<SubstepRecord> & records, const RankTwoTensor & F)
{
  _C[_qp].zero();
  const RankTwoTensor F_inv_T = F.inverse().transpose();
  for (unsigned int e = 0; e < N_STRESS; ++e)
  {
    const RankTwoTensor H = stressBasis(e);
    const RankTwoTensor dF = F_inv_T * H;
    RankTwoTensor dFp_inv_old;
    dFp_inv_old.zero();
    std::array<Real, N_SLIP> dresistance_old{};
    RankTwoTensor dSe;
    dSe.zero();
    for (const auto & record : records)
    {
      const RankTwoTensor plastic_increment =
          record.Fp_inv_old.inverse() * record.state.Fp_inv;
      const RankTwoTensor direct_dFp_inv = dFp_inv_old * plastic_increment;
      const RankTwoTensor dF_substep = record.fraction * dF;
      const RankTwoTensor direct_dFe =
          dF_substep * record.state.Fp_inv + record.F * direct_dFp_inv;
      const RankTwoTensor direct_dEe =
          0.5 * (record.state.Fe.transpose() * direct_dFe +
                 direct_dFe.transpose() * record.state.Fe);
      const RankTwoTensor direct_dSe_predicted = _elasticity_tensor[_qp] * direct_dEe;

      libMesh::DenseVector<Real> rhs(N_LOCAL), dx(N_LOCAL);
      rhs.zero();
      for (unsigned int p = 0; p < N_STRESS; ++p)
        rhs(p) = direct_dSe_predicted(voigt[p][0], voigt[p][1]);
      for (unsigned int a = 0; a < N_SLIP; ++a)
        rhs(N_STRESS + a) = dresistance_old[a];
      libMesh::DenseMatrix<Real> solve_matrix(record.jacobian);
      solve_matrix.lu_solve(rhs, dx);

      dSe.zero();
      for (unsigned int p = 0; p < N_STRESS; ++p)
        dSe += stressBasis(p) * dx(p);
      RankTwoTensor dplastic_increment;
      dplastic_increment.zero();
      for (unsigned int a = 0; a < N_SLIP; ++a)
      {
        Real dslip = 0.0;
        for (unsigned int q = 0; q < N_LOCAL; ++q)
          dslip += record.state.slip_derivative[a][q] * dx(q);
        dplastic_increment -=
            _flow_direction[_qp][a] * (record.dt * dslip);
        dresistance_old[a] = dx(N_STRESS + a);
      }
      dFp_inv_old = direct_dFp_inv + record.Fp_inv_old * dplastic_increment;
    }

    const auto & state = records.back().state;
    const Real Jp = 1.0 / state.Fp_inv.det();
    const Real dJp = -Jp * (state.Fp_inv.inverse() * dFp_inv_old).trace();
    const RankTwoTensor core = state.Fp_inv * state.Se * state.Fp_inv.transpose();
    const RankTwoTensor dS =
        dJp * core + Jp * (dFp_inv_old * state.Se * state.Fp_inv.transpose() +
                           state.Fp_inv * dSe * state.Fp_inv.transpose() +
                           state.Fp_inv * state.Se * dFp_inv_old.transpose());
    const auto k = voigt[e][0];
    const auto l = voigt[e][1];
    const Real factor = k == l ? 1.0 : 0.5;
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
      {
        _C[_qp](i, j, k, l) = factor * dS(i, j);
        _C[_qp](i, j, l, k) = factor * dS(i, j);
      }
  }
}

RankTwoTensor
ComputeLagrangianFCCCrystalPlasticityStress::referencePK2(const LocalState & state) const
{
  const Real Jp = 1.0 / state.Fp_inv.det();
  return Jp * state.Fp_inv * state.Se * state.Fp_inv.transpose();
}

Real
ComputeLagrangianFCCCrystalPlasticityStress::verifyConsistentTangent(
    const RankTwoTensor & F,
    const RankTwoTensor & F_old,
    const RankTwoTensor & Fp_inv_old,
    const RankTwoTensor & Se_old,
    const std::array<Real, N_SLIP> & resistance_old,
    Real /* local_dt */) const
{
  Real numerator2 = 0.0;
  Real denominator2 = 0.0;
  const RankTwoTensor F_inv_T = F.inverse().transpose();
  for (unsigned int e = 0; e < N_STRESS; ++e)
  {
    const RankTwoTensor H = stressBasis(e);
    const RankTwoTensor dF = F_inv_T * H;
    RankTwoTensor numerical[2];
    for (unsigned int side = 0; side < 2; ++side)
    {
      const Real sign = side == 0 ? -1.0 : 1.0;
      const RankTwoTensor perturbed_F = F + sign * _tangent_perturbation * dF;
      std::vector<SubstepRecord> perturbed_records;
      unsigned int iterations = 0;
      Real slip_increment = 0.0;
      if (!integrateIncrement(F_old,
                              perturbed_F,
                              Fp_inv_old,
                              Se_old,
                              resistance_old,
                              perturbed_records,
                              iterations,
                              slip_increment))
        return std::numeric_limits<Real>::infinity();
      numerical[side] = referencePK2(perturbed_records.back().state);
    }
    const RankTwoTensor finite_difference =
        (numerical[1] - numerical[0]) / (2.0 * _tangent_perturbation);
    const RankTwoTensor analytic = _C[_qp] * H;
    numerator2 += (analytic - finite_difference).doubleContraction(analytic - finite_difference);
    denominator2 += finite_difference.doubleContraction(finite_difference);
  }
  return std::sqrt(numerator2 / std::max(denominator2, 1e-30));
}

void
ComputeLagrangianFCCCrystalPlasticityStress::computeQpPK2Stress()
{
  const auto begin = std::chrono::steady_clock::now();
  updateFlowDirections();

  std::array<Real, N_SLIP> resistance_old;
  for (unsigned int a = 0; a < N_SLIP; ++a)
    resistance_old[a] = _slip_resistance_old[_qp].size() == N_SLIP
                            ? _slip_resistance_old[_qp][a]
                            : _initial_resistance;
  const RankTwoTensor Fp_inv_old = _Fp_old[_qp].det() > 0.0
                                       ? _Fp_old[_qp].inverse()
                                       : RankTwoTensor::Identity();

  RankTwoTensor F_old = _F_old[_qp];
  if (F_old.det() <= 0.0)
    F_old = RankTwoTensor::Identity();
  std::vector<SubstepRecord> records;
  unsigned int iterations = 0;
  Real accumulated_slip_increment = 0.0;
  if (!integrateIncrement(F_old,
                          _F[_qp],
                          Fp_inv_old,
                          _intermediate_pk2_old[_qp],
                          resistance_old,
                          records,
                          iterations,
                          accumulated_slip_increment))
    mooseException("Analytic FCC crystal-plasticity local Newton solve failed at element ",
                   _current_elem->id(),
                   ", qp ",
                   _qp,
                   ", local iteration ",
                   iterations,
                   ". Reduce the global time step.");

  const auto & final_record = records.back();
  const auto & state = final_record.state;

  _intermediate_pk2[_qp] = state.Se;
  _Fp[_qp] = state.Fp_inv.inverse();
  _slip_resistance[_qp].assign(state.resistance.begin(), state.resistance.end());
  _slip_rate[_qp].assign(state.slip_rate.begin(), state.slip_rate.end());
  _resolved_shear[_qp].assign(state.tau.begin(), state.tau.end());
  _local_newton_iterations[_qp] = iterations;
  _local_substeps[_qp] = records.size();
  _accumulated_slip_increment[_qp] = accumulated_slip_increment;

  libMesh::DenseVector<Real> residual(N_LOCAL);
  LocalState final_state;
  evaluateLocalSystem(final_record.x,
                      final_record.F,
                      final_record.Fp_inv_old,
                      final_record.resistance_old,
                      final_record.dt,
                      residual,
                      nullptr,
                      final_state);
  _local_residual_norm[_qp] = scaledResidualNorm(residual, final_record.x);

  _S[_qp] = referencePK2(state);
  if (_fe_problem.currentlyComputingJacobian() ||
      _fe_problem.currentlyComputingResidualAndJacobian())
  {
    computeConsistentTangent(records, _F[_qp]);
    if (_verify_analytic_tangent)
      _consistent_tangent_relative_error[_qp] = verifyConsistentTangent(
          _F[_qp], F_old, Fp_inv_old, _intermediate_pk2_old[_qp], resistance_old, _dt);
  }

  _elapsed_nanoseconds.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin)
          .count(),
      std::memory_order_relaxed);
  _calls.fetch_add(1, std::memory_order_relaxed);
}

Real
ComputeLagrangianFCCCrystalPlasticityStress::cumulativeSeconds()
{
  return static_cast<Real>(_elapsed_nanoseconds.load(std::memory_order_relaxed)) * 1e-9;
}

unsigned long long
ComputeLagrangianFCCCrystalPlasticityStress::cumulativeCalls()
{
  return _calls.load(std::memory_order_relaxed);
}
