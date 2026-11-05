#pragma once

#include "ComputeLagrangianStressPK2.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"

#include <array>
#include <atomic>

/**
 * Fully coupled finite-strain FCC crystal plasticity in the Total-Lagrangian PK2 interface.
 *
 * The local backward-Euler equations solve simultaneously for the six components of elastic
 * intermediate PK2 stress and all slip resistances.  The algorithmic dS/dE is obtained by
 * analytic linearization of those same discrete equations and the implicit-function theorem.
 * No Cauchy-stress adapter or numerical perturbation is used in the constitutive/Jacobian path.
 */
class ComputeLagrangianFCCCrystalPlasticityStress : public ComputeLagrangianStressPK2
{
public:
  static InputParameters validParams();
  ComputeLagrangianFCCCrystalPlasticityStress(const InputParameters & parameters);

  static Real cumulativeSeconds();
  static unsigned long long cumulativeCalls();

protected:
  void initQpStatefulProperties() override;
  void computeQpPK2Stress() override;

private:
  static constexpr unsigned int N_SLIP = 12;
  static constexpr unsigned int N_STRESS = 6;
  static constexpr unsigned int N_LOCAL = N_STRESS + N_SLIP;

  struct LocalState
  {
    RankTwoTensor Se;
    RankTwoTensor Fp_inv;
    RankTwoTensor Fe;
    std::array<Real, N_SLIP> resistance;
    std::array<Real, N_SLIP> tau;
    std::array<Real, N_SLIP> slip_rate;
    std::array<Real, N_SLIP> hardening;
    std::array<Real, N_SLIP> hardening_derivative;
    std::array<std::array<Real, N_LOCAL>, N_SLIP> slip_derivative;
  };

  struct SubstepRecord
  {
    RankTwoTensor F;
    RankTwoTensor Fp_inv_old;
    std::array<Real, N_SLIP> resistance_old;
    libMesh::DenseVector<Real> x;
    libMesh::DenseMatrix<Real> jacobian;
    LocalState state;
    Real dt;
    Real fraction;
  };

  void updateFlowDirections();
  RankTwoTensor stressTensor(const libMesh::DenseVector<Real> & x) const;
  RankTwoTensor stressBasis(unsigned int component) const;
  Real scaledResidualNorm(const libMesh::DenseVector<Real> & residual,
                          const libMesh::DenseVector<Real> & x) const;
  bool evaluateLocalSystem(const libMesh::DenseVector<Real> & x,
                           const RankTwoTensor & F,
                           const RankTwoTensor & Fp_inv_old,
                           const std::array<Real, N_SLIP> & resistance_old,
                           Real local_dt,
                           libMesh::DenseVector<Real> & residual,
                           libMesh::DenseMatrix<Real> * jacobian,
                           LocalState & state) const;
  bool solveLocalSystem(const RankTwoTensor & F,
                        const RankTwoTensor & Fp_inv_old,
                        const RankTwoTensor & Se_old,
                        const std::array<Real, N_SLIP> & resistance_old,
                        Real local_dt,
                        libMesh::DenseVector<Real> & x,
                        libMesh::DenseMatrix<Real> & jacobian,
                        LocalState & state,
                        unsigned int & iterations) const;
  bool integrateIncrement(const RankTwoTensor & F_old,
                          const RankTwoTensor & F,
                          const RankTwoTensor & Fp_inv_old,
                          const RankTwoTensor & Se_old,
                          const std::array<Real, N_SLIP> & resistance_old,
                          std::vector<SubstepRecord> & records,
                          unsigned int & total_iterations,
                          Real & accumulated_slip_increment) const;
  void computeConsistentTangent(const std::vector<SubstepRecord> & records,
                                const RankTwoTensor & F);
  RankTwoTensor referencePK2(const LocalState & state) const;
  Real verifyConsistentTangent(const RankTwoTensor & F,
                               const RankTwoTensor & F_old,
                               const RankTwoTensor & Fp_inv_old,
                               const RankTwoTensor & Se_old,
                               const std::array<Real, N_SLIP> & resistance_old,
                               Real local_dt) const;

  const MaterialProperty<RankFourTensor> & _elasticity_tensor;
  const MaterialProperty<RankTwoTensor> & _crystal_rotation;
  const MaterialProperty<RankTwoTensor> & _F_old;

  MaterialProperty<RankTwoTensor> & _Fp;
  const MaterialProperty<RankTwoTensor> & _Fp_old;
  MaterialProperty<RankTwoTensor> & _intermediate_pk2;
  const MaterialProperty<RankTwoTensor> & _intermediate_pk2_old;
  MaterialProperty<std::vector<Real>> & _slip_resistance;
  const MaterialProperty<std::vector<Real>> & _slip_resistance_old;
  MaterialProperty<std::vector<Real>> & _slip_rate;
  MaterialProperty<std::vector<Real>> & _resolved_shear;
  MaterialProperty<std::vector<RankTwoTensor>> & _flow_direction;
  MaterialProperty<Real> & _local_newton_iterations;
  MaterialProperty<Real> & _local_residual_norm;
  MaterialProperty<Real> & _consistent_tangent_relative_error;
  MaterialProperty<Real> & _local_substeps;
  MaterialProperty<Real> & _accumulated_slip_increment;

  std::array<RealVectorValue, N_SLIP> _crystal_slip_direction;
  std::array<RealVectorValue, N_SLIP> _crystal_plane_normal;

  const Real _reference_slip_rate;
  const Real _rate_sensitivity;
  const Real _initial_resistance;
  const Real _hardening_modulus;
  const Real _saturation_resistance;
  const Real _hardening_exponent;
  const Real _latent_hardening;
  const Real _slip_increment_tolerance;
  const Real _local_absolute_tolerance;
  const Real _local_relative_tolerance;
  const unsigned int _local_max_iterations;
  const unsigned int _line_search_max_iterations;
  const unsigned int _maximum_substeps;
  const bool _verify_analytic_tangent;
  const Real _tangent_perturbation;

  static std::atomic<long long> _elapsed_nanoseconds;
  static std::atomic<unsigned long long> _calls;
};
