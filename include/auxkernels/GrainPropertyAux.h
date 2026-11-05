#pragma once

#include "AuxKernel.h"

/** Maps a one-based grain subdomain ID to a constant scalar value. */
class GrainPropertyAux : public AuxKernel
{
public:
  static InputParameters validParams();
  GrainPropertyAux(const InputParameters & parameters);

protected:
  virtual Real computeValue() override;

  const std::vector<Real> & _values;
  const subdomain_id_type _first_grain_id;
};
