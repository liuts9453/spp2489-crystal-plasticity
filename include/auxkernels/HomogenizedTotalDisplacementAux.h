#pragma once

#include "AuxKernel.h"

/** Reconstructs Hbar*X plus the periodic displacement fluctuation. */
class HomogenizedTotalDisplacementAux : public AuxKernel
{
public:
  static InputParameters validParams();
  HomogenizedTotalDisplacementAux(const InputParameters & parameters);

protected:
  virtual Real computeValue() override;

  const VariableValue & _macro_gradient;
  const std::vector<const VariableValue *> _displacements;
  const int _component;
  const Point _reference_point;
};
