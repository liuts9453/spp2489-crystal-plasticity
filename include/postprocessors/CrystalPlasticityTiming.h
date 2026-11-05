#pragma once

#include "GeneralPostprocessor.h"

/** Reports cumulative time or call count from the analytic PK2 FCC material. */
class CrystalPlasticityTiming : public GeneralPostprocessor
{
public:
  static InputParameters validParams();
  CrystalPlasticityTiming(const InputParameters & parameters);
  virtual void initialize() override {}
  virtual void execute() override {}
  virtual void finalize() override;
  virtual PostprocessorValue getValue() const override;

protected:
  const MooseEnum _metric;
  PostprocessorValue _value;
};
