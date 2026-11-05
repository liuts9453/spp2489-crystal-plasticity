#include "CrystalPlasticityTiming.h"
#include "ComputeLagrangianFCCCrystalPlasticityStress.h"

registerMooseObject("Spp2489App", CrystalPlasticityTiming);

InputParameters
CrystalPlasticityTiming::validParams()
{
  InputParameters params = GeneralPostprocessor::validParams();
  params.addClassDescription("Reports cumulative analytic PK2 crystal-plasticity update timing.");
  params.addParam<MooseEnum>(
      "metric", MooseEnum("seconds calls", "seconds"), "Quantity to report.");
  return params;
}

CrystalPlasticityTiming::CrystalPlasticityTiming(const InputParameters & parameters)
  : GeneralPostprocessor(parameters), _metric(getParam<MooseEnum>("metric")), _value(0.0)
{
}

void
CrystalPlasticityTiming::finalize()
{
  _value =
      _metric == "seconds"
          ? ComputeLagrangianFCCCrystalPlasticityStress::cumulativeSeconds()
          : static_cast<Real>(ComputeLagrangianFCCCrystalPlasticityStress::cumulativeCalls());
  gatherSum(_value);
}

PostprocessorValue
CrystalPlasticityTiming::getValue() const
{
  return _value;
}
