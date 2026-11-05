#include "CrystalPlasticityTiming.h"
#include "ComputeLagrangianFCCCrystalPlasticityStress.h"

registerMooseObject("Spp2489App", CrystalPlasticityTiming);

InputParameters
CrystalPlasticityTiming::validParams()
{
  InputParameters params = GeneralPostprocessor::validParams();
  params.addClassDescription("Reports cumulative analytic PK2 crystal-plasticity update timing.");
  params.addParam<MooseEnum>("metric",
                             MooseEnum("seconds total_seconds residual_seconds jacobian_seconds "
                                       "linear_solve_seconds line_search_seconds tangent_seconds "
                                       "calls local_newton_iterations substeps "
                                       "line_search_evaluations tangent_calls grouped_calls "
                                       "legacy_calls",
                                       "seconds"),
                             "Mutually exclusive phase time or cumulative event count to report.");
  return params;
}

CrystalPlasticityTiming::CrystalPlasticityTiming(const InputParameters & parameters)
  : GeneralPostprocessor(parameters), _metric(getParam<MooseEnum>("metric")), _value(0.0)
{
}

void
CrystalPlasticityTiming::finalize()
{
  _value = ComputeLagrangianFCCCrystalPlasticityStress::cumulativeMetric(_metric);
  gatherSum(_value);
}

PostprocessorValue
CrystalPlasticityTiming::getValue() const
{
  return _value;
}
