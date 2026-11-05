#include "GrainPropertyAux.h"

registerMooseObject("Spp2489App", GrainPropertyAux);

InputParameters
GrainPropertyAux::validParams()
{
  InputParameters params = AuxKernel::validParams();
  params.addClassDescription("Outputs a grain-wise constant scalar property.");
  params.addRequiredParam<std::vector<Real>>("values", "Values ordered by grain ID.");
  params.addParam<subdomain_id_type>(
      "first_grain_id", 1, "Subdomain ID corresponding to the first value.");
  return params;
}

GrainPropertyAux::GrainPropertyAux(const InputParameters & parameters)
  : AuxKernel(parameters),
    _values(getParam<std::vector<Real>>("values")),
    _first_grain_id(getParam<subdomain_id_type>("first_grain_id"))
{
  if (isNodal())
    paramError("variable", "GrainPropertyAux requires an elemental variable.");
  if (_values.empty())
    paramError("values", "At least one grain value is required.");
}

Real
GrainPropertyAux::computeValue()
{
  const auto grain_id = _current_elem->subdomain_id();
  if (grain_id < _first_grain_id || grain_id >= _first_grain_id + _values.size())
    mooseError("No GrainPropertyAux value for subdomain ", grain_id, ".");
  return _values[grain_id - _first_grain_id];
}
