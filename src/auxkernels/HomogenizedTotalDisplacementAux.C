#include "HomogenizedTotalDisplacementAux.h"

registerMooseObject("Spp2489App", HomogenizedTotalDisplacementAux);

InputParameters
HomogenizedTotalDisplacementAux::validParams()
{
  InputParameters params = AuxKernel::validParams();
  params.addClassDescription(
      "Reconstructs total displacement from the homogenized gradient and periodic fluctuation.");
  params.addRequiredCoupledVar("macro_gradient", "Column-major homogenized displacement gradient.");
  params.addRequiredCoupledVar("displacements", "Periodic displacement fluctuation variables.");
  params.addRequiredParam<MooseEnum>(
      "component", MooseEnum("x=0 y=1 magnitude=2"), "Output component or magnitude.");
  params.addParam<Point>(
      "reference_point", Point(0, 0, 0), "Origin used by the macroscopic affine field.");
  return params;
}

HomogenizedTotalDisplacementAux::HomogenizedTotalDisplacementAux(
    const InputParameters & parameters)
  : AuxKernel(parameters),
    _macro_gradient(coupledScalarValue("macro_gradient")),
    _displacements(coupledValues("displacements")),
    _component(getParam<MooseEnum>("component")),
    _reference_point(getParam<Point>("reference_point"))
{
  if (!isNodal())
    paramError("variable", "HomogenizedTotalDisplacementAux requires a nodal variable.");
  if (_displacements.size() != 2)
    paramError("displacements", "Exactly two displacement components are required.");
}

Real
HomogenizedTotalDisplacementAux::computeValue()
{
  const Point X = *_current_node - _reference_point;
  const Real ux = (*_displacements[0])[_qp] + _macro_gradient[0] * X(0) +
                  _macro_gradient[2] * X(1);
  const Real uy = (*_displacements[1])[_qp] + _macro_gradient[1] * X(0) +
                  _macro_gradient[3] * X(1);

  if (_component == 0)
    return ux;
  if (_component == 1)
    return uy;
  return std::sqrt(ux * ux + uy * uy);
}
