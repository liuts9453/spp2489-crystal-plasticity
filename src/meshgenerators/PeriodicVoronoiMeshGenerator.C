#include "PeriodicVoronoiMeshGenerator.h"

#include "libmesh/elem.h"

registerMooseObject("Spp2489App", PeriodicVoronoiMeshGenerator);

InputParameters
PeriodicVoronoiMeshGenerator::validParams()
{
  InputParameters params = MeshGenerator::validParams();
  params.addClassDescription(
      "Assigns element blocks using nearest-seed distance on a rectangular periodic cell.");
  params.addRequiredParam<MeshGeneratorName>("input", "Input mesh to partition.");
  params.addRequiredParam<std::vector<Real>>(
      "centers", "Flattened x,y coordinates of the Voronoi seed points.");
  params.addParam<Point>("origin", Point(0, 0, 0), "Lower-left corner of the periodic cell.");
  params.addRequiredRangeCheckedParam<Real>("width", "width > 0", "Periodic-cell width.");
  params.addRequiredRangeCheckedParam<Real>("height", "height > 0", "Periodic-cell height.");
  params.addRangeCheckedParam<Real>(
      "aspect_ratio",
      1.0,
      "aspect_ratio > 0",
      "Area-preserving metric parameter: d^2=(dx/aspect_ratio)^2+(aspect_ratio*dy)^2. "
      "Values above one favor grains elongated along x.");
  params.addParam<std::vector<Real>>(
      "power_weights",
      {},
      "Optional one-per-grain additive power weights in squared-length units. The assigned "
      "grain minimizes d^2-weight; an empty vector recovers an ordinary Voronoi diagram.");
  params.addParam<ExtraElementIDName>(
      "grain_id_name", "grain_id", "Extra element integer containing one-based grain IDs.");
  return params;
}

PeriodicVoronoiMeshGenerator::PeriodicVoronoiMeshGenerator(const InputParameters & parameters)
  : MeshGenerator(parameters),
    _input(getMesh("input")),
    _centers(getParam<std::vector<Real>>("centers")),
    _origin(getParam<Point>("origin")),
    _width(getParam<Real>("width")),
    _height(getParam<Real>("height")),
    _aspect_ratio(getParam<Real>("aspect_ratio")),
    _power_weights(getParam<std::vector<Real>>("power_weights")),
    _grain_id_name(getParam<ExtraElementIDName>("grain_id_name"))
{
  if (_centers.empty() || _centers.size() % 2 != 0)
    paramError("centers", "Supply one x,y pair for every grain.");

  for (const auto grain : make_range(_centers.size() / 2))
    if (_centers[2 * grain] < _origin(0) ||
        _centers[2 * grain] >= _origin(0) + _width ||
        _centers[2 * grain + 1] < _origin(1) ||
        _centers[2 * grain + 1] >= _origin(1) + _height)
      paramError("centers", "Every seed point must lie inside the periodic cell.");

  if (!_power_weights.empty() && _power_weights.size() != _centers.size() / 2)
    paramError("power_weights", "Supply either zero weights or exactly one weight per grain.");

  for (const auto weight : _power_weights)
    if (!std::isfinite(weight))
      paramError("power_weights", "All power weights must be finite.");
}

std::unique_ptr<MeshBase>
PeriodicVoronoiMeshGenerator::generate()
{
  std::unique_ptr<MeshBase> mesh = std::move(_input);
  const auto grain_count = _centers.size() / 2;
  const auto grain_id_index = mesh->has_elem_integer(_grain_id_name)
                                  ? mesh->get_elem_integer_index(_grain_id_name)
                                  : mesh->add_elem_integer(_grain_id_name);

  for (auto * const elem : mesh->active_element_ptr_range())
  {
    const Point centroid = elem->true_centroid();
    Real minimum_distance_squared = std::numeric_limits<Real>::max();
    std::size_t closest_grain = 0;

    for (const auto grain : make_range(grain_count))
    {
      const Real raw_dx = std::abs(centroid(0) - _centers[2 * grain]);
      const Real raw_dy = std::abs(centroid(1) - _centers[2 * grain + 1]);
      const Real dx = std::min(raw_dx, _width - raw_dx);
      const Real dy = std::min(raw_dy, _height - raw_dy);
      const Real scaled_dx = dx / _aspect_ratio;
      const Real scaled_dy = _aspect_ratio * dy;
      const Real weight = _power_weights.empty() ? 0.0 : _power_weights[grain];
      const Real distance_squared = scaled_dx * scaled_dx + scaled_dy * scaled_dy - weight;
      if (distance_squared < minimum_distance_squared)
      {
        minimum_distance_squared = distance_squared;
        closest_grain = grain;
      }
    }

    const auto grain_id = static_cast<subdomain_id_type>(closest_grain + 1);
    elem->subdomain_id() = grain_id;
    elem->set_extra_integer(grain_id_index, grain_id);
  }

  for (const auto grain : make_range(grain_count))
    mesh->set_subdomain_name(grain + 1, "grain_" + std::to_string(grain + 1));

  mesh->unset_is_prepared();
  return mesh;
}
