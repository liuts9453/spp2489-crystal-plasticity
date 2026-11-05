#pragma once

#include "MeshGenerator.h"

/** Assigns structured-mesh elements to periodic Voronoi grains by their centroids. */
class PeriodicVoronoiMeshGenerator : public MeshGenerator
{
public:
  static InputParameters validParams();
  PeriodicVoronoiMeshGenerator(const InputParameters & parameters);

  virtual std::unique_ptr<MeshBase> generate() override;

protected:
  std::unique_ptr<MeshBase> & _input;
  const std::vector<Real> & _centers;
  const Point _origin;
  const Real _width;
  const Real _height;
  const Real _aspect_ratio;
  const std::vector<Real> & _power_weights;
  const ExtraElementIDName _grain_id_name;
};
