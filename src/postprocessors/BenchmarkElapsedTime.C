#include "BenchmarkElapsedTime.h"

registerMooseObject("Spp2489App", BenchmarkElapsedTime);

InputParameters
BenchmarkElapsedTime::validParams()
{
  InputParameters params = GeneralPostprocessor::validParams();
  params.addClassDescription("Reports elapsed wall time for per-load-step CP benchmarking.");
  return params;
}

BenchmarkElapsedTime::BenchmarkElapsedTime(const InputParameters & parameters)
  : GeneralPostprocessor(parameters), _start(std::chrono::steady_clock::now()), _elapsed(0.0)
{
}

void
BenchmarkElapsedTime::execute()
{
  _elapsed = std::chrono::duration<Real>(std::chrono::steady_clock::now() - _start).count();
}
