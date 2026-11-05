#pragma once

#include "GeneralPostprocessor.h"

#include <chrono>

/** Low-overhead wall clock used to resolve the cost of each completed load step. */
class BenchmarkElapsedTime : public GeneralPostprocessor
{
public:
  static InputParameters validParams();
  BenchmarkElapsedTime(const InputParameters & parameters);

  void initialize() override {}
  void execute() override;
  void finalize() override {}
  PostprocessorValue getValue() const override { return _elapsed; }

private:
  const std::chrono::steady_clock::time_point _start;
  Real _elapsed;
};
