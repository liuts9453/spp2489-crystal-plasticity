//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html
#include "Spp2489TestApp.h"
#include "Spp2489App.h"
#include "Moose.h"
#include "AppFactory.h"
#include "MooseSyntax.h"

InputParameters
Spp2489TestApp::validParams()
{
  InputParameters params = Spp2489App::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

Spp2489TestApp::Spp2489TestApp(const InputParameters & parameters) : MooseApp(parameters)
{
  Spp2489TestApp::registerAll(
      _factory, _action_factory, _syntax, getParam<bool>("allow_test_objects"));
}

Spp2489TestApp::~Spp2489TestApp() {}

void
Spp2489TestApp::registerAll(Factory & f, ActionFactory & af, Syntax & s, bool use_test_objs)
{
  Spp2489App::registerAll(f, af, s);
  if (use_test_objs)
  {
    Registry::registerObjectsTo(f, {"Spp2489TestApp"});
    Registry::registerActionsTo(af, {"Spp2489TestApp"});
  }
}

void
Spp2489TestApp::registerApps()
{
  registerApp(Spp2489App);
  registerApp(Spp2489TestApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
// External entry point for dynamic application loading
extern "C" void
Spp2489TestApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  Spp2489TestApp::registerAll(f, af, s);
}
extern "C" void
Spp2489TestApp__registerApps()
{
  Spp2489TestApp::registerApps();
}
