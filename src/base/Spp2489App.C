#include "Spp2489App.h"
#include "Moose.h"
#include "AppFactory.h"
#include "ModulesApp.h"
#include "MooseSyntax.h"

InputParameters
Spp2489App::validParams()
{
  InputParameters params = MooseApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

Spp2489App::Spp2489App(const InputParameters & parameters) : MooseApp(parameters)
{
  Spp2489App::registerAll(_factory, _action_factory, _syntax);
}

Spp2489App::~Spp2489App() {}

void
Spp2489App::registerAll(Factory & f, ActionFactory & af, Syntax & syntax)
{
  ModulesApp::registerAllObjects<Spp2489App>(f, af, syntax);
  Registry::registerObjectsTo(f, {"Spp2489App"});
  Registry::registerActionsTo(af, {"Spp2489App"});

  /* register custom execute flags, action syntax, etc. here */
}

void
Spp2489App::registerApps()
{
  registerApp(Spp2489App);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
extern "C" void
Spp2489App__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  Spp2489App::registerAll(f, af, s);
}
extern "C" void
Spp2489App__registerApps()
{
  Spp2489App::registerApps();
}
