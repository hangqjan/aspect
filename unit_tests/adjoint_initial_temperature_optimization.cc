/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2, or (at your option) any later version.
*/

#include "common.h"

#include <aspect/adjoint/optimization/initial_temperature_optimization.h>

#include <stdexcept>

using aspect::Adjoint::InitialTemperatureOptimization;
using dealii::ParameterHandler;

TEST_CASE("initial-temperature optimization owns and validates its parameter group")
{
  ParameterHandler parameters;
  InitialTemperatureOptimization<2>::declare_parameters(parameters);

  InitialTemperatureOptimization<2> optimization;
  optimization.parse_parameters(parameters);
  CHECK_FALSE(optimization.enabled());

  parameters.parse_input_from_string(
    "subsection Adjoint\n"
    "  set Solve mode = optimize\n"
    "  subsection Optimization\n"
    "    set Maximum iterations = 4\n"
    "    set L-BFGS history size = 6\n"
    "    set Lower bound = -0.2\n"
    "    set Upper bound = 0.3\n"
    "  end\n"
    "end\n");
  optimization.parse_parameters(parameters);
  CHECK(optimization.enabled());
}



TEST_CASE("initial-temperature optimization rejects inconsistent bounds")
{
  ParameterHandler parameters;
  InitialTemperatureOptimization<2>::declare_parameters(parameters);
  parameters.parse_input_from_string(
    "subsection Adjoint\n"
    "  set Solve mode = optimize\n"
    "  subsection Optimization\n"
    "    set Lower bound = 1\n"
    "    set Upper bound = -1\n"
    "  end\n"
    "end\n");

  InitialTemperatureOptimization<2> optimization;
  CHECK_THROWS_AS(optimization.parse_parameters(parameters), std::invalid_argument);
}
