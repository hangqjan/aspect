/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2, or (at your option) any later version.
*/

#include "common.h"

#include <aspect/adjoint/optimization/optimization_parameters.h>

#include <stdexcept>

namespace
{
  void
  validate_or_throw(const unsigned int history_size,
                    const double gradient_tolerance,
                    const double step_tolerance,
                    const double trust_region_radius,
                    const double lower_bound,
                    const double upper_bound)
  {
    aspect::Adjoint::validate_initial_temperature_optimization_parameters(
      history_size,
      gradient_tolerance,
      step_tolerance,
      trust_region_radius,
      lower_bound,
      upper_bound);
  }
}

TEST_CASE("initial-temperature optimizer scalar parameters reject invalid values locally")
{
  REQUIRE_NOTHROW(validate_or_throw(3, 1e-8, 1e-12, 1e-3, -1e-2, 1e-2));
  REQUIRE_THROWS_AS(validate_or_throw(3, 1e-8, 1e-12, 1e-3, 1.0, -1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(validate_or_throw(0, 1e-8, 1e-12, 1e-3, -1e-2, 1e-2),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(validate_or_throw(3, 1e-8, 1e-12, 0.0, -1e-2, 1e-2),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(validate_or_throw(3, 0.0, 1e-12, 1e-3, -1e-2, 1e-2),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(validate_or_throw(3, 1e-8, 0.0, 1e-3, -1e-2, 1e-2),
                    std::invalid_argument);
}
