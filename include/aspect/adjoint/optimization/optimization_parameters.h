/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2, or (at your option) any later version.
*/

#ifndef _aspect_adjoint_optimization_optimization_parameters_h
#define _aspect_adjoint_optimization_optimization_parameters_h

#include <stdexcept>

namespace aspect
{
  namespace Adjoint
  {
    /**
     * Validate the scalar and cross-parameter contracts for the native ROL
     * initial-temperature optimizer. This helper deliberately has no MPI,
     * Simulator, or ROL dependency, so the production validation contracts can
     * be tested locally without entering ASPECT's top-level MPI abort path.
     */
    inline void
    validate_initial_temperature_optimization_parameters(
      const unsigned int lbfgs_history_size,
      const double gradient_tolerance,
      const double step_tolerance,
      const double initial_trust_region_radius,
      const double lower_bound,
      const double upper_bound)
    {
      if (lbfgs_history_size == 0)
        throw std::invalid_argument("Optimization L-BFGS history size must be at least one.");
      if (!(gradient_tolerance > 0.0))
        throw std::invalid_argument("Optimization gradient tolerance must be positive.");
      if (!(step_tolerance > 0.0))
        throw std::invalid_argument("Optimization step tolerance must be positive.");
      if (!(initial_trust_region_radius > 0.0))
        throw std::invalid_argument("Optimization initial trust-region radius must be positive.");
      if (!(lower_bound <= upper_bound))
        throw std::invalid_argument("Optimization lower bound must not exceed upper bound.");
    }
  }
}

#endif
