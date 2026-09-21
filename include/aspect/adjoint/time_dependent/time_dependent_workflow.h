/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_adjoint_time_dependent_time_dependent_workflow_h
#define _aspect_adjoint_time_dependent_time_dependent_workflow_h

#include <aspect/global.h>
#include <aspect/linear_algebra_types.h>

#include <map>
#include <string>
#include <vector>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim> class Manager;
    template <int dim> class TrajectoryManager;

    /**
     * Coordinate one complete reverse-time adjoint evaluation.
     *
     * This workflow is the single production entry point for terminal and
     * running objective seeds, trajectory reconstruction, the three-level
     * reverse recurrence, checkpoint commits, and successful cleanup. The
     * simulator-facing Manager remains the facade and supplies the narrowly
     * scoped assembly/replay primitives used by this workflow.
     *
     * @ingroup Adjoint
     */
    template <int dim>
    class TimeDependentWorkflow
    {
      public:
        /** Execute the configured reverse-time workflow. */
        void run(Manager<dim> &manager);

      private:
        friend class Manager<dim>;
        friend class TrajectoryManager<dim>;

        std::map<std::string, double> running_stokes_objective_values;
        // Retained after finalization for diagnostics. These are copies of the
        // exact terminal covectors before the reverse recurrence mutates its
        // history slots.
        LinearAlgebra::BlockVector terminal_temperature_seed;
        std::vector<LinearAlgebra::BlockVector> terminal_composition_seeds;
        LinearAlgebra::BlockVector initial_temperature_gradient;
        std::vector<LinearAlgebra::BlockVector> initial_composition_gradients;
    };
  }
}

#endif
