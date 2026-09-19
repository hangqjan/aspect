/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_adjoint_core_instantaneous_stokes_workflow_h
#define _aspect_adjoint_core_instantaneous_stokes_workflow_h

#include <aspect/adjoint/core/kernel_calculator.h>
#include <aspect/adjoint/objective/objective_functional.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  template <int dim> class Simulator;

  namespace Adjoint
  {
    /**
     * Production workflow for objectives of one Stokes state. It owns no
     * persistent result data: the facade supplies repositories that remain
     * valid after the workflow returns.
     *
     * @ingroup Adjoint
     */
    template <int dim>
    class InstantaneousStokesWorkflow : public SimulatorAccess<dim>
    {
      public:
        InstantaneousStokesWorkflow(Simulator<dim> &simulator,
                                    Objective::Manager<dim> &objective_manager);

        void
        run(std::vector<std::unique_ptr<ObjectiveResult>> &objective_results,
            std::vector<std::unique_ptr<AdjointState>> &adjoint_states,
            KernelRepository<dim> &kernels);

      private:
        void validate_setup() const;
        void prepare_required_forward_postprocessors();

        Simulator<dim> &simulator;
        Objective::Manager<dim> &objective_manager;
    };
  }
}

#endif
