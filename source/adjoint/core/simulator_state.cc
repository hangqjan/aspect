/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/adjoint/core/simulator_state.h>
#include <aspect/simulator.h>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    ScopedSimulatorState<dim>::ScopedSimulatorState(Simulator<dim> &simulator_object)
      :
      simulator(simulator_object),
      solution(simulator.solution),
      old_solution(simulator.old_solution),
      old_old_solution(simulator.old_old_solution),
      current_linearization_point(simulator.current_linearization_point),
      system_rhs(simulator.system_rhs),
      current_constraints(simulator.current_constraints),
      time(simulator.time),
      time_step(simulator.time_step),
      old_time_step(simulator.old_time_step),
      pressure_normalization_adjustment(simulator.last_pressure_normalization_adjustment),
      pressure_scaling(simulator.pressure_scaling),
      timestep_number(simulator.timestep_number),
      nonlinear_iteration(simulator.nonlinear_iteration),
      nonlinear_solver_failures(simulator.nonlinear_solver_failures),
      linear_solver_failures(simulator.linear_solver_failures),
      rebuild_sparsity_and_matrices(simulator.rebuild_sparsity_and_matrices),
      rebuild_stokes_matrix(simulator.rebuild_stokes_matrix),
      rebuild_stokes_preconditioner(simulator.rebuild_stokes_preconditioner),
      active(true)
    {}



    template <int dim>
    void
    ScopedSimulatorState<dim>::restore()
    {
      if (!active)
        return;

      // Restore vectors before scalar time information and constraints. This
      // mirrors the order in which replay code installs a state and ensures
      // that any subsequent constraint rebuild observes the original fields.
      simulator.solution = solution;
      simulator.old_solution = old_solution;
      simulator.old_old_solution = old_old_solution;
      simulator.current_linearization_point = current_linearization_point;
      simulator.system_rhs = system_rhs;
      simulator.current_constraints.copy_from(current_constraints);
      simulator.time = time;
      simulator.time_step = time_step;
      simulator.old_time_step = old_time_step;
      simulator.last_pressure_normalization_adjustment = pressure_normalization_adjustment;
      simulator.pressure_scaling = pressure_scaling;
      simulator.timestep_number = timestep_number;
      simulator.nonlinear_iteration = nonlinear_iteration;
      simulator.nonlinear_solver_failures = nonlinear_solver_failures;
      simulator.linear_solver_failures = linear_solver_failures;
      simulator.rebuild_sparsity_and_matrices = rebuild_sparsity_and_matrices;
      simulator.rebuild_stokes_matrix = rebuild_stokes_matrix;
      simulator.rebuild_stokes_preconditioner = rebuild_stokes_preconditioner;
      active = false;
    }



    template <int dim>
    void
    ScopedSimulatorState<dim>::release()
    {
      active = false;
    }



    template <int dim>
    ScopedSimulatorState<dim>::~ScopedSimulatorState() noexcept
    {
      if (active)
        try
          {
            restore();
          }
        catch (...)
          {
            // Destructors must not throw while another exception is already
            // leaving an adjoint assembly. A failed restoration is a fatal
            // invariant violation and is reported without a second throw.
            AssertNothrow(false,
            ExcMessage("Restoring temporary adjoint Simulator state failed."));
          }
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::ScopedSimulatorState<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
