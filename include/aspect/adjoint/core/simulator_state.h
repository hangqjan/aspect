/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_adjoint_core_simulator_state_h
#define _aspect_adjoint_core_simulator_state_h

#include <aspect/global.h>

#include <deal.II/lac/affine_constraints.h>

namespace aspect
{
  template <int dim>
  class Simulator;

  namespace Adjoint
  {
    /**
     * An exception-safe guard for the mutable part of Simulator state used by
     * adjoint assembly, trajectory replay, and diagnostic perturbations.
     *
     * A number of adjoint operations temporarily install a reconstructed
     * forward state in Simulator because the existing ASPECT assembly code
     * obtains its inputs from SimulatorAccess. This class makes that mutation
     * explicit and restores the complete affected state on every exit path.
     * Keeping the restore operation in one place also prevents a newly added
     * Simulator field from being restored by one adjoint path but forgotten by
     * another.
     *
     * The guard is deliberately non-copyable and non-movable. Its lifetime
     * defines the interval during which temporary Simulator state may be
     * installed.
     *
     * @ingroup Adjoint
     */
    template <int dim>
    class ScopedSimulatorState
    {
      public:
        explicit ScopedSimulatorState(Simulator<dim> &simulator);

        ScopedSimulatorState(const ScopedSimulatorState &) = delete;
        ScopedSimulatorState &operator=(const ScopedSimulatorState &) = delete;
        ScopedSimulatorState(ScopedSimulatorState &&) = delete;
        ScopedSimulatorState &operator=(ScopedSimulatorState &&) = delete;

        /** Restore the captured state. Repeated calls have no effect. */
        void restore();

        /**
         * Keep the currently installed Simulator state when this object goes
         * out of scope. This is only appropriate when ownership of the
         * mutation has deliberately been transferred to the caller.
         */
        void release();

        ~ScopedSimulatorState() noexcept;

      private:
        Simulator<dim> &simulator;
        LinearAlgebra::BlockVector solution;
        LinearAlgebra::BlockVector old_solution;
        LinearAlgebra::BlockVector old_old_solution;
        LinearAlgebra::BlockVector current_linearization_point;
        LinearAlgebra::BlockVector system_rhs;
        AffineConstraints<double> current_constraints;
        double time;
        double time_step;
        double old_time_step;
        double pressure_normalization_adjustment;
        double pressure_scaling;
        unsigned int timestep_number;
        unsigned int nonlinear_iteration;
        unsigned int nonlinear_solver_failures;
        unsigned int linear_solver_failures;
        bool rebuild_sparsity_and_matrices;
        bool rebuild_stokes_matrix;
        bool rebuild_stokes_preconditioner;
        bool active;
    };
  }
}

#endif
