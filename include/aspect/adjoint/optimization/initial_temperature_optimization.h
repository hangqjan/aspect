/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_adjoint_optimization_initial_temperature_optimization_h
#define _aspect_adjoint_optimization_initial_temperature_optimization_h

#include <aspect/adjoint/optimization/initial_temperature_control.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/parameter_handler.h>

#include <memory>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim> class Manager;

    /**
     * Own the configuration and ROL lifecycle of reduced
     * initial-temperature optimization.
     *
     * The simulator-facing adjoint manager remains the evaluation adapter: it
     * restores the canonical forward state and supplies value/gradient pairs.
     * This class owns everything specific to the optimization algorithm,
     * including bound construction, Lin--More/L-BFGS setup, callback
     * accounting, and the optional independent final-gradient qualification.
     * Consequently, changing the optimizer does not affect the reverse-time
     * workflow or trajectory implementation.
     *
     * @ingroup Adjoint
     */
    template <int dim>
    class InitialTemperatureOptimization : public SimulatorAccess<dim>
    {
      public:
        /** Declare the solve-mode and optimization parameters. */
        static void declare_parameters(ParameterHandler &prm);

        /** Parse and validate the optimization configuration. */
        void parse_parameters(ParameterHandler &prm);

        /** Return whether reduced initial-temperature optimization is active. */
        bool enabled() const;

        /** Run one complete ROL optimization through @p adjoint_manager. */
        void run(Manager<dim> &adjoint_manager) const;

      private:
        friend class Manager<dim>;

        /** Complete pre-forward state restored before every trial control. */
        struct CanonicalState
        {
          LinearAlgebra::BlockVector solution;
          LinearAlgebra::BlockVector old_solution;
          LinearAlgebra::BlockVector old_old_solution;
          LinearAlgebra::BlockVector current_linearization_point;
          LinearAlgebra::BlockVector system_right_hand_side;
          AffineConstraints<double> current_constraints;
          double time = 0.0;
          double time_step = 0.0;
          double old_time_step = 0.0;
          double pressure_normalization_adjustment = 0.0;
          double pressure_scaling = 0.0;
          unsigned int timestep_number = 0;
          unsigned int nonlinear_iteration = 0;
          unsigned int nonlinear_solver_failures = 0;
          unsigned int linear_solver_failures = 0;
          bool rebuild_sparsity_and_matrices = false;
          bool rebuild_stokes_matrix = false;
          bool rebuild_stokes_preconditioner = false;
          bool captured = false;
        };

        CanonicalState canonical_state;
        std::unique_ptr<InitialTemperatureControl<dim>> control_space;
        std::unique_ptr<LinearAlgebra::Vector> fixed_temperature;

        bool optimization_enabled = false;
        unsigned int maximum_iterations = 1;
        double gradient_tolerance = 1e-8;
        double step_tolerance = 1e-12;
        double initial_trust_region_radius = 1e-3;
        unsigned int lbfgs_history_size = 3;
        double lower_bound = -1e-2;
        double upper_bound = 1e-2;
        unsigned int output_verbosity = 0;
        bool use_physical_temperature_bounds = false;
    };
  }
}

#endif
