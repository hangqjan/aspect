/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_adjoint_diagnostics_diagnostics_h
#define _aspect_adjoint_diagnostics_diagnostics_h

#include <aspect/adjoint/operators/advection_adjoint_calculator.h>
#include <aspect/adjoint/time_dependent/trajectory_store.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim> class Manager;

    namespace Diagnostics
    {
      /**
       * Test-driver access to perturbing reduced-control checks.
       *
       * These operations restore or rerun the forward model and therefore do
       * not belong to the public production facade. Keeping them behind this
       * adapter makes their side effects explicit to a test driver.
       *
       * @ingroup Adjoint
       */
      template <int dim>
      class InitialTemperatureOptimizationChecks
      {
        public:
          explicit InitialTemperatureOptimizationChecks(
            ::aspect::Adjoint::Manager<dim> &manager);

          void reuse() const;
          void single_evaluation() const;
          void taylor() const;
          void initialization_stage(const unsigned int stage) const;

        private:
          ::aspect::Adjoint::Manager<dim> &manager;
      };

      /**
       * Test-driver access to perturbing and high-cost checks for the
       * advection transpose. Keeping this adapter in Diagnostics prevents the
       * production operator from exposing a public family of run_* methods.
       *
       * @ingroup Adjoint
       */
      template <int dim>
      class AdvectionOperatorChecks
      {
        public:
          explicit AdvectionOperatorChecks(AdvectionAdjointCalculator<dim> &calculator);

          void velocity_transpose(const AdvectionField &advection_field,
                                  const ForwardStateView<dim> &snapshot,
                                  const LinearAlgebra::BlockVector &current_field_adjoint,
                                  const double epsilon) const;

          void transposed_constraints(const AdvectionField &advection_field,
                                      const ForwardStateView<dim> &snapshot,
                                      const LinearAlgebra::BlockVector &current_field_adjoint) const;

          void artificial_viscosity_coefficient_seed(
            const AdvectionField &advection_field,
            const ForwardStateView<dim> &snapshot,
            const LinearAlgebra::BlockVector &current_field_adjoint,
            const double epsilon) const;

          void artificial_viscosity_estimator_transpose(
            const AdvectionField &advection_field,
            const ForwardStateView<dim> &snapshot,
            const double epsilon) const;

          void local_entropy_residual_transpose(const AdvectionField &advection_field,
                                                const ForwardStateView<dim> &snapshot,
                                                const double epsilon) const;

          void global_entropy_denominator_scalar(const AdvectionField &advection_field,
                                                 const ForwardStateView<dim> &snapshot,
                                                 const double epsilon) const;

          void global_entropy_variation_transpose(const AdvectionField &advection_field,
                                                  const ForwardStateView<dim> &snapshot,
                                                  const double epsilon) const;

        private:
          AdvectionAdjointCalculator<dim> &calculator;
      };

      /**
       * Central entry point for lightweight, non-perturbing adjoint runtime
       * diagnostics. Numerical differentiation and transpose checks belong in
       * test drivers and are intentionally not part of this interface.
       *
       * @ingroup Adjoint
       */
      template <int dim>
      class Manager : public SimulatorAccess<dim>
      {
        public:
          /** Enable or disable rank-local memory checkpoints. */
          void
          set_output_memory_checkpoints(const bool enabled);

          /** Report durable trajectory storage and the active replay cache. */
          void
          report_trajectory(
            const TrajectoryStore &trajectory,
            const std::string &stage,
            const std::size_t local_cache_memory,
            const std::size_t local_current_record_memory,
            const unsigned int stokes_recovery_solves,
            const unsigned int sparse_peak_window_state_count,
            const unsigned int sparse_replay_transition_count,
            const unsigned int sparse_restart_replay_transition_count) const;

          /** Emit one allocation-free, rank-local memory checkpoint. */
          void
          report_memory_checkpoint(
            const std::string &stage,
            const LinearAlgebra::BlockVector &state,
            const std::size_t reverse_cache_size) const;

        private:
          bool output_memory_checkpoints = false;
      };
    }
  }
}

#endif
