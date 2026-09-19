/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#ifndef _aspect_adjoint_core_manager_h
#define _aspect_adjoint_core_manager_h

#include <aspect/advection_field.h>
#include <aspect/adjoint/core/kernel_calculator.h>
#include <aspect/adjoint/objective/objective_functional.h>
#include <aspect/adjoint/optimization/initial_temperature_control.h>
#include <aspect/adjoint/optimization/initial_temperature_optimization.h>
#include <aspect/adjoint/time_dependent/time_dependent_workflow.h>
#include <aspect/adjoint/time_dependent/trajectory_manager.h>
#include <aspect/adjoint/diagnostics/diagnostics.h>
#include <aspect/global.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/parameter_handler.h>

#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace aspect
{
  template <int dim> class Simulator;

  namespace Adjoint
  {
    /**
     * Simulator-facing facade for adjoint workflows.
     *
     * This class owns parameter state, initializes objective, workflow,
     * trajectory, optimization, and diagnostics components, dispatches the
     * selected steady or reverse-time workflow, and exposes completed scalar
     * objectives and gradients. Discrete operator implementations belong to
     * the corresponding component rather than to the facade.
     *
     * @ingroup Adjoint
     */
    template <int dim>
    class Manager : public SimulatorAccess<dim>
    {
      public:
        /**
         * Constructor.
         */
        Manager();

        /**
         * Declare the parameters used by the adjoint manager and its
         * subobjects.
         *
         * @param prm The parameter handler in which parameters are declared.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Initialize this manager and all objective plugins with access to the
         * simulator object. Unlike most SimulatorAccess users, the adjoint
         * manager needs mutable access to the simulator: the instantaneous
         * adjoint solve temporarily replaces the Stokes right hand side and
         * solution vectors, calls simulator solve routines, and then restores
         * the forward state.
         *
         * This function is therefore the only valid initialization path for
         * this class. It stores the mutable simulator pointer used by the
         * adjoint workflow, initializes the SimulatorAccess base class for
         * standard read-only simulator queries, and forwards simulator access
         * to the contained objective manager.
         *
         * @param simulator The simulator object that owns the forward model
         * and adjoint workflow.
         */
        void
        initialize_simulator(Simulator<dim> &simulator);

        /**
         * Override the generic SimulatorAccess initialization interface, but
         * reject it deliberately. The base class interface only provides a
         * const simulator reference, which would leave this manager without
         * the mutable simulator access required by solve_instantaneous_stokes().
         *
         * Defining this override prevents accidental initialization through a
         * SimulatorAccess pointer from silently setting only the base-class
         * simulator pointer. Call initialize_simulator(Simulator<dim> &) on
         * the concrete adjoint manager instead.
         */
        void
        initialize_simulator(const Simulator<dim> &simulator) override;

        /**
         * Read adjoint parameters from the parameter file and create the
         * selected objective plugins.
         *
         * @param prm The parameter handler from which adjoint parameters are
         * read.
         */
        void
        parse_parameters(ParameterHandler &prm);

        /**
         * Initialize the adjoint manager after parameters have been parsed.
         */
        void
        initialize();

        /**
         * Run the instantaneous Stokes adjoint workflow for the current model.
         */
        void
        solve_instantaneous_stokes();

        /**
         * Return whether the transient temperature adjoint workflow is active.
         */
        bool
        time_dependent_temperature_workflow_enabled() const;

        bool optimization_enabled() const;

        /** Run the opt-in native ROL reduced initial-temperature solve. */
        void run_initial_temperature_optimization();

        /**
         * Store one unique accepted physical state and its accepted outgoing
         * replay schedule entry in the trajectory archive.
         */
        void
        record_forward_snapshot(const double outgoing_time_step,
                                const bool has_outgoing_step);

        /** Mark the last accepted forward state as the terminal trajectory state. */
        void
        mark_forward_snapshot_terminal();

        /**
         * Run the time-dependent temperature reverse workflow and write the
         * initial-temperature gradient.
         */
        void
        finalize_time_dependent_temperature();

        /** Return whether a committed reverse checkpoint should be resumed. */
        bool
        has_pending_adjoint_restart() const;

        /**
         * Return the most recently computed physical-property kernels.
         *
         * @return The kernel repository.
         */
        const KernelRepository<dim> &
        get_kernels() const;

        /**
         * Return the most recently evaluated objective values.
         *
         * @return A map from objective functional names to objective values.
         */
        std::map<std::string, double>
        get_objective_values() const;

        /**
         * Return the latest initial-temperature gradient computed by the
         * time-dependent temperature adjoint workflow.
         */
        const LinearAlgebra::BlockVector &
        get_initial_temperature_gradient() const;

        /**
         * Return the latest initial-composition gradients computed by the
         * time-dependent temperature adjoint workflow.
         */
        const std::vector<LinearAlgebra::BlockVector> &
        get_initial_composition_gradients() const;

        /**
         * Capture the canonical pre-forward state for a later in-memory
         * optimization evaluation. This is intentionally separate from the
         * ordinary checkpoint/restart format.
         */
        void capture_optimization_initial_state();

        /**
         * Restore the canonical pre-forward state and invalidate all
         * evaluation-owned adjoint/runtime caches. This is the O3 reset
         * primitive; it does not run a forward solve itself.
         */
        void restore_optimization_initial_state();

        bool has_optimization_initial_state() const;

        /** Install T_fixed + P m after an O3 baseline restore. */
        void install_initial_temperature_control(const LinearAlgebra::Vector &control);

        /** Replace the physical baseline in T_ic = T_fixed + Pm. */
        void
        set_optimization_initial_temperature_baseline(const LinearAlgebra::Vector &baseline);

        /** Return the current physical initial-temperature baseline. */
        const LinearAlgebra::Vector &
        get_optimization_initial_temperature_baseline() const;

        /**
         * Express a representable physical temperature as the reduced
         * increment from the installed baseline.
         */
        LinearAlgebra::Vector
        reduced_control_from_physical_temperature(const LinearAlgebra::Vector &temperature);

        /** Return per-control bounds enforcing 0 <= T_fixed + Pm <= 1. */
        std::pair<LinearAlgebra::Vector, LinearAlgebra::Vector>
        initial_temperature_control_physical_bounds();

        /** Return nodal extrema of the physical field T_fixed + Pm. */
        std::pair<double, double>
        initial_temperature_control_physical_min_max(
          const LinearAlgebra::Vector &control);

        /** Return the finite-element L2 reconstruction error of T_fixed + Pm. */
        double
        initial_temperature_control_reconstruction_error_l2(
          const LinearAlgebra::Vector &control,
          const LinearAlgebra::Vector &temperature);

        /**
         * Return the immutable reduced initial-temperature control space and
         * its cached L2 metric after restoring the O3 canonical baseline.
         */
        const InitialTemperatureControl<dim> &get_initial_temperature_control_space();

        /** Capture the current terminal state as the temperature observation. */
        void set_terminal_temperature_observation_from_current_state();

        /** Capture the current terminal state for all compatible scalar objectives. */
        void capture_terminal_observation_from_current_state(const AdvectionField &field);

        /**
         * Evaluate an initial-temperature control from the canonical in-memory
         * baseline. When @p l2_gradient is non-null, run the matching reverse
         * sweep and return its reduced L2 Riesz representative in that vector.
         */
        std::map<std::string, double>
        evaluate_initial_temperature_control(const LinearAlgebra::Vector &control,
                                             LinearAlgebra::Vector *l2_gradient = nullptr);

        /**
         * Complete the reverse half of the immediately preceding
         * value-only initial-temperature control evaluation. This is an
         * O4/O5 optimization entry point: it never restores the baseline or
         * reruns the forward model, and is therefore valid only while that
         * forward trajectory is still the active manager state.
         */
        std::map<std::string, double>
        complete_initial_temperature_control_gradient(LinearAlgebra::Vector &l2_gradient);

      private:
        friend class TimeDependentWorkflow<dim>;
        friend class TrajectoryManager<dim>;
        friend class Diagnostics::InitialTemperatureOptimizationChecks<dim>;

        using ReconstructedTrajectoryState =
          typename TrajectoryManager<dim>::ReconstructedTrajectoryState;
        using ForwardReplayWindow =
          typename TrajectoryManager<dim>::ForwardReplayWindow;

        /** Focused O3 regression: evaluate m0, m1, then m0 in one process. */
        void run_initial_temperature_control_reuse_diagnostic();

        /** Focused O3 regression: evaluate only the canonical zero control. */
        void run_initial_temperature_control_single_evaluation_diagnostic();

        /** Focused O3 regression: check a reduced-space PDE Taylor expansion. */
        void run_initial_temperature_control_taylor_diagnostic();

        /** Execute one post-run O3 initialization stage and then return. */
        void run_initial_temperature_control_initialization_diagnostic(
          const unsigned int stage);

        double temperature_l2_difference(const LinearAlgebra::Vector &first,
                                         const LinearAlgebra::Vector &second) const;

        void ensure_initial_temperature_control_space();

        /**
         * Check that the transient temperature workflow is used only in the
         * narrow v1 support envelope.
         */
        void
        validate_time_dependent_temperature_setup() const;

        /**
         * Solve a Stokes adjoint system from an already assembled full-system
         * right hand side.
         */
        void
        solve_stokes_adjoint_from_rhs(
          const LinearAlgebra::BlockVector &right_hand_side,
          LinearAlgebra::BlockVector &adjoint_solution);

        /**
         * Write the latest initial-temperature gradient to one text file per
         * MPI rank.
         */
        void
        write_initial_temperature_gradient() const;

        /**
         * Write the latest initial-composition gradients to one text file per
         * compositional field and MPI rank.
         */
        void
        write_initial_composition_gradients() const;

        /**
         * Apply the transpose of the advection-field dependence in the Stokes
         * right hand side for one restored forward state.
         */
        void
        add_stokes_field_transpose(const AdvectionField &advection_field,
                                   const ForwardStateView<dim> &snapshot,
                                   const LinearAlgebra::BlockVector &stokes_adjoint_rhs,
                                   LinearAlgebra::BlockVector &field_adjoint);

        /**
         * Apply the transpose of the advection-field dependence in the Stokes
         * right hand side for a supplied Stokes test vector. This is the core
         * local action used by add_stokes_field_transpose() after the Stokes
         * adjoint solve and by the finite-difference diagnostic.
         */
        void
        add_stokes_rhs_field_transpose(const AdvectionField &advection_field,
                                       const ForwardStateView<dim> &snapshot,
                                       const LinearAlgebra::BlockVector &stokes_test_vector,
                                       LinearAlgebra::BlockVector &field_adjoint);

        /**
         * Run a finite-difference dot-product diagnostic for the composition
         * dependence of the assembled Stokes right hand side.
         */
        void
        run_composition_stokes_rhs_transpose_test(const ForwardStateView<dim> &snapshot);

        /**
         * Run a direct finite-difference diagnostic for the viscosity
         * coefficient in the Stokes residual, bypassing MaterialModelAdjoint.
         */
        void
        run_direct_stokes_viscosity_derivative_test(const ForwardStateView<dim> &snapshot);

        /**
         * Run a standalone dot-product diagnostic for the material-model VJP
         * backend on the simple material model.
         */
        void
        run_material_model_adjoint_vjp_test(const ForwardStateView<dim> &snapshot) const;

        void
        inject_lifecycle_failure(const Diagnostics::LifecycleFailurePoint point,
                                 const unsigned int next_reverse_step) const;

        /**
         * Reconstruct the full, temporary replay view required by existing
         * adjoint assembly routines. Unlike an archived record it may contain
         * indexed BDF history and reconstructed Stokes fields, but it is held
         * only for the current reverse operation.
         */
        ForwardStateView<dim>
        reconstruct_forward_snapshot(const unsigned int record);

        /** Return one state from the three-entry reverse working cache. */
        const LinearAlgebra::BlockVector &
        load_reconstructed_state(const unsigned int record);

        /** Recover velocity and pressure from archived T,C for minimal mode. */
        void
        recover_stokes_state(const unsigned int record,
                             LinearAlgebra::BlockVector &state);

        /** Compare a T,C-recovered Stokes state with the cached forward one. */
        void
        run_trajectory_stokes_recovery_diagnostic();

        /** Validate the Phase-A deterministic primal replay primitive. */
        void
        run_trajectory_replay_diagnostic();

        /** Reject configurations that have not been audited for Phase-A replay. */
        void
        validate_trajectory_replay_capability() const;

        /** Hydrate and replay one fixed-spacing interval into memory. */
        void
        materialize_forward_replay_window(const unsigned int record);

        /** Compare sparse replay states with the optional dense reference. */
        void
        compare_sparse_replay_state(const unsigned int record,
                                    const LinearAlgebra::BlockVector &state);

        /** Apply the restart-safe cleanup policy after a committed checkpoint. */
        void
        cleanup_trajectory_after_checkpoint(const AdjointRestartState &state);

        /** Print storage, working-set, and process-memory diagnostics. */
        void
        report_trajectory_statistics(const std::string &stage) const;

        /**
         * Emit a rank-local, allocation-free memory checkpoint while
         * investigating the experimental MPI reverse path.
         */
        void
        report_mpi_adjoint_memory_checkpoint(
          const std::string &stage,
          const LinearAlgebra::BlockVector *state = nullptr) const;

        /**
         * Mutable simulator pointer used only by the adjoint workflow for
         * operations that cannot be expressed through SimulatorAccess, such as
         * temporarily replacing the system right hand side or current solution
         * before an adjoint Stokes solve. The SimulatorAccess base class is
         * still initialized separately and remains the preferred route for
         * ordinary read-only simulator queries.
         */
        Simulator<dim> *simulator;
        Objective::Manager<dim> objective_manager;
        Diagnostics::Manager<dim> diagnostics_manager;
        InitialTemperatureOptimization<dim> initial_temperature_optimization;
        TimeDependentWorkflow<dim> time_dependent_workflow;
        std::vector<std::unique_ptr<ObjectiveResult>> objective_results;
        std::vector<std::unique_ptr<AdjointState>> adjoint_states;
        TrajectoryManager<dim> trajectory_manager;
        bool suppress_evaluation_side_effects = false;
        unsigned int stop_after_adjoint_checkpoint;
        bool run_trajectory_stokes_recovery_check;
        bool run_trajectory_replay_check;
        bool run_sparse_trajectory_replay_check;
        bool run_matrix_free_cold_rebuild_diagnostic;
        double trajectory_stokes_recovery_tolerance;
        double trajectory_replay_tolerance;
        Diagnostics::LifecycleFailurePoint lifecycle_failure_injection;
        Diagnostics::CheckpointFailurePoint checkpoint_transaction_failure_injection;
        int checkpoint_transaction_failure_rank;
        unsigned int checkpoint_transaction_failure_next_reverse_step;
        Diagnostics::CleanupFailurePoint trajectory_cleanup_failure_injection;
        int trajectory_cleanup_failure_rank;
        unsigned int trajectory_cleanup_failure_next_reverse_step;
        Diagnostics::AppendFailurePoint trajectory_append_failure_injection;
        int trajectory_append_failure_rank;
        unsigned int trajectory_append_failure_step;
        KernelRepository<dim> kernels;
        std::string material_model_name;
        std::string advection_preconditioner;
        bool run_advection_preconditioner_comparison;
        bool run_material_model_adjoint_vjp_diagnostic;
        double material_model_adjoint_vjp_diagnostic_epsilon;
        bool run_composition_stokes_rhs_transpose_diagnostic;
        double composition_stokes_rhs_transpose_diagnostic_epsilon;
        bool run_direct_stokes_viscosity_derivative_diagnostic;
        double direct_stokes_viscosity_derivative_diagnostic_epsilon;
        bool run_transposed_advection_constraint_diagnostic;
        bool run_artificial_viscosity_coefficient_seed_diagnostic;
        double artificial_viscosity_coefficient_seed_diagnostic_epsilon;
        bool run_artificial_viscosity_estimator_transpose_diagnostic;
        double artificial_viscosity_estimator_transpose_diagnostic_epsilon;
        bool run_local_entropy_residual_transpose_diagnostic;
        double local_entropy_residual_transpose_diagnostic_epsilon;
        bool run_global_entropy_denominator_scalar_diagnostic;
        double global_entropy_denominator_scalar_diagnostic_epsilon;
        bool run_global_entropy_variation_transpose_diagnostic;
        double global_entropy_variation_transpose_diagnostic_epsilon;
    };
  }
}

#endif
