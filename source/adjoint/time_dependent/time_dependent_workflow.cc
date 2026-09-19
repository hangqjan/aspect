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

#include <aspect/adjoint/core/manager.h>
#include <aspect/adjoint/core/instantaneous_stokes_workflow.h>
#include <aspect/adjoint/core/simulator_state.h>
#include <aspect/adjoint/operators/stokes_adjoint_solver.h>
#include <aspect/adjoint/objective/temperature_field_objective.h>

#include <aspect/geometry_model/box.h>
#include <aspect/gravity_model/vertical.h>
#include <aspect/material_model/simple.h>
#include <aspect/adjoint/operators/advection_adjoint_calculator.h>
#include <aspect/adjoint/operators/boundary_conditions.h>
#include <aspect/adjoint/operators/material_model_adjoint.h>
#include <aspect/postprocess/dynamic_topography.h>
#include <aspect/simulator.h>
#include <aspect/simulator/solver/stokes_matrix_free.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <string>

#if defined(__APPLE__)
#  include <mach/mach.h>
#endif

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    void
    TimeDependentWorkflow<dim>::run(Manager<dim> &manager)
    {
      // The accepted forward trajectory defines residuals
      //   R_n(q_n,q_{n-1},q_{n-2},u_n)=0,  n=1,...,N,
      // plus a quasi-static Stokes residual S_n(u_n,T_n,C_n)=0. Starting from
      // terminal/running objective seeds, this workflow applies the discrete
      // Jacobian transpose in descending n. No continuous adjoint equation is
      // re-discretized here: every action reverses an implemented forward
      // algebraic dependency at its recorded time level.
      if (!manager.time_dependent_temperature_workflow_enabled())
        return;

      manager.validate_time_dependent_temperature_setup();

      const bool resuming_reverse = !manager.trajectory_manager.has_archive()
                                    && manager.has_pending_adjoint_restart();
      const bool restart_enabled = !manager.suppress_evaluation_side_effects;
      const bool lifecycle_cleanup_enabled = true;
      if (resuming_reverse)
        // Opening an archive reconstructs durable schedule/payload metadata;
        // replay windows remain transient and are rebuilt only when demanded
        // by a reverse step.
        manager.trajectory_manager.create_archive(manager.trajectory_manager.configuration().storage_mode,
                                                  manager.get_output_directory(),
                                                  true,
                                                  manager.trajectory_manager.configuration().records_per_segment);

      if (resuming_reverse && manager.trajectory_manager.configuration().anchor_spacing > 1)
        {
          manager.trajectory_manager.sparse_anchor_steps.clear();
          for (unsigned int record = 0; record < manager.trajectory_manager.archive().size(); ++record)
            if (manager.trajectory_manager.archive().has_record(record)
                && (record % manager.trajectory_manager.configuration().anchor_spacing == 0
                    || record + 1 == manager.trajectory_manager.archive().size()))
              manager.trajectory_manager.sparse_anchor_steps.push_back(record);
        }

      if (resuming_reverse)
        {
          // The normal forward loop reaches start_timestep() after a generic
          // restart. The adjoint-resume path deliberately skips that loop, so
          // reproduce the runtime-only initialization that checkpoint I/O does
          // not serialize: current constraints, system storage, and the GMG
          // hierarchy/level constraints.
          manager.report_mpi_adjoint_memory_checkpoint(
            "before initialize_current_linearization_point");
          manager.simulator->initialize_current_linearization_point();
          manager.report_mpi_adjoint_memory_checkpoint(
            "after initialize_current_linearization_point");
          manager.report_mpi_adjoint_memory_checkpoint("before compute_current_constraints");
          manager.simulator->compute_current_constraints();
          manager.report_mpi_adjoint_memory_checkpoint("after compute_current_constraints");
          if (manager.simulator->rebuild_sparsity_and_matrices)
            {
              manager.simulator->setup_system_matrix(
                manager.introspection().index_sets.system_partitioning);
              manager.simulator->setup_system_preconditioner(
                manager.introspection().index_sets.system_partitioning);
              manager.simulator->rebuild_sparsity_and_matrices = false;
            }
          if (manager.simulator->stokes_matrix_free)
            manager.simulator->stokes_matrix_free->setup_dofs();
          manager.simulator->rebuild_stokes_matrix = true;
          manager.simulator->rebuild_stokes_preconditioner = true;
          manager.get_pcout()
              << "   Adjoint cold-restart Stokes runtime reconstruction complete: "
              << "timestep = " << manager.simulator->timestep_number
              << ", time = " << manager.simulator->time
              << ", next reverse step pending." << std::endl;
        }

      AssertThrow(manager.trajectory_manager.has_archive() && manager.trajectory_manager.archive().size() > 0,
                  ExcMessage("The time-dependent temperature adjoint did not record any "
                             "accepted forward time steps. Check the end time and time-stepping setup."));

      manager.report_trajectory_statistics("forward archive complete");
      if (!resuming_reverse && manager.run_trajectory_stokes_recovery_check)
        manager.run_trajectory_stokes_recovery_diagnostic();
      if (!resuming_reverse && manager.run_trajectory_replay_check)
        manager.run_trajectory_replay_diagnostic();
      manager.trajectory_manager.reset_replay_state();
      // Reverse operations temporarily install historical states, constraints,
      // time stamps, and rebuild flags in Simulator. The guard makes the whole
      // sweep transactional with respect to the caller's forward state,
      // including exceptions from solvers, diagnostics, or checkpoint I/O.
      ScopedSimulatorState<dim> simulator_state(*manager.simulator);

      // This is the authoritative reverse state. Level zero is the seed at
      // next_reverse_step; levels one and two are its two pending predecessors.
      // The old N-sized history arrays are deliberately not retained.
      AdjointRestartState reverse_state;
      reverse_state.trajectory_storage_mode = manager.trajectory_manager.configuration().storage_mode;
      reverse_state.composition_history.resize(manager.introspection().n_compositional_fields);
      const auto initialize_vector = [&manager](LinearAlgebra::BlockVector &vector)
      {
        // Reverse histories are accumulation targets for constraint
        // distribution and transpose assembly. Keep them owned-only and
        // writable; temporary ghosted views belong solely to replay states
        // used for FE evaluation.
        vector.reinit(manager.introspection().index_sets.system_partitioning,
                      manager.get_mpi_communicator());
        vector = 0.0;
      };
      for (auto &vector : reverse_state.temperature_history)
        initialize_vector(vector);
      for (auto &vector : reverse_state.stokes_right_hand_side_history)
        initialize_vector(vector);
      for (auto &field_history : reverse_state.composition_history)
        for (auto &vector : field_history)
          initialize_vector(vector);

      auto &temperature_adjoints = reverse_state.temperature_history;
      auto &composition_adjoints = reverse_state.composition_history;
      auto &stokes_adjoint_right_hand_sides =
        reverse_state.stokes_right_hand_side_history;
      if (restart_enabled)
        manager.trajectory_manager.create_restart_checkpoint_manager();

      AdvectionAdjointCalculator<dim> advection_adjoint_calculator(*manager.simulator);
      Diagnostics::AdvectionOperatorChecks<dim> advection_operator_checks(
        advection_adjoint_calculator);
      advection_adjoint_calculator.set_run_preconditioner_comparison(
        manager.run_advection_preconditioner_comparison);
      if (manager.advection_preconditioner == "none")
        advection_adjoint_calculator.set_preconditioner(
          AdvectionAdjointCalculator<dim>::Preconditioner::none);
      else if (manager.advection_preconditioner == "forward ilu experimental")
        advection_adjoint_calculator.set_preconditioner(
          AdvectionAdjointCalculator<dim>::Preconditioner::forward_ilu_experimental);
      else
        {
          AssertThrow(manager.advection_preconditioner == "transpose ilu",
                      ExcMessage("Unknown adjoint advection preconditioner."));
          advection_adjoint_calculator.set_preconditioner(
            AdvectionAdjointCalculator<dim>::Preconditioner::transpose_ilu);
        }

      if (!resuming_reverse)
        {
          manager.report_mpi_adjoint_memory_checkpoint(
            "fresh reverse: initialize_current_linearization_point not invoked");
          // Evaluate terminal objective derivatives at the final accepted forward
          // state. A checkpoint at N is the state immediately before reverse step N.
          const ForwardStateView<dim> final_snapshot =
            manager.reconstruct_forward_snapshot(manager.trajectory_manager.archive().size() - 1);
          manager.simulator->solution = final_snapshot.solution.get();
          manager.simulator->old_solution = final_snapshot.old_solution.get();
          manager.simulator->old_old_solution = final_snapshot.old_old_solution.get();
          manager.simulator->current_linearization_point = final_snapshot.stokes_linearization_point.get();
          manager.simulator->time = final_snapshot.time;
          manager.simulator->time_step = final_snapshot.time_step;
          manager.simulator->old_time_step = final_snapshot.old_time_step;
          manager.simulator->timestep_number = final_snapshot.timestep_number;
          manager.simulator->last_pressure_normalization_adjustment =
            final_snapshot.pressure_normalization_adjustment;

          manager.report_mpi_adjoint_memory_checkpoint("before terminal objective assembly",
                                                       &final_snapshot.solution.get());
          reverse_state.objective_values = manager.objective_manager.evaluate_terminal_objectives();
          manager.objective_manager.assemble_terminal_advection_field_right_hand_side(
            AdvectionField::temperature(), temperature_adjoints[0]);
          // For J=Phi(q_N)+sum_n L_n(q_n), initialize lambda_N with
          // dPhi/dq_N. Running Stokes objectives were accumulated during the
          // forward pass because their scalar values require the accepted
          // physical state at each n; their derivatives are injected below at
          // the matching reverse step.
          for (const auto &value : this->running_stokes_objective_values)
            reverse_state.objective_values[value.first] += value.second;
          manager.report_mpi_adjoint_memory_checkpoint("after terminal temperature objective assembly",
                                                       &temperature_adjoints[0]);
          for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
            manager.objective_manager.assemble_terminal_advection_field_right_hand_side(
              AdvectionField::composition(c), composition_adjoints[c][0]);

          if (manager.run_material_model_adjoint_vjp_diagnostic)
            manager.run_material_model_adjoint_vjp_test(final_snapshot);
          if (manager.run_direct_stokes_viscosity_derivative_diagnostic)
            manager.run_direct_stokes_viscosity_derivative_test(final_snapshot);

          if (manager.run_composition_stokes_rhs_transpose_diagnostic)
            {
              manager.run_composition_stokes_rhs_transpose_test(final_snapshot);
              if (final_snapshot.timestep_number <= 1)
                advection_operator_checks.velocity_transpose(
                  AdvectionField::temperature(),
                  final_snapshot,
                  temperature_adjoints[0],
                  manager.composition_stokes_rhs_transpose_diagnostic_epsilon);
              else
                manager.get_pcout() << "   Skipping temperature-advection velocity transpose diagnostic "
                                    << "on timestep "
                                    << final_snapshot.timestep_number
                                    << "; the current diagnostic is a one-step "
                                    << "u-star check and does not compare the BDF2 "
                                    << "history split." << std::endl;
            }
          if (manager.run_transposed_advection_constraint_diagnostic)
            {
              advection_operator_checks.transposed_constraints(
                AdvectionField::temperature(),
                final_snapshot,
                temperature_adjoints[0]);

              for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
                advection_operator_checks.transposed_constraints(
                  AdvectionField::composition(c),
                  final_snapshot,
                  composition_adjoints[c][0]);
            }
          if (manager.run_artificial_viscosity_coefficient_seed_diagnostic)
            {
              AssertThrow(manager.artificial_viscosity_coefficient_seed_diagnostic_epsilon > 0.0,
                          ExcMessage("Adjoint/artificial viscosity coefficient seed test epsilon must be positive."));

              advection_operator_checks.artificial_viscosity_coefficient_seed(
                AdvectionField::temperature(),
                final_snapshot,
                temperature_adjoints[0],
                manager.artificial_viscosity_coefficient_seed_diagnostic_epsilon);

              for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
                advection_operator_checks.artificial_viscosity_coefficient_seed(
                  AdvectionField::composition(c),
                  final_snapshot,
                  composition_adjoints[c][0],
                  manager.artificial_viscosity_coefficient_seed_diagnostic_epsilon);
            }
          if (manager.run_artificial_viscosity_estimator_transpose_diagnostic)
            {
              AssertThrow(manager.artificial_viscosity_estimator_transpose_diagnostic_epsilon > 0.0,
                          ExcMessage("Adjoint/artificial viscosity estimator transpose test epsilon must be positive."));

              advection_operator_checks.artificial_viscosity_estimator_transpose(
                AdvectionField::temperature(),
                final_snapshot,
                manager.artificial_viscosity_estimator_transpose_diagnostic_epsilon);

              for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
                advection_operator_checks.artificial_viscosity_estimator_transpose(
                  AdvectionField::composition(c),
                  final_snapshot,
                  manager.artificial_viscosity_estimator_transpose_diagnostic_epsilon);
            }
          if (manager.run_local_entropy_residual_transpose_diagnostic)
            {
              AssertThrow(manager.local_entropy_residual_transpose_diagnostic_epsilon > 0.0,
                          ExcMessage("Adjoint/local entropy residual transpose test epsilon must be positive."));

              advection_operator_checks.local_entropy_residual_transpose(
                AdvectionField::temperature(),
                final_snapshot,
                manager.local_entropy_residual_transpose_diagnostic_epsilon);

              for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
                advection_operator_checks.local_entropy_residual_transpose(
                  AdvectionField::composition(c),
                  final_snapshot,
                  manager.local_entropy_residual_transpose_diagnostic_epsilon);
            }
          if (manager.run_global_entropy_denominator_scalar_diagnostic)
            {
              AssertThrow(manager.global_entropy_denominator_scalar_diagnostic_epsilon > 0.0,
                          ExcMessage("Adjoint/global entropy denominator scalar test epsilon must be positive."));

              advection_operator_checks.global_entropy_denominator_scalar(
                AdvectionField::temperature(),
                final_snapshot,
                manager.global_entropy_denominator_scalar_diagnostic_epsilon);

              for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
                advection_operator_checks.global_entropy_denominator_scalar(
                  AdvectionField::composition(c),
                  final_snapshot,
                  manager.global_entropy_denominator_scalar_diagnostic_epsilon);
            }
          if (manager.run_global_entropy_variation_transpose_diagnostic)
            {
              AssertThrow(manager.global_entropy_variation_transpose_diagnostic_epsilon > 0.0,
                          ExcMessage("Adjoint/global entropy variation transpose test epsilon must be positive."));

              advection_operator_checks.global_entropy_variation_transpose(
                AdvectionField::temperature(),
                final_snapshot,
                manager.global_entropy_variation_transpose_diagnostic_epsilon);

              for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
                advection_operator_checks.global_entropy_variation_transpose(
                  AdvectionField::composition(c),
                  final_snapshot,
                  manager.global_entropy_variation_transpose_diagnostic_epsilon);
            }

          reverse_state.next_reverse_step = manager.trajectory_manager.archive().size();
          // A checkpoint stores the semantic three-level adjoint window before
          // executing step N. Publishing this initial generation makes a
          // restart independent of whether the process fails before its first
          // transpose solve.
          manager.trajectory_manager.set_restart_trajectory_identity(reverse_state);
          if (restart_enabled)
            {
              TimerOutput::Scope checkpoint_timer(manager.get_computing_timer(),
                                                  "Adjoint restart checkpoint write");
              const CheckpointWriteResult checkpoint_write =
                manager.trajectory_manager.restart_checkpoint().write(reverse_state);
              if (!checkpoint_write.committed)
                {
                  manager.get_pcout()
                      << "   Adjoint checkpoint transaction failed collectively before "
                      << "its first reverse step; "
                      << (checkpoint_write.latest_published
                          ? "the new generation was already published."
                          : "no new generation was published.")
                      << std::endl;
                  if (Utilities::MPI::n_mpi_processes(manager.get_mpi_communicator()) > 1)
                    throw QuietException();
                  AssertThrow(false,
                              ExcMessage("Adjoint checkpoint transaction failed before "
                                         "the first reverse step."));
                }
              manager.get_pcout() << "   Adjoint restart checkpoint committed before reverse step "
                                  << reverse_state.next_reverse_step << " ("
                                  << checkpoint_write.local_bytes << " rank-local bytes)." << std::endl;
            }
        }
      else
        {
          // Restore lambda_n, lambda_{n-1}, lambda_{n-2} and the corresponding
          // Stokes RHS histories exactly as they were committed. Forward replay
          // caches are intentionally absent from the checkpoint and are
          // reconstructed from the durable anchor named by its identity data.
          TimerOutput::Scope restore_timer(manager.get_computing_timer(),
                                           "Adjoint restart checkpoint restore");
          const std::size_t restore_bytes =
            manager.trajectory_manager.restart_checkpoint().restore(reverse_state);
          AssertThrow(reverse_state.trajectory_storage_mode == manager.trajectory_manager.configuration().storage_mode,
                      ExcMessage("The committed adjoint restart checkpoint has a "
                                 "different trajectory storage mode."));
          AssertThrow(reverse_state.next_reverse_step <= manager.trajectory_manager.archive().size(),
                      ExcMessage("Adjoint restart next_reverse_step is outside the "
                                 "trajectory archive."));
          manager.trajectory_manager.validate_sparse_restart_state(reverse_state);
          manager.get_pcout() << "   Restored adjoint restart checkpoint before reverse step "
                              << reverse_state.next_reverse_step << " ("
                              << restore_bytes << " rank-local bytes)." << std::endl;
        }

      // A committed safe-incremental frontier is authoritative even if the
      // previous process died before or during removal. Finish the idempotent
      // cleanup before rebuilding the disposable sparse replay window.
      if (resuming_reverse
          && manager.trajectory_manager.configuration().cleanup_policy == TrajectoryCleanupPolicy::safe_incremental
          && reverse_state.cleanup_frontier != numbers::invalid_unsigned_int)
        {
          AssertThrow(reverse_state.cleanup_frontier <= manager.trajectory_manager.archive().size(),
                      ExcMessage("Adjoint restart cleanup frontier is outside the "
                                 "trajectory archive."));
          manager.cleanup_trajectory_after_checkpoint(reverse_state);
        }

      if (resuming_reverse && manager.trajectory_manager.configuration().anchor_spacing > 1
          && reverse_state.next_reverse_step > 0)
        {
          const unsigned int anchor =
            manager.trajectory_manager.required_anchor_for_reverse_step(reverse_state.next_reverse_step);
          manager.get_pcout() << "ASPECT_ADJOINT_SPARSE_RESTART next_reverse_step="
                              << reverse_state.next_reverse_step
                              << " required_lower_anchor=" << anchor
                              << " action=replay_disposable_interval" << std::endl;
          const unsigned int replayed_before = manager.trajectory_manager.sparse_replay_transition_count;
          manager.materialize_forward_replay_window(reverse_state.next_reverse_step - 1);
          manager.trajectory_manager.sparse_restart_replay_transition_count +=
            manager.trajectory_manager.sparse_replay_transition_count - replayed_before;
        }

      manager.objective_results.clear();
      for (const auto &objective_value : reverse_state.objective_values)
        {
          auto result = std::make_unique<ObjectiveResult>();
          result->objective_name = objective_value.first;
          result->value = objective_value.second;
          result->right_hand_side.reinit(
            manager.introspection().index_sets.system_partitioning,
            manager.get_mpi_communicator());
          result->right_hand_side = 0.0;
          manager.objective_results.push_back(std::move(result));

          manager.get_pcout() << "   Time-dependent terminal objective '"
                              << objective_value.first
                              << "' = "
                              << std::setprecision(16)
                              << objective_value.second
                              << std::setprecision(6)
                              << std::endl;
        }

      TimerOutput::Scope reverse_timer(manager.get_computing_timer(),
                                       "Adjoint trajectory reverse sweep");
      while (reverse_state.next_reverse_step > 0)
        {
          // At loop entry, history slot 0 contains all seeds of q_n. Applying
          // (dR_n/dq_{n-k})^T accumulates into slots k for k=1,2. The Stokes
          // RHS history follows the same convention because velocity at an
          // older level is eliminated only after all later residuals that use
          // it have contributed.
          const unsigned int reverse_index = reverse_state.next_reverse_step;
          const unsigned int snapshot_index = reverse_index - 1;
          const ForwardStateView<dim> snapshot =
            manager.reconstruct_forward_snapshot(snapshot_index);

          if (!this->running_stokes_objective_values.empty())
            // Add dL_n/du_n to the current Stokes adjoint RHS before eliminating
            // u_n. Objective dispatch is keyed by the recorded timestep, not
            // by loop position, which is important after restart and replay.
            manager.objective_manager.assemble_running_stokes_adjoint_right_hand_side(
              snapshot.timestep_number, stokes_adjoint_right_hand_sides[0]);

          // Deliberately reproduce the runtime portion of the adjoint-resume
          // path from a clean matrix-free state. The following first Stokes
          // transpose call performs the same primal-topology assembly and
          // build_preconditioner() call as an actual cold resume. Keep this
          // narrow experiment separate from the normal resume path: it tells
          // us whether GMG can rebuild at all (A), or whether resume lacks
          // another dependency (B).
          const bool run_cold_rebuild_diagnostic_here =
            manager.run_matrix_free_cold_rebuild_diagnostic
            && reverse_index == manager.trajectory_manager.archive().size();
          if (run_cold_rebuild_diagnostic_here)
            {
              AssertThrow(manager.simulator->stokes_matrix_free != nullptr,
                          ExcMessage("The matrix-free cold rebuild diagnostic "
                                     "requires the block GMG Stokes solver."));
              manager.get_pcout()
                  << "   Adjoint matrix-free cold-rebuild diagnostic: resetting "
                  << "GMG runtime state before reverse step " << reverse_index
                  << "." << std::endl;
              manager.simulator->stokes_matrix_free->setup_dofs();
              manager.simulator->rebuild_stokes_matrix = true;
              manager.simulator->rebuild_stokes_preconditioner = true;
            }

          // reverse_index names the adjoint seed at the current state q_n.
          // snapshot_index names the forward residual that advanced the model
          // from older states to q_n. Applying the transpose of that residual
          // moves sensitivities into q_{n-1} and possibly q_{n-2}.
          if (snapshot.timestep_number == 0)
            {
              // The first recorded state has no previous accepted advection
              // residual to invert. Only the Stokes dependence can still feed
              // back to field variables, and the remaining terminal/current
              // seed is simply part of the initial-condition gradient.
              manager.add_stokes_field_transpose(AdvectionField::temperature(),
                                                 snapshot,
                                                 stokes_adjoint_right_hand_sides[0],
                                                 temperature_adjoints[1]);
              temperature_adjoints[1] += temperature_adjoints[0];

              for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
                {
                  manager.add_stokes_field_transpose(AdvectionField::composition(c),
                                                     snapshot,
                                                     stokes_adjoint_right_hand_sides[0],
                                                     composition_adjoints[c][1]);
                  composition_adjoints[c][1] += composition_adjoints[c][0];
                }
            }
          else
            {
              // A Stokes RHS already accumulated at q_n affects temperature and
              // composition through the Stokes operator assembled at q_n. Apply
              // this before propagating the advection residual so that all seeds
              // at q_n are present in temperature_adjoints[reverse_index] and
              // composition_adjoints[*][reverse_index].
              if (reverse_index == manager.trajectory_manager.archive().size())
                manager.report_mpi_adjoint_memory_checkpoint(
                  "before first reverse Stokes setup", &snapshot.solution.get());
              manager.add_stokes_field_transpose(AdvectionField::temperature(),
                                                 snapshot,
                                                 stokes_adjoint_right_hand_sides[0],
                                                 temperature_adjoints[0]);
              if (reverse_index == manager.trajectory_manager.archive().size())
                manager.report_mpi_adjoint_memory_checkpoint(
                  "after first reverse Stokes setup", &snapshot.solution.get());
              if (run_cold_rebuild_diagnostic_here)
                manager.get_pcout()
                    << "   Adjoint matrix-free cold-rebuild diagnostic: first "
                    << "reverse Stokes preconditioner rebuild completed." << std::endl;
              for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
                manager.add_stokes_field_transpose(AdvectionField::composition(c),
                                                   snapshot,
                                                   stokes_adjoint_right_hand_sides[0],
                                                   composition_adjoints[c][0]);

              std::vector<LinearAlgebra::BlockVector> previous_composition_adjoint_slice;
              std::vector<LinearAlgebra::BlockVector> previous_previous_composition_adjoint_slice;
              previous_composition_adjoint_slice.reserve(manager.introspection().n_compositional_fields);
              previous_previous_composition_adjoint_slice.reserve(manager.introspection().n_compositional_fields);
              // The temperature material-prefactor transpose can modify all
              // composition adjoints. Build per-time-level slices so the helper can
              // treat them as a compact vector over composition fields, then copy
              // them back into the time-major storage below.
              for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
                {
                  previous_composition_adjoint_slice.push_back(composition_adjoints[c][1]);
                  if (snapshot_index > 0)
                    previous_previous_composition_adjoint_slice.push_back(composition_adjoints[c][2]);
                }

              advection_adjoint_calculator.propagate_temperature_adjoint(
                snapshot,
                temperature_adjoints[0],
                temperature_adjoints[1],
                snapshot_index > 0 ? &temperature_adjoints[2] : nullptr,
                stokes_adjoint_right_hand_sides[1],
                snapshot_index > 0 ? &stokes_adjoint_right_hand_sides[2] : nullptr,
                manager.introspection().n_compositional_fields > 0 ? &previous_composition_adjoint_slice : nullptr,
                (manager.introspection().n_compositional_fields > 0 && snapshot_index > 0)
                ? &previous_previous_composition_adjoint_slice
                : nullptr);

              // Algebraically this call applies the temperature row of
              // R_n^T. For BDF2 it distributes mass-term sensitivity with the
              // recorded variable-step coefficients to both q_{n-1} and
              // q_{n-2}; it also reverses advection velocity, diffusion,
              // material prefactors, and the active artificial-viscosity map.

              // Scatter the modified slices back into the persistent adjoint
              // history after propagate_temperature_adjoint() has added BDF,
              // velocity, artificial-viscosity, and material-prefactor transpose
              // contributions.
              for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
                {
                  composition_adjoints[c][1] = previous_composition_adjoint_slice[c];
                  if (snapshot_index > 0)
                    composition_adjoints[c][2] = previous_previous_composition_adjoint_slice[c];
                }

              for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
                advection_adjoint_calculator.propagate_advection_field_adjoint(
                  AdvectionField::composition(c),
                  snapshot,
                  composition_adjoints[c][0],
                  composition_adjoints[c][1],
                  snapshot_index > 0 ? &composition_adjoints[c][2] : nullptr,
                  stokes_adjoint_right_hand_sides[1],
                  snapshot_index > 0 ? &stokes_adjoint_right_hand_sides[2] : nullptr);
            }

          // Commit the state immediately before the next step. Moving the
          // three-level window happens before the checkpoint so its semantic
          // next_reverse_step is exact and unambiguous.
          --reverse_state.next_reverse_step;
          if (reverse_state.next_reverse_step > 0)
            {
              // Rotate the fixed semantic window after all contributions from
              // R_n have landed:
              //   [bar q_n,bar q_{n-1},bar q_{n-2}]
              //       <- [bar q_{n-1},bar q_{n-2},0].
              // Assignment rather than deque growth keeps restart storage and
              // working memory independent of the number of time steps.
              temperature_adjoints[0] = temperature_adjoints[1];
              temperature_adjoints[1] = temperature_adjoints[2];
              temperature_adjoints[2] = 0.0;
              stokes_adjoint_right_hand_sides[0] = stokes_adjoint_right_hand_sides[1];
              stokes_adjoint_right_hand_sides[1] = stokes_adjoint_right_hand_sides[2];
              stokes_adjoint_right_hand_sides[2] = 0.0;
              for (auto &field_history : composition_adjoints)
                {
                  field_history[0] = field_history[1];
                  field_history[1] = field_history[2];
                  field_history[2] = 0.0;
                }
            }

          if (restart_enabled && reverse_state.next_reverse_step > 0
              && reverse_state.next_reverse_step % manager.trajectory_manager.configuration().checkpoint_interval == 0)
            {
              // The transaction order is checkpoint -> optional cleanup. Thus
              // the durable checkpoint always names every anchor it needs. If
              // cleanup fails midway, resume repeats the idempotent cleanup
              // from cleanup_frontier without losing a recoverable state.
              manager.inject_lifecycle_failure(Diagnostics::LifecycleFailurePoint::before_checkpoint_commit,
                                               reverse_state.next_reverse_step);
              if (manager.trajectory_manager.configuration().cleanup_policy
                  == TrajectoryCleanupPolicy::safe_incremental)
                // Persist the target before deleting anything. A resume can
                // therefore repeat a partially completed deletion idempotently.
                reverse_state.cleanup_frontier = reverse_state.next_reverse_step;
              manager.trajectory_manager.set_restart_trajectory_identity(reverse_state);
              TimerOutput::Scope checkpoint_timer(manager.get_computing_timer(),
                                                  "Adjoint restart checkpoint write");
              const bool inject_checkpoint_transaction =
                reverse_state.next_reverse_step
                == manager.checkpoint_transaction_failure_next_reverse_step;
              const CheckpointWriteResult checkpoint_write =
                manager.trajectory_manager.restart_checkpoint().write(
                  reverse_state,
              {
                inject_checkpoint_transaction
                ? manager.checkpoint_transaction_failure_injection
                : Diagnostics::CheckpointFailurePoint::none,
                manager.checkpoint_transaction_failure_rank
              });
              if (!checkpoint_write.committed)
                {
                  manager.get_pcout()
                      << "   Adjoint checkpoint transaction failed collectively before "
                      << "reverse step " << reverse_state.next_reverse_step
                      << "; "
                      << (checkpoint_write.latest_published
                          ? "the new generation was already published."
                          : "no new generation was published.")
                      << std::endl;
                  if (Utilities::MPI::n_mpi_processes(manager.get_mpi_communicator()) > 1)
                    throw QuietException();
                  AssertThrow(false,
                              ExcMessage("Adjoint checkpoint transaction failed before "
                                         "reverse step "
                                         + Utilities::int_to_string(
                                           reverse_state.next_reverse_step)
                                         + "."));
                }
              manager.get_pcout() << "   Adjoint restart checkpoint committed before reverse step "
                                  << reverse_state.next_reverse_step << "." << std::endl;
              manager.inject_lifecycle_failure(
                Diagnostics::LifecycleFailurePoint::after_checkpoint_commit_before_cleanup,
                reverse_state.next_reverse_step);
              if (manager.trajectory_manager.configuration().cleanup_policy
                  == TrajectoryCleanupPolicy::safe_incremental)
                manager.cleanup_trajectory_after_checkpoint(reverse_state);
              manager.inject_lifecycle_failure(Diagnostics::LifecycleFailurePoint::after_cleanup,
                                               reverse_state.next_reverse_step);
              if (reverse_state.next_reverse_step == manager.stop_after_adjoint_checkpoint)
                {
                  manager.get_pcout() << "   Controlled stop after committed adjoint checkpoint "
                                      << "before reverse step "
                                      << reverse_state.next_reverse_step
                                      << ". Restart with 'Stop after adjoint checkpoint = -1' "
                                      << "to continue." << std::endl;
                  // This is an expected M2 control-flow exit, not a scientific
                  // error. Throwing a deal.II assertion independently on every
                  // MPI rank causes its verbose exception formatter to grow
                  // unboundedly on macOS. QuietException gives all ranks the
                  // same non-zero termination path after the checkpoint has
                  // already been published.
                  if (Utilities::MPI::n_mpi_processes(manager.get_mpi_communicator()) > 1)
                    throw QuietException();
                  AssertThrow(false,
                              ExcMessage("Controlled stop after committed adjoint checkpoint "
                                         "before reverse step "
                                         + Utilities::int_to_string(reverse_state.next_reverse_step)
                                         + ". Restart with 'Stop after adjoint checkpoint = -1' "
                                         "to continue."));
                }
            }
        }

      // Any Stokes RHS left at the initial time level has no earlier
      // advection step to absorb it, so solve/scatter it directly into the
      // initial field gradients.
      if (stokes_adjoint_right_hand_sides[1].l2_norm()
          > std::numeric_limits<double>::min())
        {
          manager.add_stokes_field_transpose(AdvectionField::temperature(),
                                             manager.reconstruct_forward_snapshot(0),
                                             stokes_adjoint_right_hand_sides[1],
                                             temperature_adjoints[1]);
          for (unsigned int c = 0; c < manager.introspection().n_compositional_fields; ++c)
            manager.add_stokes_field_transpose(AdvectionField::composition(c),
                                               manager.reconstruct_forward_snapshot(0),
                                               stokes_adjoint_right_hand_sides[1],
                                               composition_adjoints[c][1]);
        }

      this->initial_temperature_gradient = temperature_adjoints[1];
      // After step zero there is no earlier residual. Slot 1 is therefore the
      // covector with respect to the initial field itself, dJ/dT_0 (and
      // analogously dJ/dC_0), in the full constrained FE space.
      this->initial_composition_gradients.clear();
      this->initial_composition_gradients.reserve(composition_adjoints.size());
      for (const auto &field_adjoints : composition_adjoints)
        this->initial_composition_gradients.push_back(field_adjoints[1]);

      manager.get_pcout() << "   Time-dependent temperature adjoint recorded "
                          << manager.trajectory_manager.archive().size()
                          << " accepted forward trajectory records." << std::endl;
      if (manager.trajectory_manager.configuration().anchor_spacing > 1)
        manager.get_pcout() << "ASPECT_ADJOINT_SPARSE_REPLAY_SUMMARY anchors_spacing="
                            << manager.trajectory_manager.configuration().anchor_spacing
                            << " schedule_count=" << manager.trajectory_manager.archive().size()
                            << " anchor_count=" << manager.trajectory_manager.sparse_anchor_steps.size()
                            << " anchor_payload_state_count="
                            << manager.trajectory_manager.archive().payload_record_count()
                            << " anchor_load_count=" << manager.trajectory_manager.sparse_anchor_load_count
                            << " interval_replay_count=" << manager.trajectory_manager.sparse_interval_replay_count
                            << " replayed_transition_count=" << manager.trajectory_manager.sparse_replay_transition_count
                            << " restart_induced_extra_replay_transitions="
                            << manager.trajectory_manager.sparse_restart_replay_transition_count
                            << " peak_window_state_count=" << manager.trajectory_manager.sparse_peak_window_state_count
                            << " max_state_relative_difference=" << manager.trajectory_manager.sparse_max_state_relative_difference
                            << " max_velocity_relative_difference=" << manager.trajectory_manager.sparse_max_velocity_relative_difference
                            << " max_gauge_pressure_relative_difference="
                            << manager.trajectory_manager.sparse_max_pressure_relative_difference << std::endl;
      manager.get_pcout() << "   Completed reverse temperature-advection transpose sweep "
                          << "and wrote the initial-temperature gradient." << std::endl;

      if (!manager.suppress_evaluation_side_effects)
        {
          manager.write_initial_temperature_gradient();
          manager.write_initial_composition_gradients();
        }

      manager.report_trajectory_statistics("reverse complete");

      // A normal return means all final scientific outputs above are durable.
      // Do not place this in a destructor: an exception must leave both the
      // archive and any committed reverse checkpoint available for diagnosis
      // or restart. Incremental reclamation is performed only after a future
      // committed reverse checkpoint; at completion all remaining records are
      // unconditionally obsolete.
      if (lifecycle_cleanup_enabled
          && manager.trajectory_manager.configuration().cleanup_policy != TrajectoryCleanupPolicy::keep)
        {
          TimerOutput::Scope cleanup_timer(manager.get_computing_timer(),
                                           "Adjoint trajectory cleanup");
          const TrajectoryStorageStatistics storage_before_cleanup =
            manager.trajectory_manager.archive().storage_statistics();
          manager.trajectory_manager.archive().remove_archive();
          manager.get_pcout() << "   Adjoint trajectory cleanup reclaimed "
                              << Utilities::MPI::sum(
                                static_cast<unsigned long long>(storage_before_cleanup.total_bytes()),
                                manager.get_mpi_communicator())
                              << " bytes after successful reverse completion."
                              << std::endl;
        }

      // simulator_state restores the forward state even if the reverse sweep,
      // checkpoint transaction, or cleanup path throws.
    }

  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::TimeDependentWorkflow<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
