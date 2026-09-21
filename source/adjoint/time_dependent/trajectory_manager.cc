/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/adjoint/time_dependent/trajectory_manager.h>

#include <aspect/adjoint/core/manager.h>
#include <aspect/simulator.h>

#include <algorithm>
#include <map>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    void
    TrajectoryManager<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Adjoint");
      prm.enter_subsection("Time dependent");
      {
        const std::string storage_mode = prm.get("Trajectory storage mode");
        AssertThrow(storage_mode == "minimal" || storage_mode == "stokes cached",
                    ExcMessage("Unknown adjoint trajectory storage mode."));
        parsed_configuration.storage_mode = storage_mode == "minimal"
                                            ? TrajectoryStorageMode::minimal
                                            : TrajectoryStorageMode::stokes_cached;

        const std::string cleanup_policy = prm.get("Trajectory cleanup policy");
        AssertThrow(cleanup_policy == "safe incremental" || cleanup_policy == "on success"
                    || cleanup_policy == "keep",
                    ExcMessage("Unknown adjoint trajectory cleanup policy."));
        parsed_configuration.cleanup_policy =
          cleanup_policy == "safe incremental" ? TrajectoryCleanupPolicy::safe_incremental
          : cleanup_policy == "on success" ? TrajectoryCleanupPolicy::on_success
          : TrajectoryCleanupPolicy::keep;

        parsed_configuration.records_per_segment =
          prm.get_integer("Trajectory records per segment");
        parsed_configuration.anchor_spacing = prm.get_integer("Trajectory anchor spacing");
        parsed_configuration.checkpoint_interval =
          prm.get_integer("Adjoint checkpoint interval");
      }
      prm.leave_subsection();
      prm.leave_subsection();
    }



    template <int dim>
    const typename TrajectoryManager<dim>::Configuration &
    TrajectoryManager<dim>::configuration() const
    {
      return parsed_configuration;
    }



    template <int dim>
    void
    TrajectoryManager<dim>::record_forward_snapshot(Manager<dim> &manager,
                                                    const double outgoing_time_step,
                                                    const bool has_outgoing_step)
    {
      // Recording is attached to accepted forward states only. A logical
      // record contains the complete time-grid metadata; its physical payload
      // may be dense or may be one member of a sparse three-state anchor
      // package used to seed deterministic replay.
      if (!manager.time_dependent_temperature_workflow_enabled())
        return;

      manager.validate_time_dependent_temperature_setup();

      if (!has_archive())
        {
          // Archive creation also starts a new objective accumulation epoch.
          // The optional dense reference is diagnostic-only and never serves
          // the production reverse sweep or restart path.
          create_archive(parsed_configuration.storage_mode,
                         manager.get_output_directory(),
                         false,
                         parsed_configuration.records_per_segment);
          sparse_anchor_steps.clear();
          if (parsed_configuration.anchor_spacing > 1
              && manager.run_sparse_trajectory_replay_check)
            create_reference_archive(
              TrajectoryStorageMode::stokes_cached,
              manager.get_output_directory() + "sparse_replay_reference/",
              parsed_configuration.records_per_segment);
          manager.time_dependent_workflow.running_stokes_objective_values.clear();
        }

      archive().set_append_failure_injection_for_testing(
      {
        manager.simulator->timestep_number == manager.trajectory_append_failure_step
        ? manager.trajectory_append_failure_injection
        : Diagnostics::AppendFailurePoint::none,
        manager.trajectory_append_failure_rank
      });

      TimerOutput::Scope write_timer(manager.get_computing_timer(),
                                     "Adjoint trajectory write");
      const ReplayStepMetadata metadata
      {
        manager.simulator->timestep_number,
                          manager.simulator->time,
                          manager.simulator->time_step,
                          manager.simulator->old_time_step,
                          outgoing_time_step,
                          has_outgoing_step
      };
      // Running Stokes objectives are quadrature-in-time contributions already
      // associated with this accepted step by the objective plugin. Store only
      // their scalar sum here; their state seeds are reconstructed again at
      // the corresponding reverse step.
      const std::map<std::string, double> running_values =
        manager.objective_manager.evaluate_running_stokes_objectives(
          manager.simulator->timestep_number);
      for (const auto &value : running_values)
        manager.time_dependent_workflow.running_stokes_objective_values[value.first]
        += value.second;

      if (parsed_configuration.anchor_spacing == 1)
        // Dense storage commits one payload per logical state.
        archive().store(metadata,
                        manager.simulator->solution,
                        manager.introspection().block_indices.temperature,
                        manager.introspection().block_indices.compositional_fields,
                        manager.introspection().block_indices.velocities,
                        manager.introspection().block_indices.pressure);
      else
        {
          // Sparse storage always commits the dense schedule but only persists
          // payloads at anchor packages. The three cached states
          // [q_a,q_{a-1},q_{a-2}] are sufficient to restart BDF2 replay from
          // anchor a without reaching into a preceding interval.
          archive().append_schedule(metadata);
          terminal_anchor_history.clear();
          terminal_anchor_history.push_back({metadata.step, manager.simulator->solution});
          if (metadata.step > 0)
            terminal_anchor_history.push_back({metadata.step - 1, manager.simulator->old_solution});
          if (metadata.step > 1)
            terminal_anchor_history.push_back({metadata.step - 2, manager.simulator->old_old_solution});
          if (metadata.step == 0 || metadata.step % parsed_configuration.anchor_spacing == 0)
            store_sparse_anchor_package(manager, metadata.step);
          if (has_reference_archive())
            reference_archive().store(metadata,
                                      manager.simulator->solution,
                                      manager.introspection().block_indices.temperature,
                                      manager.introspection().block_indices.compositional_fields,
                                      manager.introspection().block_indices.velocities,
                                      manager.introspection().block_indices.pressure);
        }
    }



    template <int dim>
    void
    TrajectoryManager<dim>::mark_forward_snapshot_terminal(Manager<dim> &manager)
    {
      if (!manager.time_dependent_temperature_workflow_enabled())
        return;

      AssertThrow(has_archive(), ExcInternalError());
      // The last state may not fall on the periodic anchor grid. Persist its
      // history package explicitly before declaring that its schedule record
      // has no outgoing step; otherwise the first reverse window could lack a
      // replay starting point.
      if (parsed_configuration.anchor_spacing > 1)
        store_sparse_anchor_package(manager, archive().size() - 1);
      archive().mark_last_record_terminal();
      if (has_reference_archive())
        reference_archive().mark_last_record_terminal();
    }



    template <int dim>
    void
    TrajectoryManager<dim>::store_sparse_anchor_package(Manager<dim> &manager,
                                                        const unsigned int anchor)
    {
      AssertThrow(has_archive(), ExcInternalError());
      AssertThrow(!terminal_anchor_history.empty()
                  && terminal_anchor_history.front().record == anchor,
                  ExcMessage("Sparse anchor history no longer matches its logical state."));
      // Store every missing member of the BDF history package idempotently.
      // Neighboring anchors may share records, and restart can repeat this
      // method after an interrupted append without duplicating payloads.
      const bool new_anchor = !archive().has_record(anchor);
      for (const ReconstructedTrajectoryState &state : terminal_anchor_history)
        if (!archive().has_record(state.record))
          archive().store_payload(
            state.record,
            state.state,
            manager.introspection().block_indices.temperature,
            manager.introspection().block_indices.compositional_fields,
            manager.introspection().block_indices.velocities,
            manager.introspection().block_indices.pressure);

      if (new_anchor)
        {
          sparse_anchor_steps.push_back(anchor);
          manager.get_pcout() << "ASPECT_ADJOINT_SPARSE_ANCHOR step=" << anchor
                              << " package_begin=" << terminal_anchor_history.back().record
                              << " package_end=" << terminal_anchor_history.front().record
                              << " stored_physical_states="
                              << archive().payload_record_count() << std::endl;
        }
    }



    template <int dim>
    void
    TrajectoryManager<dim>::reset()
    {
      restart_manager.reset();
      dense_reference_archive.reset();
      primary_archive.reset();
      reset_replay_state();
      terminal_anchor_history.clear();
      sparse_anchor_steps.clear();
    }



    template <int dim>
    void
    TrajectoryManager<dim>::reset_replay_state()
    {
      // These caches and counters describe one reverse execution only. The
      // durable archive and parsed configuration survive separately; clearing
      // them here would conflate transient replay memory with trajectory
      // ownership.
      reverse_state_cache.clear();
      forward_replay_windows.clear();
      stokes_recovery_solves = 0;
      sparse_replay_transition_count = 0;
      sparse_restart_replay_transition_count = 0;
      sparse_anchor_load_count = 0;
      sparse_interval_replay_count = 0;
      sparse_peak_window_state_count = 0;
      sparse_max_state_relative_difference = 0.0;
      sparse_max_velocity_relative_difference = 0.0;
      sparse_max_pressure_relative_difference = 0.0;
    }



    template <int dim>
    void
    TrajectoryManager<dim>::create_archive(const TrajectoryStorageMode mode,
                                           const std::string &output_directory,
                                           const bool open_existing,
                                           const unsigned int records_per_segment)
    {
      // Opening/creating a primary archive invalidates any checkpoint manager
      // bound to the previous archive directory. Recreate it explicitly only
      // after the trajectory identity is known.
      primary_archive = std::make_unique<TrajectoryStore>(
                          mode, output_directory, this->get_mpi_communicator(), open_existing,
                          records_per_segment);
      restart_manager.reset();
    }



    template <int dim>
    void
    TrajectoryManager<dim>::create_reference_archive(
      const TrajectoryStorageMode mode,
      const std::string &output_directory,
      const unsigned int records_per_segment)
    {
      dense_reference_archive = std::make_unique<TrajectoryStore>(
                                  mode, output_directory, this->get_mpi_communicator(), false,
                                  records_per_segment);
    }



    template <int dim>
    bool
    TrajectoryManager<dim>::has_archive() const
    {
      return primary_archive != nullptr;
    }



    template <int dim>
    bool
    TrajectoryManager<dim>::has_reference_archive() const
    {
      return dense_reference_archive != nullptr;
    }



    template <int dim>
    bool
    TrajectoryManager<dim>::has_restart_checkpoint_manager() const
    {
      return restart_manager != nullptr;
    }



    template <int dim>
    TrajectoryStore &
    TrajectoryManager<dim>::archive()
    {
      AssertThrow(primary_archive != nullptr, ExcInternalError());
      return *primary_archive;
    }



    template <int dim>
    const TrajectoryStore &
    TrajectoryManager<dim>::archive() const
    {
      AssertThrow(primary_archive != nullptr, ExcInternalError());
      return *primary_archive;
    }



    template <int dim>
    TrajectoryStore &
    TrajectoryManager<dim>::reference_archive()
    {
      AssertThrow(dense_reference_archive != nullptr, ExcInternalError());
      return *dense_reference_archive;
    }



    template <int dim>
    const TrajectoryStore &
    TrajectoryManager<dim>::reference_archive() const
    {
      AssertThrow(dense_reference_archive != nullptr, ExcInternalError());
      return *dense_reference_archive;
    }



    template <int dim>
    void
    TrajectoryManager<dim>::create_restart_checkpoint_manager()
    {
      AssertThrow(primary_archive != nullptr, ExcInternalError());
      restart_manager = std::make_unique<RestartCheckpointManager>(
                          primary_archive->archive_directory(), this->get_mpi_communicator());
    }



    template <int dim>
    RestartCheckpointManager &
    TrajectoryManager<dim>::restart_checkpoint()
    {
      AssertThrow(restart_manager != nullptr, ExcInternalError());
      return *restart_manager;
    }



    template <int dim>
    const RestartCheckpointManager &
    TrajectoryManager<dim>::restart_checkpoint() const
    {
      AssertThrow(restart_manager != nullptr, ExcInternalError());
      return *restart_manager;
    }



    template <int dim>
    std::pair<unsigned int, unsigned int>
    TrajectoryManager<dim>::sparse_interval_for_record(const unsigned int record) const
    {
      AssertThrow(has_archive() && archive().size() > 0, ExcInternalError());
      AssertIndexRange(record, archive().size());
      // Partition logical records into half-open anchor bins [a,a+s), clipped
      // at the terminal record. The returned end is inclusive because reverse
      // replay materializes concrete state indices rather than step edges.
      const unsigned int anchor = (record / parsed_configuration.anchor_spacing)
                                  * parsed_configuration.anchor_spacing;
      const unsigned int end = std::min(anchor + parsed_configuration.anchor_spacing,
                                        archive().size() - 1);
      AssertThrow(anchor <= record && record <= end,
                  ExcMessage("Sparse replay interval does not contain its record."));
      return {anchor, end};
    }



    template <int dim>
    unsigned int
    TrajectoryManager<dim>::required_anchor_for_reverse_step(
      const unsigned int reverse_step) const
    {
      AssertThrow(has_archive() && reverse_step > 0 && reverse_step <= archive().size(),
                  ExcMessage("A reverse step has no corresponding sparse replay interval."));

      // Reverse step r transposes the forward edge q_(r-1)->q_r and therefore
      // consumes q_(r-1). Map that state, not q_r, to an interval. As a result
      // an exact anchor-boundary reverse step belongs to the preceding
      // interval, preventing cleanup from deleting the anchor it still needs.
      return sparse_interval_for_record(reverse_step - 1).first;
    }



    template <int dim>
    void
    TrajectoryManager<dim>::set_restart_trajectory_identity(
      AdjointRestartState &state) const
    {
      AssertThrow(has_archive() && archive().size() > 0, ExcInternalError());
      // Bind the checkpoint to both sparse-layout policy and the exact dense
      // accepted-step schedule. required_sparse_anchor is a durable cleanup
      // frontier: payload reclamation may delete newer intervals only after a
      // checkpoint naming the next required anchor has committed.
      state.trajectory_anchor_spacing = parsed_configuration.anchor_spacing;
      state.trajectory_schedule_records = archive().size();
      state.trajectory_schedule_fingerprint = archive().schedule_fingerprint();
      state.required_sparse_anchor = parsed_configuration.anchor_spacing > 1
                                     && state.next_reverse_step > 0
                                     ? required_anchor_for_reverse_step(state.next_reverse_step)
                                     : numbers::invalid_unsigned_int;
    }



    template <int dim>
    void
    TrajectoryManager<dim>::validate_sparse_restart_state(
      const AdjointRestartState &state) const
    {
      // Reject a numerically plausible but structurally different replay.
      // Time-grid coefficients, record count, and anchor spacing jointly
      // define the discrete reverse operator and must all match.
      AssertThrow(state.trajectory_anchor_spacing == parsed_configuration.anchor_spacing
                  && state.trajectory_schedule_records == archive().size()
                  && state.trajectory_schedule_fingerprint == archive().schedule_fingerprint(),
                  ExcMessage("The committed adjoint restart checkpoint refers to a different "
                             "accepted-step schedule or sparse-anchor layout."));

      if (parsed_configuration.anchor_spacing > 1 && state.next_reverse_step > 0)
        {
          // Validate both the checkpoint's claimed frontier and the payload
          // needed to realize it. Boundary assertions protect the subtle
          // r-1 interval rule against later off-by-one regressions.
          const unsigned int required =
            required_anchor_for_reverse_step(state.next_reverse_step);
          AssertThrow(state.required_sparse_anchor == required,
                      ExcMessage("The committed sparse restart anchor does not match the "
                                 "authoritative reverse-step interval mapping."));
          AssertThrow(archive().has_record(required),
                      ExcMessage("The committed sparse restart anchor package is missing."));
          AssertThrow(required_anchor_for_reverse_step(1) == 0, ExcInternalError());
          if (archive().size() >= parsed_configuration.anchor_spacing)
            AssertThrow(required_anchor_for_reverse_step(parsed_configuration.anchor_spacing) == 0,
                        ExcMessage("An anchor-boundary reverse step selected the wrong interval."));
        }
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::TrajectoryManager<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
