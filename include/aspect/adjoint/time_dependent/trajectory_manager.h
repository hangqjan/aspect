/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_adjoint_time_dependent_trajectory_manager_h
#define _aspect_adjoint_time_dependent_trajectory_manager_h

#include <aspect/adjoint/time_dependent/adjoint_restart_checkpoint.h>
#include <aspect/adjoint/time_dependent/trajectory_store.h>
#include <aspect/linear_algebra_types.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/parameter_handler.h>

#include <deque>
#include <memory>
#include <vector>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim> class Manager;
    template <int dim> class TimeDependentWorkflow;

    /**
     * Own the durable storage used by a time-dependent adjoint evaluation.
     *
     * The primary archive is the source of reverse replay. An optional dense
     * reference archive exists only while sparse-replay equivalence is being
     * checked. The restart manager is created next to the primary archive so
     * that a checkpoint always identifies the journal it resumes.
     *
     * Replay-window state remains owned by TimeDependentWorkflow because it
     * is transient and must never be serialized by this class.
     *
     * @ingroup Adjoint
     */
    template <int dim>
    class TrajectoryManager : public SimulatorAccess<dim>
    {
      public:
        /** Immutable-after-setup controls for storage, replay, and restart. */
        struct Configuration
        {
          TrajectoryStorageMode storage_mode = TrajectoryStorageMode::minimal;
          TrajectoryCleanupPolicy cleanup_policy = TrajectoryCleanupPolicy::safe_incremental;
          unsigned int records_per_segment = 4;
          unsigned int anchor_spacing = 1;
          unsigned int checkpoint_interval = 50;
        };

        /** Parse the production trajectory configuration. */
        void parse_parameters(ParameterHandler &prm);

        /** Return the parsed production trajectory configuration. */
        const Configuration &configuration() const;

        /** Record one accepted forward state and its outgoing schedule data. */
        void record_forward_snapshot(Manager<dim> &manager,
                                     const double outgoing_time_step,
                                     const bool has_outgoing_step);

        /** Mark the final accepted state and materialize its sparse anchor. */
        void mark_forward_snapshot_terminal(Manager<dim> &manager);

        /** Destroy all in-memory handles without deleting durable files. */
        void reset();

        /**
         * Discard transient reconstruction windows and reset replay counters.
         * Durable trajectory and restart storage remain open.
         */
        void reset_replay_state();

        /** Create or open the primary accepted-step archive. */
        void create_archive(const TrajectoryStorageMode mode,
                            const std::string &output_directory,
                            const bool open_existing,
                            const unsigned int records_per_segment);

        /** Create the optional dense archive used by sparse-replay checks. */
        void create_reference_archive(const TrajectoryStorageMode mode,
                                      const std::string &output_directory,
                                      const unsigned int records_per_segment);

        bool has_archive() const;
        bool has_reference_archive() const;
        bool has_restart_checkpoint_manager() const;

        TrajectoryStore &archive();
        const TrajectoryStore &archive() const;
        TrajectoryStore &reference_archive();
        const TrajectoryStore &reference_archive() const;

        /** Create reverse-checkpoint storage next to the primary archive. */
        void create_restart_checkpoint_manager();

        RestartCheckpointManager &restart_checkpoint();
        const RestartCheckpointManager &restart_checkpoint() const;

      private:
        friend class Manager<dim>;
        friend class TimeDependentWorkflow<dim>;

        struct ReconstructedTrajectoryState
        {
          unsigned int record = numbers::invalid_unsigned_int;
          LinearAlgebra::BlockVector state;
        };

        struct ForwardReplayWindow
        {
          unsigned int anchor = numbers::invalid_unsigned_int;
          unsigned int end = numbers::invalid_unsigned_int;
          std::deque<ReconstructedTrajectoryState> states;
        };

        /** Return the inclusive sparse interval containing @p record. */
        std::pair<unsigned int, unsigned int>
        sparse_interval_for_record(const unsigned int record) const;

        /** Return the anchor needed to execute the given reverse step. */
        unsigned int required_anchor_for_reverse_step(
          const unsigned int reverse_step) const;

        /** Store the durable trajectory identity in a reverse checkpoint. */
        void set_restart_trajectory_identity(AdjointRestartState &state) const;

        /** Validate a reverse checkpoint against the currently open archive. */
        void validate_sparse_restart_state(const AdjointRestartState &state) const;

        /** Materialize the three-level state package for one sparse anchor. */
        void store_sparse_anchor_package(Manager<dim> &manager,
                                         const unsigned int anchor);

        std::unique_ptr<TrajectoryStore> primary_archive;
        std::unique_ptr<TrajectoryStore> dense_reference_archive;
        std::unique_ptr<RestartCheckpointManager> restart_manager;
        std::deque<ReconstructedTrajectoryState> reverse_state_cache;
        std::deque<ForwardReplayWindow> forward_replay_windows;
        std::deque<ReconstructedTrajectoryState> terminal_anchor_history;
        std::vector<unsigned int> sparse_anchor_steps;
        unsigned int stokes_recovery_solves = 0;
        unsigned int sparse_replay_transition_count = 0;
        unsigned int sparse_restart_replay_transition_count = 0;
        unsigned int sparse_anchor_load_count = 0;
        unsigned int sparse_interval_replay_count = 0;
        unsigned int sparse_peak_window_state_count = 0;
        double sparse_max_state_relative_difference = 0.0;
        double sparse_max_velocity_relative_difference = 0.0;
        double sparse_max_pressure_relative_difference = 0.0;
        Configuration parsed_configuration;
    };
  }
}

#endif
