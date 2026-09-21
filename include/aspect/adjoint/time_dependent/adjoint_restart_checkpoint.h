/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  This class deliberately does not use ASPECT's forward restart format.  It
  persists only the state required to continue the reverse recurrence.
*/

#ifndef _aspect_adjoint_time_dependent_adjoint_restart_checkpoint_h
#define _aspect_adjoint_time_dependent_adjoint_restart_checkpoint_h

#include <aspect/adjoint/time_dependent/trajectory_store.h>
#include <aspect/adjoint/core/failure_injection.h>
#include <aspect/global.h>
#include <array>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace aspect
{
  namespace Adjoint
  {
    /**
     * Semantic, rolling state of the reverse recurrence. The three entries of
     * every history are respectively the pending current, previous, and
     * previous-previous time levels for @p next_reverse_step. No cache
     * container or reconstructible forward field is serialized here.
     */
    struct AdjointRestartState
    {
      unsigned int next_reverse_step = 0;
      unsigned int cleanup_frontier = numbers::invalid_unsigned_int;
      unsigned int generation = 0;
      // Sparse replay state is deliberately metadata-only. The transient
      // replay window is reconstructed from this anchor and the schedule.
      unsigned int trajectory_anchor_spacing = 1;
      unsigned int required_sparse_anchor = numbers::invalid_unsigned_int;
      unsigned int trajectory_schedule_records = 0;
      std::uint64_t trajectory_schedule_fingerprint = 0;
      TrajectoryStorageMode trajectory_storage_mode =
        TrajectoryStorageMode::minimal;
      std::array<LinearAlgebra::BlockVector, 3> temperature_history;
      std::array<LinearAlgebra::BlockVector, 3> stokes_right_hand_side_history;
      std::vector<std::array<LinearAlgebra::BlockVector, 3>> composition_history;
      std::map<std::string, double> objective_values;
    };

    /** Result of one collective checkpoint transaction. */
    struct CheckpointWriteResult
    {
      bool committed = false;
      bool latest_published = false;
      std::size_t local_bytes = 0;
    };

    /**
     * Transactional, rank-local adjoint restart storage. A partially written
     * .tmp generation is never selected for restore. The format deliberately
     * supports only the same MPI rank count and DoF ownership that created it.
     */
    class RestartCheckpointManager
    {
      public:
        /** Create a checkpoint store next to the forward trajectory archive. */
        RestartCheckpointManager(const std::string &trajectory_directory,
                                 const MPI_Comm mpi_communicator);

        /** Return whether the collectively published latest pointer exists. */
        bool
        has_committed_checkpoint() const;

        /** Write, commit, and make @p state the latest restart generation. */
        CheckpointWriteResult
        write(
          AdjointRestartState &state,
          const Diagnostics::FailureInjection<Diagnostics::CheckpointFailurePoint> &injection = {});

        /** Restore the latest committed state into already initialized vectors. */
        std::size_t
        restore(AdjointRestartState &state) const;

        /** Remove restart generations and the pointer after normal completion. */
        void
        remove_temporary_data();

      private:
        std::string
        checkpoint_directory(const unsigned int generation) const;

        std::string
        temporary_checkpoint_directory(const unsigned int generation) const;

        static void
        write_vector(std::ostream &output,
                     const LinearAlgebra::BlockVector &vector);

        static void
        read_vector(std::istream &input,
                    LinearAlgebra::BlockVector &vector);

        std::size_t
        write_state(const std::string &path, const AdjointRestartState &state) const;

        std::size_t
        read_state(const std::string &path, AdjointRestartState &state) const;

        void
        write_metadata(const std::string &path,
                       const AdjointRestartState &state,
                       const unsigned int generation) const;

        void
        read_metadata(const std::string &path, AdjointRestartState &state) const;

        std::string
        rank_payload_filename() const;

        bool
        collective_success(const bool local_success,
                           const std::string &stage) const;

        void
        prune_old_generations(const unsigned int newest) const;

        std::string directory;
        MPI_Comm communicator;
        unsigned int rank;
        unsigned int n_processes;
    };
  }
}

#endif
