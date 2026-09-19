/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_adjoint_time_dependent_trajectory_store_h
#define _aspect_adjoint_time_dependent_trajectory_store_h

#include <aspect/global.h>
#include <aspect/adjoint/core/failure_injection.h>


#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aspect
{
  namespace Adjoint
  {
    enum class TrajectoryStorageMode
    {
      minimal,
      stokes_cached
    };

    enum class TrajectoryCleanupPolicy
    {
      safe_incremental,
      on_success,
      keep
    };

    struct ReplayStepMetadata
    {
      /** Index of the archived physical state q_n. */
      unsigned int step = numbers::invalid_unsigned_int;

      /** Physical time associated with q_n. */
      double time = 0.0;

      /**
       * Time step that produced q_n from q_(n-1). This is the simulator's
       * current time_step while q_n is archived.
       */
      double time_step = 0.0;

      /** Time step preceding @p time_step, needed by the BDF2 history. */
      double old_time_step = 0.0;

      /**
       * The accepted step size that advances q_n to q_(n+1). This field is
       * deliberately distinct from @p time_step: it is selected by the
       * forward time-step manager after q_n has been solved.
       */
      double outgoing_time_step = 0.0;

      /** Whether q_n has a subsequently solved accepted transition. */
      bool has_outgoing_step = false;
    };

    struct TrajectoryStorageStatistics
    {
      std::size_t metadata_bytes = 0;
      std::size_t temperature_bytes = 0;
      std::size_t composition_bytes = 0;
      std::size_t velocity_bytes = 0;
      std::size_t pressure_bytes = 0;

      std::size_t total_bytes() const
      {
        return metadata_bytes + temperature_bytes + composition_bytes
               + velocity_bytes + pressure_bytes;
      }
    };

    struct TrajectoryCleanupResult
    {
      bool complete = false;
      unsigned int records_deleted = 0;
      unsigned int payload_files_deleted = 0;
      TrajectoryStorageStatistics reclaimed;
    };

    /**
     * Fixed-mesh, same-layout trajectory archive. Each logical record is one
     * concatenated, rank-local payload. Ranks write non-overlapping ranges of
     * one finite segment via MPI_File_write_at_all; rank zero only publishes
     * the small manifest. Bytes not named by that manifest are never valid
     * records. This is intentionally not a repartition-capable checkpoint.
     */
    class TrajectoryStore
    {
      public:
        TrajectoryStore(const TrajectoryStorageMode mode,
                        const std::string &output_directory,
                        const MPI_Comm mpi_communicator,
                        const bool open_existing = false,
                        const unsigned int requested_records_per_segment = 4);

        ~TrajectoryStore();

        TrajectoryStore(const TrajectoryStore &) = delete;
        TrajectoryStore &operator=(const TrajectoryStore &) = delete;

        TrajectoryStorageMode mode() const;
        const std::string &archive_directory() const;
        unsigned int size() const;
        bool has_record(const unsigned int record) const;
        unsigned int first_available_record() const;
        unsigned int last_available_record() const;
        unsigned int payload_record_count() const;

        /** A stable identity for the dense accepted-step journal. */
        std::uint64_t schedule_fingerprint() const;

        /** Sum manifest and retained shared segment bytes. */
        std::size_t shared_archive_bytes() const;

        /** Number of shared payload segments named by the manifest. */
        unsigned int shared_segment_count() const;

        TrajectoryCleanupResult
        reclaim_sparse_payloads_after(
          const unsigned int last_to_keep,
          const Diagnostics::FailureInjection<Diagnostics::CleanupFailurePoint> &injection = {});

        /** Install a typed append failure from a test driver. */
        void
        set_append_failure_injection_for_testing(
          const Diagnostics::FailureInjection<Diagnostics::AppendFailurePoint> &injection);

        TrajectoryCleanupResult
        reclaim_records_after(
          const unsigned int first_to_remove,
          const Diagnostics::FailureInjection<Diagnostics::CleanupFailurePoint> &injection = {});

        void remove_archive();
        const ReplayStepMetadata &metadata(const unsigned int record) const;
        void mark_last_record_terminal();
        void append_schedule(const ReplayStepMetadata &step_metadata);

        void store_payload(const unsigned int record,
                           const LinearAlgebra::BlockVector &solution,
                           const unsigned int temperature_block,
                           const std::vector<unsigned int> &composition_blocks,
                           const unsigned int velocity_block,
                           const unsigned int pressure_block);

        void store(const ReplayStepMetadata &step_metadata,
                   const LinearAlgebra::BlockVector &solution,
                   const unsigned int temperature_block,
                   const std::vector<unsigned int> &composition_blocks,
                   const unsigned int velocity_block,
                   const unsigned int pressure_block);

        void load(const unsigned int record,
                  LinearAlgebra::BlockVector &solution,
                  const unsigned int temperature_block,
                  const std::vector<unsigned int> &composition_blocks,
                  const unsigned int velocity_block,
                  const unsigned int pressure_block) const;

        TrajectoryStorageStatistics storage_statistics() const;
        std::size_t memory_consumption() const;

      private:
        class Implementation;
        std::unique_ptr<Implementation> implementation;
    };
  }
}

#endif
