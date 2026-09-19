/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/adjoint/time_dependent/adjoint_restart_checkpoint.h>
#include <aspect/utilities.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

namespace aspect
{
  namespace Adjoint
  {
    RestartCheckpointManager::RestartCheckpointManager(
      const std::string &trajectory_directory,
      const MPI_Comm mpi_communicator)
      :
      directory(trajectory_directory + "adjoint_restart/"),
      communicator(mpi_communicator),
      rank(Utilities::MPI::this_mpi_process(mpi_communicator)),
      n_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    {}



    bool
    RestartCheckpointManager::has_committed_checkpoint() const
    {
      // A generation directory alone is not a commit record. Only the
      // atomically replaced "latest" pointer makes a checkpoint visible to
      // restore(), so abandoned temporary or complete-but-unpublished
      // generations are deliberately ignored.
      unsigned int has_checkpoint = 0;
      bool local_success = true;
      if (rank == 0)
        {
          std::error_code error;
          has_checkpoint = std::filesystem::exists(directory + "latest", error) ? 1 : 0;
          local_success = !error;
        }
      collective_success(local_success, "inspect committed checkpoint pointer");
      return Utilities::MPI::broadcast(communicator, has_checkpoint, 0) == 1;
    }



    CheckpointWriteResult
    RestartCheckpointManager::write(
      AdjointRestartState &state,
      const Diagnostics::FailureInjection<Diagnostics::CheckpointFailurePoint> &injection)
    {
      const unsigned int generation = state.generation + 1;
      const std::string checkpoint = checkpoint_directory(generation);
      const std::string temporary = temporary_checkpoint_directory(generation);
      const auto inject_here = [this, &injection]
                               (const Diagnostics::CheckpointFailurePoint point)
      {
        return injection.point == point && injection.applies_to_rank(rank);
      };

      // The on-disk transaction is
      //
      //   create generation.tmp
      //   -> write all rank payloads
      //   -> write shared metadata
      //   -> rename generation.tmp to generation
      //   -> atomically replace latest.
      //
      // Every stage is made collective before advancing. Consequently no rank
      // can publish a generation for which another rank failed to write its
      // payload. The temporary directory is intentionally harmless after a
      // crash because restore follows only the atomically published pointer.
      bool local_success = true;
      if (rank == 0)
        try
          {
            std::error_code error;
            std::filesystem::create_directories(directory, error);
            if (error)
              local_success = false;
            std::filesystem::remove_all(temporary, error);
            if (error)
              local_success = false;
            std::filesystem::create_directory(temporary, error);
            if (error)
              local_success = false;
          }
        catch (const std::exception &)
          {
            local_success = false;
          }
      if (!collective_success(local_success, "create temporary checkpoint generation"))
        return {};
      if (!collective_success(
            !inject_here(Diagnostics::CheckpointFailurePoint::after_temporary_directory_creation),
            "inject checkpoint failure after temporary directory creation"))
        return {};

      std::size_t bytes = 0;
      local_success = true;
      try
        {
          // Each rank writes only its owned pieces of the fixed three-level
          // reverse history. The metadata is withheld until all payload files
          // have closed successfully.
          bytes = write_state(temporary, state);
        }
      catch (const std::exception &)
        {
          local_success = false;
        }
      local_success = local_success
                      && !inject_here(Diagnostics::CheckpointFailurePoint::after_rank_payload_write);
      if (!collective_success(local_success, "write rank-local checkpoint payloads"))
        return {false, false, bytes};
      if (!collective_success(
            !inject_here(Diagnostics::CheckpointFailurePoint::before_metadata_publication),
            "inject checkpoint failure before metadata publication"))
        return {false, false, bytes};

      local_success = true;
      if (rank == 0)
        try
          {
            // Shared metadata binds the vector payloads to the next reverse
            // step, cleanup frontier, sparse anchor, and trajectory schedule
            // identity. Publishing a payload without these invariants would
            // not be sufficient for a reproducible resume.
            write_metadata(temporary, state, generation);
          }
        catch (const std::exception &)
          {
            local_success = false;
          }
      if (!collective_success(local_success, "write checkpoint metadata"))
        return {false, false, bytes};
      if (!collective_success(
            !inject_here(Diagnostics::CheckpointFailurePoint::after_metadata_publication),
            "inject checkpoint failure before directory commit"))
        return {false, false, bytes};

      local_success = true;
      if (rank == 0)
        try
          {
            std::error_code error;
            std::filesystem::remove_all(checkpoint, error);
            if (error)
              local_success = false;
            if (local_success)
              {
                // This rename makes the generation internally complete, but
                // it is still unreachable by restore() until "latest" is
                // replaced below. A crash in between merely leaves garbage
                // that can be reclaimed later.
                std::filesystem::rename(temporary, checkpoint, error);
                if (error)
                  local_success = false;
              }
          }
        catch (const std::exception &)
          {
            local_success = false;
          }
      if (!collective_success(local_success, "commit checkpoint generation"))
        return {false, false, bytes};
      if (!collective_success(
            !inject_here(Diagnostics::CheckpointFailurePoint::before_latest_publication),
            "inject checkpoint failure before latest publication"))
        return {false, false, bytes};

      local_success = true;
      if (rank == 0)
        try
          {
            // Replacing latest is the external commit point. The temporary
            // pointer is fully flushed/closed before rename, so readers select
            // either the preceding generation or this complete generation.
            const std::string latest_tmp = directory + "latest.tmp";
            std::ofstream latest(latest_tmp, std::ios::trunc);
            latest << generation << '\n';
            latest.close();
            if (!latest)
              local_success = false;
            if (local_success)
              {
                std::error_code error;
                std::filesystem::rename(latest_tmp, directory + "latest", error);
                if (error)
                  local_success = false;
              }
          }
        catch (const std::exception &)
          {
            local_success = false;
          }
      if (!collective_success(local_success, "publish committed checkpoint generation"))
        return {false, false, bytes};
      if (!collective_success(
            !inject_here(Diagnostics::CheckpointFailurePoint::after_latest_publication),
            "inject checkpoint failure after latest publication"))
        return {false, true, bytes};

      state.generation = generation;
      return {true, true, bytes};
    }



    std::size_t
    RestartCheckpointManager::restore(AdjointRestartState &state) const
    {
      // Resolve the commit pointer first and never scan for the numerically
      // largest directory. This preserves the transaction semantics when a
      // crash leaves a newer, complete-looking but unpublished generation.
      unsigned int generation = 0;
      bool local_success = true;
      if (rank == 0)
        try
          {
            std::ifstream latest(directory + "latest");
            latest >> generation;
            if (!latest || generation == 0)
              local_success = false;
          }
        catch (const std::exception &)
          {
            local_success = false;
          }
      collective_success(local_success, "read committed checkpoint pointer");
      generation = Utilities::MPI::broadcast(communicator, generation, 0);
      const std::string checkpoint = checkpoint_directory(generation);

      // Shared metadata is restored before rank payloads so the caller's
      // vectors and trajectory identity can be validated against the selected
      // generation. Every rank must agree on this phase before local reads.
      local_success = true;
      try
        {
          read_metadata(checkpoint, state);
          if (state.generation != generation)
            local_success = false;
        }
      catch (const std::exception &)
        {
          local_success = false;
        }
      collective_success(local_success, "restore shared checkpoint metadata");

      std::size_t bytes = 0;
      local_success = true;
      try
        {
          bytes = read_state(checkpoint, state);
        }
      catch (const std::exception &)
        {
          local_success = false;
        }
      collective_success(local_success, "restore rank-local checkpoint payloads");
      return bytes;
    }



    void
    RestartCheckpointManager::remove_temporary_data()
    {
      bool local_success = true;
      if (rank == 0)
        {
          std::error_code error;
          std::filesystem::remove_all(directory, error);
          local_success = !error;
        }
      collective_success(local_success, "remove adjoint restart data");
    }



    std::string
    RestartCheckpointManager::checkpoint_directory(const unsigned int generation) const
    {
      return directory + "checkpoint_" + Utilities::int_to_string(generation, 8) + "/";
    }



    std::string
    RestartCheckpointManager::temporary_checkpoint_directory(const unsigned int generation) const
    {
      return directory + "checkpoint_" + Utilities::int_to_string(generation, 8) + ".tmp/";
    }



    void
    RestartCheckpointManager::write_vector(std::ostream &output,
                                           const LinearAlgebra::BlockVector &vector)
    {
      // Persist global indices together with values for every locally owned
      // entry. The index makes same-size-but-different MPI partitions fail
      // loudly during restore instead of silently permuting an adjoint state;
      // ghosts are excluded because they are not owned data.
      const std::uint64_t blocks = vector.n_blocks();
      output.write(reinterpret_cast<const char *>(&blocks), sizeof(blocks));
      for (unsigned int b = 0; b < vector.n_blocks(); ++b)
        {
          const std::uint64_t count = vector.block(b).locally_owned_elements().n_elements();
          output.write(reinterpret_cast<const char *>(&count), sizeof(count));
          for (const auto index : vector.block(b).locally_owned_elements())
            {
              const std::uint64_t stored_index = index;
              const double value = vector.block(b)(index);
              output.write(reinterpret_cast<const char *>(&stored_index), sizeof(stored_index));
              output.write(reinterpret_cast<const char *>(&value), sizeof(value));
            }
        }
      AssertThrow(output, ExcMessage("Writing an adjoint restart vector failed."));
    }



    void
    RestartCheckpointManager::read_vector(std::istream &input,
                                          LinearAlgebra::BlockVector &vector)
    {
      std::uint64_t blocks = 0;
      input.read(reinterpret_cast<char *>(&blocks), sizeof(blocks));
      AssertThrow(input && blocks == vector.n_blocks(),
                  ExcMessage("Adjoint restart vector does not match the active DoF layout."));
      for (unsigned int b = 0; b < vector.n_blocks(); ++b)
        {
          std::uint64_t count = 0;
          input.read(reinterpret_cast<char *>(&count), sizeof(count));
          AssertThrow(input && count == vector.block(b).locally_owned_elements().n_elements(),
                      ExcMessage("Adjoint restart vector does not match the active MPI partitioning."));
          // IndexSet iteration order must match the writer. Checking every
          // stored global index turns partition compatibility into an exact
          // invariant rather than relying only on the local element count.
          for (const auto index : vector.block(b).locally_owned_elements())
            {
              std::uint64_t stored_index = 0;
              double value = 0;
              input.read(reinterpret_cast<char *>(&stored_index), sizeof(stored_index));
              input.read(reinterpret_cast<char *>(&value), sizeof(value));
              AssertThrow(input && stored_index == index,
                          ExcMessage("Adjoint restart vector does not match the active MPI ownership."));
              vector.block(b)(index) = value;
            }
          vector.block(b).compress(VectorOperation::insert);
        }
    }



    std::size_t
    RestartCheckpointManager::write_state(const std::string &path,
                                          const AdjointRestartState &state) const
    {
      const std::string filename = path + rank_payload_filename();
      std::ofstream output(filename, std::ios::binary | std::ios::trunc);
      AssertThrow(output, ExcMessage("Could not write rank-local adjoint restart payload <"
                                     + filename + ">."));
      const std::uint64_t magic = 0x4153504543544152ULL;
      output.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
      const auto write = [&output](const LinearAlgebra::BlockVector &vector)
      {
        write_vector(output, vector);
      };
      // Reverse BDF propagation has exactly three slots: current, previous,
      // and previous-previous. Temperature and Stokes RHS histories are stored
      // level-major; each composition field carries the same three-level
      // layout. Restore reads this schema in precisely the same order.
      for (unsigned int level = 0; level < 3; ++level)
        {
          write(state.temperature_history[level]);
          write(state.stokes_right_hand_side_history[level]);
        }
      for (const auto &composition : state.composition_history)
        for (unsigned int level = 0; level < 3; ++level)
          write(composition[level]);
      output.close();
      AssertThrow(output, ExcMessage("Writing rank-local adjoint restart payload <"
                                     + filename + "> failed."));
      return static_cast<std::size_t>(std::filesystem::file_size(filename));
    }



    std::size_t
    RestartCheckpointManager::read_state(const std::string &path,
                                         AdjointRestartState &state) const
    {
      const std::string filename = path + rank_payload_filename();
      std::ifstream input(filename, std::ios::binary | std::ios::ate);
      AssertThrow(input, ExcMessage("Could not read rank-local adjoint restart payload <"
                                    + filename + ">."));
      const std::size_t bytes = static_cast<std::size_t>(input.tellg());
      input.seekg(0);
      std::uint64_t magic = 0;
      input.read(reinterpret_cast<char *>(&magic), sizeof(magic));
      AssertThrow(input && magic == 0x4153504543544152ULL,
                  ExcMessage("Adjoint restart payload has an invalid format."));
      const auto read = [&input](LinearAlgebra::BlockVector &vector)
      {
        read_vector(input, vector);
      };
      for (unsigned int level = 0; level < 3; ++level)
        {
          read(state.temperature_history[level]);
          read(state.stokes_right_hand_side_history[level]);
        }
      for (auto &composition : state.composition_history)
        for (unsigned int level = 0; level < 3; ++level)
          read(composition[level]);
      return bytes;
    }



    void
    RestartCheckpointManager::write_metadata(const std::string &path,
                                             const AdjointRestartState &state,
                                             const unsigned int generation) const
    {
      std::ofstream output(path + "state.txt", std::ios::trunc);
      AssertThrow(output, ExcMessage("Could not write adjoint restart metadata."));
      output.precision(std::numeric_limits<double>::max_digits10);
      output << "ASPECT_ADJOINT_RESTART_V3\n"
             << generation << ' ' << state.next_reverse_step << ' '
             << state.cleanup_frontier << ' '
             << (state.trajectory_storage_mode == TrajectoryStorageMode::minimal ? 0 : 1)
             << ' ' << state.composition_history.size() << ' ' << n_processes << ' '
             << state.trajectory_anchor_spacing << ' '
             << state.required_sparse_anchor << ' '
             << state.trajectory_schedule_records << ' '
             << state.trajectory_schedule_fingerprint << '\n'
             << state.objective_values.size() << '\n';

      // Objective values are scalar accumulated results, not reconstructible
      // from the three adjoint history slots. Store their names with quoting
      // so whitespace in registered objective names round-trips exactly.
      for (const auto &entry : state.objective_values)
        output << std::quoted(entry.first) << '\t' << entry.second << '\n';
      output.close();
      AssertThrow(output, ExcMessage("Writing adjoint restart metadata failed."));
    }



    void
    RestartCheckpointManager::read_metadata(const std::string &path,
                                            AdjointRestartState &state) const
    {
      unsigned int composition_fields = 0;
      unsigned int storage_mode = 0;
      unsigned int stored_processes = 0;
      std::ifstream input(path + "state.txt");
      std::string magic;
      std::size_t objectives = 0;
      input >> magic >> state.generation >> state.next_reverse_step
      >> state.cleanup_frontier >> storage_mode >> composition_fields
      >> stored_processes >> state.trajectory_anchor_spacing
      >> state.required_sparse_anchor >> state.trajectory_schedule_records
      >> state.trajectory_schedule_fingerprint >> objectives;
      AssertThrow(input && magic == "ASPECT_ADJOINT_RESTART_V3"
                  && composition_fields == state.composition_history.size()
                  && stored_processes == n_processes,
                  ExcMessage("Adjoint restart metadata is incompatible with this MPI run."));
      AssertThrow(storage_mode == 0 || storage_mode == 1,
                  ExcMessage("Adjoint restart metadata has an invalid storage mode."));
      state.trajectory_storage_mode = storage_mode == 0
                                      ? TrajectoryStorageMode::minimal
                                      : TrajectoryStorageMode::stokes_cached;
      state.objective_values.clear();
      for (std::size_t i = 0; i < objectives; ++i)
        {
          std::string name;
          double value = 0;
          input >> std::quoted(name) >> value;
          AssertThrow(input,
                      ExcMessage("Adjoint restart objective metadata ended unexpectedly."));
          state.objective_values.emplace(name, value);
        }
    }



    std::string
    RestartCheckpointManager::rank_payload_filename() const
    {
      return "rank_" + Utilities::int_to_string(rank, 5) + ".bin";
    }



    bool
    RestartCheckpointManager::collective_success(const bool local_success,
                                                 const std::string &stage) const
    {
      const unsigned int global_success = Utilities::MPI::min(local_success ? 1U : 0U,
                                                              communicator);
      const int barrier_status = MPI_Barrier(communicator);
      (void)stage;
      return global_success == 1 && barrier_status == MPI_SUCCESS;
    }



    void
    RestartCheckpointManager::prune_old_generations(const unsigned int newest) const
    {
      if (rank != 0 || newest <= 2)
        return;
      std::error_code error;
      std::filesystem::remove_all(checkpoint_directory(newest - 2), error);
      AssertThrow(!error, ExcMessage("Could not prune obsolete adjoint checkpoint: "
                                     + error.message()));
    }
  }
}
