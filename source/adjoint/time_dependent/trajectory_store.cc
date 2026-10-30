/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/adjoint/time_dependent/trajectory_store.h>
#include <aspect/utilities.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_large_count.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

namespace aspect
{
  namespace Adjoint
  {
    class TrajectoryStore::Implementation
    {
      public:
        Implementation(const TrajectoryStorageMode mode,
                       const std::string &output_directory,
                       const MPI_Comm mpi_communicator,
                       const bool open_existing = false,
                       const unsigned int requested_records_per_segment = 4)
          :
          storage_mode(mode),
          directory(output_directory + "adjoint_trajectory/"),
          communicator(mpi_communicator),
          rank(Utilities::MPI::this_mpi_process(mpi_communicator)),
          n_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
          records_per_segment(requested_records_per_segment)
        {
          AssertThrow(records_per_segment > 0, ExcInternalError());
          if (open_existing)
            open_existing_archive();
          else
            initialize_fresh_archive();
        }

        TrajectoryStorageMode mode() const
        {
          return storage_mode;
        }
        const std::string &archive_directory() const
        {
          return directory;
        }
        unsigned int size() const
        {
          return metadata_table.size();
        }
        bool has_record(const unsigned int record) const
        {
          return record < metadata_table.size()
                 && !record_locations[record].ranks.empty();
        }
        unsigned int first_available_record() const
        {
          return first_available;
        }
        unsigned int last_available_record() const
        {
          return last_available;
        }
        unsigned int payload_record_count() const
        {
          unsigned int result = 0;
          for (const RecordLocation &location : record_locations)
            result += !location.ranks.empty();
          return result;
        }

        /** A stable identity for the dense accepted-step journal. */
        std::uint64_t schedule_fingerprint() const
        {
          // FNV-1a over the logical accepted-step journal detects a restart
          // against a trajectory with different time levels. Hash the IEEE
          // bit patterns, not formatted decimal text, so equality means the
          // replay coefficients are bitwise identical to those checkpointed.
          std::uint64_t hash = 1469598103934665603ULL;
          hash_append(hash, metadata_table.size());
          hash_append(hash, records_per_segment);
          hash_append(hash, storage_mode == TrajectoryStorageMode::minimal ? 0 : 1);
          for (const ReplayStepMetadata &metadata : metadata_table)
            {
              hash_append(hash, metadata.step);
              hash_double(hash, metadata.time);
              hash_double(hash, metadata.time_step);
              hash_double(hash, metadata.old_time_step);
              hash_double(hash, metadata.outgoing_time_step);
              hash_append(hash, metadata.has_outgoing_step ? 1 : 0);
            }
          return hash;
        }

        /** Sum manifest and retained shared segment bytes (reported on rank 0). */
        std::size_t shared_archive_bytes() const
        {
          std::size_t bytes = 0;
          if (rank == 0)
            {
              std::error_code error;
              bytes = static_cast<std::size_t>(
                        std::filesystem::file_size(directory + manifest_filename, error));
              AssertThrow(!error, ExcMessage("Could not inspect trajectory manifest."));
              for (unsigned int segment = 0; segment <= last_segment(); ++segment)
                if (std::filesystem::exists(segment_filename(segment), error))
                  {
                    AssertThrow(!error, ExcMessage("Could not inspect trajectory segment."));
                    bytes += static_cast<std::size_t>(
                               std::filesystem::file_size(segment_filename(segment), error));
                    AssertThrow(!error, ExcMessage("Could not inspect trajectory segment."));
                  }
                else
                  AssertThrow(!error, ExcMessage("Could not inspect trajectory segment."));
            }
          return Utilities::MPI::broadcast(communicator, bytes, 0);
        }

        /** Number of shared payload segments still named by the manifest. */
        unsigned int shared_segment_count() const
        {
          unsigned int count = 0;
          if (rank == 0)
            for (unsigned int segment = 0; segment <= last_segment(); ++segment)
              for (const RecordLocation &location : record_locations)
                if (location.segment == segment && !location.ranks.empty())
                  {
                    ++count;
                    break;
                  }
          return Utilities::MPI::broadcast(communicator, count, 0);
        }

        /**
         * Drop only sparse physical payloads after @p last_to_keep. The dense
         * accepted-step schedule deliberately remains intact. A missing
         * segment before manifest publication is recoverable: a restart
         * repeats this idempotent transaction using its durable frontier.
         */
        TrajectoryCleanupResult
        reclaim_sparse_payloads_after(
          const unsigned int last_to_keep,
          const Diagnostics::FailureInjection<Diagnostics::CleanupFailurePoint> &injection = {})
        {
          TrajectoryCleanupResult result;
          const auto inject_here = [this, &injection]
                                   (const Diagnostics::CleanupFailurePoint point)
          {
            return injection.point == point && injection.applies_to_rank(rank);
          };
          if (!collective_cleanup_success(
                !inject_here(Diagnostics::CleanupFailurePoint::before_cleanup)))
            return result;

          // The cleanup frontier is a restart invariant: records <=
          // last_to_keep may include the sparse anchor required to replay the
          // next reverse step. First remove only segments that contain no
          // such payload. Because segments are append-only, deleting a whole
          // obsolete file cannot damage a retained record.
          //
          // the durable restart-safe anchor. If interrupted here, the old
          // manifest may still name obsolete records, but it never loses a
          // valid restart anchor and the checkpoint frontier repeats cleanup.
          if (!delete_sparse_obsolete_segments(last_to_keep, result))
            return result;
          if (!collective_cleanup_success(
                !inject_here(Diagnostics::CleanupFailurePoint::after_segment_deletion)))
            return result;
          if (!collective_cleanup_success(
                !inject_here(Diagnostics::CleanupFailurePoint::after_rank_payload_deletion)))
            return result;

          for (unsigned int record = last_to_keep + 1; record < record_locations.size(); ++record)
            if (!record_locations[record].ranks.empty())
              {
                ++result.records_deleted;
                add_statistics(result.reclaimed, payload_bytes[record]);
                record_locations[record].ranks.clear();
                payload_bytes[record] = {};
              }
          if (!collective_cleanup_success(
                !inject_here(Diagnostics::CleanupFailurePoint::before_collective_success))
              || !collective_cleanup_success(
                !inject_here(Diagnostics::CleanupFailurePoint::before_manifest_update)))
            return result;

          // Clearing the in-memory locations is not durable until the new
          // manifest is published. The atomic rename below is the commit
          // point: after a crash readers see either the old manifest or the
          // new one, never a partially serialized record table.
          bool local_success = true;
          if (rank == 0)
            try
              {
                write_manifest_atomically();
              }
            catch (const std::exception &)
              {
                local_success = false;
              }
          if (!collective_cleanup_success(local_success))
            return result;
          if (!collective_cleanup_success(
                !inject_here(Diagnostics::CleanupFailurePoint::after_manifest_update)))
            return result;
          result.complete = true;
          return result;
        }

        /** Test-only hook for validating an uncommitted shared append. */
        void set_append_failure_injection_for_testing(
          const Diagnostics::FailureInjection<Diagnostics::AppendFailurePoint> &injection)
        {
          append_failure_injection = injection;
        }

        TrajectoryCleanupResult
        reclaim_records_after(
          const unsigned int first_to_remove,
          const Diagnostics::FailureInjection<Diagnostics::CleanupFailurePoint> &injection = {})
        {
          TrajectoryCleanupResult result;
          const auto inject_here = [this, &injection]
                                   (const Diagnostics::CleanupFailurePoint point)
          {
            return injection.point == point && injection.applies_to_rank(rank);
          };
          if (metadata_table.empty() || first_available == numbers::invalid_unsigned_int
              || first_to_remove > last_available)
            {
              result.complete = delete_fully_reclaimed_segments(result);
              return result;
            }

          const unsigned int begin = std::max(first_to_remove, first_available);
          if (!collective_cleanup_success(
                !inject_here(Diagnostics::CleanupFailurePoint::before_cleanup))
              || !collective_cleanup_success(
                !inject_here(Diagnostics::CleanupFailurePoint::after_rank_payload_deletion))
              || !collective_cleanup_success(
                !inject_here(Diagnostics::CleanupFailurePoint::before_collective_success))
              || !collective_cleanup_success(
                !inject_here(Diagnostics::CleanupFailurePoint::before_manifest_update)))
            return result;

          for (unsigned int record = begin; record <= last_available; ++record)
            {
              ++result.records_deleted;
              add_statistics(result.reclaimed, payload_bytes[record]);
            }
          const unsigned int old_last = last_available;
          last_available = begin == 0 ? numbers::invalid_unsigned_int : begin - 1;
          bool local_success = true;
          if (rank == 0)
            try
              {
                write_manifest_atomically();
              }
            catch (const std::exception &)
              {
                local_success = false;
              }
          if (!collective_cleanup_success(local_success))
            {
              last_available = old_last;
              return result;
            }
          if (!collective_cleanup_success(
                !inject_here(Diagnostics::CleanupFailurePoint::after_manifest_update)))
            return result;
          result.complete = delete_fully_reclaimed_segments(result);
          return result;
        }

        void remove_archive()
        {
          bool local_success = true;
          if (rank == 0)
            {
              std::error_code error;
              std::filesystem::remove_all(directory, error);
              local_success = !error;
            }
          AssertThrow(collective_cleanup_success(local_success),
                      ExcMessage("Could not remove the adjoint trajectory archive collectively."));
          metadata_table.clear();
          record_locations.clear();
          payload_bytes.clear();
          first_available = 0;
          last_available = numbers::invalid_unsigned_int;
        }

        const ReplayStepMetadata &metadata(const unsigned int record) const
        {
          AssertIndexRange(record, metadata_table.size());
          return metadata_table[record];
        }

        /** Mark the final stored state as having no solved successor. */
        void mark_last_record_terminal()
        {
          AssertThrow(!metadata_table.empty(), ExcInternalError());
          // A stored state normally describes the step leaving it. The final
          // accepted state has no successor, so explicitly remove that edge;
          // reverse replay must not manufacture an extra dt beyond the
          // terminal objective time.
          metadata_table.back().has_outgoing_step = false;
          metadata_table.back().outgoing_time_step = 0.0;
          bool local_success = true;
          if (rank == 0)
            try
              {
                write_manifest_atomically();
              }
            catch (const std::exception &)
              {
                local_success = false;
              }
          collective_success(local_success, "publish terminal trajectory schedule metadata");
        }

        /** Append one dense accepted-step schedule entry without a payload. */
        void append_schedule(const ReplayStepMetadata &step_metadata)
        {
          AssertThrow(step_metadata.step == metadata_table.size(),
                      ExcMessage("Adjoint trajectory records must be contiguous."));
          // The dense schedule is the authoritative time grid and exists even
          // when sparse storage omits this state's vector payload. Contiguous
          // step numbers make record index == time-step index, avoiding a
          // second mapping during reverse traversal.
          metadata_table.push_back(step_metadata);
          record_locations.emplace_back();
          payload_bytes.emplace_back();
          if (metadata_table.size() == 1)
            first_available = step_metadata.step;
          last_available = step_metadata.step;
          bool local_success = true;
          if (rank == 0)
            try
              {
                write_manifest_atomically();
              }
            catch (const std::exception &)
              {
                local_success = false;
              }
          collective_success(local_success, "publish dense trajectory schedule metadata");
        }

        /** Store the physical payload for an existing logical schedule entry. */
        void store_payload(const unsigned int record,
                           const LinearAlgebra::BlockVector &solution,
                           const unsigned int temperature_block,
                           const std::vector<unsigned int> &composition_blocks,
                           const unsigned int velocity_block,
                           const unsigned int pressure_block)
        {
          AssertIndexRange(record, metadata_table.size());
          AssertThrow(!has_record(record),
                      ExcMessage("A physical payload was already committed for this trajectory state."));
          // Append is a two-phase transaction. First every rank serializes
          // its locally owned blocks and collectively writes an unpublished
          // byte range. Only after the shared file is synchronized does rank
          // zero atomically publish the new RecordLocation in the manifest.
          // An orphaned byte range is therefore harmless after a failed
          // append: has_record() is determined solely by committed metadata.
          std::vector<char> local_payload;
          TrajectoryStorageStatistics local_bytes;
          bool local_success = true;
          try
            {
              local_bytes.temperature_bytes =
                append_vector(local_payload, solution.block(temperature_block));
              for (const unsigned int block : composition_blocks)
                local_bytes.composition_bytes += append_vector(local_payload, solution.block(block));
              if (storage_mode == TrajectoryStorageMode::stokes_cached)
                {
                  local_bytes.velocity_bytes = append_vector(local_payload, solution.block(velocity_block));
                  local_bytes.pressure_bytes = append_vector(local_payload, solution.block(pressure_block));
                }
            }
          catch (const std::exception &)
            {
              local_success = false;
            }
          collective_success(local_success, "serialize rank-local trajectory payloads");
          abort_append_for_testing(Diagnostics::AppendFailurePoint::after_local_serialization);

          const RecordLocation location = allocate_record_location(
                                            record, local_payload.size(),
                                            partition_fingerprint(solution, temperature_block, composition_blocks,
                                                                  velocity_block, pressure_block), local_bytes);
          local_success = write_shared_payload(location, local_payload);
          collective_success(local_success, "write and synchronize shared trajectory payload");
          abort_append_for_testing(
            Diagnostics::AppendFailurePoint::after_collective_payload_write);
          abort_append_for_testing(Diagnostics::AppendFailurePoint::after_payload_write);

          // This mutation is tentative until write_manifest_atomically()
          // succeeds collectively. Fault-injection after payload write tests
          // that reopening the archive still ignores these unpublished bytes.
          record_locations[record] = location;
          payload_bytes[record] = local_bytes;
          local_success = true;
          if (rank == 0)
            try
              {
                write_manifest_atomically();
              }
            catch (const std::exception &)
              {
                local_success = false;
              }
          collective_success(local_success, "publish shared trajectory record metadata");
          abort_append_for_testing(Diagnostics::AppendFailurePoint::after_manifest_publication);
        }

        void store(const ReplayStepMetadata &step_metadata,
                   const LinearAlgebra::BlockVector &solution,
                   const unsigned int temperature_block,
                   const std::vector<unsigned int> &composition_blocks,
                   const unsigned int velocity_block,
                   const unsigned int pressure_block)
        {
          append_schedule(step_metadata);
          store_payload(step_metadata.step, solution, temperature_block,
                        composition_blocks, velocity_block, pressure_block);
        }

        void load(const unsigned int record,
                  LinearAlgebra::BlockVector &solution,
                  const unsigned int temperature_block,
                  const std::vector<unsigned int> &composition_blocks,
                  const unsigned int velocity_block,
                  const unsigned int pressure_block) const
        {
          AssertThrow(has_record(record),
                      ExcMessage("Adjoint trajectory record is not available in the committed archive."));
          // The manifest fixes both the shared byte range and the per-rank
          // partition fingerprint. A restart is intentionally same-rank and
          // same-partition: values are restored directly into their original
          // locally owned global indices without repartitioning.
          const RecordLocation &location = record_locations[record];
          AssertThrow(location.ranks.size() == n_processes,
                      ExcMessage("Trajectory record has an incompatible MPI rank count."));
          const RankPayload &local = location.ranks[rank];
          collective_success(local.fingerprint == partition_fingerprint(solution, temperature_block,
                                                                        composition_blocks, velocity_block,
                                                                        pressure_block),
                             "validate the active same-rank DoF partition");
          AssertThrow(local.size <= std::numeric_limits<std::size_t>::max(),
                      ExcMessage("Trajectory payload is too large for this process."));
          std::vector<char> payload(static_cast<std::size_t>(local.size));
          collective_success(read_shared_payload(location, payload), "read shared trajectory payload");
          const char *cursor = payload.data();
          const char *const end = cursor + payload.size();
          read_vector(cursor, end, solution.block(temperature_block));
          for (const unsigned int block : composition_blocks)
            read_vector(cursor, end, solution.block(block));
          if (storage_mode == TrajectoryStorageMode::stokes_cached)
            {
              read_vector(cursor, end, solution.block(velocity_block));
              read_vector(cursor, end, solution.block(pressure_block));
            }
          AssertThrow(cursor == end, ExcMessage("Trajectory payload has an invalid schema."));
        }

        TrajectoryStorageStatistics storage_statistics() const
        {
          TrajectoryStorageStatistics result;
          if (first_available != numbers::invalid_unsigned_int)
            for (unsigned int record = first_available; record <= last_available; ++record)
              add_statistics(result, payload_bytes[record]);
          if (rank == 0)
            {
              std::error_code error;
              result.metadata_bytes = static_cast<std::size_t>(
                                        std::filesystem::file_size(directory + manifest_filename, error));
              AssertThrow(!error, ExcMessage("Could not inspect trajectory manifest."));
            }
          return result;
        }

        std::size_t memory_consumption() const
        {
          std::size_t result = sizeof(*this) + directory.capacity();
          result += metadata_table.capacity() * sizeof(ReplayStepMetadata);
          result += payload_bytes.capacity() * sizeof(TrajectoryStorageStatistics);
          for (const RecordLocation &location : record_locations)
            result += sizeof(location) + location.ranks.capacity() * sizeof(RankPayload);
          return result;
        }

      private:
        struct RankPayload
        {
          std::uint64_t offset = 0, size = 0, fingerprint = 0;
          std::uint64_t temperature_bytes = 0, composition_bytes = 0;
          std::uint64_t velocity_bytes = 0, pressure_bytes = 0;
        };
        struct RecordLocation
        {
          unsigned int segment = 0;
          std::uint64_t record_offset = 0, total_size = 0;
          std::vector<RankPayload> ranks;
        };
        static constexpr const char *manifest_filename = "manifest";
        static constexpr unsigned int schema_version = 4;
        // Small enough that ordinary short trajectory cases cross a boundary.

        static void add_statistics(TrajectoryStorageStatistics &into,
                                   const TrajectoryStorageStatistics &from)
        {
          into.temperature_bytes += from.temperature_bytes;
          into.composition_bytes += from.composition_bytes;
          into.velocity_bytes += from.velocity_bytes;
          into.pressure_bytes += from.pressure_bytes;
        }
        static void append_bytes(std::vector<char> &output, const void *data,
                                 const std::size_t size)
        {
          AssertThrow(size <= std::numeric_limits<std::size_t>::max() - output.size(),
                      ExcMessage("Trajectory payload size overflows local memory."));
          const char *begin = static_cast<const char *>(data);
          output.insert(output.end(), begin, begin + size);
        }
        template <typename T>
        static void append_scalar(std::vector<char> &output, const T &value)
        {
          append_bytes(output, &value, sizeof(value));
        }
        static std::size_t append_vector(std::vector<char> &output,
                                         const LinearAlgebra::Vector &vector)
        {
          // Store only locally owned entries in IndexSet iteration order.
          // Ghost values are reconstructed by the caller after loading and
          // would otherwise be duplicated across MPI payloads.
          const std::size_t beginning = output.size();
          const std::uint64_t count = vector.locally_owned_elements().n_elements();
          append_scalar(output, count);
          for (const auto index : vector.locally_owned_elements())
            append_scalar(output, vector(index));
          return output.size() - beginning;
        }
        template <typename T>
        static T read_scalar(const char *&cursor, const char *const end)
        {
          AssertThrow(static_cast<std::uint64_t>(end - cursor) >= sizeof(T),
                      ExcMessage("Trajectory payload ended unexpectedly."));
          T value;
          std::memcpy(&value, cursor, sizeof(T));
          cursor += sizeof(T);
          return value;
        }
        static void read_vector(const char *&cursor, const char *const end,
                                LinearAlgebra::Vector &vector)
        {
          const std::uint64_t count = read_scalar<std::uint64_t>(cursor, end);
          AssertThrow(count == vector.locally_owned_elements().n_elements(),
                      ExcMessage("Trajectory payload does not match active MPI partitioning."));
          for (const auto index : vector.locally_owned_elements())
            vector(index) = read_scalar<double>(cursor, end);
          vector.compress(VectorOperation::insert);
        }
        static void hash_append(std::uint64_t &hash, const std::uint64_t value)
        {
          for (unsigned int byte = 0; byte < sizeof(value); ++byte)
            {
              hash ^= (value >> (8 * byte)) & 0xffU;
              hash *= 1099511628211ULL;
            }
        }
        static void hash_double(std::uint64_t &hash, const double value)
        {
          std::uint64_t bits = 0;
          static_assert(sizeof(bits) == sizeof(value));
          std::memcpy(&bits, &value, sizeof(bits));
          hash_append(hash, bits);
        }
        static void hash_vector_layout(std::uint64_t &hash, const LinearAlgebra::Vector &vector)
        {
          hash_append(hash, vector.locally_owned_elements().n_elements());
          for (const auto index : vector.locally_owned_elements())
            hash_append(hash, static_cast<std::uint64_t>(index));
        }
        std::uint64_t partition_fingerprint(const LinearAlgebra::BlockVector &solution,
                                            const unsigned int temperature_block,
                                            const std::vector<unsigned int> &composition_blocks,
                                            const unsigned int velocity_block,
                                            const unsigned int pressure_block) const
        {
          // Hash block identities and every locally owned global index. Equal
          // vector lengths alone are insufficient: a differently partitioned
          // mesh could have the same local size but assign different DoFs.
          std::uint64_t hash = 1469598103934665603ULL;
          hash_append(hash, storage_mode == TrajectoryStorageMode::minimal);
          hash_append(hash, temperature_block);
          hash_vector_layout(hash, solution.block(temperature_block));
          hash_append(hash, composition_blocks.size());
          for (const unsigned int block : composition_blocks)
            {
              hash_append(hash, block);
              hash_vector_layout(hash, solution.block(block));
            }
          if (storage_mode == TrajectoryStorageMode::stokes_cached)
            {
              hash_append(hash, velocity_block);
              hash_vector_layout(hash, solution.block(velocity_block));
              hash_append(hash, pressure_block);
              hash_vector_layout(hash, solution.block(pressure_block));
            }
          return hash;
        }

        RecordLocation allocate_record_location(const unsigned int record,
                                                const std::uint64_t local_size,
                                                const std::uint64_t local_fingerprint,
                                                const TrajectoryStorageStatistics &bytes) const
        {
          // Rank payloads for one logical record occupy one contiguous shared
          // range. Exscan computes this rank's prefix, while Allgather builds
          // the complete table that is later written to the manifest. Thus
          // every rank independently derives the same total record size and
          // can participate in the collective MPI-IO call.
          std::uint64_t prefix = 0;
          int status = MPI_Exscan(&local_size, &prefix, 1, MPI_UINT64_T, MPI_SUM, communicator);
          if (rank == 0) prefix = 0;
          collective_success(status == MPI_SUCCESS, "determine shared trajectory rank offsets");
          RecordLocation result;
          result.segment = record / records_per_segment;
          result.record_offset = end_offset_for_segment(result.segment);
          result.ranks.resize(n_processes);
          std::vector<std::uint64_t> sizes(n_processes), fingerprints(n_processes);
          std::vector<std::uint64_t> temperature(n_processes), composition(n_processes);
          std::vector<std::uint64_t> velocity(n_processes), pressure(n_processes);
          const auto remember_status = [&status](const int current_status)
          {
            if (current_status != MPI_SUCCESS)
              status = current_status;
          };
          status = MPI_SUCCESS;
          remember_status(MPI_Allgather(&local_size, 1, MPI_UINT64_T,
                                        sizes.data(), 1, MPI_UINT64_T, communicator));
          remember_status(MPI_Allgather(&local_fingerprint, 1, MPI_UINT64_T,
                                        fingerprints.data(), 1, MPI_UINT64_T, communicator));
          const std::uint64_t local_fields[4] = {bytes.temperature_bytes, bytes.composition_bytes,
                                                 bytes.velocity_bytes, bytes.pressure_bytes
                                                };
          std::uint64_t *fields[4] = {temperature.data(), composition.data(), velocity.data(), pressure.data()};
          for (unsigned int i = 0; i < 4; ++i)
            remember_status(MPI_Allgather(&local_fields[i], 1, MPI_UINT64_T,
                                          fields[i], 1, MPI_UINT64_T, communicator));
          collective_success(status == MPI_SUCCESS, "exchange shared trajectory record metadata");
          std::uint64_t offset = 0;
          for (unsigned int p = 0; p < n_processes; ++p)
            {
              AssertThrow(sizes[p] <= std::numeric_limits<std::uint64_t>::max() - offset,
                          ExcMessage("Trajectory record exceeds MPI file offset range."));
              RankPayload &payload = result.ranks[p];
              payload.offset = offset;
              payload.size = sizes[p];
              payload.fingerprint = fingerprints[p];
              payload.temperature_bytes = temperature[p];
              payload.composition_bytes = composition[p];
              payload.velocity_bytes = velocity[p];
              payload.pressure_bytes = pressure[p];
              offset += sizes[p];
            }
          AssertThrow(result.ranks[rank].offset == prefix,
                      ExcMessage("MPI_Exscan offset does not match record metadata."));
          result.total_size = offset;
          return result;
        }

        bool write_shared_payload(const RecordLocation &location,
                                  const std::vector<char> &payload) const
        {
          // Collective MPI-IO writes each rank's bytes at
          // record_offset+rank_offset. File_sync and collective close precede
          // manifest publication, establishing the payload-before-metadata
          // durability order required by store_payload().
          MPI_File file = MPI_FILE_NULL;
          const std::string filename = segment_filename(location.segment);
          bool success = MPI_File_open(communicator, const_cast<char *>(filename.c_str()),
                                       MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file) == MPI_SUCCESS;
          if (!collective_cleanup_success(success)) return false;
          MPI_Offset existing_size = 0;
          success = MPI_File_get_size(file, &existing_size) == MPI_SUCCESS;
          // MPI_File_get_size is independent, but every operation below is
          // collective on @p communicator. In particular, MPI_File_set_size
          // must be called by every rank with the same size. Do not branch on
          // a rank-local observation of the current size: an inconsistent
          // branch would leave one rank in File_set_size while another enters
          // the next allreduce in collective_cleanup_success().
          if (!collective_cleanup_success(success))
            {
              MPI_File_close(&file);
              return false;
            }
          MPI_Offset global_existing_size = 0;
          success = MPI_Allreduce(&existing_size, &global_existing_size, 1,
                                  MPI_OFFSET, MPI_MAX, communicator) == MPI_SUCCESS;
          if (!collective_cleanup_success(success))
            {
              MPI_File_close(&file);
              return false;
            }
          const MPI_Offset required_size = static_cast<MPI_Offset>(
                                             location.record_offset + location.total_size);
          const MPI_Offset target_size = std::max(global_existing_size, required_size);
          success = MPI_File_set_size(file, target_size) == MPI_SUCCESS;
          if (!collective_cleanup_success(success))
            {
              MPI_File_close(&file);
              return false;
            }
          const RankPayload &local = location.ranks[rank];
          const void *buffer = payload.empty() ? nullptr : payload.data();
          success = dealii::Utilities::MPI::LargeCount::File_write_at_all_c(
                      file, static_cast<MPI_Offset>(location.record_offset + local.offset), buffer,
                      static_cast<MPI_Count>(payload.size()), MPI_BYTE, MPI_STATUS_IGNORE) == MPI_SUCCESS;
          const bool sync_success = MPI_File_sync(file) == MPI_SUCCESS;
          const bool close_success = MPI_File_close(&file) == MPI_SUCCESS;
          return collective_cleanup_success(success && sync_success && close_success);
        }
        bool read_shared_payload(const RecordLocation &location,
                                 std::vector<char> &payload) const
        {
          MPI_File file = MPI_FILE_NULL;
          const std::string filename = segment_filename(location.segment);
          bool success = MPI_File_open(communicator, const_cast<char *>(filename.c_str()),
                                       MPI_MODE_RDONLY, MPI_INFO_NULL, &file) == MPI_SUCCESS;
          if (!collective_cleanup_success(success)) return false;
          const RankPayload &local = location.ranks[rank];
          void *buffer = payload.empty() ? nullptr : payload.data();
          success = dealii::Utilities::MPI::LargeCount::File_read_at_all_c(
                      file, static_cast<MPI_Offset>(location.record_offset + local.offset), buffer,
                      static_cast<MPI_Count>(payload.size()), MPI_BYTE, MPI_STATUS_IGNORE) == MPI_SUCCESS;
          const bool close_success = MPI_File_close(&file) == MPI_SUCCESS;
          return collective_cleanup_success(success && close_success);
        }
        bool append_failure_here(const Diagnostics::AppendFailurePoint point) const
        {
          return append_failure_injection.point == point
                 && append_failure_injection.applies_to_rank(rank);
        }
        void abort_append_for_testing(const Diagnostics::AppendFailurePoint point) const
        {
          if (!collective_cleanup_success(!append_failure_here(point)))
            throw QuietException();
        }
        std::uint64_t end_offset_for_segment(const unsigned int segment) const
        {
          std::uint64_t result = 0;
          for (const RecordLocation &location : record_locations)
            if (location.segment == segment)
              result = std::max(result, location.record_offset + location.total_size);
          return result;
        }
        bool record_is_available(const unsigned int record) const
        {
          return first_available != numbers::invalid_unsigned_int
                 && record >= first_available && record <= last_available;
        }
        bool segment_is_fully_reclaimed(const unsigned int segment) const
        {
          for (unsigned int record = 0; record < record_locations.size(); ++record)
            if (record_locations[record].segment == segment && record_is_available(record)) return false;
          return true;
        }
        bool segment_is_sparse_obsolete(const unsigned int segment,
                                        const unsigned int last_to_keep) const
        {
          for (unsigned int record = 0; record < record_locations.size(); ++record)
            if (record_locations[record].segment == segment
                && !record_locations[record].ranks.empty()
                && record <= last_to_keep)
              return false;
          return true;
        }
        bool delete_sparse_obsolete_segments(const unsigned int last_to_keep,
                                             TrajectoryCleanupResult &result) const
        {
          bool success = true;
          if (rank == 0)
            try
              {
                for (unsigned int segment = 0; segment <= last_segment(); ++segment)
                  if (segment_is_sparse_obsolete(segment, last_to_keep))
                    {
                      std::error_code error;
                      const std::string filename = segment_filename(segment);
                      if (std::filesystem::exists(filename, error))
                        {
                          if (error || !std::filesystem::remove(filename, error) || error)
                            {
                              success = false;
                              break;
                            }
                          ++result.payload_files_deleted;
                        }
                      else if (error)
                        {
                          success = false;
                          break;
                        }
                    }
              }
            catch (const std::exception &)
              {
                success = false;
              }
          return collective_cleanup_success(success);
        }
        bool delete_fully_reclaimed_segments(TrajectoryCleanupResult &result) const
        {
          bool success = true;
          if (rank == 0)
            try
              {
                for (unsigned int segment = 0; segment <= last_segment(); ++segment)
                  if (segment_is_fully_reclaimed(segment))
                    {
                      std::error_code error;
                      const std::string filename = segment_filename(segment);
                      if (std::filesystem::exists(filename, error))
                        {
                          if (error || !std::filesystem::remove(filename, error) || error)
                            {
                              success = false;
                              break;
                            }
                          ++result.payload_files_deleted;
                        }
                      else if (error)
                        {
                          success = false;
                          break;
                        }
                    }
              }
            catch (const std::exception &)
              {
                success = false;
              }
          return collective_cleanup_success(success);
        }
        unsigned int last_segment() const
        {
          unsigned int result = 0;
          for (const RecordLocation &location : record_locations)
            result = std::max(result, location.segment);
          return result;
        }

        void initialize_fresh_archive()
        {
          bool success = true;
          if (rank == 0)
            try
              {
                std::error_code error;
                if (std::filesystem::exists(directory, error))
                  {
                    // Never recursively replace an arbitrary user directory.
                    // At least one recognized trajectory marker must prove
                    // that this path belongs to a previous generated archive.
                    const bool marker = std::filesystem::exists(directory + manifest_filename, error)
                                        || std::filesystem::exists(directory + "manifest.txt", error)
                                        || std::filesystem::exists(directory + "metadata.txt", error);
                    AssertThrow(!error && marker,
                                ExcMessage("Refusing to reset a non-trajectory directory."));
                    std::filesystem::remove_all(directory, error);
                    AssertThrow(!error, ExcMessage("Could not reset stale trajectory archive."));
                  }
              }
            catch (const std::exception &)
              {
                success = false;
              }
          collective_success(success, "initialize shared trajectory archive");
          Utilities::create_directory(directory, communicator, true);
          success = true;
          if (rank == 0)
            try
              {
                write_manifest_atomically();
              }
            catch (const std::exception &)
              {
                success = false;
              }
          collective_success(success, "publish empty shared trajectory manifest");
        }
        void open_existing_archive()
        {
          AssertThrow(Utilities::fexists(directory + manifest_filename, communicator),
                      ExcMessage("Cannot resume an adjoint trajectory: shared manifest is missing."));
          read_manifest();
        }
        void write_manifest_atomically() const
        {
          AssertThrow(rank == 0, ExcInternalError());
          // Serialize the complete logical journal and every committed byte
          // location to a sibling temporary file. POSIX rename is the sole
          // publication point, so a reader never observes a truncated table.
          const std::string temporary = directory + std::string(manifest_filename) + ".tmp";
          std::ofstream output(temporary, std::ios::trunc);
          AssertThrow(output, ExcMessage("Could not write temporary trajectory manifest."));
          output.precision(std::numeric_limits<double>::max_digits10);
          output << "ASPECT_ADJOINT_TRAJECTORY_SHARED_MPI_V4\n"
                 << "schema_version " << schema_version << '\n'
                 << "storage_mode " << (storage_mode == TrajectoryStorageMode::minimal ? "minimal" : "stokes_cached") << '\n'
                 << "records_per_segment " << records_per_segment << '\n'
                 << "records " << metadata_table.size() << '\n'
                 << "first_available " << first_available << '\n'
                 << "last_available " << last_available << '\n';
          for (unsigned int record = 0; record < metadata_table.size(); ++record)
            {
              const ReplayStepMetadata &metadata = metadata_table[record];
              const RecordLocation &location = record_locations[record];
              output << "record " << metadata.step << ' ' << metadata.time << ' ' << metadata.time_step << ' '
                     << metadata.old_time_step << ' ' << metadata.outgoing_time_step << ' '
                     << (metadata.has_outgoing_step ? 1 : 0) << ' '
                     << (storage_mode == TrajectoryStorageMode::minimal ? 0 : 1) << ' '
                     << location.segment << ' ' << location.record_offset << ' ' << location.total_size << ' '
                     << location.ranks.size() << '\n';
              for (unsigned int p = 0; p < location.ranks.size(); ++p)
                {
                  const RankPayload &payload = location.ranks[p];
                  output << "rank " << p << ' ' << payload.offset << ' ' << payload.size << ' '
                         << payload.fingerprint << ' ' << payload.temperature_bytes << ' '
                         << payload.composition_bytes << ' ' << payload.velocity_bytes << ' '
                         << payload.pressure_bytes << '\n';
                }
            }
          output.close();
          AssertThrow(output, ExcMessage("Writing temporary trajectory manifest failed."));
          std::error_code error;
          std::filesystem::rename(temporary, directory + manifest_filename, error);
          AssertThrow(!error, ExcMessage("Could not publish trajectory manifest: " + error.message()));
        }
        void read_manifest()
        {
          // Rank zero validates and parses the text manifest. The parsed
          // scalar/table entries are then broadcast, avoiding independent
          // filesystem parsing and ensuring all ranks enter later collective
          // MPI-IO operations with identical record locations.
          bool success = true;
          if (rank == 0)
            try
              {
                std::ifstream input(directory + manifest_filename);
                std::string magic, key, mode;
                unsigned int version = 0, segment_records = 0, records = 0;
                input >> magic >> key >> version >> key >> mode >> key >> segment_records
                >> key >> records >> key >> first_available >> key >> last_available;
                AssertThrow(input && magic == "ASPECT_ADJOINT_TRAJECTORY_SHARED_MPI_V4"
                            && key == "last_available" && version == schema_version
                            && segment_records == records_per_segment,
                            ExcMessage("Invalid or incompatible shared trajectory manifest."));
                AssertThrow((mode == "minimal" || mode == "stokes_cached")
                            && (mode == "minimal") == (storage_mode == TrajectoryStorageMode::minimal),
                            ExcMessage("Trajectory storage mode differs from the archive."));
                metadata_table.resize(records);
                record_locations.resize(records);
                payload_bytes.resize(records);
                for (unsigned int record = 0; record < records; ++record)
                  {
                    std::string record_key;
                    unsigned int has_outgoing_step = 0, mode_code = 0, stored_processes = 0;
                    RecordLocation &location = record_locations[record];
                    input >> record_key >> metadata_table[record].step >> metadata_table[record].time
                    >> metadata_table[record].time_step >> metadata_table[record].old_time_step
                    >> metadata_table[record].outgoing_time_step >> has_outgoing_step
                    >> mode_code >> location.segment
                    >> location.record_offset >> location.total_size >> stored_processes;
                    AssertThrow(input && record_key == "record" && metadata_table[record].step == record
                                && (has_outgoing_step == 0 || has_outgoing_step == 1)
                                && mode_code == (storage_mode == TrajectoryStorageMode::minimal ? 0 : 1)
                                && (stored_processes == 0 || stored_processes == n_processes),
                                ExcMessage("Shared trajectory record is incompatible with this MPI run."));
                    metadata_table[record].has_outgoing_step = has_outgoing_step == 1;
                    location.ranks.resize(stored_processes);
                    for (unsigned int p = 0; p < stored_processes; ++p)
                      {
                        std::string rank_key;
                        unsigned int stored_rank = 0;
                        RankPayload &payload = location.ranks[p];
                        input >> rank_key >> stored_rank >> payload.offset >> payload.size >> payload.fingerprint
                        >> payload.temperature_bytes >> payload.composition_bytes
                        >> payload.velocity_bytes >> payload.pressure_bytes;
                        AssertThrow(input && rank_key == "rank" && stored_rank == p,
                                    ExcMessage("Invalid shared trajectory rank index."));
                      }
                    if (stored_processes != 0)
                      set_local_payload_bytes(record);
                  }
              }
            catch (const std::exception &)
              {
                success = false;
              }
          collective_success(success, "read shared trajectory manifest");
          unsigned int records = rank == 0 ? metadata_table.size() : 0;
          records = Utilities::MPI::broadcast(communicator, records, 0);
          first_available = Utilities::MPI::broadcast(communicator, first_available, 0);
          last_available = Utilities::MPI::broadcast(communicator, last_available, 0);
          if (rank != 0)
            {
              metadata_table.resize(records);
              record_locations.resize(records);
              payload_bytes.resize(records);
            }
          for (unsigned int record = 0; record < records; ++record)
            {
              ReplayStepMetadata &metadata = metadata_table[record];
              RecordLocation &location = record_locations[record];
              metadata.step = Utilities::MPI::broadcast(communicator, metadata.step, 0);
              metadata.time = Utilities::MPI::broadcast(communicator, metadata.time, 0);
              metadata.time_step = Utilities::MPI::broadcast(communicator, metadata.time_step, 0);
              metadata.old_time_step = Utilities::MPI::broadcast(communicator, metadata.old_time_step, 0);
              metadata.outgoing_time_step = Utilities::MPI::broadcast(communicator, metadata.outgoing_time_step, 0);
              unsigned int has_outgoing_step = metadata.has_outgoing_step ? 1 : 0;
              has_outgoing_step = Utilities::MPI::broadcast(communicator, has_outgoing_step, 0);
              metadata.has_outgoing_step = has_outgoing_step == 1;
              location.segment = Utilities::MPI::broadcast(communicator, location.segment, 0);
              location.record_offset = Utilities::MPI::broadcast(communicator, location.record_offset, 0);
              location.total_size = Utilities::MPI::broadcast(communicator, location.total_size, 0);
              unsigned int stored_processes = rank == 0 ? location.ranks.size() : 0;
              stored_processes = Utilities::MPI::broadcast(communicator, stored_processes, 0);
              AssertThrow(stored_processes == 0 || stored_processes == n_processes,
                          ExcMessage("Trajectory payload has an incompatible MPI rank count."));
              if (rank != 0) location.ranks.resize(stored_processes);
              for (unsigned int p = 0; p < stored_processes; ++p)
                {
                  RankPayload &payload = location.ranks[p];
                  payload.offset = Utilities::MPI::broadcast(communicator, payload.offset, 0);
                  payload.size = Utilities::MPI::broadcast(communicator, payload.size, 0);
                  payload.fingerprint = Utilities::MPI::broadcast(communicator, payload.fingerprint, 0);
                  payload.temperature_bytes = Utilities::MPI::broadcast(communicator, payload.temperature_bytes, 0);
                  payload.composition_bytes = Utilities::MPI::broadcast(communicator, payload.composition_bytes, 0);
                  payload.velocity_bytes = Utilities::MPI::broadcast(communicator, payload.velocity_bytes, 0);
                  payload.pressure_bytes = Utilities::MPI::broadcast(communicator, payload.pressure_bytes, 0);
                }
              if (stored_processes != 0)
                set_local_payload_bytes(record);
            }
        }
        void set_local_payload_bytes(const unsigned int record)
        {
          const RankPayload &local = record_locations[record].ranks[rank];
          payload_bytes[record].temperature_bytes = local.temperature_bytes;
          payload_bytes[record].composition_bytes = local.composition_bytes;
          payload_bytes[record].velocity_bytes = local.velocity_bytes;
          payload_bytes[record].pressure_bytes = local.pressure_bytes;
        }
        std::string segment_filename(const unsigned int segment) const
        {
          return directory + "trajectory_" + Utilities::int_to_string(segment, 6) + ".bin";
        }
        void collective_success(const bool local_success, const std::string &stage) const
        {
          // Convert rank-local I/O errors into one collective decision before
          // any rank proceeds to the next transaction stage. The barrier is
          // intentional: without it one rank could publish metadata while
          // another rank is still unwinding a failed payload write.
          const unsigned int global_success = Utilities::MPI::min(local_success ? 1U : 0U, communicator);
          const int barrier_status = MPI_Barrier(communicator);
          AssertThrow(global_success == 1 && barrier_status == MPI_SUCCESS,
                      ExcMessage("Adjoint MPI trajectory operation failed collectively while attempting to "
                                 + stage + ". The record was not published."));
        }
        bool collective_cleanup_success(const bool local_success) const
        {
          const unsigned int global_success = Utilities::MPI::min(local_success ? 1U : 0U, communicator);
          return global_success == 1 && MPI_Barrier(communicator) == MPI_SUCCESS;
        }

        TrajectoryStorageMode storage_mode;
        std::string directory;
        MPI_Comm communicator;
        unsigned int rank, n_processes;
        unsigned int records_per_segment;
        std::vector<ReplayStepMetadata> metadata_table;
        std::vector<RecordLocation> record_locations;
        std::vector<TrajectoryStorageStatistics> payload_bytes;
        unsigned int first_available = 0;
        unsigned int last_available = numbers::invalid_unsigned_int;
        Diagnostics::FailureInjection<Diagnostics::AppendFailurePoint>
        append_failure_injection;
    };

    TrajectoryStore::TrajectoryStore(const TrajectoryStorageMode mode,
                                     const std::string &output_directory,
                                     const MPI_Comm mpi_communicator,
                                     const bool open_existing,
                                     const unsigned int requested_records_per_segment)
      :
      implementation(std::make_unique<Implementation>(
                       mode, output_directory, mpi_communicator, open_existing,
                       requested_records_per_segment))
    {}

    TrajectoryStore::~TrajectoryStore() = default;

    TrajectoryStorageMode TrajectoryStore::mode() const
    {
      return implementation->mode();
    }
    const std::string &TrajectoryStore::archive_directory() const
    {
      return implementation->archive_directory();
    }
    unsigned int TrajectoryStore::size() const
    {
      return implementation->size();
    }
    bool TrajectoryStore::has_record(const unsigned int record) const
    {
      return implementation->has_record(record);
    }
    unsigned int TrajectoryStore::first_available_record() const
    {
      return implementation->first_available_record();
    }
    unsigned int TrajectoryStore::last_available_record() const
    {
      return implementation->last_available_record();
    }
    unsigned int TrajectoryStore::payload_record_count() const
    {
      return implementation->payload_record_count();
    }
    std::uint64_t TrajectoryStore::schedule_fingerprint() const
    {
      return implementation->schedule_fingerprint();
    }
    std::size_t TrajectoryStore::shared_archive_bytes() const
    {
      return implementation->shared_archive_bytes();
    }
    unsigned int TrajectoryStore::shared_segment_count() const
    {
      return implementation->shared_segment_count();
    }

    TrajectoryCleanupResult
    TrajectoryStore::reclaim_sparse_payloads_after(
      const unsigned int last_to_keep,
      const Diagnostics::FailureInjection<Diagnostics::CleanupFailurePoint> &injection)
    {
      return implementation->reclaim_sparse_payloads_after(last_to_keep, injection);
    }

    void
    TrajectoryStore::set_append_failure_injection_for_testing(
      const Diagnostics::FailureInjection<Diagnostics::AppendFailurePoint> &injection)
    {
      implementation->set_append_failure_injection_for_testing(injection);
    }

    TrajectoryCleanupResult
    TrajectoryStore::reclaim_records_after(
      const unsigned int first_to_remove,
      const Diagnostics::FailureInjection<Diagnostics::CleanupFailurePoint> &injection)
    {
      return implementation->reclaim_records_after(first_to_remove, injection);
    }

    void TrajectoryStore::remove_archive()
    {
      implementation->remove_archive();
    }
    const ReplayStepMetadata &TrajectoryStore::metadata(const unsigned int record) const
    {
      return implementation->metadata(record);
    }
    void TrajectoryStore::mark_last_record_terminal()
    {
      implementation->mark_last_record_terminal();
    }
    void TrajectoryStore::append_schedule(const ReplayStepMetadata &metadata)
    {
      implementation->append_schedule(metadata);
    }

    void
    TrajectoryStore::store_payload(const unsigned int record,
                                   const LinearAlgebra::BlockVector &solution,
                                   const unsigned int temperature_block,
                                   const std::vector<unsigned int> &composition_blocks,
                                   const unsigned int velocity_block,
                                   const unsigned int pressure_block)
    {
      implementation->store_payload(record, solution, temperature_block,
                                    composition_blocks, velocity_block,
                                    pressure_block);
    }

    void
    TrajectoryStore::store(const ReplayStepMetadata &metadata,
                           const LinearAlgebra::BlockVector &solution,
                           const unsigned int temperature_block,
                           const std::vector<unsigned int> &composition_blocks,
                           const unsigned int velocity_block,
                           const unsigned int pressure_block)
    {
      implementation->store(metadata, solution, temperature_block,
                            composition_blocks, velocity_block, pressure_block);
    }

    void
    TrajectoryStore::load(const unsigned int record,
                          LinearAlgebra::BlockVector &solution,
                          const unsigned int temperature_block,
                          const std::vector<unsigned int> &composition_blocks,
                          const unsigned int velocity_block,
                          const unsigned int pressure_block) const
    {
      implementation->load(record, solution, temperature_block,
                           composition_blocks, velocity_block, pressure_block);
    }

    TrajectoryStorageStatistics TrajectoryStore::storage_statistics() const
    {
      return implementation->storage_statistics();
    }

    std::size_t TrajectoryStore::memory_consumption() const
    {
      return sizeof(*this) + implementation->memory_consumption();
    }

  }
}
