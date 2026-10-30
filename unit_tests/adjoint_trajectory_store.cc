/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.
*/

#include "common.h"

#include <aspect/adjoint/time_dependent/trajectory_store.h>

#include <deal.II/base/mpi.h>

#include <filesystem>

TEST_CASE("adjoint trajectory manifest and terminal schedule")
{
  using namespace aspect::Adjoint;

  const std::filesystem::path directory =
    std::filesystem::temp_directory_path() / "aspect-adjoint-trajectory-unit-test";
  std::error_code error;
  std::filesystem::remove_all(directory, error);
  REQUIRE_FALSE(error);
  std::filesystem::create_directories(directory, error);
  REQUIRE_FALSE(error);

  std::uint64_t fingerprint_before_terminal = 0;
  {
    TrajectoryStore trajectory(TrajectoryStorageMode::minimal,
                               directory.string() + "/",
                               MPI_COMM_SELF,
                               false,
                               2);
    trajectory.append_schedule({0, 0.0, 0.0, 0.0, 0.25, true});
    trajectory.append_schedule({1, 0.25, 0.25, 0.0, 0.2, true});

    REQUIRE(trajectory.size() == 2);
    REQUIRE_FALSE(trajectory.has_record(0));
    REQUIRE(trajectory.metadata(1).outgoing_time_step == Approx(0.2));
    fingerprint_before_terminal = trajectory.schedule_fingerprint();

    trajectory.mark_last_record_terminal();
    REQUIRE_FALSE(trajectory.metadata(1).has_outgoing_step);
    REQUIRE(trajectory.metadata(1).outgoing_time_step == Approx(0.0));
    REQUIRE(trajectory.schedule_fingerprint() != fingerprint_before_terminal);
  }

  {
    TrajectoryStore reopened(TrajectoryStorageMode::minimal,
                             directory.string() + "/",
                             MPI_COMM_SELF,
                             true,
                             2);
    REQUIRE(reopened.size() == 2);
    REQUIRE(reopened.metadata(0).has_outgoing_step);
    REQUIRE_FALSE(reopened.metadata(1).has_outgoing_step);
    REQUIRE(reopened.payload_record_count() == 0);
    reopened.remove_archive();
  }

  REQUIRE_FALSE(std::filesystem::exists(directory / "adjoint_trajectory"));
  std::filesystem::remove_all(directory, error);
  REQUIRE_FALSE(error);
}


namespace adjoint_trajectory_store_mpi_tests
{
  dealii::IndexSet
  owned_range(const dealii::types::global_dof_index global_size,
              const dealii::types::global_dof_index begin,
              const dealii::types::global_dof_index end)
  {
    dealii::IndexSet owned(global_size);
    owned.add_range(begin, end);
    return owned;
  }


  aspect::LinearAlgebra::BlockVector
  make_solution(const dealii::IndexSet &owned,
                const double value_offset)
  {
    aspect::LinearAlgebra::BlockVector result({owned, owned, owned, owned}, MPI_COMM_WORLD);
    for (unsigned int block = 0; block < result.n_blocks(); ++block)
      {
        for (const auto index : result.block(block).locally_owned_elements())
          result.block(block)[index] = value_offset + 10.0 * block + index;
        result.block(block).compress(dealii::VectorOperation::insert);
      }
    return result;
  }


  void
  check_stored_values(const aspect::LinearAlgebra::BlockVector &solution,
                      const double value_offset)
  {
    for (unsigned int block = 0; block < solution.n_blocks(); ++block)
      for (const auto index : solution.block(block).locally_owned_elements())
        REQUIRE(solution.block(block)[index] == Approx(value_offset + 10.0 * block + index));
  }


  TEST_CASE("MPI trajectory payloads use one collective order for unequal local partitions")
  {
    using namespace aspect::Adjoint;

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    if (n_ranks != 2)
      {
        SUCCEED("This focused collective-I/O regression is exercised with two MPI ranks.");
        return;
      }

    struct Case
    {
      const char *name;
      dealii::types::global_dof_index global_size;
      dealii::types::global_dof_index rank_zero_end;
    };
    const std::vector<Case> cases = {
      {"both-nonempty", 6, 3},
      {"rank-one-empty", 6, 6},
      {"unequal-payloads", 7, 1}
    };

    for (const Case &test_case : cases)
      {
        const dealii::types::global_dof_index begin = rank == 0 ? 0 : test_case.rank_zero_end;
        const dealii::types::global_dof_index end = rank == 0 ? test_case.rank_zero_end : test_case.global_size;
        const dealii::IndexSet owned = owned_range(test_case.global_size, begin, end);
        const std::filesystem::path directory =
          std::filesystem::temp_directory_path()
          / (std::string("aspect-adjoint-trajectory-mpi-") + test_case.name);

        if (rank == 0)
          {
            std::error_code error;
            std::filesystem::remove_all(directory, error);
            REQUIRE_FALSE(error);
          }
        MPI_Barrier(MPI_COMM_WORLD);

        const double value_offset = 100.0 * (&test_case - cases.data() + 1);
        {
          TrajectoryStore trajectory(TrajectoryStorageMode::stokes_cached,
                                     directory.string() + "/",
                                     MPI_COMM_WORLD,
                                     false,
                                     2);
          auto solution = make_solution(owned, value_offset);
          trajectory.store({0, 0.0, 0.0, 0.0, 1.0, true}, solution, 0, {1}, 2, 3);

          REQUIRE(trajectory.has_record(0));
          REQUIRE(trajectory.payload_record_count() == 1);
          const std::size_t local_vector_bytes =
            sizeof(std::uint64_t) + owned.n_elements() * sizeof(double);
          const auto statistics = trajectory.storage_statistics();
          REQUIRE(statistics.temperature_bytes == local_vector_bytes);
          REQUIRE(statistics.composition_bytes == local_vector_bytes);
          REQUIRE(statistics.velocity_bytes == local_vector_bytes);
          REQUIRE(statistics.pressure_bytes == local_vector_bytes);

          const std::size_t expected_payload_size =
            dealii::Utilities::MPI::sum(4 * local_vector_bytes, MPI_COMM_WORLD);
          std::size_t payload_file_size = 0;
          if (rank == 0)
            {
              std::error_code error;
              const std::filesystem::path payload_file =
                directory / "adjoint_trajectory" / "trajectory_000000.bin";
              REQUIRE(std::filesystem::exists(payload_file, error));
              REQUIRE_FALSE(error);
              payload_file_size = std::filesystem::file_size(payload_file, error);
              REQUIRE_FALSE(error);
            }
          payload_file_size = dealii::Utilities::MPI::broadcast(MPI_COMM_WORLD,
                                                                 payload_file_size, 0);
          REQUIRE(payload_file_size == expected_payload_size);

          for (unsigned int block = 0; block < solution.n_blocks(); ++block)
            solution.block(block) = 0.0;
          trajectory.load(0, solution, 0, {1}, 2, 3);
          check_stored_values(solution, value_offset);
          trajectory.remove_archive();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
          REQUIRE_FALSE(std::filesystem::exists(directory / "adjoint_trajectory"));
      }
  }
}
