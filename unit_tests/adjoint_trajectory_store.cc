/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.
*/

#include "common.h"

#include <aspect/adjoint/time_dependent/trajectory_store.h>

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
