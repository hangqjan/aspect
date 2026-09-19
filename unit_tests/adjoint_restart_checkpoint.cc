/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.
*/

#include "common.h"

#include <aspect/adjoint/time_dependent/adjoint_restart_checkpoint.h>

#include <filesystem>

TEST_CASE("adjoint restart checkpoint transaction")
{
  using namespace aspect::Adjoint;

  const std::filesystem::path directory =
    std::filesystem::temp_directory_path() / "aspect-adjoint-restart-unit-test";
  std::error_code error;
  std::filesystem::remove_all(directory, error);
  REQUIRE_FALSE(error);
  std::filesystem::create_directories(directory, error);
  REQUIRE_FALSE(error);

  RestartCheckpointManager checkpoints(directory.string() + "/", MPI_COMM_SELF);
  REQUIRE_FALSE(checkpoints.has_committed_checkpoint());

  AdjointRestartState written;
  written.next_reverse_step = 7;
  written.cleanup_frontier = 4;
  written.trajectory_anchor_spacing = 3;
  written.required_sparse_anchor = 6;
  written.trajectory_schedule_records = 9;
  written.trajectory_schedule_fingerprint = 0x12345678ULL;
  written.trajectory_storage_mode = TrajectoryStorageMode::stokes_cached;
  written.objective_values.emplace("terminal temperature", 2.5);

  const CheckpointWriteResult result = checkpoints.write(written);
  REQUIRE(result.committed);
  REQUIRE(result.latest_published);
  REQUIRE(written.generation == 1);
  REQUIRE(checkpoints.has_committed_checkpoint());

  AdjointRestartState restored;
  const std::size_t bytes = checkpoints.restore(restored);
  REQUIRE(bytes > 0);
  REQUIRE(restored.generation == written.generation);
  REQUIRE(restored.next_reverse_step == written.next_reverse_step);
  REQUIRE(restored.cleanup_frontier == written.cleanup_frontier);
  REQUIRE(restored.trajectory_anchor_spacing == written.trajectory_anchor_spacing);
  REQUIRE(restored.required_sparse_anchor == written.required_sparse_anchor);
  REQUIRE(restored.trajectory_schedule_records == written.trajectory_schedule_records);
  REQUIRE(restored.trajectory_schedule_fingerprint
          == written.trajectory_schedule_fingerprint);
  REQUIRE(restored.trajectory_storage_mode == TrajectoryStorageMode::stokes_cached);
  REQUIRE(restored.objective_values.at("terminal temperature") == Approx(2.5));

  checkpoints.remove_temporary_data();
  REQUIRE_FALSE(checkpoints.has_committed_checkpoint());
  std::filesystem::remove_all(directory, error);
  REQUIRE_FALSE(error);
}
