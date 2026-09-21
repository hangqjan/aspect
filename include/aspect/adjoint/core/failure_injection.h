/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_adjoint_core_failure_injection_h
#define _aspect_adjoint_core_failure_injection_h

#include <aspect/global.h>

#include <string>

namespace aspect
{
  namespace Adjoint
  {
    namespace Diagnostics
    {
      /** Transaction boundaries at which a lifecycle test may interrupt. */
      enum class LifecycleFailurePoint
      {
        none,
        before_checkpoint_commit,
        after_checkpoint_commit_before_cleanup,
        during_cleanup,
        after_cleanup
      };

      /** Collective boundaries of the reverse-checkpoint transaction. */
      enum class CheckpointFailurePoint
      {
        none,
        after_temporary_directory_creation,
        after_rank_payload_write,
        before_metadata_publication,
        after_metadata_publication,
        before_latest_publication,
        after_latest_publication
      };

      /** Collective boundaries of a trajectory cleanup transaction. */
      enum class CleanupFailurePoint
      {
        none,
        before_cleanup,
        after_rank_payload_deletion,
        before_collective_success,
        before_manifest_update,
        after_segment_deletion,
        after_manifest_update
      };

      /** Collective boundaries of a shared trajectory append transaction. */
      enum class AppendFailurePoint
      {
        none,
        after_local_serialization,
        after_collective_payload_write,
        after_payload_write,
        after_manifest_publication
      };

      /**
       * Test-only description of one rank-local failure. A negative rank
       * consistently selects rank zero, matching the historical test driver.
       */
      template <typename FailurePoint>
      struct FailureInjection
      {
        FailurePoint point = FailurePoint::none;
        int rank = -1;

        bool
        applies_to_rank(const unsigned int current_rank) const
        {
          const unsigned int selected_rank =
            rank < 0 ? 0 : static_cast<unsigned int>(rank);
          return point != FailurePoint::none && current_rank == selected_rank;
        }
      };

      LifecycleFailurePoint parse_lifecycle_failure_point(const std::string &name);
      CheckpointFailurePoint parse_checkpoint_failure_point(const std::string &name);
      CleanupFailurePoint parse_cleanup_failure_point(const std::string &name);
      AppendFailurePoint parse_append_failure_point(const std::string &name);

      std::string to_string(const LifecycleFailurePoint point);
    }
  }
}

#endif
