/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/adjoint/core/failure_injection.h>

namespace aspect
{
  namespace Adjoint
  {
    namespace Diagnostics
    {
      LifecycleFailurePoint
      parse_lifecycle_failure_point(const std::string &name)
      {
        if (name == "none")
          return LifecycleFailurePoint::none;
        if (name == "before checkpoint commit")
          return LifecycleFailurePoint::before_checkpoint_commit;
        if (name == "after checkpoint commit before cleanup")
          return LifecycleFailurePoint::after_checkpoint_commit_before_cleanup;
        if (name == "during cleanup")
          return LifecycleFailurePoint::during_cleanup;
        if (name == "after cleanup")
          return LifecycleFailurePoint::after_cleanup;
        AssertThrow(false, ExcMessage("Unknown adjoint lifecycle failure point: " + name));
        return LifecycleFailurePoint::none;
      }



      CheckpointFailurePoint
      parse_checkpoint_failure_point(const std::string &name)
      {
        if (name == "none")
          return CheckpointFailurePoint::none;
        if (name == "after temporary checkpoint directory creation")
          return CheckpointFailurePoint::after_temporary_directory_creation;
        if (name == "after rank-local checkpoint payload writing")
          return CheckpointFailurePoint::after_rank_payload_write;
        if (name == "after all payloads before metadata publication")
          return CheckpointFailurePoint::before_metadata_publication;
        if (name == "after checkpoint metadata publication")
          return CheckpointFailurePoint::after_metadata_publication;
        if (name == "after committed checkpoint directory before latest publication")
          return CheckpointFailurePoint::before_latest_publication;
        if (name == "after latest publication")
          return CheckpointFailurePoint::after_latest_publication;
        AssertThrow(false, ExcMessage("Unknown adjoint checkpoint failure point: " + name));
        return CheckpointFailurePoint::none;
      }



      CleanupFailurePoint
      parse_cleanup_failure_point(const std::string &name)
      {
        if (name == "none")
          return CleanupFailurePoint::none;
        if (name == "before cleanup starts")
          return CleanupFailurePoint::before_cleanup;
        if (name == "after rank-local cleanup payload deletion")
          return CleanupFailurePoint::after_rank_payload_deletion;
        if (name == "before cleanup collective success")
          return CleanupFailurePoint::before_collective_success;
        if (name == "after rank-local cleanup success before manifest update")
          return CleanupFailurePoint::before_manifest_update;
        if (name == "after physical segment deletion before manifest update")
          return CleanupFailurePoint::after_segment_deletion;
        if (name == "after cleanup manifest update")
          return CleanupFailurePoint::after_manifest_update;
        AssertThrow(false, ExcMessage("Unknown adjoint cleanup failure point: " + name));
        return CleanupFailurePoint::none;
      }



      AppendFailurePoint
      parse_append_failure_point(const std::string &name)
      {
        if (name == "none")
          return AppendFailurePoint::none;
        if (name == "after local serialization before shared-file write")
          return AppendFailurePoint::after_local_serialization;
        if (name == "after collective payload write before manifest")
          return AppendFailurePoint::after_collective_payload_write;
        if (name == "after payload write before manifest")
          return AppendFailurePoint::after_payload_write;
        if (name == "after manifest publication")
          return AppendFailurePoint::after_manifest_publication;
        AssertThrow(false, ExcMessage("Unknown adjoint append failure point: " + name));
        return AppendFailurePoint::none;
      }



      std::string
      to_string(const LifecycleFailurePoint point)
      {
        switch (point)
          {
            case LifecycleFailurePoint::none:
              return "none";
            case LifecycleFailurePoint::before_checkpoint_commit:
              return "before checkpoint commit";
            case LifecycleFailurePoint::after_checkpoint_commit_before_cleanup:
              return "after checkpoint commit before cleanup";
            case LifecycleFailurePoint::during_cleanup:
              return "during cleanup";
            case LifecycleFailurePoint::after_cleanup:
              return "after cleanup";
          }

        Assert(false, ExcInternalError());
        return "none";
      }
    }
  }
}
