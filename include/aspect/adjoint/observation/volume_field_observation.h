/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.
*/

#ifndef _aspect_adjoint_observation_volume_field_observation_h
#define _aspect_adjoint_observation_volume_field_observation_h

#include <aspect/adjoint/observation/full_field_observation.h>

namespace aspect
{
  namespace Adjoint
  {
    namespace Observation
    {
      /**
       * Compatibility name for the Stage 1--5 full-field observation API.
       * New code should use FullFieldObservation, which makes the mathematical
       * representation explicit without changing source semantics.
       */
      template <int dim>
      using VolumeFieldObservation = FullFieldObservation<dim>;
    }
  }
}

#endif
