/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.
*/

#ifndef _aspect_adjoint_observation_spherical_harmonic_observation_h
#define _aspect_adjoint_observation_spherical_harmonic_observation_h

#include <aspect/postprocess/geoid_harmonics.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>

namespace aspect
{
  namespace Adjoint
  {
    namespace Observation
    {
      /** Cached coefficient data using ASPECT's canonical (l,m,C,S) layout. */
      template <int dim>
      class SphericalHarmonicObservation : public SimulatorAccess<dim>
      {
        public:
          SphericalHarmonicObservation();

          static void declare_parameters(ParameterHandler &prm);
          void parse_parameters(ParameterHandler &prm);
          void initialize(const unsigned int minimum_degree,
                          const unsigned int maximum_degree) const;

          const Postprocess::SphericalHarmonicCoefficients &coefficients() const;

        private:
          enum class Format
          {
            generic, aspect_native_geoid
          };

          Format format;
          std::string observation_file;
          std::string reference_output_directory;
          std::string time_selection;
          unsigned int timestep;
          mutable std::unique_ptr<Postprocess::SphericalHarmonicCoefficients> observation;

          std::string discover_file() const;
      };
    }
  }
}

#endif
