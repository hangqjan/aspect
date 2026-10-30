/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#ifndef _aspect_adjoint_objective_geoid_objective_h
#define _aspect_adjoint_objective_geoid_objective_h

#include <aspect/adjoint/objective/objective_functional.h>
#include <aspect/adjoint/observation/spherical_harmonic_observation.h>
#include <aspect/postprocess/geoid_harmonics.h>

namespace aspect
{
  namespace Adjoint
  {
    /**
     * A coefficient-space geoid objective for the non-self-gravitating
     * mantle-density, surface-dynamic-topography, and CMB-dynamic-topography
     * contributions. The topography branches contribute to one shared
     * physical Stokes adjoint; the mantle-density branch is a direct
     * observation-operator transpose.
     */
    template <int dim>
    class GeoidObjective : public Objective::Interface<dim>,
      public Objective::InstantaneousInterface<dim>
    {
      public:
        static
        void
        declare_parameters (ParameterHandler &prm);

        void
        parse_parameters (ParameterHandler &prm) override;

        void initialize_simulator(const Simulator<dim> &simulator) override;

        double
        evaluate() const override;

        /** Assemble all enabled dynamic-topography contributions to the physical Stokes adjoint. */
        void
        assemble_adjoint_right_hand_side(LinearAlgebra::BlockVector &right_hand_side) const override;

        bool
        requires_physical_stokes_adjoint() const override;

        std::set<std::string>
        required_forward_postprocessors() const override;

        void
        add_direct_kernel_contributions(const std::string &objective_name,
                                        KernelRepository<dim> &kernels) const override;

      private:
        /** Return the spherical-shell radius and the surface gravity magnitude. */
        std::pair<double, double>
        operator_context() const;

        Postprocess::SphericalHarmonicCoefficients
        forward_coefficients() const;

        /** Assemble the G0-compatible seed for the surface topography branch. */
        void
        assemble_surface_topography_seed(LinearAlgebra::BlockVector &topography_adjoint,
                                         double &top_area,
                                         double &surface_pressure_weight) const;

        /** Assemble the G0-compatible seed for the CMB topography branch. */
        void
        assemble_cmb_topography_seed(LinearAlgebra::BlockVector &topography_adjoint,
                                     double &bottom_area,
                                     double &bottom_pressure_weight) const;

        /** Construct q_N=W(N-N_obs) in the shared coefficient layout. */
        Postprocess::SphericalHarmonicCoefficients
        coefficient_seed() const;

        unsigned int minimum_degree;
        unsigned int maximum_degree;
        mutable Postprocess::SphericalHarmonicCoefficients observed_coefficients;
        mutable Observation::SphericalHarmonicObservation<dim> observation;
        bool use_file_observation;
        /** One nonnegative diagonal weight per degree/order, shared by cos/sin. */
        std::vector<double> weights;
        bool include_surface_topography;
        bool include_cmb_topography;
        double density_above;
        double density_below;
    };
  }
}

#endif
