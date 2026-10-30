/*
 Copyright (C) 2015 - 2022 by the authors of the ASPECT code.

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


#ifndef _aspect_postprocess_geoid_h
#define _aspect_postprocess_geoid_h

#include <aspect/postprocess/geoid_harmonics.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>


namespace aspect
{
  namespace Postprocess
  {
    /**
     * A postprocessor that computes the geoid anomaly at the surface.
     *
     * @ingroup Postprocessing
     */
    template <int dim>
    class Geoid : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Evaluate the solution for the geoid in spherical harmonics and then transfer it to grid output.
         */
        std::pair<std::string,std::string>
        execute (TableHandler &statistics) override;

        /**
         * Register with the simulator the other postprocessors that we need
         * (namely: dynamic topography).
         */
        std::list<std::string>
        required_other_postprocessors() const override;

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        void
        parse_parameters (ParameterHandler &prm) override;

        /**
         * Find if the top or bottom boundaries are free surfaces.
         */
        void initialize() override;

        /**
         * Evaluate the geoid solution at a point. The evaluation point
         * must be outside of the model domain, and it must be called
         * after execute().
         */
        double
        evaluate (const Point<dim> &) const;

      private:
        /** Harmonic topography and its adjacent-layer mean density. */
        struct BoundaryTopographyHarmonics
        {
          double average_density = 0.0;
          SphericalHarmonicCoefficients topography;
        };

        /** Surface and CMB inputs prepared for conventional geoid scaling. */
        struct TopographyHarmonics
        {
          BoundaryTopographyHarmonics surface;
          BoundaryTopographyHarmonics cmb;
        };

        /**
         * Parameters to set the maximum and minimum degree when computing geoid from spherical harmonics
         */
        unsigned int maximum_degree;
        unsigned int minimum_degree;

        /**
         * A parameter to control whether to output the data in geographical coordinates.
         * If true, output the data in longitudes and latitudes; if false, output data in x y z.
         */
        bool output_in_lat_lon;

        /**
         * A parameter to control whether to output the spherical harmonic coefficients of the geoid anomaly
         */
        bool output_geoid_anomaly_coefficients;

        /**
         * A parameter to control whether to output the spherical harmonic coefficients of the surface topography contribution
         */
        bool output_surface_topography_contribution_coefficients;

        /**
         * A parameter to control whether to output the spherical harmonic coefficients of the CMB topography contribution
         */
        bool output_cmb_topography_contribution_coefficients;

        /**
         * A parameter to control whether to output the spherical harmonic coefficients of the density anomaly
         */
        bool output_density_anomaly_contribution_coefficients;

        /**
         * A parameter to control whether to output the free-air gravity anomaly
         */
        bool output_gravity_anomaly;

        /**
         * Parameters to set the density value out of the surface and CMB boundary
         */
        double density_above;
        double density_below;

        /**
         * A parameter to control whether to include the surface topography contribution on geoid
         */
        bool include_surface_topography_contribution;

        /**
         * A parameter to control whether to include the CMB topography contribution on geoid
         */
        bool include_cmb_topography_contribution;

        /**
         * A parameter to specify if the top boundary is an active free surface
         */
        bool use_free_surface_topography;

        /**
         * A parameter to specify if the bottom boundary is an active free surface
         */
        bool use_free_cmb_topography;

        /**
         * Function to compute the real spherical harmonic coefficients (cos and sin part) from min degree to max degree
         * The input spherical_function is a vector of vectors.
         * The inner vector stores theta, phi, spherical infinitesimal, and function value on a spherical surface.
         * The outer vector stores the inner vector associated with each quadrature point on a spherical surface.
         */
        SphericalHarmonicCoefficients
        to_spherical_harmonic_coefficients(const std::vector<std::vector<double>> &spherical_function) const;

        /**
         * Function to compute the surface and CMB topography contribution in spherical harmonic expansion
         * The input inner and outer radius are used to calculate spherical infinitesimal area, i.e., sin(theta)*d_theta*d_phi
         * associated with each quadrature point on surface and bottom respectively.
         * This function returns a pair containing surface and CMB topography's real spherical harmonic coefficients (cos and sin part)
         * from min degree to max degree. The surface and CMB average density are also included as the first single element of each subpair respectively.
         * The topography is based on the dynamic topography postprocessor in case of no free surface,
         * and based on the real surface from the geometry model in case of a free surface.
         */
        TopographyHarmonics
        topography_contribution(const double &outer_radius,
                                const double &inner_radius) const;

        /** Complete, validated spherical-harmonic representation of the geoid. */
        SphericalHarmonicCoefficients geoid_coefficients;
    };
  }
}


#endif
