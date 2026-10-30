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

#ifndef _aspect_postprocess_geoid_harmonics_h
#define _aspect_postprocess_geoid_harmonics_h

#include <aspect/global.h>
#include <aspect/simulator_access.h>

#include <deal.II/lac/vector.h>

#include <iosfwd>
#include <vector>

namespace aspect
{
  namespace Postprocess
  {
    /**
     * Real spherical-harmonic coefficients in degree-major order. For every
     * degree l=l_min,...,l_max, orders m=0,...,l are stored consecutively;
     * cosine and sine components use the same index. The index is
     * (l*(l+1)-l_min*(l_min+1))/2+m. Harmonics are evaluated with ASPECT's
     * fully normalized real convention (Dahlen and Tromp, Appendix B.6),
     * including the Condon--Shortley phase.
     */
    class SphericalHarmonicCoefficients
    {
      public:
        /** Construct the valid degree-zero layout. */
        SphericalHarmonicCoefficients();

        /** Construct a zero-initialized coefficient set for [l_min,l_max]. */
        SphericalHarmonicCoefficients(const unsigned int minimum_degree,
                                      const unsigned int maximum_degree);

        /** Reset the degree range and zero all coefficient components. */
        void
        reinit(const unsigned int minimum_degree,
               const unsigned int maximum_degree);

        unsigned int minimum_degree() const;
        unsigned int maximum_degree() const;
        unsigned int size() const;

        double &cosine(const unsigned int index);
        double cosine(const unsigned int index) const;
        double &sine(const unsigned int index);
        double sine(const unsigned int index) const;

        const std::vector<double> &cosine_coefficients() const;
        const std::vector<double> &sine_coefficients() const;

        /** Replace both components after validating their common layout. */
        void
        set_coefficients(const std::vector<double> &cosine,
                         const std::vector<double> &sine);

        /** Return the shared storage index for a degree/order pair. */
        unsigned int
        coefficient_index (const unsigned int degree,
                           const unsigned int order) const;

        /** Evaluate the real spherical-harmonic expansion at (theta,phi). */
        double
        evaluate (const double theta,
                  const double phi) const;

      private:
        static unsigned int
        coefficient_count(const unsigned int minimum_degree,
                          const unsigned int maximum_degree);

        unsigned int stored_minimum_degree;
        unsigned int stored_maximum_degree;
        std::vector<double> cosine_values;
        std::vector<double> sine_values;
    };



    /**
     * Write one coefficient set as degree, order, cosine, and sine columns.
     *
     * Iteration follows the invariant degree-major layout owned by
     * SphericalHarmonicCoefficients, so callers never maintain a parallel
     * manual coefficient index.
     */
    void
    write_spherical_harmonic_coefficients(
      std::ostream &output,
      const SphericalHarmonicCoefficients &coefficients);



    /**
     * Compute the density-only contribution to the geoid at @p reference_radius.
     * The result is the forward operator A_rho applied to the current material
     * density field, including the r^{-1}(r/R)^{l+1} radial weighting,
     * 4*pi*G/(g_s*(2*l+1)) scaling, and MPI reduction.
     */
    template <int dim>
    SphericalHarmonicCoefficients
    compute_density_geoid_harmonics (const SimulatorAccess<dim> &simulator_access,
                                     const unsigned int minimum_degree,
                                     const unsigned int maximum_degree,
                                     const double reference_radius,
                                     const double surface_gravity);

    /**
     * Apply the discrete transpose of compute_density_geoid_harmonics() to
     * @p coefficient_seed. The returned values are cell averages, indexed by
     * active-cell index, and use the same volume quadrature as the forward
     * density integral. Only locally owned cell entries are populated.
     */
    template <int dim>
    Vector<double>
    compute_density_geoid_transpose (const SimulatorAccess<dim> &simulator_access,
                                     const SphericalHarmonicCoefficients &coefficient_seed,
                                     const double reference_radius,
                                     const double surface_gravity);

    /**
     * Apply the discrete density transpose of one of the two raw volume
     * moments used by the coupled self-gravitating geoid operator.  The
     * surface moment uses r^l/R^(l+1), whereas the CMB moment uses
     * R_c^l/r^(l+1).  Returned values are locally owned cell averages,
     * consistent with compute_density_geoid_transpose().
     */
    enum class SelfGravitatingDensityMoment
    {
      surface,
      cmb
    };

    template <int dim>
    Vector<double>
    compute_self_gravitating_density_moment_transpose(
      const SimulatorAccess<dim> &simulator_access,
      const SphericalHarmonicCoefficients &coefficient_seed,
      const double outer_radius,
      const double inner_radius,
      const SelfGravitatingDensityMoment moment);

    /** Compute the surface dynamic-topography contribution to the geoid. */
    SphericalHarmonicCoefficients
    scale_surface_topography_geoid_harmonics(
      const SphericalHarmonicCoefficients &topography_coefficients,
      const double reference_radius,
      const double surface_gravity,
      const double surface_density_contrast);

    /** Compute the CMB topography contribution from projected coefficients. */
    SphericalHarmonicCoefficients
    scale_cmb_topography_geoid_harmonics(
      const SphericalHarmonicCoefficients &topography_coefficients,
      const double reference_radius,
      const double cmb_radius,
      const double surface_gravity,
      const double cmb_density_contrast);

    /** Compute the surface dynamic-topography contribution to the geoid. */
    template <int dim>
    SphericalHarmonicCoefficients
    compute_surface_topography_geoid_harmonics(
      const SimulatorAccess<dim> &simulator_access,
      const LinearAlgebra::BlockVector &topography,
      const unsigned int minimum_degree,
      const unsigned int maximum_degree,
      const double reference_radius,
      const double surface_gravity,
      const double surface_density_contrast);

    /** Apply the coefficient-space transpose of the surface geoid operator. */
    SphericalHarmonicCoefficients
    compute_surface_topography_geoid_transpose(
      const SphericalHarmonicCoefficients &coefficient_seed,
      const double reference_radius,
      const double surface_gravity,
      const double surface_density_contrast);

    /** Compute the CMB dynamic-topography contribution to the geoid. */
    template <int dim>
    SphericalHarmonicCoefficients
    compute_cmb_topography_geoid_harmonics(
      const SimulatorAccess<dim> &simulator_access,
      const LinearAlgebra::BlockVector &topography,
      const unsigned int minimum_degree,
      const unsigned int maximum_degree,
      const double reference_radius,
      const double cmb_radius,
      const double surface_gravity,
      const double cmb_density_contrast);

    /** Apply the coefficient-space transpose of the CMB geoid operator. */
    SphericalHarmonicCoefficients
    compute_cmb_topography_geoid_transpose(
      const SphericalHarmonicCoefficients &coefficient_seed,
      const double reference_radius,
      const double cmb_radius,
      const double surface_gravity,
      const double cmb_density_contrast);
  }
}

#endif
