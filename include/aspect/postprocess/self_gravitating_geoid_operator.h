/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#ifndef _aspect_postprocess_self_gravitating_geoid_operator_h
#define _aspect_postprocess_self_gravitating_geoid_operator_h

#include <aspect/postprocess/geoid_harmonics.h>

#include <array>
#include <utility>

namespace aspect
{
  namespace Postprocess
  {
    /** Raw density moments used by the coupled self-gravity forward system. */
    struct SelfGravitatingDensityMoments
    {
      SphericalHarmonicCoefficients at_surface;
      SphericalHarmonicCoefficients at_cmb;
    };

    /** Inputs whose material dependence defines the degree-wise 4x4 matrix. */
    struct SelfGravitatingGeoidOperatorInputs
    {
      SelfGravitatingDensityMoments density_moments;
      SphericalHarmonicCoefficients surface_reduced_stress;
      SphericalHarmonicCoefficients cmb_reduced_stress;
      double outer_radius;
      double inner_radius;
      double surface_gravity;
      double top_boundary_density;
      double bottom_boundary_density;
      double density_above;
      double density_below;
    };

    /** The four explicit fields x=(surface topo, CMB topo, surface geoid, CMB geoid). */
    struct SelfGravitatingGeoidCoefficients
    {
      SphericalHarmonicCoefficients surface_topography;
      SphericalHarmonicCoefficients cmb_topography;
      SphericalHarmonicCoefficients surface_geoid;
      SphericalHarmonicCoefficients cmb_geoid;
      double maximum_normalized_residual = 0.0;
    };

    /**
     * Spectral reverse seeds for the four physical channels of the forward
     * right-hand side (-tau_t, +tau_b, Gamma I_t, Gamma I_b).
     */
    struct SelfGravitatingGeoidRHSSeeds
    {
      SphericalHarmonicCoefficients surface_reduced_stress;
      SphericalHarmonicCoefficients cmb_reduced_stress;
      SphericalHarmonicCoefficients surface_density_moment;
      SphericalHarmonicCoefficients cmb_density_moment;
    };

    /**
     * Solution of A_l^T z=q_x. The four z fields correspond, in order, to
     * surface/CMB topography and surface/CMB geoid forward outputs.
     */
    struct SelfGravitatingGeoidTransposeCoefficients
    {
      SphericalHarmonicCoefficients surface_topography;
      SphericalHarmonicCoefficients cmb_topography;
      SphericalHarmonicCoefficients surface_geoid;
      SphericalHarmonicCoefficients cmb_geoid;
      SelfGravitatingGeoidRHSSeeds rhs_seeds;
      double maximum_normalized_residual = 0.0;
    };

    /** The two raw-density-controlled boundary means entering the coupled matrix. */
    enum class SelfGravitatingMatrixDensityParameter
    {
      top_boundary_density,
      bottom_boundary_density
    };

    /** Accumulated -z^T (dA/d rho_bar) x contractions over all real modes. */
    struct SelfGravitatingMatrixDensitySensitivity
    {
      double top_boundary_density = 0.0;
      double bottom_boundary_density = 0.0;
    };

    /**
     * Coupled self-gravity spectral operator. It implements the audited l>=2
     * real-harmonic 4x4 forward solve and its fixed-matrix spectral transpose.
     * Fixed-A spectral propagation is complemented by explicit density-mean
     * matrix derivatives; geometry, gravity, and external-density derivatives
     * remain outside this class.
     */
    class SelfGravitatingGeoidOperator
    {
      public:
        SelfGravitatingGeoidCoefficients
        apply(const SelfGravitatingGeoidOperatorInputs &inputs) const;

        /**
         * Solve A_l^T z=q_x for arbitrary seeds on the four forward outputs.
         * The returned RHS seeds account for the fixed coupled-RHS signs and
         * Gamma_l factors, but do not apply CBF or density-space transposes.
         */
        SelfGravitatingGeoidTransposeCoefficients
        apply_transpose(const SelfGravitatingGeoidOperatorInputs &inputs,
                        const SelfGravitatingGeoidCoefficients &solution_seed) const;

        /** Exact dA_l/d rho_bar for one of the two boundary density means. */
        static std::array<std::array<double, 4>, 4>
        matrix_density_derivative_for_degree(
          const SelfGravitatingGeoidOperatorInputs &inputs,
          unsigned int degree,
          SelfGravitatingMatrixDensityParameter parameter);

        /** Accumulate the contractions -z^T(dA/d rho_bar)x. */
        SelfGravitatingMatrixDensitySensitivity
        matrix_density_sensitivity(const SelfGravitatingGeoidOperatorInputs &inputs,
                                   const SelfGravitatingGeoidCoefficients &forward_solution,
                                   const SelfGravitatingGeoidTransposeCoefficients &transpose_solution) const;

      private:
        static std::array<std::array<double, 4>, 4>
        matrix_for_degree(const SelfGravitatingGeoidOperatorInputs &inputs,
                          unsigned int degree);

        static std::array<std::array<double, 4>, 4>
        inverse(const std::array<std::array<double, 4>, 4> &matrix);
    };

    /** Project the two raw volume-density moments I_t and I_b in the shared harmonic layout. */
    template <int dim>
    SelfGravitatingDensityMoments
    compute_self_gravitating_density_moments(const SimulatorAccess<dim> &simulator_access,
                                             unsigned int minimum_degree,
                                             unsigned int maximum_degree,
                                             double outer_radius,
                                             double inner_radius);

    /**
     * Assemble all forward inputs from the active ASPECT postprocessors and
     * apply the coupled operator.  This is intentionally forward-only: it is
     * the forward production entry point, while the adjoint uses the
     * corresponding transpose below.
     */
    template <int dim>
    SelfGravitatingGeoidOperatorInputs
    compute_self_gravitating_geoid_inputs(const SimulatorAccess<dim> &simulator_access,
                                          unsigned int minimum_degree,
                                          unsigned int maximum_degree);

    template <int dim>
    SelfGravitatingGeoidCoefficients
    compute_self_gravitating_geoid_harmonics(const SimulatorAccess<dim> &simulator_access,
                                             unsigned int minimum_degree,
                                             unsigned int maximum_degree);
  }
}

#endif
