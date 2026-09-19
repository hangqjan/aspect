/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#ifndef _aspect_postprocess_forward_boundary_reduced_stress_h
#define _aspect_postprocess_forward_boundary_reduced_stress_h

#include <aspect/postprocess/geoid_harmonics.h>

namespace aspect
{
  namespace Postprocess
  {
    /**
     * Harmonic projections of the pressure-gauge-reduced CBF normal stress
     * on the top and bottom boundaries.  Both coefficient fields use the
     * shared G5 degree-major real-harmonic layout.
     */
    struct BoundaryReducedStressHarmonics
    {
      SphericalHarmonicCoefficients surface;
      SphericalHarmonicCoefficients cmb;
    };

    /** The selected boundary for the reduced-stress forward/transpose pair. */
    enum class DynamicTopographyBoundary
    {
      surface,
      cmb
    };

    /**
     * Forward-only provider for the reduced stresses used by the coupled
     * self-gravity operator.  It deliberately projects the velocity block of
     * DynamicTopography::topography_vector(), whose documented contents are
     * the established CBF traction, rather than reassembling CBF traction.
     */
    template <int dim>
    class ForwardBoundaryReducedStressProvider
    {
      public:
        static BoundaryReducedStressHarmonics
        compute(const SimulatorAccess<dim> &simulator_access,
                unsigned int minimum_degree,
                unsigned int maximum_degree,
                double outer_radius,
                double inner_radius);
    };

    /**
     * Convert a coefficient-space seed for the reduced CBF stress into the
     * temperature-support-point topography seed consumed by the validated
     * DynamicTopographyBoundaryOperator.  This is the exact transpose of the
     * harmonic projection in ForwardBoundaryReducedStressProvider::compute:
     * the latter integrates over dOmega=dA/r^2 and the former accounts for
     * h=tau/((rho-rho_exterior)|g|).  @p pressure_weight is the matching
     * quadrature-space seed for the selected-boundary mean-pressure transpose.
     */
    template <int dim>
    void
    assemble_boundary_reduced_stress_transpose(
      const SimulatorAccess<dim> &simulator_access,
      const SphericalHarmonicCoefficients &coefficient_seed,
      const DynamicTopographyBoundary boundary,
      const double boundary_radius,
      const double density_exterior,
      LinearAlgebra::BlockVector &topography_adjoint,
      double &boundary_area,
      double &pressure_weight);
  }
}

#endif
