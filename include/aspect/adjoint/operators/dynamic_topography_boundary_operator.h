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

#ifndef _aspect_adjoint_operators_dynamic_topography_boundary_operator_h
#define _aspect_adjoint_operators_dynamic_topography_boundary_operator_h

#include <aspect/adjoint/core/kernel_calculator.h>
#include <aspect/postprocess/forward_boundary_reduced_stress.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  template <int dim> class Simulator;

  namespace Adjoint
  {
    /**
     * Apply the transpose and the explicit material derivatives of the
     * dynamic-topography postprocessing chain at either the top or bottom
     * boundary. The boundary kind is explicit so that the CBF denominator,
     * outward normal, and pressure-mean correction remain tied to the
     * selected physical boundary.
     *
     * The supplied seed is a ghosted, distributed full-system block vector.
     * Its meaningful entries are temperature-component DoFs on the selected
     * boundary face
     * support points and contain the assembled derivative with respect to
     * topography. @p pressure_weight is the corresponding
     * quadrature-space seed contracted with
     * \f$((rho-rho_{above})|g|)^{-1}\f$, which is needed by the transpose of
     * the selected-boundary mean pressure subtraction.
     */
    template <int dim>
    class DynamicTopographyBoundaryOperator : public SimulatorAccess<dim>
    {
      public:
        explicit DynamicTopographyBoundaryOperator(const Simulator<dim> &simulator);

        /** Add the state transpose to an existing physical adjoint RHS. */
        void
        add_state_transpose(const LinearAlgebra::BlockVector &topography_adjoint,
                            const double boundary_area,
                            const double pressure_weight,
                            const double density_exterior,
                            LinearAlgebra::BlockVector &rhs,
                            const Postprocess::DynamicTopographyBoundary boundary) const;

        /** Add direct raw-density and raw-viscosity kernel contributions. */
        void
        add_direct_kernel_contributions(const LinearAlgebra::BlockVector &topography_adjoint,
                                        const LinearAlgebra::BlockVector &topography_vector,
                                        const double density_exterior,
                                        const std::string &objective_name,
                                        KernelRepository<dim> &kernels,
                                        const Postprocess::DynamicTopographyBoundary boundary) const;
    };
  }
}

#endif
