/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of the ASPECT code.
*/

#ifndef _aspect_adjoint_operators_boundary_conditions_h
#define _aspect_adjoint_operators_boundary_conditions_h

#include <aspect/adjoint/core/state.h>

namespace aspect
{
  namespace Adjoint
  {
    /**
     * Rebuild the constraint space used by an adjoint calculation from an
     * accepted forward snapshot.  The forward constraint graph is retained
     * exactly (in particular the affine Cartesian-component relations used by
     * curved free-slip boundaries), while every inhomogeneity is removed.
     *
     * Fixed boundary data are not controls in the v1 adjoint. Consequently
     * their variations, and hence the corresponding adjoint essential values,
     * are zero. Fixed heat-flux boundaries do not contribute constraint lines
     * and therefore remain natural boundaries here.
     */
    template <int dim>
    class BoundaryConditions
    {
      public:
        void
        build_homogeneous_constraints(const ForwardStateView<dim> &snapshot,
                                      AffineConstraints<double> &constraints) const
        {
          build_homogeneous_constraints(snapshot.constraints, constraints);
        }

        void
        build_homogeneous_constraints(const AffineConstraints<double> &forward_constraints,
                                      AffineConstraints<double> &constraints) const
        {
          constraints.copy_from(forward_constraints);

          // Do not use set_zero() on a solution vector here. A free-slip
          // constraint on a curved face can be an affine relation between
          // velocity components; preserving its entries is essential.
          for (const auto &line : constraints.get_lines())
            constraints.set_inhomogeneity(line.index, 0.0);

          // Keep this verification next to the transformation: it guards
          // against a future "homogenization" shortcut that would retain
          // only constrained indices and silently drop affine entries (for
          // example curved free-slip, hanging-node, or periodic relations).
          const auto &forward_lines = forward_constraints.get_lines();
          const auto &homogeneous_lines = constraints.get_lines();
          AssertThrow(forward_lines.size() == homogeneous_lines.size(),
                      ExcInternalError());
          for (unsigned int line_number = 0; line_number < forward_lines.size(); ++line_number)
            {
              AssertThrow(forward_lines[line_number].index
                          == homogeneous_lines[line_number].index,
                          ExcInternalError());
              AssertThrow(forward_lines[line_number].entries
                          == homogeneous_lines[line_number].entries,
                          ExcInternalError());
              AssertThrow(homogeneous_lines[line_number].inhomogeneity == 0.0,
                          ExcInternalError());
            }

          constraints.close();
        }

        void
        build_stokes_adjoint_constraints(const ForwardStateView<dim> &snapshot,
                                         AffineConstraints<double> &constraints) const
        {
          build_homogeneous_constraints(snapshot, constraints);
        }

        void
        build_temperature_adjoint_constraints(const ForwardStateView<dim> &snapshot,
                                              AffineConstraints<double> &constraints) const
        {
          build_homogeneous_constraints(snapshot, constraints);
        }
    };
  }
}

#endif
