/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_adjoint_operators_stokes_adjoint_solver_h
#define _aspect_adjoint_operators_stokes_adjoint_solver_h

#include <aspect/global.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  template <int dim> class Simulator;

  namespace Adjoint
  {
    /**
     * Apply the transpose of the assembled Stokes operator using ASPECT's
     * existing Stokes solver. The operation temporarily installs homogeneous
     * adjoint constraints and is strongly exception safe.
     *
     * @ingroup Adjoint
     */
    template <int dim>
    class StokesAdjointSolver : public SimulatorAccess<dim>
    {
      public:
        explicit StokesAdjointSolver(Simulator<dim> &simulator);

        void
        solve(const LinearAlgebra::BlockVector &right_hand_side,
              LinearAlgebra::BlockVector &adjoint_solution);

      private:
        Simulator<dim> &simulator;
    };
  }
}

#endif
