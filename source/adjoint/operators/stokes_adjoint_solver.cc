/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/adjoint/operators/boundary_conditions.h>
#include <aspect/adjoint/core/simulator_state.h>
#include <aspect/adjoint/operators/stokes_adjoint_solver.h>
#include <aspect/simulator.h>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    StokesAdjointSolver<dim>::StokesAdjointSolver(Simulator<dim> &simulator_object)
      :
      simulator(simulator_object)
    {
      this->initialize_simulator(simulator_object);
    }



    template <int dim>
    void
    StokesAdjointSolver<dim>::solve(
      const LinearAlgebra::BlockVector &right_hand_side,
      LinearAlgebra::BlockVector &adjoint_solution)
    {
      // Under the supported incompressible formulation the linearized mixed
      // Stokes operator has the saddle-point form
      //
      //       [ A  B^T ] [u] = [f],
      //       [ B   0  ] [p]   [g],
      //
      // with symmetric viscous block A and matching divergence/gradient
      // blocks. Its discrete transpose is therefore the same constrained
      // operator. We can reuse Simulator::solve_stokes after replacing the
      // primal RHS by the objective seed and homogenizing all affine boundary
      // data. ScopedSimulatorState restores every overwritten solver field on
      // normal return and exception paths.
      ScopedSimulatorState<dim> simulator_state(simulator);
      const AffineConstraints<double> forward_constraints(simulator.current_constraints);
      const unsigned int velocity_block = this->introspection().block_indices.velocities;
      const unsigned int pressure_block = this->introspection().block_indices.pressure;

      // The exported adjoint vector is owned-only because it is accumulated
      // into kernels. The solver temporary is ghosted because pressure
      // normalization reads locally relevant pressure values.
      adjoint_solution.reinit(this->introspection().index_sets.system_partitioning,
                              this->get_mpi_communicator());
      adjoint_solution = 0.0;
      LinearAlgebra::BlockVector solver_solution(
        this->introspection().index_sets.system_partitioning,
        this->introspection().index_sets.system_relevant_partitioning,
        this->get_mpi_communicator());
      solver_solution = 0.0;

      BoundaryConditions<dim> boundary_conditions;
      // Adjoint perturbations satisfy homogeneous essential conditions. Keep
      // the component relations of the forward AffineConstraints (periodic,
      // hanging-node, and tangential free-slip couplings), but remove their
      // inhomogeneous values before zeroing constrained RHS entries.
      boundary_conditions.build_homogeneous_constraints(forward_constraints,
                                                        simulator.current_constraints);
      simulator.system_rhs = right_hand_side;
      simulator.current_constraints.set_zero(simulator.system_rhs);

      // Advected fields remain at the forward state, preserving eta(T,C) and
      // rho(T,C) at the primal linearization point. Only the Stokes trial
      // blocks are cleared to form a zero adjoint initial guess; changing T or
      // C here would solve a different transpose operator.
      simulator.current_linearization_point.block(velocity_block) = 0.0;
      simulator.current_linearization_point.block(pressure_block) = 0.0;
      simulator.solution.block(velocity_block) = 0.0;
      simulator.solution.block(pressure_block) = 0.0;

      // The adjoint pressure has its own gauge. A forward normalization offset
      // would otherwise contaminate a zero adjoint initial guess.
      simulator.last_pressure_normalization_adjustment = 0.0;
      simulator.solve_stokes(solver_solution);

      // The Simulator solve uses a locally relevant vector because pressure
      // normalization and constraints inspect ghosts. Export only owned
      // entries: kernel assembly will create its own short-lived read view,
      // and retaining ghosts in a vector that is later accumulated is unsafe.
      for (unsigned int block = 0; block < adjoint_solution.n_blocks(); ++block)
        for (const auto index : adjoint_solution.block(block).locally_owned_elements())
          adjoint_solution.block(block)(index) = solver_solution.block(block)(index);
      adjoint_solution.compress(VectorOperation::insert);
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template class Adjoint::StokesAdjointSolver<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
