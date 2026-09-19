/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include "common.h"

#include <aspect/adjoint/optimization/initial_temperature_control.h>
#include <aspect/adjoint/optimization/rol_vector.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>

#include <iostream>

namespace adjoint_initial_temperature_control_mpi_tests
{
  void finish_owned_write(aspect::LinearAlgebra::Vector &vector)
  {
    vector.compress(dealii::VectorOperation::insert);
  }


  TEST_CASE("initial-temperature control has distributed reduced ownership and L2 Riesz map")
  {
    constexpr int dim = 2;
    dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
    dealii::GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(2);
    dealii::FE_Q<dim> fe(1);
    dealii::DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);
    const auto owned = dof_handler.locally_owned_dofs();
    const auto relevant = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

    dealii::AffineConstraints<double> constraints(owned, relevant);
    constraints.close();
    aspect::Adjoint::InitialTemperatureControl<dim> control(
      dof_handler, dealii::ComponentMask(1, true), constraints, MPI_COMM_WORLD);

    const auto &owned_control = control.locally_owned_control_dofs();
    REQUIRE(dealii::Utilities::MPI::sum(owned_control.n_elements(), MPI_COMM_WORLD)
            == control.n_control_dofs());
    REQUIRE(control.n_control_dofs() == dof_handler.n_dofs());

    aspect::LinearAlgebra::Vector m(owned_control, MPI_COMM_WORLD);
    for (const auto index : m.locally_owned_elements())
      m[index] = 1.0 + index;
    finish_owned_write(m);

    aspect::LinearAlgebra::Vector perturbation(owned, MPI_COMM_WORLD);
    control.inject_perturbation(m, perturbation);
    for (const auto index : owned)
      REQUIRE(perturbation[index] == Approx(1.0 + index).margin(1e-13));

    aspect::LinearAlgebra::Vector raw(owned, MPI_COMM_WORLD);
    for (const auto index : owned)
      raw[index] = 3.0 - .125 * index;
    finish_owned_write(raw);
    aspect::LinearAlgebra::Vector reduced(owned_control, MPI_COMM_WORLD);
    control.restrict_covector(raw, reduced);
    const double pt_identity_error = std::abs(raw * perturbation - reduced * m);
    REQUIRE(pt_identity_error < 1e-12);

    dealii::MappingQ1<dim> mapping;
    control.assemble_mass_matrix(mapping, dealii::QGauss<dim>(2));
    const auto mass = control.mass_matrix();
    aspect::LinearAlgebra::Vector l2_gradient(owned_control, MPI_COMM_WORLD);
    control.solve_l2_riesz(reduced, l2_gradient);
    aspect::LinearAlgebra::Vector residual(owned_control, MPI_COMM_WORLD);
    mass->vmult(residual, l2_gradient);
    residual -= reduced;
    const double riesz_residual = residual.l2_norm() / reduced.l2_norm();
    REQUIRE(riesz_residual < 1e-10);

    aspect::Adjoint::ROLReducedL2Vector rol_gradient(l2_gradient, mass);
    aspect::Adjoint::ROLReducedL2Vector rol_direction(m, mass);
    const double full_identity_error =
      std::abs(raw * perturbation - rol_gradient.dot(rol_direction));
    REQUIRE(full_identity_error < 1e-10);

    std::cout << "rank "
              << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
              << " owns " << owned_control.n_elements() << " reduced controls\n";
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "distributed controls=" << control.n_control_dofs()
                << ", P/PT error=" << pt_identity_error
                << ", Riesz residual=" << riesz_residual
                << ", full identity error=" << full_identity_error << '\n';
  }
}
