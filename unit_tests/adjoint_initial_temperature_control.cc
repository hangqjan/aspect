/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include "common.h"

#include <aspect/adjoint/optimization/initial_temperature_control.h>
#include <aspect/adjoint/optimization/rol_vector.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <iostream>

namespace
{
  void finish_insert(aspect::LinearAlgebra::Vector &vector)
  {
    vector.compress(dealii::VectorOperation::insert);
  }
}

TEST_CASE("initial-temperature control eliminates affine rows and uses P transpose")
{
  constexpr int dim = 2;
  dealii::Triangulation<dim> triangulation;
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  dealii::FE_Q<dim> fe(1);
  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  const auto owned = dof_handler.locally_owned_dofs();
  const auto relevant = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

  dealii::AffineConstraints<double> constraints(owned, relevant);
  constraints.add_line(2);
  constraints.add_entry(2, 0, .25);
  constraints.add_entry(2, 1, .75);
  constraints.set_inhomogeneity(2, 4.0);
  constraints.add_line(3); // prescribed/Dirichlet row
  constraints.set_inhomogeneity(3, 11.0);
  constraints.close();
  REQUIRE(constraints.is_constrained(2));
  REQUIRE(constraints.is_constrained(3));

  aspect::Adjoint::InitialTemperatureControl<dim> control(
    dof_handler, dealii::ComponentMask(1, true), constraints, MPI_COMM_SELF);
  REQUIRE(control.n_control_dofs() == dof_handler.n_dofs() - 2);
  REQUIRE(control.expansion(2).size() == 2);
  REQUIRE(control.expansion(3).empty());

  aspect::LinearAlgebra::Vector m(control.locally_owned_control_dofs(), MPI_COMM_SELF);
  for (const auto index : m.locally_owned_elements())
    m[index] = 1.0 + index;
  finish_insert(m);

  aspect::LinearAlgebra::Vector fixed(owned, relevant, MPI_COMM_SELF);
  fixed = -17.0;
  finish_insert(fixed);
  aspect::LinearAlgebra::Vector physical(owned, relevant, MPI_COMM_SELF);
  control.inject_physical(m, fixed, physical);
  REQUIRE(physical[2] == Approx(4.0 + .25 * physical[0] + .75 * physical[1]).margin(1e-13));
  REQUIRE(physical[3] == Approx(11.0).margin(1e-13));

  aspect::LinearAlgebra::Vector perturbation(owned, relevant, MPI_COMM_SELF);
  control.inject_perturbation(m, perturbation);
  REQUIRE(perturbation[2] == Approx(.25 * perturbation[0] + .75 * perturbation[1]).margin(1e-13));
  REQUIRE(perturbation[3] == Approx(0.0).margin(1e-13));

  aspect::LinearAlgebra::Vector raw(owned, relevant, MPI_COMM_SELF);
  for (const auto index : owned)
    raw[index] = 2.0 + index;
  finish_insert(raw);
  aspect::LinearAlgebra::Vector reduced(control.locally_owned_control_dofs(), MPI_COMM_SELF);
  control.restrict_covector(raw, reduced);
  REQUIRE(reduced[0] == Approx(3.0).margin(1e-13));
  REQUIRE(reduced[1] == Approx(6.0).margin(1e-13));
  REQUIRE(reduced[2] == Approx(6.0).margin(1e-13));
  std::cout << "P^T microtest expected=[3,6,6], computed=["
            << reduced[0] << ',' << reduced[1] << ',' << reduced[2] << "]\n";

  double raw_dot_perturbation = raw * perturbation;
  double reduced_dot_control = reduced * m;
  std::cout << "P/PT identity error="
            << std::abs(raw_dot_perturbation - reduced_dot_control) << '\n';
  REQUIRE(raw_dot_perturbation == Approx(reduced_dot_control).margin(1e-12));

  dealii::MappingQ1<dim> mapping;
  control.assemble_mass_matrix(mapping, dealii::QGauss<dim>(2));
  const auto mass = control.mass_matrix();
  aspect::LinearAlgebra::Vector mass_m(m);
  mass->vmult(mass_m, m);
  REQUIRE(m * mass_m > 0.0);

  aspect::LinearAlgebra::Vector v(m);
  for (const auto index : v.locally_owned_elements())
    v[index] = 2.0 - index;
  finish_insert(v);
  aspect::LinearAlgebra::Vector mass_v(v);
  mass->vmult(mass_v, v);
  const double mass_symmetry_error = std::abs(m * mass_v - v * mass_m);
  std::cout << "mass symmetry error=" << mass_symmetry_error
            << ", positive form=" << m *mass_m << '\n';
  REQUIRE(mass_symmetry_error < 1e-12);

  dealii::DynamicSparsityPattern full_dsp(relevant);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  std::vector<dealii::types::global_dof_index> cell_dofs(dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(cell_dofs);
      for (const auto row : cell_dofs)
        for (const auto column : cell_dofs)
          full_dsp.add(row, column);
    }
  aspect::LinearAlgebra::SparseMatrix full_mass;
  full_mass.reinit(owned, full_dsp, MPI_COMM_SELF);
  dealii::FEValues<dim> fe_values(mapping, fe, dealii::QGauss<dim>(2),
                                  dealii::update_values | dealii::update_JxW_values);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(cell_dofs);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
            full_mass.add(cell_dofs[i], cell_dofs[j],
                          fe_values.shape_value(i,q) * fe_values.shape_value(j,q)
                          * fe_values.JxW(q));
    }
  full_mass.compress(dealii::VectorOperation::add);
  aspect::LinearAlgebra::Vector perturbation_v(owned, relevant, MPI_COMM_SELF);
  control.inject_perturbation(v, perturbation_v);
  aspect::LinearAlgebra::Vector full_mass_v(perturbation_v);
  full_mass.vmult(full_mass_v, perturbation_v);
  const double reduced_mass_error = std::abs(m * mass_v - perturbation * full_mass_v);
  std::cout << "M_c=P^T M_T P error=" << reduced_mass_error << '\n';
  REQUIRE(reduced_mass_error < 1e-12);

  aspect::LinearAlgebra::Vector l2_gradient(control.locally_owned_control_dofs(), MPI_COMM_SELF);
  control.solve_l2_riesz(reduced, l2_gradient);
  aspect::LinearAlgebra::Vector residual(m);
  mass->vmult(residual, l2_gradient);
  residual -= reduced;
  const double riesz_relative_residual = residual.l2_norm() / reduced.l2_norm();
  const double full_identity_error = std::abs(raw_dot_perturbation - l2_gradient * mass_m);
  std::cout << "Riesz relative residual=" << riesz_relative_residual
            << ", full identity error=" << full_identity_error << '\n';
  REQUIRE(riesz_relative_residual < 1e-10);
  REQUIRE(full_identity_error < 1e-10);

  aspect::Adjoint::ROLReducedL2Vector rol_gradient(l2_gradient, mass);
  aspect::Adjoint::ROLReducedL2Vector rol_direction(m, mass);
  const double rol_dot_error = std::abs(rol_gradient.dot(rol_direction) - raw_dot_perturbation);
  std::cout << "ROL L2 dot error=" << rol_dot_error << '\n';
  REQUIRE(rol_dot_error < 1e-10);
}
