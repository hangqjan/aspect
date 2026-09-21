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

#include <aspect/adjoint/core/manager.h>
#include <aspect/adjoint/core/instantaneous_stokes_workflow.h>
#include <aspect/adjoint/core/simulator_state.h>
#include <aspect/adjoint/operators/stokes_adjoint_solver.h>
#include <aspect/adjoint/objective/temperature_field_objective.h>

#include <aspect/geometry_model/box.h>
#include <aspect/gravity_model/vertical.h>
#include <aspect/material_model/simple.h>
#include <aspect/adjoint/operators/advection_adjoint_calculator.h>
#include <aspect/adjoint/operators/boundary_conditions.h>
#include <aspect/adjoint/operators/material_model_adjoint.h>
#include <aspect/postprocess/dynamic_topography.h>
#include <aspect/simulator.h>
#include <aspect/simulator/solver/stokes_matrix_free.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <string>

#if defined(__APPLE__)
#  include <mach/mach.h>
#endif

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    void
    Manager<dim>::solve_stokes_adjoint_from_rhs(
      const LinearAlgebra::BlockVector &right_hand_side,
      LinearAlgebra::BlockVector &adjoint_solution)
    {
      StokesAdjointSolver<dim>(*simulator).solve(right_hand_side,
                                                 adjoint_solution);
    }



    template <int dim>
    void
    Manager<dim>::write_initial_temperature_gradient() const
    {
      if (time_dependent_workflow.initial_temperature_gradient.size() == 0)
        return;

      const unsigned int rank =
        Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
      const unsigned int temperature_block_index =
        this->introspection().block_indices.temperature;

      const std::string filename =
        this->get_output_directory()
        + "initial_temperature_gradient_rank_"
        + Utilities::int_to_string(rank, 5)
        + ".txt";

      std::ofstream out(filename);
      out << "# global_dof_index\tgradient";
      for (unsigned int d = 0; d < dim; ++d)
        out << "\tx" << d;
      out << '\n';
      out << std::setprecision(16);

      (void)temperature_block_index;
      const IndexSet locally_owned_dofs =
        this->get_dof_handler().locally_owned_dofs();
      const std::vector<Point<dim>> unit_support_points =
        Utilities::get_unit_support_points(*this);
      const Quadrature<dim> support_quadrature(unit_support_points);
      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              support_quadrature,
                              update_quadrature_points);
      std::vector<types::global_dof_index> local_dof_indices(this->get_fe().dofs_per_cell);
      std::set<types::global_dof_index> written_dofs;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);

            for (unsigned int i = 0; i < this->get_fe().dofs_per_cell; ++i)
              {
                const unsigned int component =
                  this->get_fe().system_to_component_index(i).first;
                if (this->introspection().component_masks.temperature[component] == false)
                  continue;

                const types::global_dof_index global_index = local_dof_indices[i];
                if (locally_owned_dofs.is_element(global_index) == false
                    || written_dofs.insert(global_index).second == false)
                  continue;

                out << global_index << '\t'
                    << time_dependent_workflow.initial_temperature_gradient(global_index);
                for (unsigned int d = 0; d < dim; ++d)
                  out << '\t' << fe_values.quadrature_point(i)[d];
                out << '\n';
              }
          }
    }



    template <int dim>
    void
    Manager<dim>::write_terminal_seed_diagnostics() const
    {
      const unsigned int rank =
        Utilities::MPI::this_mpi_process(this->get_mpi_communicator());

      const auto write_seed = [this, rank](const std::string &name,
                                           const LinearAlgebra::BlockVector &seed)
      {
        AssertThrow(seed.n_blocks() == this->introspection().index_sets.system_partitioning.size(),
                    ExcInternalError());
        std::ofstream out(this->get_output_directory()
                          + "terminal_seed_" + name + "_rank_"
                          + Utilities::int_to_string(rank, 5) + ".txt");
        out << "# block\tblock_dof\tvalue\n" << std::setprecision(17);
        double local_square = 0.0;
        for (unsigned int block = 0; block < seed.n_blocks(); ++block)
          {
            double local_block_square = 0.0;
            for (const auto index : seed.block(block).locally_owned_elements())
              {
                const double value = seed.block(block)(index);
                out << block << '\t' << index << '\t' << value << '\n';
                local_block_square += value * value;
              }
            local_square += local_block_square;
            this->get_pcout() << "ASPECT_ADJOINT_TERMINAL_SEED field=" << name
                              << " block=" << block
                              << " l2=" << std::setprecision(17)
                              << std::sqrt(Utilities::MPI::sum(local_block_square,
                                                               this->get_mpi_communicator()))
                              << std::setprecision(6) << std::endl;
          }
        this->get_pcout() << "ASPECT_ADJOINT_TERMINAL_SEED field=" << name
                          << " full_l2=" << std::setprecision(17)
                          << std::sqrt(Utilities::MPI::sum(local_square,
                                                           this->get_mpi_communicator()))
                          << std::setprecision(6) << std::endl;
      };

      write_seed("temperature", time_dependent_workflow.terminal_temperature_seed);
      for (unsigned int c = 0; c < time_dependent_workflow.terminal_composition_seeds.size(); ++c)
        write_seed("composition_" + this->introspection().name_for_compositional_index(c),
                   time_dependent_workflow.terminal_composition_seeds[c]);
    }



    template <int dim>
    void
    Manager<dim>::write_initial_composition_gradients() const
    {
      if (time_dependent_workflow.initial_composition_gradients.empty())
        return;

      const unsigned int rank =
        Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
      const IndexSet locally_owned_dofs =
        this->get_dof_handler().locally_owned_dofs();
      const std::vector<Point<dim>> unit_support_points =
        Utilities::get_unit_support_points(*this);
      const Quadrature<dim> support_quadrature(unit_support_points);
      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              support_quadrature,
                              update_quadrature_points);
      std::vector<types::global_dof_index> local_dof_indices(this->get_fe().dofs_per_cell);

      for (unsigned int c = 0; c < time_dependent_workflow.initial_composition_gradients.size(); ++c)
        {
          const AdvectionField composition_field = AdvectionField::composition(c);
          const unsigned int composition_component =
            composition_field.component_index(this->introspection());
          const std::string filename =
            this->get_output_directory()
            + "initial_composition_gradient_"
            + this->introspection().name_for_compositional_index(c)
            + "_rank_"
            + Utilities::int_to_string(rank, 5)
            + ".txt";

          std::ofstream out(filename);
          out << "# global_dof_index\tgradient";
          for (unsigned int d = 0; d < dim; ++d)
            out << "\tx" << d;
          out << '\n';
          out << std::setprecision(16);

          std::set<types::global_dof_index> written_dofs;
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                for (unsigned int i = 0; i < this->get_fe().dofs_per_cell; ++i)
                  {
                    const unsigned int component =
                      this->get_fe().system_to_component_index(i).first;
                    if (component != composition_component)
                      continue;

                    const types::global_dof_index global_index = local_dof_indices[i];
                    if (locally_owned_dofs.is_element(global_index) == false
                        || written_dofs.insert(global_index).second == false)
                      continue;

                    out << global_index << '\t'
                        << time_dependent_workflow.initial_composition_gradients[c](global_index);
                    for (unsigned int d = 0; d < dim; ++d)
                      out << '\t' << fe_values.quadrature_point(i)[d];
                    out << '\n';
                  }
              }
        }
    }



    template <int dim>
    void
    Manager<dim>::add_stokes_field_transpose(const AdvectionField &advection_field,
                                             const ForwardStateView<dim> &snapshot,
                                             const LinearAlgebra::BlockVector &stokes_adjoint_rhs,
                                             LinearAlgebra::BlockVector &field_adjoint)
    {
      // At time n the quasi-static Stokes equation S_n(u_n,p_n;q_n)=0 depends
      // on the advected field q_n through rho and eta. Given a Stokes RHS seed
      // accumulated from later advection steps, solve
      //
      //   (partial S_n/partial[u,p])^T psi_n = bar([u,p]_n)
      //
      // and then add -(partial S_n/partial q_n)^T psi_n to bar(q_n). The
      // zero-velocity test avoids rebuilding an operator for an empty edge in
      // the reverse graph.
      const unsigned int velocity_block_index =
        this->introspection().block_indices.velocities;

      if (stokes_adjoint_rhs.block(velocity_block_index).l2_norm()
          <= std::numeric_limits<double>::min())
        return;

      ScopedSimulatorState<dim> simulator_state(*simulator);

      simulator->solution = snapshot.solution.get();
      simulator->old_solution = snapshot.old_solution.get();
      simulator->old_old_solution = snapshot.old_old_solution.get();
      simulator->current_linearization_point = snapshot.stokes_linearization_point.get();
      simulator->time = snapshot.time;
      simulator->time_step = snapshot.time_step;
      simulator->old_time_step = snapshot.old_time_step;
      simulator->timestep_number = snapshot.timestep_number;
      simulator->last_pressure_normalization_adjustment =
        snapshot.pressure_normalization_adjustment;

      // Matrix-free Stokes preconditioners must be built with the primal
      // constraint topology: homogeneous adjoint constraints leave some
      // eliminated mass-matrix diagonal entries at zero. This is also the
      // topology used by the uninterrupted forward path whose preconditioner
      // was previously reused by the adjoint solve.
      simulator->current_constraints.copy_from(snapshot.constraints);
      // This scalar is intentionally runtime-only. The normal forward Stokes
      // path computes it immediately before every assembly; a cold adjoint
      // resume skips that path and therefore retains the constructor's NaN.
      // Recompute it from this restored primal snapshot, rather than storing
      // any solver or matrix-free state in the checkpoint.
      simulator->pressure_scaling = simulator->compute_pressure_scaling_factor();
      simulator->rebuild_stokes_matrix = true;
      simulator->rebuild_stokes_preconditioner = true;
      simulator->assemble_stokes_system();

      // Preconditioner data may depend on constrained block diagonals and the
      // pressure scaling but not on the adjoint RHS. Build it once from the
      // restored primal topology before replacing constraints by their
      // homogeneous adjoint form.
      if (simulator->stokes_matrix_free)
        simulator->stokes_matrix_free->build_preconditioner();
      else
        simulator->build_stokes_preconditioner();

      BoundaryConditions<dim> boundary_conditions;
      boundary_conditions.build_stokes_adjoint_constraints(snapshot,
                                                           simulator->current_constraints);
      // Assemble the adjoint matrix/RHS under homogeneous constraints, but
      // retain the already initialized primal-topology preconditioner.
      simulator->rebuild_stokes_matrix = true;
      simulator->rebuild_stokes_preconditioner = false;
      simulator->assemble_stokes_system();

      LinearAlgebra::BlockVector stokes_adjoint_solution;
      solve_stokes_adjoint_from_rhs(stokes_adjoint_rhs,
                                    stokes_adjoint_solution);

      // The implicit-state solve has removed the [u,p] dependence. What
      // remains is the material-input VJP at fixed primal velocity, evaluated
      // for the selected temperature or composition component.
      add_stokes_rhs_field_transpose(advection_field,
                                     snapshot,
                                     stokes_adjoint_solution,
                                     field_adjoint);

    }



    template <int dim>
    void
    Manager<dim>::add_stokes_rhs_field_transpose(const AdvectionField &advection_field,
                                                 const ForwardStateView<dim> &snapshot,
                                                 const LinearAlgebra::BlockVector &stokes_test_vector,
                                                 LinearAlgebra::BlockVector &field_adjoint)
    {
      // For the incompressible weak residual
      //
      //   S(v)=integral [2 eta(q) eps(u):eps(v)-rho(q) g.v- p div(v)] dx,
      //
      // contraction with the Stokes adjoint psi seeds material outputs as
      //
      //   bar(rho)= +(g.psi),
      //   bar(eta)= -2 eps(u):eps(psi),
      //
      // under the b-Ax sign convention used by the reverse sweep. The
      // material-model transpose then maps these seeds to q.
      const unsigned int velocity_block_index =
        this->introspection().block_indices.velocities;

      if (stokes_test_vector.block(velocity_block_index).l2_norm()
          <= std::numeric_limits<double>::min())
        return;

      // This is an FE read of the Stokes adjoint. The solve exports an
      // owned-only vector so it can be accumulated safely; materialize its
      // locally-relevant view only at this read boundary.
      LinearAlgebra::BlockVector stokes_test_vector_read(
        this->introspection().index_sets.system_partitioning,
        this->introspection().index_sets.system_relevant_partitioning,
        this->get_mpi_communicator());
      stokes_test_vector_read = stokes_test_vector;

      ScopedSimulatorState<dim> simulator_state(*simulator);

      simulator->solution = snapshot.solution.get();
      simulator->old_solution = snapshot.old_solution.get();
      simulator->old_old_solution = snapshot.old_old_solution.get();
      simulator->current_linearization_point = snapshot.stokes_linearization_point.get();
      simulator->time = snapshot.time;
      simulator->time_step = snapshot.time_step;
      simulator->old_time_step = snapshot.old_time_step;
      simulator->timestep_number = snapshot.timestep_number;
      simulator->last_pressure_normalization_adjustment =
        snapshot.pressure_normalization_adjustment;

      const Quadrature<dim> &quadrature_formula =
        this->introspection().quadratures.velocities;
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      const FEValuesExtractors::Scalar field_extractor =
        advection_field.scalar_extractor(this->introspection());
      const unsigned int field_component =
        advection_field.component_index(this->introspection());

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_values |
                              update_gradients |
                              update_quadrature_points |
                              update_JxW_values);

      std::vector<Tensor<1,dim>> psi_velocity_values(quadrature_formula.size());
      std::vector<SymmetricTensor<2,dim>> forward_velocity_symmetric_gradients(quadrature_formula.size());
      std::vector<SymmetricTensor<2,dim>> psi_velocity_symmetric_gradients(quadrature_formula.size());
      Vector<double> local_field_adjoint(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[this->introspection().extractors.velocities]
            .get_function_symmetric_gradients(simulator->solution,
                                              forward_velocity_symmetric_gradients);
            fe_values[this->introspection().extractors.velocities]
            .get_function_values(stokes_test_vector_read,
                                 psi_velocity_values);
            fe_values[this->introspection().extractors.velocities]
            .get_function_symmetric_gradients(stokes_test_vector_read,
                                              psi_velocity_symmetric_gradients);

            local_field_adjoint = 0.0;
            cell->get_dof_indices(local_dof_indices);

            MaterialModel::MaterialModelInputs<dim> in(fe_values,
                                                       cell,
                                                       this->introspection(),
                                                       simulator->current_linearization_point);
            MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points,
                                                         this->n_compositional_fields());
            in.requested_properties =
              static_cast<MaterialModel::MaterialProperties::Property>(
                static_cast<int>(MaterialModel::MaterialProperties::equation_of_state_properties)
                | static_cast<int>(MaterialModel::MaterialProperties::viscosity));
            this->get_material_model().evaluate(in, out);

            MaterialModelOutputAdjoints<dim> bar_out(fe_values.n_quadrature_points);
            MaterialModelInputAdjoints<dim> bar_in(fe_values.n_quadrature_points,
                                                   this->n_compositional_fields());

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                // Include JxW in the material-output seed so the following VJP
                // is already a quadrature contribution. This is the exact
                // transpose of material outputs being sampled inside the weak
                // Stokes assembly, not a pointwise continuum derivative.
                const Tensor<1,dim> gravity =
                  this->get_gravity_model().gravity_vector(fe_values.quadrature_point(q));
                bar_out.densities[q] =
                  (gravity * psi_velocity_values[q]) * fe_values.JxW(q);
                bar_out.viscosities[q] =
                  -2.0
                  * (forward_velocity_symmetric_gradients[q]
                     * psi_velocity_symmetric_gradients[q])
                  * fe_values.JxW(q);
              }

            const MaterialModelAdjoint<dim> material_model_adjoint(*simulator);
            material_model_adjoint.apply_derivative_transpose(in,
                                                              out,
                                                              bar_out,
                                                              bar_in);

            for (const double pressure_adjoint : bar_in.pressure)
              AssertThrow(pressure_adjoint == 0.0,
                          ExcMessage("The Stokes material adjoint generated a non-zero pressure input "
                                     "adjoint, but the v1 Stokes-material scatter supports only "
                                     "temperature and composition states."));

            // Scatter the selected material input seed with the scalar FE
            // evaluation transpose: bar(q_i)+=bar(q(x_q)) phi_i(x_q).
            // Material dependence on other advected fields is handled by
            // separate calls, so each call selects exactly one component.
            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double seed =
                  advection_field.is_temperature()
                  ? bar_in.temperature[q]
                  : bar_in.composition[q][advection_field.compositional_variable];
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  if (this->get_fe().system_to_component_index(i).first == field_component)
                    local_field_adjoint(i) +=
                      seed
                      * fe_values[field_extractor].value(i, q);
              }

            this->get_current_constraints().distribute_local_to_global(local_field_adjoint,
                                                                       local_dof_indices,
                                                                       field_adjoint);
          }

      field_adjoint.compress(VectorOperation::add);

    }

  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::Manager<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
