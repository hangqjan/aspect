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
    Manager<dim>::run_composition_stokes_rhs_transpose_test(const ForwardStateView<dim> &snapshot)
    {
      // Verify the material-mediated Stokes residual edge C -> (b-Au):
      //
      //   psi^T [R(C+eps delta_C)-R(C-eps delta_C)]/(2 eps)
      //       = delta_C^T (partial R/partial C)^T psi.
      //
      // The deterministic perturbation and test velocity are distributed FE
      // fields; all scalar contractions are reduced over owned DoFs only.
      if (this->introspection().n_compositional_fields == 0)
        return;

      const double epsilon =
        composition_stokes_rhs_transpose_diagnostic_epsilon;
      AssertThrow(epsilon > 0.0,
                  ExcMessage("Adjoint/C to Stokes RHS transpose test epsilon must be positive."));

      const AdvectionField composition_field = AdvectionField::composition(0);
      const unsigned int composition_component =
        composition_field.component_index(this->introspection());
      const unsigned int velocity_block_index =
        this->introspection().block_indices.velocities;

      LinearAlgebra::BlockVector delta_composition;
      delta_composition.reinit(this->introspection().index_sets.system_partitioning,
                               this->introspection().index_sets.system_relevant_partitioning,
                               this->get_mpi_communicator());
      delta_composition = 0.0;

      const IndexSet locally_owned_dofs =
        this->get_dof_handler().locally_owned_dofs();
      std::vector<types::global_dof_index> local_dof_indices(this->get_fe().dofs_per_cell);
      std::set<types::global_dof_index> written_dofs;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < this->get_fe().dofs_per_cell; ++i)
              if (this->get_fe().system_to_component_index(i).first == composition_component)
                {
                  const types::global_dof_index global_index = local_dof_indices[i];
                  if (locally_owned_dofs.is_element(global_index) == false
                      || written_dofs.insert(global_index).second == false)
                    continue;

                  delta_composition(global_index) =
                    std::sin(0.173 * static_cast<double>(global_index + 1));
                }
          }

      delta_composition.compress(VectorOperation::insert);
      // The trigonometric direction is deterministic across partitions and
      // nonzero on a broad set of DoFs. Distributing constraints turns it into
      // an admissible perturbation before either side of the identity uses it.
      this->get_current_constraints().distribute(delta_composition);

      // Use the primal velocity as a reproducible test covector psi and zero
      // every non-velocity block. This probes both density buoyancy and
      // viscosity bilinear-form contributions without a pressure test seed.
      LinearAlgebra::BlockVector stokes_test_vector(snapshot.solution.get());
      for (unsigned int block = 0; block < stokes_test_vector.n_blocks(); ++block)
        if (block != velocity_block_index)
          stokes_test_vector.block(block) = 0.0;
      stokes_test_vector.compress(VectorOperation::insert);

      LinearAlgebra::BlockVector adjoint_field;
      adjoint_field.reinit(this->introspection().index_sets.system_partitioning,
                           this->introspection().index_sets.system_relevant_partitioning,
                           this->get_mpi_communicator());
      adjoint_field = 0.0;
      add_stokes_rhs_field_transpose(composition_field,
                                     snapshot,
                                     stokes_test_vector,
                                     adjoint_field);

      const LinearAlgebra::BlockVector saved_solution = simulator->solution;
      const LinearAlgebra::BlockVector saved_old_solution = simulator->old_solution;
      const LinearAlgebra::BlockVector saved_old_old_solution = simulator->old_old_solution;
      const LinearAlgebra::BlockVector saved_linearization_point = simulator->current_linearization_point;
      const LinearAlgebra::BlockVector saved_system_rhs = simulator->system_rhs;
      const bool saved_rebuild_stokes_matrix = simulator->rebuild_stokes_matrix;
      const double saved_time = simulator->time;
      const double saved_time_step = simulator->time_step;
      const double saved_old_time_step = simulator->old_time_step;
      const unsigned int saved_timestep_number = simulator->timestep_number;
      const double saved_pressure_normalization_adjustment =
        simulator->last_pressure_normalization_adjustment;

      auto restore_state = [&]()
      {
        simulator->solution = saved_solution;
        simulator->old_solution = saved_old_solution;
        simulator->old_old_solution = saved_old_old_solution;
        simulator->current_linearization_point = saved_linearization_point;
        simulator->system_rhs = saved_system_rhs;
        simulator->rebuild_stokes_matrix = saved_rebuild_stokes_matrix;
        simulator->time = saved_time;
        simulator->time_step = saved_time_step;
        simulator->old_time_step = saved_old_time_step;
        simulator->timestep_number = saved_timestep_number;
        simulator->last_pressure_normalization_adjustment =
          saved_pressure_normalization_adjustment;
      };

      auto assemble_stokes_residual_for_composition_perturbation =
        [&](const double sign)
      {
        // Reassemble b(C)-A(C)u at fixed primal u for C+/-eps delta_C. This is
        // the finite-difference counterpart of add_stokes_rhs_field_transpose;
        // solving a perturbed Stokes system here would test an additional
        // implicit inverse and obscure residual-level errors.
        simulator->solution = snapshot.solution.get();
        simulator->old_solution = snapshot.old_solution.get();
        simulator->old_old_solution = snapshot.old_old_solution.get();
        simulator->current_linearization_point = snapshot.stokes_linearization_point.get();
        simulator->current_linearization_point.add(sign * epsilon,
                                                   delta_composition);
        simulator->time = snapshot.time;
        simulator->time_step = snapshot.time_step;
        simulator->old_time_step = snapshot.old_time_step;
        simulator->timestep_number = snapshot.timestep_number;
        simulator->last_pressure_normalization_adjustment =
          snapshot.pressure_normalization_adjustment;

        simulator->rebuild_stokes_matrix = true;
        simulator->assemble_stokes_system();
        LinearAlgebra::BlockVector matrix_times_solution;
        matrix_times_solution.reinit(simulator->system_rhs);
        simulator->system_matrix.vmult(matrix_times_solution,
                                       snapshot.solution.get());

        LinearAlgebra::BlockVector residual(simulator->system_rhs);
        residual.add(-1.0, matrix_times_solution);
        return residual;
      };

      const LinearAlgebra::BlockVector residual_plus =
        assemble_stokes_residual_for_composition_perturbation(1.0);
      const LinearAlgebra::BlockVector residual_minus =
        assemble_stokes_residual_for_composition_perturbation(-1.0);

      double local_coefficient_fd_dot = 0.0;
      const MaterialModel::NonlinearDependence::ModelDependence model_dependence =
        this->get_material_model().get_model_dependence();
      const bool viscosity_depends_on_composition =
        (static_cast<int>(model_dependence.viscosity)
         & static_cast<int>(MaterialModel::NonlinearDependence::compositional_fields)) != 0;
      if (viscosity_depends_on_composition)
        {
          // When eta(C) changes, composition affects the matrix A as well as
          // buoyancy. Isolate the coefficient contraction
          // -2 eps(u):eps(psi) D eta[delta_C] so it can be compared directly
          // with the material VJP returned by the production transpose.
          LinearAlgebra::BlockVector linearization_plus(snapshot.stokes_linearization_point.get());
          LinearAlgebra::BlockVector linearization_minus(snapshot.stokes_linearization_point.get());
          linearization_plus.add(epsilon, delta_composition);
          linearization_minus.add(-epsilon, delta_composition);

          const Quadrature<dim> &quadrature_formula =
            this->introspection().quadratures.velocities;
          FEValues<dim> fe_values(this->get_mapping(),
                                  this->get_fe(),
                                  quadrature_formula,
                                  update_values |
                                  update_gradients |
                                  update_quadrature_points |
                                  update_JxW_values);
          std::vector<SymmetricTensor<2,dim>> forward_velocity_symmetric_gradients(quadrature_formula.size());
          std::vector<SymmetricTensor<2,dim>> psi_velocity_symmetric_gradients(quadrature_formula.size());

          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                fe_values[this->introspection().extractors.velocities]
                .get_function_symmetric_gradients(snapshot.solution.get(),
                                                  forward_velocity_symmetric_gradients);
                fe_values[this->introspection().extractors.velocities]
                .get_function_symmetric_gradients(stokes_test_vector,
                                                  psi_velocity_symmetric_gradients);

                MaterialModel::MaterialModelInputs<dim> in_plus(fe_values,
                                                                cell,
                                                                this->introspection(),
                                                                linearization_plus);
                MaterialModel::MaterialModelInputs<dim> in_minus(fe_values,
                                                                 cell,
                                                                 this->introspection(),
                                                                 linearization_minus);
                MaterialModel::MaterialModelOutputs<dim> out_plus(fe_values.n_quadrature_points,
                                                                  this->n_compositional_fields());
                MaterialModel::MaterialModelOutputs<dim> out_minus(fe_values.n_quadrature_points,
                                                                   this->n_compositional_fields());
                in_plus.requested_properties = MaterialModel::MaterialProperties::viscosity;
                in_minus.requested_properties = MaterialModel::MaterialProperties::viscosity;
                this->get_material_model().evaluate(in_plus, out_plus);
                this->get_material_model().evaluate(in_minus, out_minus);

                for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                  {
                    const double bar_eta =
                      -2.0
                      * (forward_velocity_symmetric_gradients[q]
                         * psi_velocity_symmetric_gradients[q])
                      * fe_values.JxW(q);
                    local_coefficient_fd_dot +=
                      bar_eta
                      * (out_plus.viscosities[q] - out_minus.viscosities[q])
                      / (2.0 * epsilon);
                  }
              }
        }

      double local_fd_dot = 0.0;
      double local_adjoint_dot = 0.0;
      written_dofs.clear();

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < this->get_fe().dofs_per_cell; ++i)
              {
                const unsigned int component =
                  this->get_fe().system_to_component_index(i).first;
                const types::global_dof_index global_index = local_dof_indices[i];
                if (locally_owned_dofs.is_element(global_index) == false
                    || written_dofs.insert(global_index).second == false)
                  continue;

                if (this->introspection().component_masks.velocities[component])
                  // Left side: psi^T D_C R delta_C over owned Stokes rows.
                  local_fd_dot += stokes_test_vector(global_index)
                                  * (residual_plus(global_index) - residual_minus(global_index))
                                  / (2.0 * epsilon);

                if (component == composition_component)
                  // Right side: delta_C^T D_C R^T psi over owned C rows.
                  local_adjoint_dot += delta_composition(global_index)
                                       * adjoint_field(global_index);
              }
          }

      const double fd_dot =
        Utilities::MPI::sum(local_fd_dot, this->get_mpi_communicator());
      const double adjoint_dot =
        Utilities::MPI::sum(local_adjoint_dot, this->get_mpi_communicator());
      if (viscosity_depends_on_composition)
        {
          const double coefficient_fd_dot =
            Utilities::MPI::sum(local_coefficient_fd_dot, this->get_mpi_communicator());
          const double coefficient_relative_error =
            std::abs(coefficient_fd_dot - adjoint_dot)
            / std::max({std::abs(coefficient_fd_dot), std::abs(adjoint_dot), 1e-30});

          this->get_pcout() << "   C-to-Stokes-viscosity-coefficient transpose diagnostic for field '"
                            << this->introspection().name_for_compositional_index(0)
                            << "': FD = "
                            << std::setprecision(16) << coefficient_fd_dot
                            << ", adjoint = " << adjoint_dot
                            << ", relative error = " << coefficient_relative_error
                            << std::setprecision(6) << std::endl;

          AssertThrow(coefficient_relative_error <= 1e-5,
                      ExcMessage("The C-to-Stokes viscosity-coefficient transpose diagnostic failed."));

          this->get_pcout() << "   Skipping legacy C-to-Stokes RHS/solve transpose diagnostics for field '"
                            << this->introspection().name_for_compositional_index(0)
                            << "' because composition-dependent viscosity changes the Stokes matrix, "
                            << "not only the assembled right hand side." << std::endl;
          restore_state();
          return;
        }
      const double relative_error =
        std::abs(fd_dot - adjoint_dot)
        / std::max({std::abs(fd_dot), std::abs(adjoint_dot), 1e-30});

      this->get_pcout() << "   C-to-Stokes-residual transpose diagnostic for field '"
                        << this->introspection().name_for_compositional_index(0)
                        << "': FD = "
                        << std::setprecision(16) << fd_dot
                        << ", adjoint = " << adjoint_dot
                        << ", relative error = " << relative_error
                        << std::setprecision(6) << std::endl;

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
      simulator->rebuild_stokes_matrix = true;
      simulator->assemble_stokes_system();

      // A second check includes the implicit Stokes solution map. For a probe
      // r, solve A^T psi=r and compare r^T D_C u delta_C with the same material
      // transpose contraction using psi. This validates the composition edge
      // as it is actually consumed by the time-dependent reverse workflow.
      LinearAlgebra::BlockVector stokes_probe_rhs;
      stokes_probe_rhs.reinit(this->introspection().index_sets.system_partitioning,
                              this->introspection().index_sets.system_relevant_partitioning,
                              this->get_mpi_communicator());
      stokes_probe_rhs = 0.0;
      written_dofs.clear();
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < this->get_fe().dofs_per_cell; ++i)
              {
                const unsigned int component =
                  this->get_fe().system_to_component_index(i).first;
                const types::global_dof_index global_index = local_dof_indices[i];
                if (locally_owned_dofs.is_element(global_index) == false
                    || written_dofs.insert(global_index).second == false)
                  continue;

                if (this->introspection().component_masks.velocities[component])
                  stokes_probe_rhs(global_index) =
                    std::cos(0.137 * static_cast<double>(global_index + 1));
              }
          }
      stokes_probe_rhs.compress(VectorOperation::insert);
      this->get_current_constraints().distribute(stokes_probe_rhs);

      LinearAlgebra::BlockVector stokes_adjoint_probe;
      solve_stokes_adjoint_from_rhs(stokes_probe_rhs,
                                    stokes_adjoint_probe);

      auto solve_stokes_for_composition_perturbation =
        [&](const double sign)
      {
        simulator->solution = snapshot.solution.get();
        simulator->old_solution = snapshot.old_solution.get();
        simulator->old_old_solution = snapshot.old_old_solution.get();
        simulator->current_linearization_point = snapshot.stokes_linearization_point.get();
        simulator->current_linearization_point.add(sign * epsilon,
                                                   delta_composition);
        simulator->time = snapshot.time;
        simulator->time_step = snapshot.time_step;
        simulator->old_time_step = snapshot.old_time_step;
        simulator->timestep_number = snapshot.timestep_number;
        simulator->last_pressure_normalization_adjustment =
          snapshot.pressure_normalization_adjustment;

        simulator->rebuild_stokes_matrix = true;
        simulator->assemble_stokes_system();

        LinearAlgebra::BlockVector perturbed_solution(snapshot.solution.get());
        simulator->solve_stokes(perturbed_solution);
        return perturbed_solution;
      };

      const LinearAlgebra::BlockVector stokes_solution_plus =
        solve_stokes_for_composition_perturbation(1.0);
      const LinearAlgebra::BlockVector stokes_solution_minus =
        solve_stokes_for_composition_perturbation(-1.0);

      LinearAlgebra::BlockVector solved_adjoint_field;
      solved_adjoint_field.reinit(this->introspection().index_sets.system_partitioning,
                                  this->introspection().index_sets.system_relevant_partitioning,
                                  this->get_mpi_communicator());
      solved_adjoint_field = 0.0;
      add_stokes_rhs_field_transpose(composition_field,
                                     snapshot,
                                     stokes_adjoint_probe,
                                     solved_adjoint_field);

      double local_solved_fd_dot = 0.0;
      double local_solved_adjoint_dot = 0.0;
      written_dofs.clear();
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < this->get_fe().dofs_per_cell; ++i)
              {
                const unsigned int component =
                  this->get_fe().system_to_component_index(i).first;
                const types::global_dof_index global_index = local_dof_indices[i];
                if (locally_owned_dofs.is_element(global_index) == false
                    || written_dofs.insert(global_index).second == false)
                  continue;

                if (this->introspection().component_masks.velocities[component])
                  local_solved_fd_dot += stokes_probe_rhs(global_index)
                                         * (stokes_solution_plus(global_index)
                                            - stokes_solution_minus(global_index))
                                         / (2.0 * epsilon);

                if (component == composition_component)
                  local_solved_adjoint_dot += delta_composition(global_index)
                                              * solved_adjoint_field(global_index);
              }
          }

      const double solved_fd_dot =
        Utilities::MPI::sum(local_solved_fd_dot, this->get_mpi_communicator());
      const double solved_adjoint_dot =
        Utilities::MPI::sum(local_solved_adjoint_dot, this->get_mpi_communicator());
      const double solved_relative_error =
        std::abs(solved_fd_dot - solved_adjoint_dot)
        / std::max({std::abs(solved_fd_dot), std::abs(solved_adjoint_dot), 1e-30});

      this->get_pcout() << "   C-to-Stokes-solve transpose diagnostic for field '"
                        << this->introspection().name_for_compositional_index(0)
                        << "': FD = "
                        << std::setprecision(16) << solved_fd_dot
                        << ", adjoint = " << solved_adjoint_dot
                        << ", relative error = " << solved_relative_error
                        << std::setprecision(6) << std::endl;

      restore_state();
    }



    template <int dim>
    void
    Manager<dim>::run_direct_stokes_viscosity_derivative_test(const ForwardStateView<dim> &snapshot)
    {
      // Isolate viscosity as an independent quadrature property in the weak
      // Stokes residual. The analytic contraction
      // 2 delta_eta eps(u):eps(psi) is compared with a central difference of
      // the assembled residual while density and all state variables remain
      // fixed. This separates assembly signs/factors from material-model VJP.
      const double epsilon =
        direct_stokes_viscosity_derivative_diagnostic_epsilon;
      AssertThrow(epsilon > 0.0,
                  ExcMessage("Adjoint/direct Stokes viscosity derivative test epsilon must be positive."));

      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      const Quadrature<dim> &quadrature_formula =
        this->introspection().quadratures.velocities;

      LinearAlgebra::BlockVector stokes_test_vector(snapshot.solution.get());
      stokes_test_vector = 0.0;

      const IndexSet locally_owned_dofs =
        this->get_dof_handler().locally_owned_dofs();
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      std::set<types::global_dof_index> written_dofs;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int component =
                  this->get_fe().system_to_component_index(i).first;
                const types::global_dof_index global_index = local_dof_indices[i];
                if (locally_owned_dofs.is_element(global_index) == false
                    || written_dofs.insert(global_index).second == false)
                  continue;

                if (this->introspection().component_masks.velocities[component])
                  stokes_test_vector(global_index) =
                    std::cos(0.211 * static_cast<double>(global_index + 1));
              }
          }

      stokes_test_vector.compress(VectorOperation::insert);
      this->get_current_constraints().distribute(stokes_test_vector);

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_values |
                              update_gradients |
                              update_quadrature_points |
                              update_JxW_values);

      std::vector<SymmetricTensor<2,dim>> forward_velocity_symmetric_gradients(quadrature_formula.size());
      std::vector<SymmetricTensor<2,dim>> psi_velocity_symmetric_gradients(quadrature_formula.size());
      Vector<double> local_residual_plus(dofs_per_cell);
      Vector<double> local_residual_minus(dofs_per_cell);

      LinearAlgebra::BlockVector residual_plus;
      residual_plus.reinit(this->introspection().index_sets.system_partitioning,
                           this->introspection().index_sets.system_relevant_partitioning,
                           this->get_mpi_communicator());
      residual_plus = 0.0;

      LinearAlgebra::BlockVector residual_minus;
      residual_minus.reinit(this->introspection().index_sets.system_partitioning,
                            this->introspection().index_sets.system_relevant_partitioning,
                            this->get_mpi_communicator());
      residual_minus = 0.0;

      double local_direct_dot = 0.0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[this->introspection().extractors.velocities]
            .get_function_symmetric_gradients(snapshot.solution.get(),
                                              forward_velocity_symmetric_gradients);
            fe_values[this->introspection().extractors.velocities]
            .get_function_symmetric_gradients(stokes_test_vector,
                                              psi_velocity_symmetric_gradients);

            local_residual_plus = 0.0;
            local_residual_minus = 0.0;
            cell->get_dof_indices(local_dof_indices);

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                // delta_eta is an independent quadrature perturbation. The
                // direct expression is -2 delta_eta eps(u):eps(psi) JxW, while
                // local_residual_+/- store the corresponding weak residual
                // perturbations for the central-difference contraction.
                const double delta_eta =
                  std::sin(0.317 * static_cast<double>(cell->active_cell_index() + 1)
                           + 0.193 * static_cast<double>(q + 1));

                local_direct_dot +=
                  -2.0
                  * (forward_velocity_symmetric_gradients[q]
                     * psi_velocity_symmetric_gradients[q])
                  * delta_eta
                  * fe_values.JxW(q);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  if (this->introspection().component_masks.velocities[
                        this->get_fe().system_to_component_index(i).first])
                    {
                      const SymmetricTensor<2,dim> epsilon_phi_i =
                        fe_values[this->introspection().extractors.velocities]
                        .symmetric_gradient(i, q);
                      const double residual_contribution =
                        -2.0
                        * delta_eta
                        * (epsilon_phi_i
                           * forward_velocity_symmetric_gradients[q])
                        * fe_values.JxW(q);

                      local_residual_plus(i) += epsilon * residual_contribution;
                      local_residual_minus(i) -= epsilon * residual_contribution;
                    }
              }

            this->get_current_constraints().distribute_local_to_global(local_residual_plus,
                                                                       local_dof_indices,
                                                                       residual_plus);
            this->get_current_constraints().distribute_local_to_global(local_residual_minus,
                                                                       local_dof_indices,
                                                                       residual_minus);
          }

      residual_plus.compress(VectorOperation::add);
      residual_minus.compress(VectorOperation::add);

      double local_fd_dot = 0.0;
      written_dofs.clear();
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int component =
                  this->get_fe().system_to_component_index(i).first;
                const types::global_dof_index global_index = local_dof_indices[i];
                if (locally_owned_dofs.is_element(global_index) == false
                    || written_dofs.insert(global_index).second == false)
                  continue;

                if (this->introspection().component_masks.velocities[component])
                  local_fd_dot +=
                    stokes_test_vector(global_index)
                    * (residual_plus(global_index) - residual_minus(global_index))
                    / (2.0 * epsilon);
              }
          }

      const double fd_dot =
        Utilities::MPI::sum(local_fd_dot, this->get_mpi_communicator());
      const double direct_dot =
        Utilities::MPI::sum(local_direct_dot, this->get_mpi_communicator());
      const double relative_error =
        std::abs(fd_dot - direct_dot)
        / std::max({std::abs(fd_dot), std::abs(direct_dot), 1e-30});

      this->get_pcout() << "   Direct Stokes viscosity derivative diagnostic: FD = "
                        << std::setprecision(16) << fd_dot
                        << ", direct = " << direct_dot
                        << ", relative error = " << relative_error
                        << std::setprecision(6) << std::endl;

      AssertThrow(relative_error <= 1e-5,
                  ExcMessage("The direct Stokes viscosity derivative diagnostic failed."));
    }



    template <int dim>
    void
    Manager<dim>::run_material_model_adjoint_vjp_test(const ForwardStateView<dim> &snapshot) const
    {
      // Check the local material-model vector-Jacobian product by the standard
      // dot-product identity
      //
      //   bar(y)^T [F(x+eps delta_x)-F(x-eps delta_x)]/(2 eps)
      //       = [DF(x)^T bar(y)]^T delta_x.
      //
      // The diagnostic requests exactly the output properties supported by
      // the production adjoint backend and performs MPI reduction only after
      // forming rank-local quadrature contractions.
      AssertThrow(material_model_name == "simple"
                  || material_model_name == "adjoint material model",
                  ExcMessage("The standalone material model adjoint VJP diagnostic is currently "
                             "restricted to Material model/Model name = simple or "
                             "adjoint material model."));
      const double epsilon = material_model_adjoint_vjp_diagnostic_epsilon;
      AssertThrow(epsilon > 0.0,
                  ExcMessage("Adjoint/material model VJP test epsilon must be positive."));

      const LinearAlgebra::BlockVector saved_solution = simulator->solution;
      const LinearAlgebra::BlockVector saved_old_solution = simulator->old_solution;
      const LinearAlgebra::BlockVector saved_old_old_solution = simulator->old_old_solution;
      const LinearAlgebra::BlockVector saved_linearization_point = simulator->current_linearization_point;
      const LinearAlgebra::BlockVector saved_system_rhs = simulator->system_rhs;
      const double saved_time = simulator->time;
      const double saved_time_step = simulator->time_step;
      const double saved_old_time_step = simulator->old_time_step;
      const unsigned int saved_timestep_number = simulator->timestep_number;
      const double saved_pressure_normalization_adjustment =
        simulator->last_pressure_normalization_adjustment;

      simulator->solution = snapshot.solution.get();
      simulator->old_solution = snapshot.old_solution.get();
      simulator->old_old_solution = snapshot.old_old_solution.get();
      simulator->current_linearization_point = snapshot.advection_linearization_point.get();
      simulator->time = snapshot.time;
      simulator->time_step = snapshot.time_step;
      simulator->old_time_step = snapshot.old_time_step;
      simulator->timestep_number = snapshot.timestep_number;
      simulator->last_pressure_normalization_adjustment =
        snapshot.pressure_normalization_adjustment;

      const unsigned int quadrature_degree =
        this->get_fe().base_element(this->introspection().base_elements.temperature).degree + 1;
      const QGauss<dim> quadrature_formula(quadrature_degree);

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_values |
                              update_gradients |
                              update_quadrature_points);

      auto requested_density_cp_eta =
        []()
      {
        return static_cast<MaterialModel::MaterialProperties::Property>(
                 static_cast<int>(MaterialModel::MaterialProperties::density)
                 | static_cast<int>(MaterialModel::MaterialProperties::specific_heat)
                 | static_cast<int>(MaterialModel::MaterialProperties::viscosity));
      };

      enum class SeedCase
      {
        density,
        specific_heat,
        density_and_specific_heat,
        viscosity,
        density_and_viscosity
      };

      auto run_case = [&](const SeedCase seed_case)
      {
        // Each seed case chooses a different output covector bar(y). Testing
        // combined seeds catches accumulation errors that single-property
        // checks cannot reveal (for example overwriting rather than +=).
        double local_adjoint_dot = 0.0;
        double local_fd_dot = 0.0;
        bool tested_cell = false;

        for (const auto &cell : this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              if (tested_cell)
                break;
              tested_cell = true;

              fe_values.reinit(cell);

              MaterialModel::MaterialModelInputs<dim> in(fe_values,
                                                         cell,
                                                         this->introspection(),
                                                         simulator->current_linearization_point);
              MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points,
                                                           this->n_compositional_fields());
              in.requested_properties = requested_density_cp_eta();
              this->get_material_model().evaluate(in, out);

              MaterialModelOutputAdjoints<dim> bar_out(fe_values.n_quadrature_points);
              for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                {
                  if (seed_case == SeedCase::density
                      || seed_case == SeedCase::density_and_specific_heat
                      || seed_case == SeedCase::density_and_viscosity)
                    bar_out.densities[q] =
                      0.25 + 0.03 * static_cast<double>(q + 1);
                  if (seed_case == SeedCase::specific_heat
                      || seed_case == SeedCase::density_and_specific_heat)
                    bar_out.specific_heat[q] =
                      -0.4 + 0.02 * static_cast<double>(q + 1);
                  if (seed_case == SeedCase::viscosity
                      || seed_case == SeedCase::density_and_viscosity)
                    bar_out.viscosities[q] =
                      0.7 - 0.025 * static_cast<double>(q + 1);
                }

              MaterialModelInputAdjoints<dim> bar_in(fe_values.n_quadrature_points,
                                                     this->n_compositional_fields());
              const MaterialModelAdjoint<dim> material_model_adjoint(*simulator);
              material_model_adjoint.apply_derivative_transpose(in,
                                                                out,
                                                                bar_out,
                                                                bar_in);

              // Construct one input direction delta_x and simultaneously form
              // the adjoint contraction delta_x^T bar(x). The same direction
              // perturbs the two material-model evaluations below.
              MaterialModel::MaterialModelInputs<dim> in_plus = in;
              MaterialModel::MaterialModelInputs<dim> in_minus = in;
              // Contract the central output difference with the prescribed
              // bar(y), producing the finite-difference side of the identity.
              for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                {
                  const Point<dim> &point = fe_values.quadrature_point(q);
                  const double temperature_direction =
                    0.1 + 0.01 * static_cast<double>(q + 1) + 0.001 * point[0];
                  local_adjoint_dot += temperature_direction * bar_in.temperature[q];
                  in_plus.temperature[q] += epsilon * temperature_direction;
                  in_minus.temperature[q] -= epsilon * temperature_direction;

                  for (unsigned int c = 0; c < this->introspection().n_compositional_fields; ++c)
                    {
                      const double composition_direction =
                        -0.2 + 0.015 * static_cast<double>(c + 1)
                        + 0.004 * static_cast<double>(q + 1);
                      local_adjoint_dot +=
                        composition_direction * bar_in.composition[q][c];
                      in_plus.composition[q][c] += epsilon * composition_direction;
                      in_minus.composition[q][c] -= epsilon * composition_direction;
                    }
                }

              MaterialModel::MaterialModelOutputs<dim> out_plus(fe_values.n_quadrature_points,
                                                                this->n_compositional_fields());
              MaterialModel::MaterialModelOutputs<dim> out_minus(fe_values.n_quadrature_points,
                                                                 this->n_compositional_fields());
              in_plus.requested_properties = requested_density_cp_eta();
              in_minus.requested_properties = requested_density_cp_eta();
              this->get_material_model().evaluate(in_plus, out_plus);
              this->get_material_model().evaluate(in_minus, out_minus);

              for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                local_fd_dot +=
                  bar_out.densities[q]
                  * (out_plus.densities[q] - out_minus.densities[q])
                  / (2.0 * epsilon)
                  + bar_out.specific_heat[q]
                  * (out_plus.specific_heat[q] - out_minus.specific_heat[q])
                  / (2.0 * epsilon)
                  + bar_out.viscosities[q]
                  * (out_plus.viscosities[q] - out_minus.viscosities[q])
                  / (2.0 * epsilon);
            }

        const double adjoint_dot =
          Utilities::MPI::sum(local_adjoint_dot, this->get_mpi_communicator());
        const double fd_dot =
          Utilities::MPI::sum(local_fd_dot, this->get_mpi_communicator());
        const double relative_error =
          std::abs(adjoint_dot - fd_dot)
          / std::max({std::abs(adjoint_dot), std::abs(fd_dot), 1e-30});

        const std::string seed_label =
          seed_case == SeedCase::density
          ? "density"
          : (seed_case == SeedCase::specific_heat
             ? "specific heat"
             : (seed_case == SeedCase::density_and_specific_heat
                ? "density + specific heat"
                : (seed_case == SeedCase::viscosity
                   ? "viscosity"
                   : "density + viscosity")));

        this->get_pcout() << "   MaterialModelAdjoint "
                          << material_model_name
                          << " VJP diagnostic ("
                          << seed_label
                          << " seeds): FD = "
                          << std::setprecision(16) << fd_dot
                          << ", adjoint = " << adjoint_dot
                          << ", relative error = " << relative_error
                          << std::setprecision(6) << std::endl;

        AssertThrow(relative_error < 1e-5,
                    ExcMessage("The standalone MaterialModelAdjoint VJP diagnostic failed."));
      };

      run_case(SeedCase::density);
      if (material_model_name == "adjoint material model")
        run_case(SeedCase::specific_heat);
      run_case(SeedCase::density_and_specific_heat);
      if (material_model_name == "adjoint material model")
        {
          run_case(SeedCase::viscosity);
          run_case(SeedCase::density_and_viscosity);
        }

      simulator->solution = saved_solution;
      simulator->old_solution = saved_old_solution;
      simulator->old_old_solution = saved_old_old_solution;
      simulator->current_linearization_point = saved_linearization_point;
      simulator->system_rhs = saved_system_rhs;
      simulator->time = saved_time;
      simulator->time_step = saved_time_step;
      simulator->old_time_step = saved_old_time_step;
      simulator->timestep_number = saved_timestep_number;
      simulator->last_pressure_normalization_adjustment =
        saved_pressure_normalization_adjustment;
    }

  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::Manager<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
