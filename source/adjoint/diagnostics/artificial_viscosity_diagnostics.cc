/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  Perturbing finite-difference checks for the entropy-viscosity transpose are
  compiled separately from the production coefficient and estimator actions.
*/

#include <aspect/adjoint/operators/artificial_viscosity_adjoint.h>
#include <aspect/adjoint/operators/material_model_adjoint.h>
#include <aspect/advection_field.h>
#include <aspect/simulator.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <set>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    void
    ArtificialViscosityAdjoint<dim>::run_coefficient_seed_diagnostic(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const LinearAlgebra::BlockVector &transpose_solution,
      const std::vector<double> &coefficient_seed,
      const double epsilon) const
    {
      // Verify the directional-derivative identity for the cell coefficient
      // map nu -> R(q,nu):
      //
      //   z^T [R(nu+eps delta_nu)-R(nu-eps delta_nu)]/(2 eps)
      //       = delta_nu^T bar(nu),
      //
      // where bar(nu) is produced by assemble_coefficient_seed(). Restrict the
      // test to cells whose physical/artificial max branch is unambiguous.
      AssertThrow(epsilon > 0.0,
                  ExcMessage("Adjoint/artificial-viscosity coefficient seed test epsilon must be positive."));

      const Vector<double> &artificial_viscosity_per_cell =
        simulator->last_advection_artificial_viscosity_per_cell;
      const Vector<double> &minimum_physical_diffusion_per_cell =
        simulator->last_advection_physical_diffusion_min_per_cell;
      const Vector<double> &maximum_physical_diffusion_per_cell =
        simulator->last_advection_physical_diffusion_max_per_cell;
      AssertThrow(coefficient_seed.size() == artificial_viscosity_per_cell.size(),
                  ExcMessage("The artificial-viscosity coefficient seed diagnostic needs "
                             "a coefficient seed vector from the same advection assembly."));

      const unsigned int quadrature_degree =
        this->get_fe().base_element(advection_field.base_element(this->introspection())).degree + 1;
      const QGauss<dim> quadrature_formula(quadrature_degree);
      const FEValuesExtractors::Scalar field_extractor =
        advection_field.scalar_extractor(this->introspection());

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_gradients |
                              update_JxW_values);

      std::vector<Tensor<1, dim>> field_gradients(quadrature_formula.size());
      std::vector<Tensor<1, dim>> transpose_gradients(quadrature_formula.size());

      double local_fd_dot = 0.0;
      double local_adjoint_dot = 0.0;
      unsigned int local_tested_cells = 0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int active_cell_index = cell->active_cell_index();
            const double artificial_viscosity =
              artificial_viscosity_per_cell[active_cell_index];

            if (artificial_viscosity <= minimum_physical_diffusion_per_cell[active_cell_index])
              continue;

            AssertThrow(artificial_viscosity > maximum_physical_diffusion_per_cell[active_cell_index],
                        ExcMessage("The artificial-viscosity coefficient seed diagnostic currently "
                                   "supports only cells where max(k, nu_art) selects the artificial "
                                   "diffusion branch at every quadrature point."));

            const double delta_nu =
              std::sin(0.217 * static_cast<double>(active_cell_index + 1));
            if (delta_nu == 0.0)
              continue;

            fe_values.reinit(cell);
            fe_values[field_extractor]
            .get_function_gradients(this->get_solution(),
                                    field_gradients);
            fe_values[field_extractor]
            .get_function_gradients(transpose_solution,
                                    transpose_gradients);

            // Contract the perturbed coefficient with the same local kernel
            // used by assemble_coefficient_seed(). This isolates coefficient
            // differentiation from the nonsmooth entropy estimator itself.
            double plus_functional = 0.0;
            double minus_functional = 0.0;
            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double local_kernel =
                  -snapshot.time_step
                  * (transpose_gradients[q] * field_gradients[q])
                  * fe_values.JxW(q);
                plus_functional +=
                  (artificial_viscosity + epsilon * delta_nu) * local_kernel;
                minus_functional +=
                  (artificial_viscosity - epsilon * delta_nu) * local_kernel;
              }

            local_fd_dot +=
              (plus_functional - minus_functional) / (2.0 * epsilon);
            local_adjoint_dot +=
              delta_nu * coefficient_seed[active_cell_index];
            ++local_tested_cells;
          }

      // Both contractions sum disjoint locally owned cells. Reduce them before
      // forming the relative error so every rank checks one global identity.
      const double fd_dot =
        Utilities::MPI::sum(local_fd_dot, this->get_mpi_communicator());
      const double adjoint_dot =
        Utilities::MPI::sum(local_adjoint_dot, this->get_mpi_communicator());
      const unsigned int tested_cells =
        Utilities::MPI::sum(local_tested_cells, this->get_mpi_communicator());
      const double relative_error =
        std::abs(fd_dot - adjoint_dot)
        / std::max({std::abs(fd_dot), std::abs(adjoint_dot), 1e-30});

      this->get_pcout() << "   "
                        << (advection_field.is_temperature() ? "Temperature" : "Composition")
                        << "-advection artificial-viscosity coefficient seed diagnostic: tested cells = "
                        << tested_cells
                        << ", FD = "
                        << std::setprecision(16) << fd_dot
                        << ", adjoint = " << adjoint_dot
                        << ", relative error = " << relative_error
                        << std::setprecision(6) << std::endl;
    }


    template <int dim>
    void
    ArtificialViscosityAdjoint<dim>::run_global_entropy_variation_transpose_diagnostic(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const double epsilon) const
    {
      // Isolate Delta_E(q_{n-1},q_{n-2}) while freezing q_avg. The test checks
      //
      //   D Delta_E[delta q_old,delta q_old_old]
      //     = <bar(q_old),delta q_old>
      //       + <bar(q_old_old),delta q_old_old>.
      //
      // Freezing q_avg deliberately excludes the separate extrapolated-range
      // transpose so a failure identifies this one nonlocal max/mean path.
      if (this->get_parameters().stabilization_alpha != 2
          || snapshot.timestep_number <= 1)
        return;

      AssertThrow(epsilon > 0.0,
                  ExcMessage("Adjoint/global entropy variation transpose test epsilon must be positive."));

      const Quadrature<dim> &quadrature_formula =
        advection_field.is_temperature()
        ? this->introspection().quadratures.temperature
        : this->introspection().quadratures.compositional_fields[advection_field.compositional_variable];
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      const FEValuesExtractors::Scalar field_extractor =
        advection_field.scalar_extractor(this->introspection());

      const std::pair<double, double> global_field_range =
        simulator->get_extrapolated_advection_field_range(advection_field);
      const double average_field =
        0.5 * (global_field_range.first + global_field_range.second);
      const double base_entropy_variation =
        simulator->get_entropy_variation(average_field, advection_field);

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_values |
                              update_JxW_values);

      std::vector<double> old_field_values(quadrature_formula.size());
      std::vector<double> old_old_field_values(quadrature_formula.size());
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      double local_min_entropy = std::numeric_limits<double>::max();
      double local_max_entropy = std::numeric_limits<double>::lowest();
      double local_entropy_integrated = 0.0;
      double local_area = 0.0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[field_extractor].get_function_values(simulator->old_solution,
                                                           old_field_values);
            fe_values[field_extractor].get_function_values(simulator->old_old_solution,
                                                           old_old_field_values);
            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double field_history =
                  0.5 * (old_field_values[q] + old_old_field_values[q]);
                const double entropy =
                  (field_history - average_field) * (field_history - average_field);
                local_min_entropy = std::min(local_min_entropy, entropy);
                local_max_entropy = std::max(local_max_entropy, entropy);
                local_entropy_integrated += entropy * fe_values.JxW(q);
                local_area += fe_values.JxW(q);
              }
          }

      const double global_min_entropy =
        -Utilities::MPI::max(-local_min_entropy, this->get_mpi_communicator());
      const double global_max_entropy =
        Utilities::MPI::max(local_max_entropy, this->get_mpi_communicator());
      const double global_entropy_integrated =
        Utilities::MPI::sum(local_entropy_integrated, this->get_mpi_communicator());
      const double global_area =
        Utilities::MPI::sum(local_area, this->get_mpi_communicator());
      const double average_entropy =
        global_entropy_integrated / global_area;

      const bool upper_branch =
        (global_max_entropy - average_entropy)
        >= (average_entropy - global_min_entropy);
      // Seed Delta_E by one and reverse the selected outer max. The following
      // FE loop then reverses E=(q_mid-q_avg)^2 and the midpoint map; q_avg is
      // intentionally frozen in this isolated diagnostic.
      const double bar_entropy_variation = 1.0;
      const double bar_max_entropy = upper_branch ? bar_entropy_variation : 0.0;
      const double bar_min_entropy = upper_branch ? 0.0 : -bar_entropy_variation;
      const double bar_average_entropy = upper_branch
                                         ? -bar_entropy_variation
                                         : bar_entropy_variation;

      LinearAlgebra::BlockVector old_field_adjoint;
      old_field_adjoint.reinit(this->introspection().index_sets.system_partitioning,
                               this->introspection().index_sets.system_relevant_partitioning,
                               this->get_mpi_communicator());
      old_field_adjoint = 0.0;
      LinearAlgebra::BlockVector old_old_field_adjoint;
      old_old_field_adjoint.reinit(this->introspection().index_sets.system_partitioning,
                                   this->introspection().index_sets.system_relevant_partitioning,
                                   this->get_mpi_communicator());
      old_old_field_adjoint = 0.0;

      Vector<double> local_old(dofs_per_cell);
      Vector<double> local_old_old(dofs_per_cell);
      unsigned int local_active_max_locations = 0;
      unsigned int local_active_min_locations = 0;
      const double extremum_tolerance =
        1e-12 * std::max({1.0,
                          std::abs(global_min_entropy),
                          std::abs(global_max_entropy)
                         });

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            fe_values[field_extractor].get_function_values(simulator->old_solution,
                                                           old_field_values);
            fe_values[field_extractor].get_function_values(simulator->old_old_solution,
                                                           old_old_field_values);
            local_old = 0.0;
            local_old_old = 0.0;

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double field_history =
                  0.5 * (old_field_values[q] + old_old_field_values[q]);
                const double entropy =
                  (field_history - average_field) * (field_history - average_field);

                double bar_entropy =
                  bar_average_entropy * fe_values.JxW(q) / global_area;
                if (upper_branch
                    && std::abs(entropy - global_max_entropy) <= extremum_tolerance)
                  {
                    bar_entropy += bar_max_entropy;
                    ++local_active_max_locations;
                  }
                if (!upper_branch
                    && std::abs(entropy - global_min_entropy) <= extremum_tolerance)
                  {
                    bar_entropy += bar_min_entropy;
                    ++local_active_min_locations;
                  }

                const double bar_field_history =
                  2.0 * (field_history - average_field) * bar_entropy;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  if (this->get_fe().system_to_component_index(i).first
                      == advection_field.component_index(this->introspection()))
                    {
                      const double contribution =
                        0.5 * bar_field_history
                        * fe_values[field_extractor].value(i, q);
                      local_old(i) += contribution;
                      local_old_old(i) += contribution;
                    }
              }

            this->get_current_constraints().distribute_local_to_global(local_old,
                                                                       local_dof_indices,
                                                                       old_field_adjoint);
            this->get_current_constraints().distribute_local_to_global(local_old_old,
                                                                       local_dof_indices,
                                                                       old_old_field_adjoint);
          }

      old_field_adjoint.compress(VectorOperation::add);
      old_old_field_adjoint.compress(VectorOperation::add);

      LinearAlgebra::BlockVector delta_old;
      delta_old.reinit(this->introspection().index_sets.system_partitioning,
                       this->introspection().index_sets.system_relevant_partitioning,
                       this->get_mpi_communicator());
      delta_old = 0.0;
      LinearAlgebra::BlockVector delta_old_old;
      delta_old_old.reinit(this->introspection().index_sets.system_partitioning,
                           this->introspection().index_sets.system_relevant_partitioning,
                           this->get_mpi_communicator());
      delta_old_old = 0.0;

      const unsigned int field_component =
        advection_field.component_index(this->introspection());
      const IndexSet locally_owned_dofs =
        this->get_dof_handler().locally_owned_dofs();
      std::set<types::global_dof_index> written_dofs;
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const types::global_dof_index global_index = local_dof_indices[i];
                if (locally_owned_dofs.is_element(global_index) == false
                    || written_dofs.insert(global_index).second == false
                    || this->get_fe().system_to_component_index(i).first != field_component)
                  continue;

                delta_old(global_index) =
                  std::sin(0.223 * static_cast<double>(global_index + 1));
                delta_old_old(global_index) =
                  std::cos(0.181 * static_cast<double>(global_index + 1));
              }
          }
      delta_old.compress(VectorOperation::insert);
      delta_old_old.compress(VectorOperation::insert);
      this->get_current_constraints().distribute(delta_old);
      this->get_current_constraints().distribute(delta_old_old);

      const LinearAlgebra::BlockVector saved_old_solution = simulator->old_solution;
      const LinearAlgebra::BlockVector saved_old_old_solution = simulator->old_old_solution;

      auto compute_entropy_variation_with_frozen_average =
        [&](const double sign)
      {
        // Perturb both history vectors along admissible FE directions while
        // passing the unperturbed q_avg to the production forward helper.
        simulator->old_solution = snapshot.old_solution.get();
        simulator->old_old_solution = snapshot.old_old_solution.get();
        simulator->old_solution.add(sign * epsilon, delta_old);
        simulator->old_old_solution.add(sign * epsilon, delta_old_old);
        this->get_current_constraints().distribute(simulator->old_solution);
        this->get_current_constraints().distribute(simulator->old_old_solution);

        return simulator->get_entropy_variation(average_field, advection_field);
      };

      const double entropy_variation_plus =
        compute_entropy_variation_with_frozen_average(1.0);
      const double entropy_variation_minus =
        compute_entropy_variation_with_frozen_average(-1.0);

      simulator->old_solution = saved_old_solution;
      simulator->old_old_solution = saved_old_old_solution;

      double local_adjoint_dot = 0.0;
      written_dofs.clear();
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const types::global_dof_index global_index = local_dof_indices[i];
                if (locally_owned_dofs.is_element(global_index) == false
                    || written_dofs.insert(global_index).second == false
                    || this->get_fe().system_to_component_index(i).first != field_component)
                  continue;

                local_adjoint_dot +=
                  delta_old(global_index) * old_field_adjoint(global_index)
                  + delta_old_old(global_index) * old_old_field_adjoint(global_index);
              }
          }

      const double fd_dot =
        (entropy_variation_plus - entropy_variation_minus) / (2.0 * epsilon);
      const double adjoint_dot =
        Utilities::MPI::sum(local_adjoint_dot, this->get_mpi_communicator());
      const unsigned int active_max_locations =
        Utilities::MPI::sum(local_active_max_locations, this->get_mpi_communicator());
      const unsigned int active_min_locations =
        Utilities::MPI::sum(local_active_min_locations, this->get_mpi_communicator());
      const double relative_error =
        std::abs(fd_dot - adjoint_dot)
        / std::max({std::abs(fd_dot), std::abs(adjoint_dot), 1e-30});

      this->get_pcout() << "   "
                        << (advection_field.is_temperature() ? "Temperature" : "Composition")
                        << "-advection global entropy variation transpose diagnostic: "
                        << "base = "
                        << std::setprecision(16) << base_entropy_variation
                        << ", branch = " << (upper_branch ? "max-average" : "average-min")
                        << ", active max/min locations = "
                        << active_max_locations << "/" << active_min_locations
                        << ", FD = " << fd_dot
                        << ", adjoint = " << adjoint_dot
                        << ", relative error = " << relative_error
                        << std::setprecision(6) << std::endl;
    }


    template <int dim>
    void
    ArtificialViscosityAdjoint<dim>::run_global_entropy_denominator_scalar_diagnostic(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const double epsilon) const
    {
      // Treat the global entropy variation Delta_E as an independent scalar in
      // nu_E=c_R h^2 M/Delta_E. Central differences must match the analytic
      // denominator derivative -bar(nu_E) nu_E/Delta_E. This test freezes all
      // local residuals and active min branches.
      if (this->get_parameters().stabilization_alpha != 2
          || snapshot.timestep_number <= 1)
        return;

      AssertThrow(epsilon > 0.0,
                  ExcMessage("Adjoint/global entropy denominator scalar test epsilon must be positive."));

      const Vector<double> &artificial_viscosity_per_cell =
        simulator->last_advection_artificial_viscosity_per_cell;
      const Vector<double> &minimum_physical_diffusion_per_cell =
        simulator->last_advection_physical_diffusion_min_per_cell;
      const Vector<double> &maximum_physical_diffusion_per_cell =
        simulator->last_advection_physical_diffusion_max_per_cell;

      const unsigned int quadrature_degree =
        this->get_fe().base_element(advection_field.base_element(this->introspection())).degree
        + (this->get_parameters().stokes_velocity_degree + 1) / 2;
      const QGauss<dim> quadrature_formula(quadrature_degree);
      const FEValuesExtractors::Scalar field_extractor =
        advection_field.scalar_extractor(this->introspection());

      const std::pair<double, double> global_field_range =
        simulator->get_extrapolated_advection_field_range(advection_field);
      const double average_field =
        0.5 * (global_field_range.first + global_field_range.second);
      const double global_entropy_variation =
        simulator->get_entropy_variation(average_field, advection_field);
      if (std::abs(global_entropy_variation) < 1e-50)
        return;

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_values |
                              update_gradients |
                              update_hessians |
                              update_quadrature_points);

      std::vector<double> old_field_values(quadrature_formula.size());
      std::vector<double> old_old_field_values(quadrature_formula.size());
      std::vector<Tensor<1, dim>> old_field_gradients(quadrature_formula.size());
      std::vector<Tensor<1, dim>> old_old_field_gradients(quadrature_formula.size());
      std::vector<double> old_field_laplacians(quadrature_formula.size());
      std::vector<double> old_old_field_laplacians(quadrature_formula.size());
      std::vector<Tensor<1, dim>> old_velocity_values(quadrature_formula.size());
      std::vector<Tensor<1, dim>> old_old_velocity_values(quadrature_formula.size());
      std::vector<SymmetricTensor<2, dim>> old_strain_rates(quadrature_formula.size());
      std::vector<SymmetricTensor<2, dim>> old_old_strain_rates(quadrature_formula.size());

      // The same global denominator is consumed by every active cell. Its
      // transpose seed is consequently the sum of all cell contributions,
      // while plus/minus dots vary only this scalar input.
      double local_bar_entropy_variation = 0.0;
      double local_plus_dot = 0.0;
      double local_minus_dot = 0.0;
      unsigned int local_tested_cells = 0;
      const double delta_entropy_variation = 0.371;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int active_cell_index = cell->active_cell_index();
            const double artificial_viscosity =
              artificial_viscosity_per_cell[active_cell_index];
            if (artificial_viscosity <= minimum_physical_diffusion_per_cell[active_cell_index])
              continue;

            AssertThrow(artificial_viscosity > maximum_physical_diffusion_per_cell[active_cell_index],
                        ExcMessage("The global entropy denominator scalar diagnostic currently "
                                   "supports only cells where max(k, nu_art) selects the artificial branch."));

            fe_values.reinit(cell);
            fe_values[field_extractor].get_function_values(simulator->old_solution,
                                                           old_field_values);
            fe_values[field_extractor].get_function_values(simulator->old_old_solution,
                                                           old_old_field_values);
            fe_values[field_extractor].get_function_gradients(simulator->old_solution,
                                                              old_field_gradients);
            fe_values[field_extractor].get_function_gradients(simulator->old_old_solution,
                                                              old_old_field_gradients);
            fe_values[field_extractor].get_function_laplacians(simulator->old_solution,
                                                               old_field_laplacians);
            fe_values[field_extractor].get_function_laplacians(simulator->old_old_solution,
                                                               old_old_field_laplacians);
            fe_values[this->introspection().extractors.velocities]
            .get_function_values(simulator->old_solution,
                                 old_velocity_values);
            fe_values[this->introspection().extractors.velocities]
            .get_function_values(simulator->old_old_solution,
                                 old_old_velocity_values);
            fe_values[this->introspection().extractors.velocities]
            .get_function_symmetric_gradients(simulator->old_solution,
                                              old_strain_rates);
            fe_values[this->introspection().extractors.velocities]
            .get_function_symmetric_gradients(simulator->old_old_solution,
                                              old_old_strain_rates);

            double max_residual_measure = 0.0;
            double max_velocity = 0.0;
            const double h = cell->diameter();
            const double gamma = this->get_parameters().stabilization_gamma;

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double field_dt =
                  (old_field_values[q] - old_old_field_values[q])
                  / snapshot.old_time_step;
                const Tensor<1, dim> velocity =
                  0.5 * (old_velocity_values[q] + old_old_velocity_values[q]);
                const Tensor<1, dim> field_gradient =
                  0.5 * (old_field_gradients[q] + old_old_field_gradients[q]);
                const double laplacian =
                  0.5 * (old_field_laplacians[q] + old_old_field_laplacians[q]);
                const double residual =
                  field_dt + velocity * field_gradient
                  - maximum_physical_diffusion_per_cell[active_cell_index] * laplacian;
                const double history_field =
                  0.5 * (old_field_values[q] + old_old_field_values[q]);
                max_residual_measure =
                  std::max(max_residual_measure,
                           std::abs(residual) * std::abs(history_field - average_field));

                const SymmetricTensor<2, dim> strain_rate =
                  0.5 * (old_strain_rates[q] + old_old_strain_rates[q]);
                max_velocity =
                  std::max(max_velocity,
                           velocity.norm() + gamma * h * strain_rate.norm());
              }

            const double beta =
              this->get_parameters().stabilization_beta[advection_field.field_index()];
            const double c_R =
              this->get_parameters().stabilization_c_R[advection_field.field_index()];
            const double maximum_viscosity = beta * max_velocity * h;
            const double entropy_viscosity =
              c_R * h * h * max_residual_measure / global_entropy_variation;
            if (maximum_viscosity < entropy_viscosity)
              continue;

            const double bar_nu =
              std::cos(0.311 * static_cast<double>(active_cell_index + 1));
            local_bar_entropy_variation +=
              -bar_nu * entropy_viscosity / global_entropy_variation;

            local_plus_dot +=
              bar_nu * c_R * h * h * max_residual_measure
              / (global_entropy_variation + epsilon * delta_entropy_variation);
            local_minus_dot +=
              bar_nu * c_R * h * h * max_residual_measure
              / (global_entropy_variation - epsilon * delta_entropy_variation);
            ++local_tested_cells;
          }

      const double bar_entropy_variation =
        Utilities::MPI::sum(local_bar_entropy_variation,
                            this->get_mpi_communicator());
      const double fd_dot =
        Utilities::MPI::sum((local_plus_dot - local_minus_dot) / (2.0 * epsilon),
                            this->get_mpi_communicator());
      const double adjoint_dot = bar_entropy_variation * delta_entropy_variation;
      const unsigned int tested_cells =
        Utilities::MPI::sum(local_tested_cells, this->get_mpi_communicator());
      const double relative_error =
        std::abs(fd_dot - adjoint_dot)
        / std::max({std::abs(fd_dot), std::abs(adjoint_dot), 1e-30});

      this->get_pcout() << "   "
                        << (advection_field.is_temperature() ? "Temperature" : "Composition")
                        << "-advection global entropy denominator scalar diagnostic: tested cells = "
                        << tested_cells
                        << ", global_entropy_variation = "
                        << std::setprecision(16) << global_entropy_variation
                        << ", bar_global_entropy_variation = "
                        << bar_entropy_variation
                        << ", FD = " << fd_dot
                        << ", adjoint = " << adjoint_dot
                        << ", relative error = " << relative_error
                        << std::setprecision(6) << std::endl;
    }


    template <int dim>
    void
    ArtificialViscosityAdjoint<dim>::run_local_entropy_residual_transpose_diagnostic(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const double epsilon) const
    {
      // Test the numerator M=max_q |R_q| |q_mid-q_avg| with the global
      // denominator frozen. Perturb old/old-old field and velocity states and
      // compare the central directional derivative of the selected local-max
      // branch with the FE adjoint contractions. A second scalar comparison
      // isolates the derivative with respect to q_avg.
      if (this->get_parameters().stabilization_alpha != 2
          || snapshot.timestep_number <= 1)
        return;

      AssertThrow(epsilon > 0.0,
                  ExcMessage("Adjoint/local entropy residual transpose test epsilon must be positive."));

      const Vector<double> &artificial_viscosity_per_cell =
        simulator->last_advection_artificial_viscosity_per_cell;
      const Vector<double> &minimum_physical_diffusion_per_cell =
        simulator->last_advection_physical_diffusion_min_per_cell;
      const Vector<double> &maximum_physical_diffusion_per_cell =
        simulator->last_advection_physical_diffusion_max_per_cell;

      const unsigned int quadrature_degree =
        this->get_fe().base_element(advection_field.base_element(this->introspection())).degree
        + (this->get_parameters().stokes_velocity_degree + 1) / 2;
      const QGauss<dim> quadrature_formula(quadrature_degree);
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      const FEValuesExtractors::Scalar field_extractor =
        advection_field.scalar_extractor(this->introspection());

      const std::pair<double, double> global_field_range =
        simulator->get_extrapolated_advection_field_range(advection_field);
      const double average_field =
        0.5 * (global_field_range.first + global_field_range.second);
      const double global_entropy_variation =
        simulator->get_entropy_variation(average_field, advection_field);

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_values |
                              update_gradients |
                              update_hessians |
                              update_quadrature_points);

      std::vector<double> old_field_values(quadrature_formula.size());
      std::vector<double> old_old_field_values(quadrature_formula.size());
      std::vector<Tensor<1, dim>> old_field_gradients(quadrature_formula.size());
      std::vector<Tensor<1, dim>> old_old_field_gradients(quadrature_formula.size());
      std::vector<double> old_field_laplacians(quadrature_formula.size());
      std::vector<double> old_old_field_laplacians(quadrature_formula.size());
      std::vector<Tensor<1, dim>> old_velocity_values(quadrature_formula.size());
      std::vector<Tensor<1, dim>> old_old_velocity_values(quadrature_formula.size());
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      // Seed only supported active entropy-viscosity cells. The production
      // transpose independently reconstructs the local numerator branch from
      // these coefficients and the archived history states.
      std::vector<double> coefficient_seed(artificial_viscosity_per_cell.size(), 0.0);
      unsigned int local_tested_cells = 0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int active_cell_index = cell->active_cell_index();
            const double artificial_viscosity =
              artificial_viscosity_per_cell[active_cell_index];
            if (artificial_viscosity <= minimum_physical_diffusion_per_cell[active_cell_index])
              continue;

            AssertThrow(artificial_viscosity > maximum_physical_diffusion_per_cell[active_cell_index],
                        ExcMessage("The local entropy residual diagnostic currently supports only "
                                   "cells where max(k, nu_art) selects the artificial branch."));

            fe_values.reinit(cell);
            fe_values[field_extractor].get_function_values(simulator->old_solution,
                                                           old_field_values);
            fe_values[field_extractor].get_function_values(simulator->old_old_solution,
                                                           old_old_field_values);
            fe_values[field_extractor].get_function_gradients(simulator->old_solution,
                                                              old_field_gradients);
            fe_values[field_extractor].get_function_gradients(simulator->old_old_solution,
                                                              old_old_field_gradients);
            fe_values[field_extractor].get_function_laplacians(simulator->old_solution,
                                                               old_field_laplacians);
            fe_values[field_extractor].get_function_laplacians(simulator->old_old_solution,
                                                               old_old_field_laplacians);
            fe_values[this->introspection().extractors.velocities]
            .get_function_values(simulator->old_solution,
                                 old_velocity_values);
            fe_values[this->introspection().extractors.velocities]
            .get_function_values(simulator->old_old_solution,
                                 old_old_velocity_values);

            double max_residual_measure = 0.0;
            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double field_dt =
                  (old_field_values[q] - old_old_field_values[q])
                  / snapshot.old_time_step;
                const Tensor<1, dim> velocity =
                  0.5 * (old_velocity_values[q] + old_old_velocity_values[q]);
                const Tensor<1, dim> field_gradient =
                  0.5 * (old_field_gradients[q] + old_old_field_gradients[q]);
                const double laplacian =
                  0.5 * (old_field_laplacians[q] + old_old_field_laplacians[q]);
                const double residual =
                  field_dt + velocity * field_gradient
                  - maximum_physical_diffusion_per_cell[active_cell_index] * laplacian;
                const double history_field =
                  0.5 * (old_field_values[q] + old_old_field_values[q]);
                max_residual_measure =
                  std::max(max_residual_measure,
                           std::abs(residual) * std::abs(history_field - average_field));
              }

            const double beta =
              this->get_parameters().stabilization_beta[advection_field.field_index()];
            const double c_R =
              this->get_parameters().stabilization_c_R[advection_field.field_index()];
            const double h = cell->diameter();
            const double gamma = this->get_parameters().stabilization_gamma;
            double max_velocity = 0.0;
            std::vector<SymmetricTensor<2, dim>> old_strain_rates(quadrature_formula.size());
            std::vector<SymmetricTensor<2, dim>> old_old_strain_rates(quadrature_formula.size());
            fe_values[this->introspection().extractors.velocities]
            .get_function_symmetric_gradients(simulator->old_solution,
                                              old_strain_rates);
            fe_values[this->introspection().extractors.velocities]
            .get_function_symmetric_gradients(simulator->old_old_solution,
                                              old_old_strain_rates);
            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              max_velocity =
                std::max(max_velocity,
                         (0.5 * (old_velocity_values[q] + old_old_velocity_values[q])).norm()
                         + gamma * h
                         * (0.5 * (old_strain_rates[q] + old_old_strain_rates[q])).norm());

            const double maximum_viscosity = beta * max_velocity * h;
            const double entropy_viscosity =
              c_R * h * h * max_residual_measure / global_entropy_variation;
            if (maximum_viscosity < entropy_viscosity)
              continue;

            const double bar_rmax =
              std::cos(0.271 * static_cast<double>(active_cell_index + 1));
            coefficient_seed[active_cell_index] =
              bar_rmax * global_entropy_variation / (c_R * h * h);
            ++local_tested_cells;
          }

      LinearAlgebra::BlockVector previous_field_adjoint;
      previous_field_adjoint.reinit(this->introspection().index_sets.system_partitioning,
                                    this->introspection().index_sets.system_relevant_partitioning,
                                    this->get_mpi_communicator());
      previous_field_adjoint = 0.0;
      LinearAlgebra::BlockVector previous_previous_field_adjoint;
      previous_previous_field_adjoint.reinit(this->introspection().index_sets.system_partitioning,
                                             this->introspection().index_sets.system_relevant_partitioning,
                                             this->get_mpi_communicator());
      previous_previous_field_adjoint = 0.0;
      LinearAlgebra::BlockVector previous_stokes_adjoint_rhs;
      previous_stokes_adjoint_rhs.reinit(this->introspection().index_sets.system_partitioning,
                                         this->introspection().index_sets.system_relevant_partitioning,
                                         this->get_mpi_communicator());
      previous_stokes_adjoint_rhs = 0.0;
      LinearAlgebra::BlockVector previous_previous_stokes_adjoint_rhs;
      previous_previous_stokes_adjoint_rhs.reinit(this->introspection().index_sets.system_partitioning,
                                                  this->introspection().index_sets.system_relevant_partitioning,
                                                  this->get_mpi_communicator());
      previous_previous_stokes_adjoint_rhs = 0.0;

      add_estimator_transpose(advection_field,
                              snapshot,
                              coefficient_seed,
                              previous_field_adjoint,
                              &previous_previous_field_adjoint,
                              previous_stokes_adjoint_rhs,
                              &previous_previous_stokes_adjoint_rhs);

      LinearAlgebra::BlockVector delta_old;
      delta_old.reinit(this->introspection().index_sets.system_partitioning,
                       this->introspection().index_sets.system_relevant_partitioning,
                       this->get_mpi_communicator());
      delta_old = 0.0;
      LinearAlgebra::BlockVector delta_old_old;
      delta_old_old.reinit(this->introspection().index_sets.system_partitioning,
                           this->introspection().index_sets.system_relevant_partitioning,
                           this->get_mpi_communicator());
      delta_old_old = 0.0;

      const unsigned int field_component =
        advection_field.component_index(this->introspection());
      const IndexSet locally_owned_dofs =
        this->get_dof_handler().locally_owned_dofs();
      std::set<types::global_dof_index> written_dofs;
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const types::global_dof_index global_index = local_dof_indices[i];
                if (locally_owned_dofs.is_element(global_index) == false
                    || written_dofs.insert(global_index).second == false
                    || this->get_fe().system_to_component_index(i).first != field_component)
                  continue;

                delta_old(global_index) =
                  std::sin(0.193 * static_cast<double>(global_index + 1));
                delta_old_old(global_index) =
                  std::cos(0.157 * static_cast<double>(global_index + 1));
              }
          }
      delta_old.compress(VectorOperation::insert);
      delta_old_old.compress(VectorOperation::insert);
      this->get_current_constraints().distribute(delta_old);
      this->get_current_constraints().distribute(delta_old_old);

      auto local_rmax_dot =
        [&](const double sign)
      {
        simulator->old_solution = snapshot.old_solution.get();
        simulator->old_old_solution = snapshot.old_old_solution.get();
        simulator->old_solution.add(sign * epsilon, delta_old);
        simulator->old_old_solution.add(sign * epsilon, delta_old_old);
        this->get_current_constraints().distribute(simulator->old_solution);
        this->get_current_constraints().distribute(simulator->old_old_solution);

        double local_dot = 0.0;
        for (const auto &cell : this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const unsigned int active_cell_index = cell->active_cell_index();
              if (coefficient_seed[active_cell_index] == 0.0)
                continue;

              fe_values.reinit(cell);
              fe_values[field_extractor].get_function_values(simulator->old_solution,
                                                             old_field_values);
              fe_values[field_extractor].get_function_values(simulator->old_old_solution,
                                                             old_old_field_values);
              fe_values[field_extractor].get_function_gradients(simulator->old_solution,
                                                                old_field_gradients);
              fe_values[field_extractor].get_function_gradients(simulator->old_old_solution,
                                                                old_old_field_gradients);
              fe_values[field_extractor].get_function_laplacians(simulator->old_solution,
                                                                 old_field_laplacians);
              fe_values[field_extractor].get_function_laplacians(simulator->old_old_solution,
                                                                 old_old_field_laplacians);
              fe_values[this->introspection().extractors.velocities]
              .get_function_values(simulator->old_solution,
                                   old_velocity_values);
              fe_values[this->introspection().extractors.velocities]
              .get_function_values(simulator->old_old_solution,
                                   old_old_velocity_values);

              double max_residual_measure = 0.0;
              for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                {
                  const double field_dt =
                    (old_field_values[q] - old_old_field_values[q])
                    / snapshot.old_time_step;
                  const Tensor<1, dim> velocity =
                    0.5 * (old_velocity_values[q] + old_old_velocity_values[q]);
                  const Tensor<1, dim> field_gradient =
                    0.5 * (old_field_gradients[q] + old_old_field_gradients[q]);
                  const double laplacian =
                    0.5 * (old_field_laplacians[q] + old_old_field_laplacians[q]);
                  const double residual =
                    field_dt + velocity * field_gradient
                    - maximum_physical_diffusion_per_cell[active_cell_index] * laplacian;
                  const double history_field =
                    0.5 * (old_field_values[q] + old_old_field_values[q]);
                  max_residual_measure =
                    std::max(max_residual_measure,
                             std::abs(residual) * std::abs(history_field - average_field));
                }

              const double c_R =
                this->get_parameters().stabilization_c_R[advection_field.field_index()];
              const double h = cell->diameter();
              const double bar_rmax =
                coefficient_seed[active_cell_index]
                * c_R * h * h / global_entropy_variation;
              local_dot += bar_rmax * max_residual_measure;
            }
        return local_dot;
      };

      const LinearAlgebra::BlockVector saved_old_solution = simulator->old_solution;
      const LinearAlgebra::BlockVector saved_old_old_solution = simulator->old_old_solution;
      const double plus_dot = local_rmax_dot(1.0);
      const double minus_dot = local_rmax_dot(-1.0);
      simulator->old_solution = saved_old_solution;
      simulator->old_old_solution = saved_old_old_solution;

      // Recompute the selected numerator maximum under a scalar perturbation
      // of q_avg. This isolates the entropy-weight derivative from all field
      // and velocity directions used in the main local-residual check.
      auto local_rmax_dot_with_average =
        [&](const double perturbed_average_field)
      {
        double local_dot = 0.0;
        for (const auto &cell : this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const unsigned int active_cell_index = cell->active_cell_index();
              if (coefficient_seed[active_cell_index] == 0.0)
                continue;

              fe_values.reinit(cell);
              fe_values[field_extractor].get_function_values(simulator->old_solution,
                                                             old_field_values);
              fe_values[field_extractor].get_function_values(simulator->old_old_solution,
                                                             old_old_field_values);
              fe_values[field_extractor].get_function_gradients(simulator->old_solution,
                                                                old_field_gradients);
              fe_values[field_extractor].get_function_gradients(simulator->old_old_solution,
                                                                old_old_field_gradients);
              fe_values[field_extractor].get_function_laplacians(simulator->old_solution,
                                                                 old_field_laplacians);
              fe_values[field_extractor].get_function_laplacians(simulator->old_old_solution,
                                                                 old_old_field_laplacians);
              fe_values[this->introspection().extractors.velocities]
              .get_function_values(simulator->old_solution,
                                   old_velocity_values);
              fe_values[this->introspection().extractors.velocities]
              .get_function_values(simulator->old_old_solution,
                                   old_old_velocity_values);

              double max_residual_measure = 0.0;
              for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                {
                  const double field_dt =
                    (old_field_values[q] - old_old_field_values[q])
                    / snapshot.old_time_step;
                  const Tensor<1, dim> velocity =
                    0.5 * (old_velocity_values[q] + old_old_velocity_values[q]);
                  const Tensor<1, dim> field_gradient =
                    0.5 * (old_field_gradients[q] + old_old_field_gradients[q]);
                  const double laplacian =
                    0.5 * (old_field_laplacians[q] + old_old_field_laplacians[q]);
                  const double residual =
                    field_dt + velocity * field_gradient
                    - maximum_physical_diffusion_per_cell[active_cell_index] * laplacian;
                  const double history_field =
                    0.5 * (old_field_values[q] + old_old_field_values[q]);
                  max_residual_measure =
                    std::max(max_residual_measure,
                             std::abs(residual)
                             * std::abs(history_field - perturbed_average_field));
                }

              const double c_R =
                this->get_parameters().stabilization_c_R[advection_field.field_index()];
              const double h = cell->diameter();
              const double bar_rmax =
                coefficient_seed[active_cell_index]
                * c_R * h * h / global_entropy_variation;
              local_dot += bar_rmax * max_residual_measure;
            }
        return local_dot;
      };

      double local_bar_average_field = 0.0;
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int active_cell_index = cell->active_cell_index();
            if (coefficient_seed[active_cell_index] == 0.0)
              continue;

            fe_values.reinit(cell);
            fe_values[field_extractor].get_function_values(simulator->old_solution,
                                                           old_field_values);
            fe_values[field_extractor].get_function_values(simulator->old_old_solution,
                                                           old_old_field_values);
            fe_values[field_extractor].get_function_gradients(simulator->old_solution,
                                                              old_field_gradients);
            fe_values[field_extractor].get_function_gradients(simulator->old_old_solution,
                                                              old_old_field_gradients);
            fe_values[field_extractor].get_function_laplacians(simulator->old_solution,
                                                               old_field_laplacians);
            fe_values[field_extractor].get_function_laplacians(simulator->old_old_solution,
                                                               old_old_field_laplacians);
            fe_values[this->introspection().extractors.velocities]
            .get_function_values(simulator->old_solution,
                                 old_velocity_values);
            fe_values[this->introspection().extractors.velocities]
            .get_function_values(simulator->old_old_solution,
                                 old_old_velocity_values);

            double max_residual_measure = 0.0;
            unsigned int max_residual_q = numbers::invalid_unsigned_int;
            double max_residual_value = 0.0;
            double max_history_field = 0.0;
            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double field_dt =
                  (old_field_values[q] - old_old_field_values[q])
                  / snapshot.old_time_step;
                const Tensor<1, dim> velocity =
                  0.5 * (old_velocity_values[q] + old_old_velocity_values[q]);
                const Tensor<1, dim> field_gradient =
                  0.5 * (old_field_gradients[q] + old_old_field_gradients[q]);
                const double laplacian =
                  0.5 * (old_field_laplacians[q] + old_old_field_laplacians[q]);
                const double residual =
                  field_dt + velocity * field_gradient
                  - maximum_physical_diffusion_per_cell[active_cell_index] * laplacian;
                const double history_field =
                  0.5 * (old_field_values[q] + old_old_field_values[q]);
                const double residual_measure =
                  std::abs(residual) * std::abs(history_field - average_field);
                if (residual_measure > max_residual_measure)
                  {
                    max_residual_measure = residual_measure;
                    max_residual_q = q;
                    max_residual_value = residual;
                    max_history_field = history_field;
                  }
              }

            if (max_residual_q != numbers::invalid_unsigned_int)
              {
                const double c_R =
                  this->get_parameters().stabilization_c_R[advection_field.field_index()];
                const double h = cell->diameter();
                const double bar_rmax =
                  coefficient_seed[active_cell_index]
                  * c_R * h * h / global_entropy_variation;
                const double entropy_weight_sign =
                  (max_history_field - average_field > 0.0) ? 1.0 : -1.0;
                local_bar_average_field -=
                  entropy_weight_sign * std::abs(max_residual_value) * bar_rmax;
              }
          }

      double local_adjoint_dot = 0.0;
      written_dofs.clear();
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const types::global_dof_index global_index = local_dof_indices[i];
                if (locally_owned_dofs.is_element(global_index) == false
                    || written_dofs.insert(global_index).second == false
                    || this->get_fe().system_to_component_index(i).first != field_component)
                  continue;

                local_adjoint_dot +=
                  delta_old(global_index) * previous_field_adjoint(global_index)
                  + delta_old_old(global_index) * previous_previous_field_adjoint(global_index);
              }
          }

      const double fd_dot =
        Utilities::MPI::sum((plus_dot - minus_dot) / (2.0 * epsilon),
                            this->get_mpi_communicator());
      const double adjoint_dot =
        Utilities::MPI::sum(local_adjoint_dot, this->get_mpi_communicator());
      const double average_delta = 0.379;
      const double average_fd_dot =
        Utilities::MPI::sum((local_rmax_dot_with_average(average_field + epsilon * average_delta)
                             - local_rmax_dot_with_average(average_field - epsilon * average_delta))
                            / (2.0 * epsilon),
                            this->get_mpi_communicator());
      const double average_adjoint_dot =
        Utilities::MPI::sum(local_bar_average_field, this->get_mpi_communicator())
        * average_delta;
      const unsigned int tested_cells =
        Utilities::MPI::sum(local_tested_cells, this->get_mpi_communicator());
      const double relative_error =
        std::abs(fd_dot - adjoint_dot)
        / std::max({std::abs(fd_dot), std::abs(adjoint_dot), 1e-30});

      this->get_pcout() << "   "
                        << (advection_field.is_temperature() ? "Temperature" : "Composition")
                        << "-advection local entropy-residual transpose diagnostic: tested cells = "
                        << tested_cells
                        << ", FD = "
                        << std::setprecision(16) << fd_dot
                        << ", adjoint = " << adjoint_dot
                        << ", relative error = " << relative_error
                        << ", average-field FD/adjoint = "
                        << average_fd_dot << "/" << average_adjoint_dot
                        << std::setprecision(6) << std::endl;
    }


    template <int dim>
    void
    ArtificialViscosityAdjoint<dim>::run_estimator_transpose_diagnostic(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const double epsilon) const
    {
      // Exercise the complete estimator map
      //
      //   (q_old,q_old_old,u_old,u_old_old,C_old,C_old_old) -> {nu_K}
      //
      // by checking <bar(nu),D nu[delta x]>=<D nu^T bar(nu),delta x>.
      // Deterministic trigonometric directions cover every locally owned DoF;
      // active-set changes at min/max kinks are excluded by the same support
      // assertions as the production transpose.
      AssertThrow(epsilon > 0.0,
                  ExcMessage("Adjoint/artificial-viscosity estimator transpose test epsilon must be positive."));

      const Vector<double> &artificial_viscosity_per_cell =
        simulator->last_advection_artificial_viscosity_per_cell;
      const Vector<double> &minimum_physical_diffusion_per_cell =
        simulator->last_advection_physical_diffusion_min_per_cell;
      const Vector<double> &maximum_physical_diffusion_per_cell =
        simulator->last_advection_physical_diffusion_max_per_cell;
      AssertThrow(artificial_viscosity_per_cell.size()
                  == this->get_dof_handler().get_triangulation().n_active_cells(),
                  ExcMessage("The artificial-viscosity estimator transpose diagnostic "
                             "requires a restored advection assembly with cached "
                             "cellwise artificial-viscosity coefficients."));

      // A deterministic cellwise bar(nu_K) pattern exercises accumulation
      // across the complete estimator graph without depending on MPI rank
      // numbering. Unsupported physical-diffusion cells retain zero seed.
      std::vector<double> coefficient_seed(artificial_viscosity_per_cell.size(), 0.0);
      unsigned int local_tested_cells = 0;
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int active_cell_index = cell->active_cell_index();
            const double artificial_viscosity =
              artificial_viscosity_per_cell[active_cell_index];

            if (artificial_viscosity <= minimum_physical_diffusion_per_cell[active_cell_index])
              continue;

            AssertThrow(artificial_viscosity > maximum_physical_diffusion_per_cell[active_cell_index],
                        ExcMessage("The artificial-viscosity estimator transpose diagnostic currently "
                                   "supports only cells where max(k, nu_art) selects the artificial "
                                   "diffusion branch at every quadrature point."));

            coefficient_seed[active_cell_index] =
              std::cos(0.311 * static_cast<double>(active_cell_index + 1));
            ++local_tested_cells;
          }

      LinearAlgebra::BlockVector previous_field_adjoint;
      previous_field_adjoint.reinit(this->introspection().index_sets.system_partitioning,
                                    this->introspection().index_sets.system_relevant_partitioning,
                                    this->get_mpi_communicator());
      previous_field_adjoint = 0.0;

      LinearAlgebra::BlockVector previous_previous_field_adjoint;
      previous_previous_field_adjoint.reinit(this->introspection().index_sets.system_partitioning,
                                             this->introspection().index_sets.system_relevant_partitioning,
                                             this->get_mpi_communicator());
      previous_previous_field_adjoint = 0.0;

      LinearAlgebra::BlockVector previous_stokes_adjoint_rhs;
      previous_stokes_adjoint_rhs.reinit(this->introspection().index_sets.system_partitioning,
                                         this->introspection().index_sets.system_relevant_partitioning,
                                         this->get_mpi_communicator());
      previous_stokes_adjoint_rhs = 0.0;

      LinearAlgebra::BlockVector previous_previous_stokes_adjoint_rhs;
      previous_previous_stokes_adjoint_rhs.reinit(this->introspection().index_sets.system_partitioning,
                                                  this->introspection().index_sets.system_relevant_partitioning,
                                                  this->get_mpi_communicator());
      previous_previous_stokes_adjoint_rhs = 0.0;

      add_estimator_transpose(advection_field,
                              snapshot,
                              coefficient_seed,
                              previous_field_adjoint,
                              snapshot.timestep_number > 1 ? &previous_previous_field_adjoint : nullptr,
                              previous_stokes_adjoint_rhs,
                              snapshot.timestep_number > 1 ? &previous_previous_stokes_adjoint_rhs : nullptr);

      const unsigned int field_component =
        advection_field.component_index(this->introspection());
      const IndexSet locally_owned_dofs =
        this->get_dof_handler().locally_owned_dofs();
      std::vector<types::global_dof_index> local_dof_indices(this->get_fe().dofs_per_cell);
      std::set<types::global_dof_index> written_dofs;

      LinearAlgebra::BlockVector delta_old;
      delta_old.reinit(this->introspection().index_sets.system_partitioning,
                       this->introspection().index_sets.system_relevant_partitioning,
                       this->get_mpi_communicator());
      delta_old = 0.0;

      LinearAlgebra::BlockVector delta_old_old;
      delta_old_old.reinit(this->introspection().index_sets.system_partitioning,
                           this->introspection().index_sets.system_relevant_partitioning,
                           this->get_mpi_communicator());
      delta_old_old = 0.0;

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

                if (component == field_component)
                  {
                    delta_old(global_index) =
                      std::sin(0.193 * static_cast<double>(global_index + 1));
                    delta_old_old(global_index) =
                      snapshot.timestep_number > 1
                      ? std::cos(0.157 * static_cast<double>(global_index + 1))
                      : delta_old(global_index);
                  }
                else if (this->introspection().component_masks.velocities[component])
                  {
                    delta_old(global_index) =
                      std::sin(0.113 * static_cast<double>(global_index + 1));
                    delta_old_old(global_index) =
                      snapshot.timestep_number > 1
                      ? std::cos(0.097 * static_cast<double>(global_index + 1))
                      : delta_old(global_index);
                  }
              }
          }

      delta_old.compress(VectorOperation::insert);
      delta_old_old.compress(VectorOperation::insert);
      this->get_current_constraints().distribute(delta_old);
      this->get_current_constraints().distribute(delta_old_old);

      double local_adjoint_dot = 0.0;
      double local_field_adjoint_dot = 0.0;
      double local_velocity_adjoint_dot = 0.0;
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

                if (component == field_component)
                  {
                    const double contribution =
                      delta_old(global_index)
                      * previous_field_adjoint(global_index);
                    local_adjoint_dot += contribution;
                    local_field_adjoint_dot += contribution;

                    if (snapshot.timestep_number > 1)
                      {
                        const double old_old_contribution =
                          delta_old_old(global_index)
                          * previous_previous_field_adjoint(global_index);
                        local_adjoint_dot += old_old_contribution;
                        local_field_adjoint_dot += old_old_contribution;
                      }
                  }
                else if (this->introspection().component_masks.velocities[component])
                  {
                    const double contribution =
                      delta_old(global_index)
                      * previous_stokes_adjoint_rhs(global_index);
                    local_adjoint_dot += contribution;
                    local_velocity_adjoint_dot += contribution;

                    if (snapshot.timestep_number > 1)
                      {
                        const double old_old_contribution =
                          delta_old_old(global_index)
                          * previous_previous_stokes_adjoint_rhs(global_index);
                        local_adjoint_dot += old_old_contribution;
                        local_velocity_adjoint_dot += old_old_contribution;
                      }
                  }
              }
          }

      const LinearAlgebra::BlockVector saved_old_solution = simulator->old_solution;
      const LinearAlgebra::BlockVector saved_old_old_solution = simulator->old_old_solution;

      auto recompute_artificial_viscosity_dot =
        [&](const double sign,
            const bool perturb_field,
            const bool perturb_velocity)
      {
        simulator->old_solution = snapshot.old_solution.get();
        simulator->old_old_solution = snapshot.old_old_solution.get();
        if (perturb_field || perturb_velocity)
          {
            LinearAlgebra::BlockVector masked_delta_old(delta_old);
            LinearAlgebra::BlockVector masked_delta_old_old(delta_old_old);
            const IndexSet locally_owned_dofs =
              this->get_dof_handler().locally_owned_dofs();
            std::vector<types::global_dof_index> local_dof_indices(this->get_fe().dofs_per_cell);
            std::set<types::global_dof_index> written_dofs;
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

                      const bool keep =
                        (perturb_field && component == field_component)
                        || (perturb_velocity
                            && this->introspection().component_masks.velocities[component]);
                      if (keep == false)
                        {
                          masked_delta_old(global_index) = 0.0;
                          masked_delta_old_old(global_index) = 0.0;
                        }
                    }
                }
            masked_delta_old.compress(VectorOperation::insert);
            masked_delta_old_old.compress(VectorOperation::insert);
            simulator->old_solution.add(sign * epsilon, masked_delta_old);
            simulator->old_old_solution.add(sign * epsilon, masked_delta_old_old);
          }
        this->get_current_constraints().distribute(simulator->old_solution);
        this->get_current_constraints().distribute(simulator->old_old_solution);

        Vector<double> perturbed_viscosity;
        perturbed_viscosity.reinit(artificial_viscosity_per_cell.size());
        simulator->get_artificial_viscosity(perturbed_viscosity,
                                            advection_field);

        double local_dot = 0.0;
        for (const auto &cell : this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const unsigned int active_cell_index = cell->active_cell_index();
              local_dot += coefficient_seed[active_cell_index]
                           * perturbed_viscosity[active_cell_index];
            }

        return local_dot;
      };

      const double plus_dot = recompute_artificial_viscosity_dot(1.0, true, true);
      const double minus_dot = recompute_artificial_viscosity_dot(-1.0, true, true);
      const double field_plus_dot = recompute_artificial_viscosity_dot(1.0, true, false);
      const double field_minus_dot = recompute_artificial_viscosity_dot(-1.0, true, false);
      const double velocity_plus_dot = recompute_artificial_viscosity_dot(1.0, false, true);
      const double velocity_minus_dot = recompute_artificial_viscosity_dot(-1.0, false, true);

      simulator->old_solution = saved_old_solution;
      simulator->old_old_solution = saved_old_old_solution;

      const double fd_dot =
        Utilities::MPI::sum((plus_dot - minus_dot) / (2.0 * epsilon),
                            this->get_mpi_communicator());
      const double adjoint_dot =
        Utilities::MPI::sum(local_adjoint_dot, this->get_mpi_communicator());
      const double field_fd_dot =
        Utilities::MPI::sum((field_plus_dot - field_minus_dot) / (2.0 * epsilon),
                            this->get_mpi_communicator());
      const double field_adjoint_dot =
        Utilities::MPI::sum(local_field_adjoint_dot, this->get_mpi_communicator());
      const double velocity_fd_dot =
        Utilities::MPI::sum((velocity_plus_dot - velocity_minus_dot) / (2.0 * epsilon),
                            this->get_mpi_communicator());
      const double velocity_adjoint_dot =
        Utilities::MPI::sum(local_velocity_adjoint_dot, this->get_mpi_communicator());
      const unsigned int tested_cells =
        Utilities::MPI::sum(local_tested_cells, this->get_mpi_communicator());
      const double relative_error =
        std::abs(fd_dot - adjoint_dot)
        / std::max({std::abs(fd_dot), std::abs(adjoint_dot), 1e-30});

      this->get_pcout() << "   "
                        << (advection_field.is_temperature() ? "Temperature" : "Composition")
                        << "-advection artificial-viscosity estimator transpose diagnostic: tested cells = "
                        << tested_cells
                        << ", FD = "
                        << std::setprecision(16) << fd_dot
                        << ", adjoint = " << adjoint_dot
                        << ", relative error = " << relative_error
                        << ", field FD/adjoint = " << field_fd_dot
                        << "/" << field_adjoint_dot
                        << ", velocity FD/adjoint = " << velocity_fd_dot
                        << "/" << velocity_adjoint_dot
                        << std::setprecision(6) << std::endl;
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template void Adjoint::ArtificialViscosityAdjoint<dim>::run_coefficient_seed_diagnostic( \
      const AdvectionField &, const Adjoint::ForwardStateView<dim> &, \
      const LinearAlgebra::BlockVector &, const std::vector<double> &, const double) const; \
  template void Adjoint::ArtificialViscosityAdjoint<dim>::run_global_entropy_variation_transpose_diagnostic( \
      const AdvectionField &, const Adjoint::ForwardStateView<dim> &, const double) const; \
  template void Adjoint::ArtificialViscosityAdjoint<dim>::run_global_entropy_denominator_scalar_diagnostic( \
      const AdvectionField &, const Adjoint::ForwardStateView<dim> &, const double) const; \
  template void Adjoint::ArtificialViscosityAdjoint<dim>::run_local_entropy_residual_transpose_diagnostic( \
      const AdvectionField &, const Adjoint::ForwardStateView<dim> &, const double) const; \
  template void Adjoint::ArtificialViscosityAdjoint<dim>::run_estimator_transpose_diagnostic( \
      const AdvectionField &, const Adjoint::ForwardStateView<dim> &, const double) const;

  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
