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
    ArtificialViscosityAdjoint<dim>::ArtificialViscosityAdjoint(Simulator<dim> &simulator_object)
      :
      simulator(&simulator_object)
    {
      this->initialize_simulator(simulator_object);
    }



    template <int dim>
    std::vector<double>
    ArtificialViscosityAdjoint<dim>::assemble_coefficient_seed(const AdvectionField &advection_field,
                                                               const ForwardStateView<dim> &snapshot,
                                                               const LinearAlgebra::BlockVector &transpose_solution) const
    {
      AssertThrow(this->get_parameters().advection_stabilization_method
                  == Parameters<dim>::AdvectionStabilizationMethod::entropy_viscosity,
                  ExcMessage("ArtificialViscosityAdjoint currently supports only entropy-viscosity stabilization."));

      const Vector<double> &artificial_viscosity_per_cell =
        simulator->last_advection_artificial_viscosity_per_cell;
      const Vector<double> &minimum_physical_diffusion_per_cell =
        simulator->last_advection_physical_diffusion_min_per_cell;
      const Vector<double> &maximum_physical_diffusion_per_cell =
        simulator->last_advection_physical_diffusion_max_per_cell;
      AssertThrow(artificial_viscosity_per_cell.size()
                  == this->get_dof_handler().get_triangulation().n_active_cells(),
                  ExcMessage("The artificial-viscosity adjoint seed requires the advection "
                             "system to be assembled first so that the cellwise stabilization "
                             "coefficients are available."));
      AssertThrow(minimum_physical_diffusion_per_cell.size() == artificial_viscosity_per_cell.size() &&
                  maximum_physical_diffusion_per_cell.size() == artificial_viscosity_per_cell.size(),
                  ExcMessage("The artificial-viscosity adjoint seed requires cached physical diffusion "
                             "bounds from the same temperature advection assembly."));

      std::vector<double> coefficient_seed(artificial_viscosity_per_cell.size(), 0.0);

      const unsigned int quadrature_degree =
        this->get_fe().base_element(advection_field.base_element(this->introspection())).degree + 1;
      const QGauss<dim> quadrature_formula(quadrature_degree);
      const FEValuesExtractors::Scalar field_extractor =
        advection_field.scalar_extractor(this->introspection());

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_gradients |
                              update_quadrature_points |
                              update_JxW_values);

      std::vector<Tensor<1, dim>> field_gradients(quadrature_formula.size());
      std::vector<Tensor<1, dim>> transpose_gradients(quadrature_formula.size());

      // Reverse the stabilized diffusion contribution on each cell K. In the
      // forward residual the artificial coefficient enters as
      //
      //   R_i^K += dt * integral_K nu_K grad(q_n) . grad(phi_i) dx,
      //
      // where nu_K=max(kappa,nu_art,K) is constant on K. Contracting its
      // derivative with the advection adjoint z gives
      //
      //   bar(nu_art,K) = -dt * integral_K grad(z) . grad(q_n) dx.
      //
      // The sign is the one induced by the residual convention used by the
      // reverse time step. Cells where physical diffusion wins the
      // max(kappa,nu_art,K) active set receive no artificial-viscosity seed.
      // A strict inequality is required here: differentiating through a tie
      // would require a separately specified generalized derivative.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int active_cell_index = cell->active_cell_index();
            const double artificial_viscosity =
              artificial_viscosity_per_cell[active_cell_index];
            const double minimum_physical_diffusion =
              minimum_physical_diffusion_per_cell[active_cell_index];
            const double maximum_physical_diffusion =
              maximum_physical_diffusion_per_cell[active_cell_index];

            if (artificial_viscosity <= minimum_physical_diffusion)
              continue;

            AssertThrow(artificial_viscosity > maximum_physical_diffusion,
                        ExcMessage("The artificial-viscosity adjoint seed currently supports only cells "
                                   "where max(k, nu_art) selects the same branch at every quadrature point. "
                                   "A cell with mixed physical/artificial diffusion branches needs a "
                                   "quadrature-point active-set cache."));

            fe_values.reinit(cell);

            fe_values[field_extractor]
            .get_function_gradients(this->get_solution(),
                                    field_gradients);
            fe_values[field_extractor]
            .get_function_gradients(transpose_solution,
                                    transpose_gradients);

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              coefficient_seed[active_cell_index] -=
                snapshot.time_step
                * (transpose_gradients[q] * field_gradients[q])
                * fe_values.JxW(q);
          }

      return coefficient_seed;
    }







    template <int dim>
    void
    ArtificialViscosityAdjoint<dim>::add_global_entropy_variation_transpose(
      const AdvectionField &advection_field,
      const double bar_entropy_variation,
      double &bar_average_field,
      LinearAlgebra::BlockVector &previous_field_adjoint,
      LinearAlgebra::BlockVector *previous_previous_field_adjoint) const
    {
      if (bar_entropy_variation == 0.0)
        return;

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

      // For alpha=2 ASPECT normalizes the entropy residual by
      //
      //   Delta_E = max(max_x E(x)-<E>, <E>-min_x E(x)),
      //   E(x)     = (q_h(x)-q_avg)^2,
      //   q_h      = (q_{n-1}+q_{n-2})/2.
      //
      // This first pass reconstructs the four global scalars that define the
      // selected branch. Their MPI reductions are part of the discrete
      // operator: the transpose must use the same globally selected extrema
      // on every rank.

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
      // Reverse the selected branch of Delta_E. For the upper branch,
      // bar(E_max)=bar(Delta_E) and bar(<E>)=-bar(Delta_E); the signs reverse
      // for the lower branch. At equality the forward comparison selects the
      // upper branch, which also fixes the adjoint convention at the kink.
      const double bar_max_entropy = upper_branch ? bar_entropy_variation : 0.0;
      const double bar_min_entropy = upper_branch ? 0.0 : -bar_entropy_variation;
      const double bar_average_entropy = upper_branch
                                         ? -bar_entropy_variation
                                         : bar_entropy_variation;
      const double extremum_tolerance =
        1e-12 * std::max({1.0,
                          std::abs(global_min_entropy),
                          std::abs(global_max_entropy)
                         });

      Vector<double> local_previous(dofs_per_cell);
      Vector<double> local_previous_previous(dofs_per_cell);
      double local_bar_average_field = 0.0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            fe_values[field_extractor].get_function_values(simulator->old_solution,
                                                           old_field_values);
            fe_values[field_extractor].get_function_values(simulator->old_old_solution,
                                                           old_old_field_values);
            local_previous = 0.0;
            local_previous_previous = 0.0;

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double field_history =
                  0.5 * (old_field_values[q] + old_old_field_values[q]);
                const double entropy =
                  (field_history - average_field) * (field_history - average_field);

                // <E> is a volume average, so its transpose distributes the
                // scalar seed as JxW/|Omega|. The seed of a global extremum is
                // placed at all quadrature points in its tolerance-defined
                // active set. This mirrors the forward quadrature sampling;
                // it is not a derivative of the continuous pointwise max.
                double bar_entropy =
                  bar_average_entropy * fe_values.JxW(q) / global_area;
                if (upper_branch
                    && std::abs(entropy - global_max_entropy) <= extremum_tolerance)
                  bar_entropy += bar_max_entropy;
                if (!upper_branch
                    && std::abs(entropy - global_min_entropy) <= extremum_tolerance)
                  bar_entropy += bar_min_entropy;

                // E=(q_h-q_avg)^2 gives
                //   bar(q_h)   =  2(q_h-q_avg) bar(E),
                //   bar(q_avg) = -2(q_h-q_avg) bar(E).
                // Because q_h is the arithmetic midpoint, both stored
                // histories receive one half of bar(q_h).
                const double bar_field_history =
                  2.0 * (field_history - average_field) * bar_entropy;
                local_bar_average_field +=
                  -2.0 * (field_history - average_field) * bar_entropy;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  if (this->get_fe().system_to_component_index(i).first
                      == advection_field.component_index(this->introspection()))
                    {
                      const double contribution =
                        0.5 * bar_field_history
                        * fe_values[field_extractor].value(i, q);
                      local_previous(i) += contribution;
                      local_previous_previous(i) += contribution;
                    }
              }

            this->get_current_constraints().distribute_local_to_global(local_previous,
                                                                       local_dof_indices,
                                                                       previous_field_adjoint);
            AssertThrow(previous_previous_field_adjoint != nullptr,
                        ExcInternalError());
            this->get_current_constraints().distribute_local_to_global(local_previous_previous,
                                                                       local_dof_indices,
                                                                       *previous_previous_field_adjoint);
          }

      previous_field_adjoint.compress(VectorOperation::add);
      previous_previous_field_adjoint->compress(VectorOperation::add);
      bar_average_field +=
        Utilities::MPI::sum(local_bar_average_field, this->get_mpi_communicator());
    }



    template <int dim>
    void
    ArtificialViscosityAdjoint<dim>::add_extrapolated_field_average_transpose(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const double bar_average_field,
      LinearAlgebra::BlockVector &previous_field_adjoint,
      LinearAlgebra::BlockVector *previous_previous_field_adjoint) const
    {
      if (bar_average_field == 0.0)
        return;

      if (snapshot.timestep_number > 1)
        AssertThrow(previous_previous_field_adjoint != nullptr,
                    ExcInternalError());

      const QIterated<dim> quadrature_formula(QTrapezoid<1>(),
                                              advection_field.polynomial_degree(this->introspection()));
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      const FEValuesExtractors::Scalar field_extractor =
        advection_field.scalar_extractor(this->introspection());

      const std::pair<double, double> global_field_range =
        simulator->get_extrapolated_advection_field_range(advection_field);
      const double global_min_field = global_field_range.first;
      const double global_max_field = global_field_range.second;
      const double extremum_tolerance =
        1e-12 * std::max({1.0,
                          std::abs(global_min_field),
                          std::abs(global_max_field)
                         });

      // q_avg=(q_min+q_max)/2 is formed from extrema of the extrapolated field
      //
      //   q* = (1+dt/dt_old) q_{n-1} - (dt/dt_old) q_{n-2}.
      //
      // The min/max operation is nonsmooth. Split each extremum seed evenly
      // over every sampled quadrature location in its tolerance-defined
      // active set. This deterministic generalized derivative is the same
      // convention used by the finite-difference diagnostics.
      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_values);

      std::vector<double> old_field_values(quadrature_formula.size());
      std::vector<double> old_old_field_values(quadrature_formula.size());
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      unsigned int local_min_count = 0;
      unsigned int local_max_count = 0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[field_extractor].get_function_values(simulator->old_solution,
                                                           old_field_values);
            if (snapshot.timestep_number > 1)
              fe_values[field_extractor].get_function_values(simulator->old_old_solution,
                                                             old_old_field_values);

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double extrapolated_field =
                  (snapshot.timestep_number > 1)
                  ? (1.0 + snapshot.time_step / snapshot.old_time_step) * old_field_values[q]
                  - snapshot.time_step / snapshot.old_time_step * old_old_field_values[q]
                  : old_field_values[q];
                if (std::abs(extrapolated_field - global_min_field) <= extremum_tolerance)
                  ++local_min_count;
                if (std::abs(extrapolated_field - global_max_field) <= extremum_tolerance)
                  ++local_max_count;
              }
          }

      const unsigned int global_min_count =
        Utilities::MPI::sum(local_min_count, this->get_mpi_communicator());
      const unsigned int global_max_count =
        Utilities::MPI::sum(local_max_count, this->get_mpi_communicator());
      AssertThrow(global_min_count > 0 && global_max_count > 0,
                  ExcInternalError());

      this->get_pcout() << "   "
                        << (advection_field.is_temperature() ? "Temperature" : "Composition")
                        << "-advection extrapolated field range active min/max locations = "
                        << global_min_count << "/" << global_max_count
                        << std::endl;

      // Since q_avg=0.5(q_min+q_max), both extrema receive half of the
      // incoming seed. Division by the active-set cardinality ensures that a
      // plateau represents one global extremum rather than multiplying the
      // derivative by the number of MPI-owned samples on that plateau.
      const double bar_min_extrapolated =
        0.5 * bar_average_field / static_cast<double>(global_min_count);
      const double bar_max_extrapolated =
        0.5 * bar_average_field / static_cast<double>(global_max_count);
      const double old_factor =
        (snapshot.timestep_number > 1)
        ? 1.0 + snapshot.time_step / snapshot.old_time_step
        : 1.0;
      const double old_old_factor =
        (snapshot.timestep_number > 1)
        ? -snapshot.time_step / snapshot.old_time_step
        : 0.0;

      // Apply the transpose of the extrapolation map. BDF1 has only q_{n-1};
      // BDF2 sends old_factor*bar(q*) to q_{n-1} and
      // old_old_factor*bar(q*) to q_{n-2}. The FE scatter below is precisely
      // the transpose of evaluating those history fields at quadrature
      // points.
      Vector<double> local_previous(dofs_per_cell);
      Vector<double> local_previous_previous(dofs_per_cell);

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            fe_values[field_extractor].get_function_values(simulator->old_solution,
                                                           old_field_values);
            if (snapshot.timestep_number > 1)
              fe_values[field_extractor].get_function_values(simulator->old_old_solution,
                                                             old_old_field_values);
            local_previous = 0.0;
            local_previous_previous = 0.0;

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double extrapolated_field =
                  (snapshot.timestep_number > 1)
                  ? old_factor * old_field_values[q]
                  + old_old_factor * old_old_field_values[q]
                  : old_field_values[q];
                double bar_extrapolated_field = 0.0;
                if (std::abs(extrapolated_field - global_min_field) <= extremum_tolerance)
                  bar_extrapolated_field += bar_min_extrapolated;
                if (std::abs(extrapolated_field - global_max_field) <= extremum_tolerance)
                  bar_extrapolated_field += bar_max_extrapolated;

                if (bar_extrapolated_field == 0.0)
                  continue;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  if (this->get_fe().system_to_component_index(i).first
                      == advection_field.component_index(this->introspection()))
                    {
                      const double phi_field =
                        fe_values[field_extractor].value(i, q);
                      local_previous(i) +=
                        old_factor * bar_extrapolated_field * phi_field;
                      if (snapshot.timestep_number > 1)
                        local_previous_previous(i) +=
                          old_old_factor * bar_extrapolated_field * phi_field;
                    }
              }

            this->get_current_constraints().distribute_local_to_global(local_previous,
                                                                       local_dof_indices,
                                                                       previous_field_adjoint);
            if (snapshot.timestep_number > 1)
              this->get_current_constraints().distribute_local_to_global(local_previous_previous,
                                                                         local_dof_indices,
                                                                         *previous_previous_field_adjoint);
          }

      previous_field_adjoint.compress(VectorOperation::add);
      if (snapshot.timestep_number > 1)
        previous_previous_field_adjoint->compress(VectorOperation::add);
    }



















    template <int dim>
    void
    ArtificialViscosityAdjoint<dim>::add_estimator_transpose(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const std::vector<double> &coefficient_seed,
      LinearAlgebra::BlockVector &previous_field_adjoint,
      LinearAlgebra::BlockVector *previous_previous_field_adjoint,
      LinearAlgebra::BlockVector &previous_stokes_adjoint_rhs,
      LinearAlgebra::BlockVector *previous_previous_stokes_adjoint_rhs,
      std::vector<LinearAlgebra::BlockVector> *previous_composition_adjoints,
      std::vector<LinearAlgebra::BlockVector> *previous_previous_composition_adjoints) const
    {
      // The transpose of a zero coefficient seed is exactly zero. Returning
      // here also avoids calling compress(add) on untouched vectors, whose
      // preceding action may legitimately have been an insert from another
      // reverse contribution.
      const bool locally_active_seed =
        std::any_of(coefficient_seed.begin(), coefficient_seed.end(),
                    [](const double value)
      {
        return value != 0.0;
      });
      if (Utilities::MPI::max(locally_active_seed ? 1 : 0,
                              this->get_mpi_communicator()) == 0)
        return;

      const unsigned int stabilization_alpha =
        this->get_parameters().stabilization_alpha;
      AssertThrow(stabilization_alpha == 1 || stabilization_alpha == 2,
                  ExcMessage("The strict artificial-viscosity estimator adjoint supports only alpha = 1 or alpha = 2."));
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
      const double global_field_variation =
        global_field_range.second - global_field_range.first;
      const double global_entropy_variation =
        stabilization_alpha == 2
        ? simulator->get_entropy_variation(average_field, advection_field)
        : numbers::signaling_nan<double>();
      const double global_max_velocity =
        simulator->get_maximal_velocity(simulator->old_solution);

      // The forward cell coefficient is the smaller of a transport cap and
      // an entropy-residual estimate:
      //
      //   nu_max = beta h a_max u_max,
      //   nu_E2  = c_R h^2 max_q(|R_q| |q_q-q_avg|) / Delta_E,
      //   nu_E1  = c_R h L u_max max_q|R_q|
      //            / (u_global (q_global,max-q_global,min)),
      //   nu_art = min(nu_max,nu_Ealpha).
      //
      // Here a=rho*C_p for temperature and one for composition. The remainder
      // of this routine first reconstructs the active quadrature points and
      // branch, then traverses these expressions in reverse order.

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
      std::vector<double> old_temperature_values(quadrature_formula.size());
      std::vector<double> old_old_temperature_values(quadrature_formula.size());
      std::vector<double> old_pressure_values(quadrature_formula.size());
      std::vector<double> old_old_pressure_values(quadrature_formula.size());
      std::vector<std::vector<double>> old_composition_values(this->introspection().n_compositional_fields,
                                                               std::vector<double>(quadrature_formula.size()));
      std::vector<std::vector<double>> old_old_composition_values(this->introspection().n_compositional_fields,
                                                                   std::vector<double>(quadrature_formula.size()));

      Vector<double> local_previous_field(dofs_per_cell);
      Vector<double> local_previous_previous_field(dofs_per_cell);
      Vector<double> local_previous_stokes(dofs_per_cell);
      Vector<double> local_previous_previous_stokes(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      const Vector<double> &minimum_physical_diffusion_per_cell =
        simulator->last_advection_physical_diffusion_min_per_cell;
      const Vector<double> &maximum_physical_diffusion_per_cell =
        simulator->last_advection_physical_diffusion_max_per_cell;
      double local_bar_global_entropy_variation = 0.0;
      double local_bar_average_field = 0.0;
      const bool use_bdf2_scheme = (snapshot.timestep_number > 1);

      // Work cell by cell because the forward estimator stores one
      // artificial-viscosity coefficient per active cell. Only cells with a
      // nonzero seed and an artificial-diffusion active branch participate in
      // the reverse pass.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int active_cell_index = cell->active_cell_index();
            const double bar_nu = coefficient_seed[active_cell_index];
            if (bar_nu == 0.0)
              continue;

            const double artificial_viscosity =
              simulator->last_advection_artificial_viscosity_per_cell[active_cell_index];
            if (artificial_viscosity <= minimum_physical_diffusion_per_cell[active_cell_index])
              continue;

            AssertThrow(artificial_viscosity > maximum_physical_diffusion_per_cell[active_cell_index],
                        ExcMessage("The artificial-viscosity estimator adjoint currently supports only cells "
                                   "where max(k, nu_art) selects the same branch at every quadrature point."));

            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);

            fe_values[field_extractor]
            .get_function_values(simulator->old_solution,
                                 old_field_values);
            fe_values[field_extractor]
            .get_function_values(simulator->old_old_solution,
                                 old_old_field_values);
            fe_values[field_extractor]
            .get_function_gradients(simulator->old_solution,
                                    old_field_gradients);
            fe_values[field_extractor]
            .get_function_gradients(simulator->old_old_solution,
                                    old_old_field_gradients);
            fe_values[field_extractor]
            .get_function_laplacians(simulator->old_solution,
                                     old_field_laplacians);
            fe_values[field_extractor]
            .get_function_laplacians(simulator->old_old_solution,
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
            if (advection_field.is_temperature())
              {
                fe_values[this->introspection().extractors.temperature]
                .get_function_values(simulator->old_solution,
                                     old_temperature_values);
                fe_values[this->introspection().extractors.temperature]
                .get_function_values(simulator->old_old_solution,
                                     old_old_temperature_values);
                fe_values[this->introspection().extractors.pressure]
                .get_function_values(simulator->old_solution,
                                     old_pressure_values);
                fe_values[this->introspection().extractors.pressure]
                .get_function_values(simulator->old_old_solution,
                                     old_old_pressure_values);
                for (unsigned int c = 0; c < this->introspection().n_compositional_fields; ++c)
                  {
                    fe_values[this->introspection().extractors.compositional_fields[c]]
                    .get_function_values(simulator->old_solution,
                                         old_composition_values[c]);
                    fe_values[this->introspection().extractors.compositional_fields[c]]
                    .get_function_values(simulator->old_old_solution,
                                         old_old_composition_values[c]);
                  }
              }

            // The estimator contains nested max operations. Retain the
            // winning quadrature locations and the primitive values needed by
            // their vector-Jacobian products. This deliberately follows the
            // same strict '>' tie-breaking as the forward estimator.
            double max_velocity = 0.0;
            unsigned int max_velocity_q = numbers::invalid_unsigned_int;
            double max_velocity_norm = 0.0;
            double max_strain_rate_norm = 0.0;
            double max_residual_measure = 0.0;
            unsigned int max_residual_q = numbers::invalid_unsigned_int;
            double max_residual_value = 0.0;
            double max_residual_entropy_weight = 0.0;
            Tensor<1, dim> max_residual_velocity;
            Tensor<1, dim> max_residual_temperature_gradient;
            double max_residual_advection_prefactor = 1.0;
            double max_residual_transport_residual = 0.0;
            unsigned int max_advection_prefactor_q = numbers::invalid_unsigned_int;

            const double physical_diffusion =
              maximum_physical_diffusion_per_cell[active_cell_index];
            const double h = cell->diameter();
            const double gamma = this->get_parameters().stabilization_gamma;
            double max_advection_prefactor = 1.0;
            std::vector<double> advection_prefactor_values(fe_values.n_quadrature_points, 1.0);
            std::vector<double> conductivity_values(fe_values.n_quadrature_points, physical_diffusion);

            if (advection_field.is_temperature())
              {
                // For temperature, the entropy residual uses material
                // prefactors evaluated at the time-centered old states. These
                // values are needed both to decide the active max locations
                // and to push derivatives back through rho*C_p where needed.
                MaterialModel::MaterialModelInputs<dim> in(fe_values,
                                                           cell,
                                                           this->introspection(),
                                                           simulator->current_linearization_point);
                MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points,
                                                             this->n_compositional_fields());
                in.requested_properties =
                  MaterialModel::MaterialProperties::equation_of_state_properties |
                  MaterialModel::MaterialProperties::thermal_conductivity;

                for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                  {
                    in.temperature[q] =
                      0.5 * (old_temperature_values[q] + old_old_temperature_values[q]);
                    in.pressure[q] =
                      0.5 * (old_pressure_values[q] + old_old_pressure_values[q]);
                    in.velocity[q] =
                      0.5 * (old_velocity_values[q] + old_old_velocity_values[q]);
                    in.strain_rate[q] =
                      0.5 * (old_strain_rates[q] + old_old_strain_rates[q]);
                    for (unsigned int c = 0; c < this->introspection().n_compositional_fields; ++c)
                      in.composition[q][c] =
                        0.5 * (old_composition_values[c][q]
                               + old_old_composition_values[c][q]);
                  }
                this->get_material_model().evaluate(in, out);

                if (this->get_parameters().formulation_temperature_equation ==
                    Parameters<dim>::Formulation::TemperatureEquation::reference_density_profile)
                  for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                    out.densities[q] =
                      this->get_adiabatic_conditions().density(fe_values.quadrature_point(q));

                max_advection_prefactor = 0.0;
                for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                  {
                    advection_prefactor_values[q] =
                      out.densities[q] * out.specific_heat[q];
                    conductivity_values[q] = out.thermal_conductivities[q];
                    if (advection_prefactor_values[q] > max_advection_prefactor)
                      {
                        max_advection_prefactor =
                          advection_prefactor_values[q];
                        max_advection_prefactor_q = q;
                      }
                  }
              }

            auto add_advection_prefactor_composition_transpose =
              [&](const unsigned int q,
                  const double bar_advection_prefactor)
            {
              if (bar_advection_prefactor == 0.0
                  || !advection_field.is_temperature())
                return;

              if (this->get_parameters().formulation_temperature_equation ==
                  Parameters<dim>::Formulation::TemperatureEquation::reference_density_profile)
                return;

              // Local VJP for rho*C_p at a single quadrature point of the
              // entropy-viscosity estimator. The scatter weights are 0.5 and
              // 0.5 because this estimator evaluates material inputs at the
              // midpoint of old and old-old states rather than the BDF
              // extrapolation used in the advection matrix.
              MaterialModel::MaterialModelInputs<dim> in(fe_values,
                                                         cell,
                                                         this->introspection(),
                                                         simulator->current_linearization_point);
              MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points,
                                                           this->n_compositional_fields());

              in.requested_properties =
                MaterialModel::MaterialProperties::equation_of_state_properties;
              for (unsigned int qp = 0; qp < fe_values.n_quadrature_points; ++qp)
                {
                  in.temperature[qp] =
                    0.5 * (old_temperature_values[qp] + old_old_temperature_values[qp]);
                  in.pressure[qp] =
                    0.5 * (old_pressure_values[qp] + old_old_pressure_values[qp]);
                  in.velocity[qp] =
                    0.5 * (old_velocity_values[qp] + old_old_velocity_values[qp]);
                  in.strain_rate[qp] =
                    0.5 * (old_strain_rates[qp] + old_old_strain_rates[qp]);
                  for (unsigned int cc = 0; cc < this->introspection().n_compositional_fields; ++cc)
                    in.composition[qp][cc] =
                      0.5 * (old_composition_values[cc][qp]
                             + old_old_composition_values[cc][qp]);
                }

              this->get_material_model().evaluate(in, out);

              MaterialModelOutputAdjoints<dim> bar_out(fe_values.n_quadrature_points);
              bar_out.densities[q] += out.specific_heat[q] * bar_advection_prefactor;
              bar_out.specific_heat[q] += out.densities[q] * bar_advection_prefactor;

              MaterialModelInputAdjoints<dim> bar_in(fe_values.n_quadrature_points,
                                                     this->n_compositional_fields());
              const MaterialModelAdjoint<dim> material_model_adjoint(*simulator);
              material_model_adjoint.apply_derivative_transpose(in,
                                                                out,
                                                                bar_out,
                                                                bar_in);

              const MaterialModelInputAdjointScatter<dim> material_input_scatter(*simulator);
              material_input_scatter.scatter_temperature_advection_inputs(
                fe_values,
                local_dof_indices,
                bar_in,
                0.5,
                0.5,
                previous_field_adjoint,
                use_bdf2_scheme ? previous_previous_field_adjoint : nullptr,
                previous_composition_adjoints,
                use_bdf2_scheme ? previous_previous_composition_adjoints : nullptr);
            };

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                // Reconstruct the forward active quantities: the maximum
                // velocity measure and the maximum residual measure. Their
                // quadrature locations are saved so the adjoint can follow the
                // same nonsmooth max branches selected in the forward
                // estimator.
                const Tensor<1, dim> velocity =
                  0.5 * (old_velocity_values[q] + old_old_velocity_values[q]);
                const double velocity_norm = velocity.norm();
                const SymmetricTensor<2, dim> strain_rate =
                  0.5 * (old_strain_rates[q] + old_old_strain_rates[q]);
                const double strain_rate_norm = strain_rate.norm();
                const double velocity_measure =
                  velocity_norm + gamma * h * strain_rate_norm;

                if (velocity_measure > max_velocity)
                  {
                    max_velocity = velocity_measure;
                    max_velocity_q = q;
                    max_velocity_norm = velocity_norm;
                    max_strain_rate_norm = strain_rate_norm;
                  }

                // At a quadrature point the strong residual reconstructed by
                // the estimator is
                //
                //   R = a[(q_{n-1}-q_{n-2})/dt_old
                //         + u_mid.grad(q_mid)] - kappa Laplace(q_mid).
                //
                // alpha=2 multiplies |R| by |q_mid-q_avg| before the cellwise
                // maximum; alpha=1 uses |R| directly.
                const double field_dt =
                  snapshot.old_time_step == 0.0
                  ? 0.0
                  : (old_field_values[q] - old_old_field_values[q])
                  / snapshot.old_time_step;
                const Tensor<1, dim> field_gradient =
                  0.5 * (old_field_gradients[q] + old_old_field_gradients[q]);
                const double laplacian =
                  0.5 * (old_field_laplacians[q] + old_old_field_laplacians[q]);
                const double transport_residual =
                  field_dt + velocity * field_gradient;
                const double residual =
                  advection_prefactor_values[q] * transport_residual
                  - conductivity_values[q] * laplacian;
                const double field =
                  0.5 * (old_field_values[q] + old_old_field_values[q]);
                const double entropy_weight =
                  stabilization_alpha == 2
                  ? std::abs(field - average_field)
                  : 1.0;
                const double residual_measure = std::abs(residual) * entropy_weight;

                if (residual_measure > max_residual_measure)
                  {
                    max_residual_measure = residual_measure;
                    max_residual_q = q;
                    max_residual_value = residual;
                    max_residual_entropy_weight = entropy_weight;
                    max_residual_velocity = velocity;
                    max_residual_temperature_gradient = field_gradient;
                    max_residual_advection_prefactor =
                      advection_prefactor_values[q];
                    max_residual_transport_residual = transport_residual;
                  }
              }

            const double beta =
              this->get_parameters().stabilization_beta[advection_field.field_index()];
            const double c_R =
              this->get_parameters().stabilization_c_R[advection_field.field_index()];
            const double maximum_viscosity =
              beta * max_advection_prefactor * max_velocity * h;

            double bar_max_velocity = 0.0;
            double bar_max_advection_prefactor = 0.0;
            double bar_residual_measure = 0.0;

            // Reverse nu_art=min(nu_max,nu_E). Degenerate first-step or zero
            // normalization cases use the same cap fallback as the forward
            // code. Away from a tie, exactly one branch receives bar(nu).
            if (std::abs(global_max_velocity) < 1e-50)
              continue;

            if (snapshot.timestep_number <= 1
                || (stabilization_alpha == 2
                    && std::abs(global_entropy_variation) < 1e-50)
                || std::abs(global_field_variation) < 1e-50)
              {
                // Degenerate/first-step fallback: ASPECT's forward estimator
                // uses the maximum-viscosity cap directly, so the seed only
                // reaches the selected max-velocity and max-prefactor paths.
                bar_max_velocity += beta * max_advection_prefactor * h * bar_nu;
                bar_max_advection_prefactor += beta * max_velocity * h * bar_nu;
              }
            else
              {
                const double entropy_viscosity =
                  (stabilization_alpha == 2)
                  ? c_R * h * h * max_residual_measure / global_entropy_variation
                  : c_R * h * simulator->global_Omega_diameter
                  * max_velocity * max_residual_measure
                  / (global_max_velocity * global_field_variation);

                if (maximum_viscosity < entropy_viscosity)
                  {
                    // The beta*h*u_max cap is active.
                    bar_max_velocity += beta * max_advection_prefactor * h * bar_nu;
                    bar_max_advection_prefactor += beta * max_velocity * h * bar_nu;
                  }
                else if (stabilization_alpha == 2)
                  {
                    // The alpha=2 entropy-viscosity branch is active. Its
                    // denominator is a global entropy variation, so accumulate
                    // a scalar seed and apply its transpose after the cell
                    // loop/MPI reduction.
                    const double entropy_viscosity =
                      c_R * h * h * max_residual_measure / global_entropy_variation;
                    bar_residual_measure +=
                      c_R * h * h / global_entropy_variation * bar_nu;
                    local_bar_global_entropy_variation +=
                      -bar_nu * entropy_viscosity / global_entropy_variation;
                  }
                else
                  {
                    // The alpha=1 entropy-viscosity branch uses a global
                    // field range and global velocity normalization. This
                    // implementation propagates the local max-velocity and
                    // residual-measure pieces used by the supported v1 tests.
                    bar_max_velocity +=
                      c_R * h * simulator->global_Omega_diameter
                      * max_residual_measure
                      / (global_max_velocity * global_field_variation)
                      * bar_nu;
                    bar_residual_measure +=
                      c_R * h * simulator->global_Omega_diameter
                      * max_velocity
                      / (global_max_velocity * global_field_variation)
                      * bar_nu;
                  }
              }

            local_previous_field = 0.0;
            local_previous_previous_field = 0.0;
            local_previous_stokes = 0.0;
            local_previous_previous_stokes = 0.0;

            if (bar_max_velocity != 0.0
                && max_velocity_q != numbers::invalid_unsigned_int
                && max_velocity > 0.0)
              {
                const Tensor<1, dim> velocity =
                  0.5 * (old_velocity_values[max_velocity_q]
                         + old_old_velocity_values[max_velocity_q]);
                const SymmetricTensor<2, dim> strain_rate =
                  0.5 * (old_strain_rates[max_velocity_q]
                         + old_old_strain_rates[max_velocity_q]);
                // u_measure=|u|+gamma h |eps(u)|. Its VJP uses the normalized
                // velocity and strain tensors; a zero norm contributes the
                // zero subgradient. Both inputs are midpoint states, hence
                // the factor 0.5 in the FE scatter to n-1 and n-2.
                const Tensor<1, dim> bar_velocity =
                  (max_velocity_norm > 0.0)
                  ? bar_max_velocity * velocity / max_velocity_norm
                  : Tensor<1, dim>();
                const SymmetricTensor<2, dim> bar_strain_rate =
                  (max_strain_rate_norm > 0.0)
                  ? bar_max_velocity * gamma * h * strain_rate / max_strain_rate_norm
                  : SymmetricTensor<2, dim>();

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  if (this->introspection().component_masks.velocities[
                        this->get_fe().system_to_component_index(i).first])
                    {
                      const Tensor<1, dim> phi_u =
                        fe_values[this->introspection().extractors.velocities].value(i, max_velocity_q);
                      const SymmetricTensor<2, dim> eps_phi_u =
                        fe_values[this->introspection().extractors.velocities].symmetric_gradient(i, max_velocity_q);
                      const double contribution =
                        0.5 * ((bar_velocity * phi_u) + (bar_strain_rate * eps_phi_u));
                      local_previous_stokes(i) += contribution;
                      if (snapshot.timestep_number > 1)
                        local_previous_previous_stokes(i) += contribution;
                      else
                        local_previous_stokes(i) += contribution;
                    }
              }

            if (bar_max_advection_prefactor != 0.0
                && max_advection_prefactor_q != numbers::invalid_unsigned_int)
              add_advection_prefactor_composition_transpose(max_advection_prefactor_q,
                                                            bar_max_advection_prefactor);

            if (bar_residual_measure != 0.0
                && max_residual_q != numbers::invalid_unsigned_int)
              {
                // Reverse the selected residual-measure quadrature point:
                // abs(residual) contributes sign(residual), alpha=2 also
                // contributes through abs(field-average_field), and the
                // transport residual splits into velocity, field gradient, and
                // time-derivative terms.
                const unsigned int q = max_residual_q;
                const double residual_sign =
                  (max_residual_value > 0.0) ? 1.0 : -1.0;
                const double field =
                  0.5 * (old_field_values[q] + old_old_field_values[q]);
                const double entropy_weight_sign =
                  (field - average_field > 0.0) ? 1.0 : -1.0;

                // Reverse
                //   M=|R| w,  w=|q_mid-q_avg| (alpha=2),
                //   R=a r_transport-kappa Laplace(q_mid),
                //   r_transport=q_t+u_mid.grad(q_mid).
                // The scalar/tensor seeds below are the partial derivatives
                // of this local computation graph before FE transposition.
                const double bar_residual =
                  residual_sign * max_residual_entropy_weight * bar_residual_measure;
                const double bar_field =
                  (stabilization_alpha == 2)
                  ? entropy_weight_sign * std::abs(max_residual_value) * bar_residual_measure
                  : 0.0;
                if (stabilization_alpha == 2)
                  local_bar_average_field -=
                    entropy_weight_sign * std::abs(max_residual_value) * bar_residual_measure;
                const double bar_transport_residual =
                  max_residual_advection_prefactor * bar_residual;
                const double bar_advection_prefactor =
                  max_residual_transport_residual * bar_residual;
                const Tensor<1, dim> bar_velocity =
                  bar_transport_residual * max_residual_temperature_gradient;
                const Tensor<1, dim> bar_temperature_gradient =
                  bar_transport_residual * max_residual_velocity;
                const double bar_laplacian =
                  -conductivity_values[q] * bar_residual;

                add_advection_prefactor_composition_transpose(q,
                                                              bar_advection_prefactor);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const unsigned int component =
                      this->get_fe().system_to_component_index(i).first;

                    if (component == advection_field.component_index(this->introspection()))
                      {
                        const double phi_field =
                          fe_values[field_extractor].value(i, q);
                        const Tensor<1, dim> grad_phi_field =
                          fe_values[field_extractor].gradient(i, q);
                        const double laplacian_phi_field =
                          trace(fe_values[field_extractor].hessian(i, q));

                        // q_mid contributes through the entropy weight,
                        // gradient, and Laplacian. The time difference
                        // contributes +bar(r_transport)/dt_old to q_{n-1}
                        // and the negative counterpart to q_{n-2}.
                        double previous_contribution =
                          0.5 * bar_field * phi_field
                          + 0.5 * (bar_temperature_gradient * grad_phi_field)
                          + 0.5 * bar_laplacian * laplacian_phi_field;
                        double previous_previous_contribution =
                          previous_contribution;

                        if (snapshot.old_time_step != 0.0)
                          {
                            previous_contribution += bar_transport_residual / snapshot.old_time_step * phi_field;
                            previous_previous_contribution -= bar_transport_residual / snapshot.old_time_step * phi_field;
                          }

                        local_previous_field(i) += previous_contribution;
                        if (snapshot.timestep_number > 1)
                          local_previous_previous_field(i) += previous_previous_contribution;
                        else
                          // On the first step ASPECT stores only q_{n-1}; both
                          // midpoint/history contributions collapse onto that
                          // single previous state.
                          local_previous_field(i) += previous_previous_contribution;
                      }

                    if (this->introspection().component_masks.velocities[component])
                      {
                        const Tensor<1, dim> phi_u =
                          fe_values[this->introspection().extractors.velocities].value(i, q);
                        const double contribution = 0.5 * (bar_velocity * phi_u);
                        local_previous_stokes(i) += contribution;
                        if (snapshot.timestep_number > 1)
                          local_previous_previous_stokes(i) += contribution;
                        else
                          // Same BDF1 collapse as for field values: there is no
                          // separate q_{n-2} Stokes state on the first step.
                          local_previous_stokes(i) += contribution;
                      }
                  }
              }

            this->get_current_constraints().distribute_local_to_global(local_previous_field,
                                                                       local_dof_indices,
                                                                       previous_field_adjoint);
            this->get_current_constraints().distribute_local_to_global(local_previous_stokes,
                                                                       local_dof_indices,
                                                                       previous_stokes_adjoint_rhs);

            if (snapshot.timestep_number > 1)
              {
                AssertThrow(previous_previous_field_adjoint != nullptr &&
                            previous_previous_stokes_adjoint_rhs != nullptr,
                            ExcInternalError());
                this->get_current_constraints().distribute_local_to_global(local_previous_previous_field,
                                                                           local_dof_indices,
                                                                           *previous_previous_field_adjoint);
                this->get_current_constraints().distribute_local_to_global(local_previous_previous_stokes,
                                                                           local_dof_indices,
                                                                           *previous_previous_stokes_adjoint_rhs);
              }
          }

      previous_field_adjoint.compress(VectorOperation::add);
      previous_stokes_adjoint_rhs.compress(VectorOperation::add);
      if (previous_composition_adjoints != nullptr)
        for (auto &composition_adjoint : *previous_composition_adjoints)
          composition_adjoint.compress(VectorOperation::add);

      if (snapshot.timestep_number > 1)
        {
          previous_previous_field_adjoint->compress(VectorOperation::add);
          previous_previous_stokes_adjoint_rhs->compress(VectorOperation::add);
          if (previous_previous_composition_adjoints != nullptr)
            for (auto &composition_adjoint : *previous_previous_composition_adjoints)
              composition_adjoint.compress(VectorOperation::add);
        }

      if (stabilization_alpha == 2 && snapshot.timestep_number > 1)
        {
          // Finish the two nonlocal paths only after the cell loop. Every
          // rank may contribute to bar(Delta_E) and bar(q_avg), while the
          // primal quantities were MPI-global. Summing the scalar seeds first
          // is therefore the transpose of broadcasting those primal scalars;
          // the two helpers then transpose the entropy variation and the
          // extrapolated min/max range, respectively.
          const double bar_global_entropy_variation =
            Utilities::MPI::sum(local_bar_global_entropy_variation,
                                this->get_mpi_communicator());
          double bar_average_field =
            Utilities::MPI::sum(local_bar_average_field,
                                this->get_mpi_communicator());
          add_global_entropy_variation_transpose(advection_field,
                                                 bar_global_entropy_variation,
                                                 bar_average_field,
                                                 previous_field_adjoint,
                                                 previous_previous_field_adjoint);
          add_extrapolated_field_average_transpose(advection_field,
                                                   snapshot,
                                                   bar_average_field,
                                                   previous_field_adjoint,
                                                   previous_previous_field_adjoint);
        }
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template class Adjoint::ArtificialViscosityAdjoint<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
