/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  Perturbing finite-difference and constraint checks for the advection
  transpose are compiled separately from the production operator.
*/

#include <aspect/adjoint/operators/advection_adjoint_calculator.h>
#include <aspect/adjoint/operators/artificial_viscosity_adjoint.h>
#include <aspect/adjoint/operators/boundary_conditions.h>
#include <aspect/adjoint/core/simulator_state.h>
#include <aspect/advection_field.h>
#include <aspect/simulator.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::run_transposed_advection_constraint_diagnostic(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const LinearAlgebra::BlockVector &current_field_adjoint) const
    {
      if (snapshot.timestep_number == 0
          || current_field_adjoint.l2_norm() <= std::numeric_limits<double>::min())
        return;

      // Diagnostics also restore the snapshot into the Simulator because the
      // forward assembly routines cache constraint and stabilization state as
      // a side effect. The saved state is restored at the end so the caller
      // can continue the reverse sweep from its current checkpoint.
      ScopedSimulatorState<dim> simulator_state(*simulator);

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

      BoundaryConditions<dim> boundary_conditions;
      boundary_conditions.build_temperature_adjoint_constraints(snapshot,
                                                                simulator->current_constraints);

      simulator->assemble_advection_system(advection_field);

      LinearAlgebra::BlockVector z;
      solve_transposed_advection_system(advection_field,
                                        current_field_adjoint,
                                        z);

      // This scalar-field regression checks the essential constrained DoFs in
      // the active field block. The BoundaryConditions helper separately
      // verifies the full affine graph before it is distributed, because an
      // affine relation must not be tested merely by forcing its slave entry
      // to zero.
      const unsigned int field_component =
        advection_field.component_index(this->introspection());
      const IndexSet locally_owned_dofs =
        this->get_dof_handler().locally_owned_dofs();
      std::vector<types::global_dof_index> local_dof_indices(this->get_fe().dofs_per_cell);
      std::set<types::global_dof_index> written_dofs;

      double local_max_abs_constrained_value = 0.0;
      unsigned int local_constrained_dofs = 0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < this->get_fe().dofs_per_cell; ++i)
              {
                const unsigned int component =
                  this->get_fe().system_to_component_index(i).first;
                const types::global_dof_index global_index = local_dof_indices[i];
                if (component != field_component
                    || locally_owned_dofs.is_element(global_index) == false
                    || written_dofs.insert(global_index).second == false
                    || this->get_current_constraints().is_constrained(global_index) == false)
                  continue;

                local_max_abs_constrained_value =
                  std::max(local_max_abs_constrained_value,
                           std::abs(z(global_index)));
                ++local_constrained_dofs;
              }
          }

      const double max_abs_constrained_value =
        Utilities::MPI::max(local_max_abs_constrained_value,
                            this->get_mpi_communicator());
      const unsigned int constrained_dofs =
        Utilities::MPI::sum(local_constrained_dofs,
                            this->get_mpi_communicator());

      this->get_pcout() << "   "
                        << (advection_field.is_temperature() ? "Temperature" : "Composition")
                        << "-advection transposed-solve constrained DoF diagnostic: constrained DoFs = "
                        << constrained_dofs
                        << ", max |z| = "
                        << std::setprecision(16) << max_abs_constrained_value
                        << std::setprecision(6) << std::endl;

      AssertThrow(max_abs_constrained_value <= 1e-14,
                  ExcMessage("The transposed advection solve produced nonzero values "
                             "on constrained adjoint DoFs. Homogeneous adjoint "
                             "constraints must be distributed."));

    }


    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::run_artificial_viscosity_coefficient_seed_diagnostic(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const LinearAlgebra::BlockVector &current_field_adjoint,
      const double epsilon) const
    {
      if (snapshot.timestep_number == 0
          || current_field_adjoint.l2_norm() <= std::numeric_limits<double>::min()
          || this->get_parameters().advection_stabilization_method
          != Parameters<dim>::AdvectionStabilizationMethod::entropy_viscosity)
        return;

      // This diagnostic checks only the residual -> artificial-viscosity
      // coefficient layer. It first solves the transposed advection block to
      // obtain z, then asks ArtificialViscosityAdjoint for the cellwise seed
      // dR/dnu_art^T z and compares that seed with a local finite-difference
      // perturbation of independent cell coefficients.
      ScopedSimulatorState<dim> simulator_state(*simulator);

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

      BoundaryConditions<dim> boundary_conditions;
      boundary_conditions.build_temperature_adjoint_constraints(snapshot,
                                                                simulator->current_constraints);

      simulator->assemble_advection_system(advection_field);

      LinearAlgebra::BlockVector z;
      solve_transposed_advection_system(advection_field,
                                        current_field_adjoint,
                                        z);

      ArtificialViscosityAdjoint<dim> artificial_viscosity_adjoint(*simulator);
      const std::vector<double> artificial_viscosity_seed =
        artificial_viscosity_adjoint.assemble_coefficient_seed(advection_field,
                                                               snapshot,
                                                               z);
      artificial_viscosity_adjoint.run_coefficient_seed_diagnostic(advection_field,
                                                                   snapshot,
                                                                   z,
                                                                   artificial_viscosity_seed,
                                                                   epsilon);

    }


    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::run_artificial_viscosity_estimator_transpose_diagnostic(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const double epsilon) const
    {
      // Reassemble the forward estimator at the archived time level, then
      // delegate the full dot-product identity to the diagnostic operator.
      // Saving Simulator fields is necessary because assembly refreshes
      // coefficient caches as a side effect of constructing the test map.
      if (snapshot.timestep_number == 0
          || this->get_parameters().advection_stabilization_method
          != Parameters<dim>::AdvectionStabilizationMethod::entropy_viscosity)
        return;

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

      simulator->assemble_advection_system(advection_field);

      ArtificialViscosityAdjoint<dim> artificial_viscosity_adjoint(*simulator);
      artificial_viscosity_adjoint.run_estimator_transpose_diagnostic(advection_field,
                                                                      snapshot,
                                                                      epsilon);

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


    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::run_global_entropy_variation_transpose_diagnostic(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const double epsilon) const
    {
      // Wrapper for the frozen-average Delta_E transpose test. The alpha/BDF
      // guards are part of its mathematical domain: alpha=1 has no entropy
      // denominator and the first step has no independent old-old history.
      if (snapshot.timestep_number <= 1
          || this->get_parameters().advection_stabilization_method
          != Parameters<dim>::AdvectionStabilizationMethod::entropy_viscosity
          || this->get_parameters().stabilization_alpha != 2)
        return;

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

      simulator->assemble_advection_system(advection_field);

      ArtificialViscosityAdjoint<dim> artificial_viscosity_adjoint(*simulator);
      artificial_viscosity_adjoint.run_global_entropy_variation_transpose_diagnostic(advection_field,
                                                                                     snapshot,
                                                                                     epsilon);

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


    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::run_global_entropy_denominator_scalar_diagnostic(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const double epsilon) const
    {
      // Wrapper for the scalar derivative of 1/Delta_E in the active
      // entropy-viscosity branch. Restoring the snapshot before assembly makes
      // the cached cell coefficients identical to the reverse step under test.
      if (snapshot.timestep_number <= 1
          || this->get_parameters().advection_stabilization_method
          != Parameters<dim>::AdvectionStabilizationMethod::entropy_viscosity
          || this->get_parameters().stabilization_alpha != 2)
        return;

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

      simulator->assemble_advection_system(advection_field);

      ArtificialViscosityAdjoint<dim> artificial_viscosity_adjoint(*simulator);
      artificial_viscosity_adjoint.run_global_entropy_denominator_scalar_diagnostic(advection_field,
                                                                                    snapshot,
                                                                                    epsilon);

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


    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::run_local_entropy_residual_transpose_diagnostic(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const double epsilon) const
    {
      // Wrapper for the local numerator/max transpose. This deliberately
      // isolates the strong-residual computation from both global entropy
      // variation and extrapolated-range derivatives.
      if (snapshot.timestep_number <= 1
          || this->get_parameters().advection_stabilization_method
          != Parameters<dim>::AdvectionStabilizationMethod::entropy_viscosity
          || this->get_parameters().stabilization_alpha != 2)
        return;

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

      simulator->assemble_advection_system(advection_field);

      ArtificialViscosityAdjoint<dim> artificial_viscosity_adjoint(*simulator);
      artificial_viscosity_adjoint.run_local_entropy_residual_transpose_diagnostic(advection_field,
                                                                                   snapshot,
                                                                                   epsilon);

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


    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::run_velocity_transpose_diagnostic(
      const AdvectionField &advection_field,
      const ForwardStateView<dim> &snapshot,
      const LinearAlgebra::BlockVector &current_field_adjoint,
      const double epsilon) const
    {
      if (snapshot.timestep_number == 0
          || current_field_adjoint.l2_norm() <= std::numeric_limits<double>::min())
        return;

      // This one-step check compares the explicit velocity transpose against
      // finite differences of the assembled advection residual. It is meant
      // for BDF1/u* checks; Manager skips it for later BDF2 steps because the
      // diagnostic does not test the split between q_{n-1} and q_{n-2}.
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

      simulator->assemble_advection_system(advection_field);

      LinearAlgebra::BlockVector z;
      solve_transposed_advection_system(advection_field,
                                        current_field_adjoint,
                                        z);

      LinearAlgebra::BlockVector velocity_adjoint;
      velocity_adjoint.reinit(this->introspection().index_sets.system_partitioning,
                              this->introspection().index_sets.system_relevant_partitioning,
                              this->get_mpi_communicator());
      velocity_adjoint = 0.0;
      LinearAlgebra::BlockVector previous_previous_velocity_adjoint;
      previous_previous_velocity_adjoint.reinit(this->introspection().index_sets.system_partitioning,
                                                this->introspection().index_sets.system_relevant_partitioning,
                                                this->get_mpi_communicator());
      previous_previous_velocity_adjoint = 0.0;
      add_velocity_transpose(advection_field,
                             snapshot,
                             z,
                             velocity_adjoint,
                             snapshot.timestep_number > 1 ? &previous_previous_velocity_adjoint : nullptr);
      const unsigned int field_component =
        advection_field.component_index(this->introspection());

      LinearAlgebra::BlockVector delta_velocity;
      delta_velocity.reinit(this->introspection().index_sets.system_partitioning,
                            this->introspection().index_sets.system_relevant_partitioning,
                            this->get_mpi_communicator());
      delta_velocity = 0.0;

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

                if (this->introspection().component_masks.velocities[component])
                  delta_velocity(global_index) =
                    std::sin(0.191 * static_cast<double>(global_index + 1));
              }
          }
      delta_velocity.compress(VectorOperation::insert);
      this->get_current_constraints().distribute(delta_velocity);

      auto assemble_advection_residual_for_velocity_perturbation =
        [&](const double sign)
      {
        simulator->solution = snapshot.solution.get();
        simulator->old_solution = snapshot.old_solution.get();
        simulator->old_old_solution = snapshot.old_old_solution.get();
        simulator->current_linearization_point = snapshot.advection_linearization_point.get();
        simulator->current_linearization_point.add(sign * epsilon,
                                                   delta_velocity);
        simulator->time = snapshot.time;
        simulator->time_step = snapshot.time_step;
        simulator->old_time_step = snapshot.old_time_step;
        simulator->timestep_number = snapshot.timestep_number;
        simulator->last_pressure_normalization_adjustment =
          snapshot.pressure_normalization_adjustment;

        simulator->assemble_advection_system(advection_field);

        LinearAlgebra::BlockVector matrix_times_solution;
        matrix_times_solution.reinit(this->introspection().index_sets.system_partitioning,
                                     this->get_mpi_communicator());
        simulator->system_matrix.vmult(matrix_times_solution,
                                       snapshot.solution.get());

        LinearAlgebra::BlockVector residual(simulator->system_rhs);
        residual.add(-1.0, matrix_times_solution);
        return residual;
      };

      const LinearAlgebra::BlockVector residual_plus =
        assemble_advection_residual_for_velocity_perturbation(1.0);
      const LinearAlgebra::BlockVector residual_minus =
        assemble_advection_residual_for_velocity_perturbation(-1.0);

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

                if (component == field_component
                    && this->get_current_constraints().is_constrained(global_index) == false)
                  local_fd_dot += z(global_index)
                                  * (residual_plus(global_index) - residual_minus(global_index))
                                  / (2.0 * epsilon);

                if (this->introspection().component_masks.velocities[component])
                  local_adjoint_dot += delta_velocity(global_index)
                                       * velocity_adjoint(global_index);
              }
          }

      const double fd_dot =
        Utilities::MPI::sum(local_fd_dot, this->get_mpi_communicator());
      const double adjoint_dot =
        Utilities::MPI::sum(local_adjoint_dot, this->get_mpi_communicator());
      const double relative_error =
        std::abs(fd_dot - adjoint_dot)
        / std::max({std::abs(fd_dot), std::abs(adjoint_dot), 1e-30});

      this->get_pcout() << "   "
                        << (advection_field.is_temperature() ? "Temperature" : "Composition")
                        << "-advection velocity transpose diagnostic: FD = "
                        << std::setprecision(16) << fd_dot
                        << ", adjoint = " << adjoint_dot
                        << ", relative error = " << relative_error
                        << std::setprecision(6) << std::endl;

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
#define INSTANTIATE(dim) \
  template void Adjoint::AdvectionAdjointCalculator<dim>::run_transposed_advection_constraint_diagnostic( \
      const AdvectionField &, const Adjoint::ForwardStateView<dim> &, \
      const LinearAlgebra::BlockVector &) const; \
  template void Adjoint::AdvectionAdjointCalculator<dim>::run_artificial_viscosity_coefficient_seed_diagnostic( \
      const AdvectionField &, const Adjoint::ForwardStateView<dim> &, \
      const LinearAlgebra::BlockVector &, const double) const; \
  template void Adjoint::AdvectionAdjointCalculator<dim>::run_artificial_viscosity_estimator_transpose_diagnostic( \
      const AdvectionField &, const Adjoint::ForwardStateView<dim> &, const double) const; \
  template void Adjoint::AdvectionAdjointCalculator<dim>::run_global_entropy_variation_transpose_diagnostic( \
      const AdvectionField &, const Adjoint::ForwardStateView<dim> &, const double) const; \
  template void Adjoint::AdvectionAdjointCalculator<dim>::run_global_entropy_denominator_scalar_diagnostic( \
      const AdvectionField &, const Adjoint::ForwardStateView<dim> &, const double) const; \
  template void Adjoint::AdvectionAdjointCalculator<dim>::run_local_entropy_residual_transpose_diagnostic( \
      const AdvectionField &, const Adjoint::ForwardStateView<dim> &, const double) const; \
  template void Adjoint::AdvectionAdjointCalculator<dim>::run_velocity_transpose_diagnostic( \
      const AdvectionField &, const Adjoint::ForwardStateView<dim> &, \
      const LinearAlgebra::BlockVector &, const double) const;

  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
