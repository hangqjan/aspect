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
    Manager<dim>::capture_optimization_initial_state()
    {
      AssertThrow(simulator != nullptr, ExcInternalError());
      AssertThrow(!initial_temperature_optimization.canonical_state.captured,
                  ExcMessage("The canonical optimization initial state was captured twice."));

      // Refresh constraints at the capture boundary. The control-space object
      // itself is built lazily after a restore, because it references this
      // mutable simulator constraint object.
      // This snapshot is the canonical evaluation origin. Every ROL trial must
      // start from identical solution/history, time integrator counters,
      // constraints, and solver-failure counters; otherwise J(c) would depend
      // on callback order rather than only on the control c.
      simulator->compute_current_constraints();
      const unsigned int temperature_block = this->introspection().block_indices.temperature;
      initial_temperature_optimization.fixed_temperature = std::make_unique<LinearAlgebra::Vector>(
                                                             this->introspection().index_sets.system_partitioning[temperature_block],
                                                             this->get_mpi_communicator());
      *initial_temperature_optimization.fixed_temperature = simulator->solution.block(temperature_block);

      initial_temperature_optimization.canonical_state.solution = simulator->solution;
      initial_temperature_optimization.canonical_state.old_solution = simulator->old_solution;
      initial_temperature_optimization.canonical_state.old_old_solution = simulator->old_old_solution;
      initial_temperature_optimization.canonical_state.current_linearization_point =
        simulator->current_linearization_point;
      initial_temperature_optimization.canonical_state.system_right_hand_side = simulator->system_rhs;
      initial_temperature_optimization.canonical_state.current_constraints.copy_from(simulator->current_constraints);
      initial_temperature_optimization.canonical_state.time = simulator->time;
      initial_temperature_optimization.canonical_state.time_step = simulator->time_step;
      initial_temperature_optimization.canonical_state.old_time_step = simulator->old_time_step;
      initial_temperature_optimization.canonical_state.pressure_normalization_adjustment =
        simulator->last_pressure_normalization_adjustment;
      initial_temperature_optimization.canonical_state.pressure_scaling = simulator->pressure_scaling;
      initial_temperature_optimization.canonical_state.timestep_number = simulator->timestep_number;
      initial_temperature_optimization.canonical_state.nonlinear_iteration = simulator->nonlinear_iteration;
      initial_temperature_optimization.canonical_state.nonlinear_solver_failures = simulator->nonlinear_solver_failures;
      initial_temperature_optimization.canonical_state.linear_solver_failures = simulator->linear_solver_failures;
      initial_temperature_optimization.canonical_state.rebuild_sparsity_and_matrices =
        simulator->rebuild_sparsity_and_matrices;
      initial_temperature_optimization.canonical_state.rebuild_stokes_matrix = simulator->rebuild_stokes_matrix;
      initial_temperature_optimization.canonical_state.rebuild_stokes_preconditioner =
        simulator->rebuild_stokes_preconditioner;
      initial_temperature_optimization.canonical_state.captured = true;
    }



    template <int dim>
    void
    Manager<dim>::restore_optimization_initial_state()
    {
      AssertThrow(simulator != nullptr && initial_temperature_optimization.canonical_state.captured,
                  ExcMessage("No canonical optimization initial state is available."));

      // Restore the complete canonical state before installing a new control.
      // Time-stepping plugins may keep mutable decision history outside the
      // solution vectors, so their runtime state is reset explicitly as part
      // of the reduced-objective transaction.
      simulator->time_stepping_manager.reset_runtime_state_for_new_evaluation();
      simulator->solution = initial_temperature_optimization.canonical_state.solution;
      simulator->old_solution = initial_temperature_optimization.canonical_state.old_solution;
      simulator->old_old_solution = initial_temperature_optimization.canonical_state.old_old_solution;
      simulator->current_linearization_point =
        initial_temperature_optimization.canonical_state.current_linearization_point;
      simulator->system_rhs = initial_temperature_optimization.canonical_state.system_right_hand_side;
      simulator->current_constraints.copy_from(initial_temperature_optimization.canonical_state.current_constraints);
      simulator->time = initial_temperature_optimization.canonical_state.time;
      simulator->time_step = initial_temperature_optimization.canonical_state.time_step;
      simulator->old_time_step = initial_temperature_optimization.canonical_state.old_time_step;
      simulator->last_pressure_normalization_adjustment =
        initial_temperature_optimization.canonical_state.pressure_normalization_adjustment;
      simulator->pressure_scaling = initial_temperature_optimization.canonical_state.pressure_scaling;
      simulator->timestep_number = initial_temperature_optimization.canonical_state.timestep_number;
      simulator->nonlinear_iteration = initial_temperature_optimization.canonical_state.nonlinear_iteration;
      simulator->nonlinear_solver_failures = initial_temperature_optimization.canonical_state.nonlinear_solver_failures;
      simulator->linear_solver_failures = initial_temperature_optimization.canonical_state.linear_solver_failures;

      // Solver objects are runtime state, not canonical baseline state. Force
      // their normal lazy reconstruction after control installation.
      simulator->rebuild_sparsity_and_matrices = true;
      simulator->rebuild_stokes_matrix = true;
      simulator->rebuild_stokes_preconditioner = true;

      // All products below are control-dependent. Discarding them here keeps
      // objective, trajectory, reverse histories, and kernels in the same
      // generation as the newly restored forward state.
      objective_results.clear();
      adjoint_states.clear();
      trajectory_manager.reset();
      trajectory_manager.reverse_state_cache.clear();
      trajectory_manager.forward_replay_windows.clear();
      trajectory_manager.terminal_anchor_history.clear();
      trajectory_manager.sparse_anchor_steps.clear();
      if (time_dependent_workflow.initial_temperature_gradient.size() != 0)
        time_dependent_workflow.initial_temperature_gradient = 0.0;
      time_dependent_workflow.initial_composition_gradients.clear();
      kernels = KernelRepository<dim>();
    }



    template <int dim>
    bool
    Manager<dim>::has_optimization_initial_state() const
    {
      return initial_temperature_optimization.canonical_state.captured;
    }



    template <int dim>
    void
    Manager<dim>::ensure_initial_temperature_control_space()
    {
      AssertThrow(initial_temperature_optimization.canonical_state.captured
                  && initial_temperature_optimization.fixed_temperature != nullptr,
                  ExcMessage("Initial-temperature control space requires a captured O3 baseline."));

      // The control map depends on the current DoF partition and constraint
      // graph. Build it lazily after canonical restore and keep it for later
      // evaluations because optimization is restricted to a fixed mesh.
      const unsigned int temperature_block = this->introspection().block_indices.temperature;
      simulator->compute_current_constraints();
      if (initial_temperature_optimization.control_space == nullptr)
        {
          types::global_dof_index temperature_block_first_dof = 0;
          for (unsigned int block = 0; block < temperature_block; ++block)
            temperature_block_first_dof += this->introspection().system_dofs_per_block[block];
          initial_temperature_optimization.control_space = std::make_unique<InitialTemperatureControl<dim>>(
                                                             simulator->dof_handler,
                                                             this->introspection().component_masks.temperature,
                                                             simulator->current_constraints,
                                                             this->get_mpi_communicator(),
                                                             temperature_block_first_dof,
                                                             this->introspection().index_sets.system_partitioning[temperature_block],
                                                             this->introspection().index_sets.system_relevant_partitioning[temperature_block]);
        }
    }



    template <int dim>
    const InitialTemperatureControl<dim> &
    Manager<dim>::get_initial_temperature_control_space()
    {
      restore_optimization_initial_state();
      ensure_initial_temperature_control_space();
      const QGauss<dim> mass_quadrature(simulator->dof_handler.get_fe().degree + 1);
      initial_temperature_optimization.control_space->assemble_mass_matrix(*simulator->mapping, mass_quadrature);
      return *initial_temperature_optimization.control_space;
    }



    template <int dim>
    void
    Manager<dim>::set_terminal_temperature_observation_from_current_state()
    {
      capture_terminal_observation_from_current_state(AdvectionField::temperature());
    }

    template <int dim>
    void
    Manager<dim>::capture_terminal_observation_from_current_state(
      const AdvectionField &field)
    {
      set_terminal_observation(field, simulator->solution);
    }

    template <int dim>
    void
    Manager<dim>::set_terminal_observation(
      const AdvectionField &field,
      const LinearAlgebra::BlockVector &observation)
    {
      bool installed = false;
      for (const auto &objective : objective_manager.get_active_plugins())
        if (auto *scalar_objective =
              dynamic_cast<TerminalScalarFieldMisfit<dim> *>(objective.get());
            scalar_objective != nullptr && scalar_objective->observes_field(field))
          {
            scalar_objective->set_discrete_terminal_observation(observation);
            installed = true;
          }
      AssertThrow(installed,
                  ExcMessage("A compatible terminal scalar-field objective is required "
                             "for a synthetic observation."));
    }

    template <int dim>
    const LinearAlgebra::BlockVector &
    Manager<dim>::get_terminal_advection_field_seed(const AdvectionField &field) const
    {
      if (field.is_temperature())
        return time_dependent_workflow.terminal_temperature_seed;

      AssertIndexRange(field.compositional_variable,
                       time_dependent_workflow.terminal_composition_seeds.size());
      return time_dependent_workflow.terminal_composition_seeds[field.compositional_variable];
    }



    template <int dim>
    void
    Manager<dim>::install_initial_temperature_control(const LinearAlgebra::Vector &control)
    {
      ensure_initial_temperature_control_space();

      const unsigned int temperature_block = this->introspection().block_indices.temperature;
      LinearAlgebra::Vector controlled_temperature(
        this->introspection().index_sets.system_partitioning[temperature_block],
        this->get_mpi_communicator());
      // Install T_0(c)=T_fixed+P c, including physical inhomogeneous boundary
      // rows. The reduced control never overwrites prescribed Dirichlet data.
      initial_temperature_optimization.control_space->inject_physical(
        control, *initial_temperature_optimization.fixed_temperature, controlled_temperature);
      // solution is a ghosted read vector in distributed ASPECT runs. Build
      // the control in owned-only storage and use the normal ASPECT transfer
      // into that ghosted block instead of writing through its ghost map.
      simulator->solution.block(temperature_block) = controlled_temperature;

      // The initial BDF history represents the same controlled physical
      // state. Rebuilding it from the installed state avoids carrying a
      // previous trial's temperature into the first accepted step.
      simulator->old_solution.block(temperature_block) = simulator->solution.block(temperature_block);
      simulator->old_old_solution.block(temperature_block) = simulator->solution.block(temperature_block);
      simulator->current_linearization_point.block(temperature_block) =
        simulator->solution.block(temperature_block);
      simulator->compute_initial_pressure_field();
      // Pressure and the nonlinear linearization point depend on the newly
      // controlled temperature. Reinitialize them before the first Stokes
      // solve rather than carrying values from the canonical baseline.
      simulator->initialize_current_linearization_point();
      simulator->rebuild_sparsity_and_matrices = true;
      simulator->rebuild_stokes_matrix = true;
      simulator->rebuild_stokes_preconditioner = true;
    }



    template <int dim>
    void
    Manager<dim>::set_optimization_initial_temperature_baseline(
      const LinearAlgebra::Vector &baseline)
    {
      AssertThrow(initial_temperature_optimization.canonical_state.captured,
                  ExcMessage("An optimization baseline must be captured before replacing "
                             "its physical initial temperature."));
      const unsigned int temperature_block = this->introspection().block_indices.temperature;
      const auto &owned = this->introspection().index_sets.system_partitioning[temperature_block];
      AssertThrow(baseline.size() == owned.size()
                  && baseline.locally_owned_elements() == owned,
                  ExcMessage("The replacement initial-temperature baseline has an "
                             "incompatible block layout."));
      initial_temperature_optimization.fixed_temperature = std::make_unique<LinearAlgebra::Vector>(owned,
                                                           this->get_mpi_communicator());
      *initial_temperature_optimization.fixed_temperature = baseline;
    }



    template <int dim>
    const LinearAlgebra::Vector &
    Manager<dim>::get_optimization_initial_temperature_baseline() const
    {
      AssertThrow(initial_temperature_optimization.fixed_temperature != nullptr,
                  ExcMessage("No physical initial-temperature baseline is installed."));
      return *initial_temperature_optimization.fixed_temperature;
    }



    template <int dim>
    LinearAlgebra::Vector
    Manager<dim>::reduced_control_from_physical_temperature(
      const LinearAlgebra::Vector &temperature)
    {
      ensure_initial_temperature_control_space();
      const auto &control_space = *initial_temperature_optimization.control_space;
      const auto &temperature_owned = control_space.temperature_block_owned_dofs();
      AssertThrow(temperature.size() == temperature_owned.size()
                  && temperature.locally_owned_elements() == temperature_owned,
                  ExcMessage("The physical temperature has an incompatible block layout."));

      // Invert T=T_fixed+P c only on free rows, where P is the identity by
      // construction. Constrained rows are consequences of these coordinates
      // and must not create independent control values.
      LinearAlgebra::Vector control(control_space.locally_owned_control_dofs(),
                                    this->get_mpi_communicator());
      control = 0.0;
      types::global_dof_index temperature_first_dof = 0;
      const unsigned int temperature_block = this->introspection().block_indices.temperature;
      for (unsigned int block = 0; block < temperature_block; ++block)
        temperature_first_dof += this->introspection().system_dofs_per_block[block];
      for (const auto local_index : temperature_owned)
        {
          const auto temperature_dof = temperature_first_dof + local_index;
          if (simulator->current_constraints.is_constrained(temperature_dof))
            continue;
          const auto expansion = control_space.expansion(temperature_dof);
          AssertThrow(expansion.size() == 1 && expansion.front().second == 1.0,
                      ExcMessage("A free initial-temperature DoF must map directly "
                                 "to one reduced control coefficient."));
          AssertThrow(control.locally_owned_elements().is_element(expansion.front().first),
                      ExcInternalError());
          control[expansion.front().first] =
            temperature[local_index] - (*initial_temperature_optimization.fixed_temperature)[local_index];
        }
      control.compress(VectorOperation::insert);
      return control;
    }



    template <int dim>
    std::pair<LinearAlgebra::Vector, LinearAlgebra::Vector>
    Manager<dim>::initial_temperature_control_physical_bounds()
    {
      ensure_initial_temperature_control_space();
      const auto &control_space = *initial_temperature_optimization.control_space;
      // Physical nondimensional bounds 0<=T_fixed+c<=1 translate on every
      // free coordinate to -T_fixed<=c<=1-T_fixed. Constraint-dependent rows
      // carry no independent box bound because they are not control DoFs.
      LinearAlgebra::Vector lower(control_space.locally_owned_control_dofs(),
                                  this->get_mpi_communicator());
      LinearAlgebra::Vector upper(control_space.locally_owned_control_dofs(),
                                  this->get_mpi_communicator());
      lower = 0.0;
      upper = 0.0;
      types::global_dof_index temperature_first_dof = 0;
      const unsigned int temperature_block = this->introspection().block_indices.temperature;
      for (unsigned int block = 0; block < temperature_block; ++block)
        temperature_first_dof += this->introspection().system_dofs_per_block[block];
      for (const auto local_index : control_space.temperature_block_owned_dofs())
        {
          const auto temperature_dof = temperature_first_dof + local_index;
          if (simulator->current_constraints.is_constrained(temperature_dof))
            continue;
          const auto expansion = control_space.expansion(temperature_dof);
          AssertThrow(expansion.size() == 1 && expansion.front().second == 1.0,
                      ExcMessage("A free initial-temperature DoF must map directly "
                                 "to one reduced control coefficient."));
          lower[expansion.front().first] = -(*initial_temperature_optimization.fixed_temperature)[local_index];
          upper[expansion.front().first] = 1.0 - (*initial_temperature_optimization.fixed_temperature)[local_index];
        }
      lower.compress(VectorOperation::insert);
      upper.compress(VectorOperation::insert);
      return {std::move(lower), std::move(upper)};
    }



    template <int dim>
    std::pair<double, double>
    Manager<dim>::initial_temperature_control_physical_min_max(
      const LinearAlgebra::Vector &control)
    {
      ensure_initial_temperature_control_space();
      LinearAlgebra::Vector physical(initial_temperature_optimization.control_space->temperature_block_owned_dofs(),
                                     this->get_mpi_communicator());
      initial_temperature_optimization.control_space->inject_physical(control,
                                                                      *initial_temperature_optimization.fixed_temperature,
                                                                      physical);
      return {physical.min(), physical.max()};
    }



    template <int dim>
    double
    Manager<dim>::temperature_l2_difference(const LinearAlgebra::Vector &first,
                                            const LinearAlgebra::Vector &second) const
    {
      // Evaluate ||T_a-T_b||_L2 by FE quadrature, not by the coefficient-vector
      // Euclidean norm. This makes reconstruction diagnostics invariant under
      // basis scaling and mesh-cell volume.
      const unsigned int temperature_block = this->introspection().block_indices.temperature;
      LinearAlgebra::BlockVector difference = simulator->solution;
      difference.block(temperature_block) = first;
      difference.block(temperature_block) -= second;

      const QGauss<dim> quadrature(simulator->dof_handler.get_fe().degree + 1);
      FEValues<dim> fe_values(*simulator->mapping, simulator->finite_element,
                              quadrature, update_values | update_JxW_values);
      std::vector<double> values(quadrature.size());
      double local_norm_squared = 0.0;
      for (const auto &cell : simulator->dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[this->introspection().extractors.temperature].get_function_values(
              difference, values);
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              local_norm_squared += values[q] * values[q] * fe_values.JxW(q);
          }
      return std::sqrt(Utilities::MPI::sum(local_norm_squared,
                                           this->get_mpi_communicator()));
    }



    template <int dim>
    double
    Manager<dim>::initial_temperature_control_reconstruction_error_l2(
      const LinearAlgebra::Vector &control,
      const LinearAlgebra::Vector &temperature)
    {
      ensure_initial_temperature_control_space();
      LinearAlgebra::Vector reconstructed(
        initial_temperature_optimization.control_space->temperature_block_owned_dofs(),
        this->get_mpi_communicator());
      initial_temperature_optimization.control_space->inject_physical(control,
                                                                      *initial_temperature_optimization.fixed_temperature,
                                                                      reconstructed);
      return temperature_l2_difference(reconstructed, temperature);
    }



    template <int dim>
    std::map<std::string, double>
    Manager<dim>::evaluate_initial_temperature_control(
      const LinearAlgebra::Vector &control,
      LinearAlgebra::Vector *l2_gradient)
    {
      AssertThrow(time_dependent_temperature_workflow_enabled(),
                  ExcMessage("Initial-temperature control evaluation requires the "
                             "time-dependent temperature adjoint workflow."));
      AssertThrow(initial_temperature_optimization.canonical_state.captured,
                  ExcMessage("Initial-temperature control evaluation requires a "
                             "captured O3 baseline."));
      AssertThrow(simulator->particle_managers.empty()
                  && simulator->mesh_deformation == nullptr,
                  ExcMessage("Reusable in-memory control evaluation supports only "
                             "fixed meshes without particles."));
      AssertThrow(this->get_parameters().adaptive_refinement_interval == 0
                  && this->get_parameters().additional_refinement_times.empty(),
                  ExcMessage("Reusable in-memory control evaluation requires a fixed mesh."));

      // One reduced value evaluation is a deterministic transaction:
      // restore canonical state -> install T_0(c) -> run the complete forward
      // time loop -> either retain its trajectory or immediately reverse it.
      restore_optimization_initial_state();
      install_initial_temperature_control(control);

      unsigned int max_refinement_level = this->get_parameters().initial_global_refinement
                                          + this->get_parameters().initial_adaptive_refinement;
      std::time_t last_checkpoint_time = std::time(nullptr);
      AssertThrow(!simulator->run_forward_time_loop(max_refinement_level,
                                                    last_checkpoint_time,
                                                    true),
                  ExcMessage("Reusable in-memory control evaluation cannot cross "
                             "an initial adaptive-refinement boundary."));

      if (l2_gradient == nullptr)
        {
          // Value-only callbacks evaluate the terminal scalar but leave the
          // just-recorded trajectory live. ROLReducedObjective may consume it
          // in a following gradient() call without a second forward solve.
          return objective_manager.evaluate_terminal_objectives();
        }

      return complete_initial_temperature_control_gradient(*l2_gradient);
    }



    template <int dim>
    std::map<std::string, double>
    Manager<dim>::complete_initial_temperature_control_gradient(
      LinearAlgebra::Vector &l2_gradient)
    {
      AssertThrow(time_dependent_temperature_workflow_enabled()
                  && initial_temperature_optimization.control_space != nullptr,
                  ExcMessage("A completed initial-temperature forward evaluation is required "
                             "before its reverse gradient can be requested."));

      // Reverse evaluation reconstructs/reclaims trajectory data and writes
      // ordinary adjoint artifacts. Suppress those external side effects for
      // optimizer callbacks while retaining the in-memory objective/gradient.
      suppress_evaluation_side_effects = true;
      try
        {
          finalize_time_dependent_temperature();
        }
      catch (...)
        {
          suppress_evaluation_side_effects = false;
          throw;
        }
      suppress_evaluation_side_effects = false;

      // The reverse sweep returns a full-space covector dJ/dT_0. First apply
      // P^T to obtain the reduced covector, then solve
      // (P^T M_T P) g=P^T dJ/dT_0 for the L2 Riesz representative expected by
      // ROLReducedL2Vector.
      l2_gradient.reinit(initial_temperature_optimization.control_space->locally_owned_control_dofs(),
                         this->get_mpi_communicator());
      LinearAlgebra::Vector reduced_covector(
        initial_temperature_optimization.control_space->locally_owned_control_dofs(),
        this->get_mpi_communicator());
      const unsigned int temperature_block = this->introspection().block_indices.temperature;
      initial_temperature_optimization.control_space->restrict_covector(
        time_dependent_workflow.initial_temperature_gradient.block(temperature_block), reduced_covector);
      const QGauss<dim> mass_quadrature(simulator->dof_handler.get_fe().degree + 1);
      initial_temperature_optimization.control_space->assemble_mass_matrix(*simulator->mapping, mass_quadrature);
      initial_temperature_optimization.control_space->solve_l2_riesz(reduced_covector, l2_gradient);
      return get_objective_values();
    }



    template <int dim>
    void
    Manager<dim>::run_initial_temperature_control_reuse_diagnostic()
    {
      AssertThrow(initial_temperature_optimization.canonical_state.captured,
                  ExcMessage("The O3 reuse diagnostic requires a captured baseline."));
      restore_optimization_initial_state();
      simulator->compute_current_constraints();
      const unsigned int temperature_block = this->introspection().block_indices.temperature;
      types::global_dof_index temperature_block_first_dof = 0;
      for (unsigned int block = 0; block < temperature_block; ++block)
        temperature_block_first_dof += this->introspection().system_dofs_per_block[block];
      initial_temperature_optimization.control_space = std::make_unique<InitialTemperatureControl<dim>>(
                                                         simulator->dof_handler, this->introspection().component_masks.temperature,
                                                         simulator->current_constraints, this->get_mpi_communicator(),
                                                         temperature_block_first_dof,
                                                         this->introspection().index_sets.system_partitioning[temperature_block],
                                                         this->introspection().index_sets.system_relevant_partitioning[temperature_block]);
      LinearAlgebra::Vector m0(initial_temperature_optimization.control_space->locally_owned_control_dofs(),
                               this->get_mpi_communicator());
      m0 = 0.0;
      LinearAlgebra::Vector m1(m0);
      for (const auto index : m1.locally_owned_elements())
        m1[index] = 1e-3;
      m1.compress(VectorOperation::insert);
      LinearAlgebra::Vector g0;
      LinearAlgebra::Vector g1;
      LinearAlgebra::Vector g0_again;
      LinearAlgebra::Vector g1_again;
      LinearAlgebra::Vector g0_final;
      LinearAlgebra::Vector raw_g0;
      LinearAlgebra::Vector raw_g1;
      LinearAlgebra::Vector raw_g0_again;
      LinearAlgebra::Vector raw_g1_again;
      LinearAlgebra::Vector raw_g0_final;
      const auto evaluate = [this, &temperature_block](
                              const LinearAlgebra::Vector &control,
                              LinearAlgebra::Vector &l2_gradient,
                              LinearAlgebra::Vector &raw_gradient)
      {
        const auto objective = evaluate_initial_temperature_control(control, &l2_gradient);
        raw_gradient.reinit(initial_temperature_optimization.control_space->locally_owned_control_dofs(),
                            this->get_mpi_communicator());
        initial_temperature_optimization.control_space->restrict_covector(
          time_dependent_workflow.initial_temperature_gradient.block(temperature_block), raw_gradient);
        return objective;
      };
      const auto j0 = evaluate(m0, g0, raw_g0);
      const auto j1 = evaluate(m1, g1, raw_g1);
      const auto j0_again = evaluate(m0, g0_again, raw_g0_again);
      const auto j1_again = evaluate(m1, g1_again, raw_g1_again);
      const auto j0_final = evaluate(m0, g0_final, raw_g0_final);
      AssertThrow(j0 == j0_again && j0 == j0_final && j1 == j1_again,
                  ExcMessage("O3 reuse diagnostic found non-reproducible terminal objectives."));
      AssertThrow(j0 != j1,
                  ExcMessage("O3 reuse diagnostic control perturbation did not change the terminal objective."));

      const auto relative_difference = [](const LinearAlgebra::Vector &reference,
                                          const LinearAlgebra::Vector &comparison)
      {
        LinearAlgebra::Vector difference(reference);
        difference -= comparison;
        return difference.l2_norm()
               / std::max(reference.l2_norm(), std::numeric_limits<double>::min());
      };
      const double l2_m0_difference = std::max(relative_difference(g0, g0_again),
                                               relative_difference(g0, g0_final));
      const double l2_m1_difference = relative_difference(g1, g1_again);
      const double raw_m0_difference = std::max(relative_difference(raw_g0, raw_g0_again),
                                                relative_difference(raw_g0, raw_g0_final));
      const double raw_m1_difference = relative_difference(raw_g1, raw_g1_again);
      AssertThrow(l2_m0_difference <= 1e-10 && l2_m1_difference <= 1e-10,
                  ExcMessage("O3 reuse diagnostic found a non-reproducible L2 gradient."));
      AssertThrow(raw_m0_difference <= 1e-10 && raw_m1_difference <= 1e-10,
                  ExcMessage("O3 reuse diagnostic found a non-reproducible reduced raw covector."));
      this->get_pcout() << "ASPECT_O3_REUSE_DIAGNOSTIC objectives=" << j0.size()
                        << " l2_m0_relative_difference=" << l2_m0_difference
                        << " l2_m1_relative_difference=" << l2_m1_difference
                        << " raw_m0_relative_difference=" << raw_m0_difference
                        << " raw_m1_relative_difference=" << raw_m1_difference
                        << " control_difference=" << m1.l2_norm() << std::endl;
    }



    template <int dim>
    void
    Manager<dim>::run_initial_temperature_control_single_evaluation_diagnostic()
    {
      AssertThrow(initial_temperature_optimization.canonical_state.captured,
                  ExcMessage("The O3 single-evaluation diagnostic requires a captured baseline."));
      restore_optimization_initial_state();
      simulator->compute_current_constraints();
      const unsigned int temperature_block = this->introspection().block_indices.temperature;
      types::global_dof_index temperature_block_first_dof = 0;
      for (unsigned int block = 0; block < temperature_block; ++block)
        temperature_block_first_dof += this->introspection().system_dofs_per_block[block];
      initial_temperature_optimization.control_space = std::make_unique<InitialTemperatureControl<dim>>(
                                                         simulator->dof_handler, this->introspection().component_masks.temperature,
                                                         simulator->current_constraints, this->get_mpi_communicator(),
                                                         temperature_block_first_dof,
                                                         this->introspection().index_sets.system_partitioning[temperature_block],
                                                         this->introspection().index_sets.system_relevant_partitioning[temperature_block]);
      LinearAlgebra::Vector m0(initial_temperature_optimization.control_space->locally_owned_control_dofs(),
                               this->get_mpi_communicator());
      m0 = 0.0;
      LinearAlgebra::Vector l2_gradient;
      const auto objective = evaluate_initial_temperature_control(m0, &l2_gradient);
      LinearAlgebra::Vector raw_gradient(
        initial_temperature_optimization.control_space->locally_owned_control_dofs(),
        this->get_mpi_communicator());
      initial_temperature_optimization.control_space->restrict_covector(
        time_dependent_workflow.initial_temperature_gradient.block(temperature_block), raw_gradient);
      this->get_pcout() << "ASPECT_O3_SINGLE_DIAGNOSTIC objectives=" << objective.size()
                        << " raw_gradient_norm=" << raw_gradient.l2_norm()
                        << " l2_gradient_norm=" << l2_gradient.l2_norm()
                        << std::endl;
    }



    template <int dim>
    void
    Manager<dim>::run_initial_temperature_control_taylor_diagnostic()
    {
      AssertThrow(initial_temperature_optimization.canonical_state.captured,
                  ExcMessage("The O3 Taylor diagnostic requires a captured baseline."));
      restore_optimization_initial_state();
      simulator->compute_current_constraints();
      const unsigned int temperature_block = this->introspection().block_indices.temperature;
      types::global_dof_index temperature_block_first_dof = 0;
      for (unsigned int block = 0; block < temperature_block; ++block)
        temperature_block_first_dof += this->introspection().system_dofs_per_block[block];
      initial_temperature_optimization.control_space = std::make_unique<InitialTemperatureControl<dim>>(
                                                         simulator->dof_handler, this->introspection().component_masks.temperature,
                                                         simulator->current_constraints, this->get_mpi_communicator(),
                                                         temperature_block_first_dof,
                                                         this->introspection().index_sets.system_partitioning[temperature_block],
                                                         this->introspection().index_sets.system_relevant_partitioning[temperature_block]);

      LinearAlgebra::Vector m0(initial_temperature_optimization.control_space->locally_owned_control_dofs(),
                               this->get_mpi_communicator());
      m0 = 0.0;
      LinearAlgebra::Vector direction(m0);
      for (const auto index : direction.locally_owned_elements())
        direction[index] = 1.0 + static_cast<double>(index % 3);
      direction.compress(VectorOperation::insert);

      LinearAlgebra::Vector l2_gradient;
      const auto objective_at_m0 = evaluate_initial_temperature_control(m0, &l2_gradient);
      AssertThrow(objective_at_m0.size() == 1,
                  ExcMessage("The O3 Taylor diagnostic requires exactly one objective."));
      const double j0 = objective_at_m0.begin()->second;
      LinearAlgebra::Vector raw_gradient(
        initial_temperature_optimization.control_space->locally_owned_control_dofs(),
        this->get_mpi_communicator());
      initial_temperature_optimization.control_space->restrict_covector(
        time_dependent_workflow.initial_temperature_gradient.block(temperature_block), raw_gradient);
      LinearAlgebra::Vector mass_times_direction(direction);
      initial_temperature_optimization.control_space->mass_matrix()->vmult(mass_times_direction, direction);
      const double raw_directional_derivative = raw_gradient * direction;
      const double l2_directional_derivative = l2_gradient * mass_times_direction;
      const double derivative_scale = std::max({std::fabs(raw_directional_derivative),
                                                std::fabs(l2_directional_derivative),
                                                std::numeric_limits<double>::min()
                                               });
      const double derivative_relative_difference =
        std::fabs(raw_directional_derivative - l2_directional_derivative) / derivative_scale;
      AssertThrow(derivative_relative_difference <= 1e-10,
                  ExcMessage("O3 Taylor diagnostic found inconsistent raw and L2 directional derivatives."));

      const std::vector<double> step_lengths = {1e-3, 5e-4, 2.5e-4, 1.25e-4};
      std::vector<double> remainders;
      remainders.reserve(step_lengths.size());
      for (const double step_length : step_lengths)
        {
          LinearAlgebra::Vector control(m0);
          control.add(step_length, direction);
          const auto objective = evaluate_initial_temperature_control(control);
          AssertThrow(objective.size() == 1,
                      ExcMessage("The O3 Taylor diagnostic requires exactly one objective."));
          const double remainder = std::fabs(objective.begin()->second - j0
                                             - step_length * raw_directional_derivative);
          remainders.push_back(remainder);
          this->get_pcout() << "ASPECT_O3_TAYLOR_DIAGNOSTIC h=" << step_length
                            << " objective_difference=" << objective.begin()->second - j0
                            << " remainder=" << remainder << std::endl;
        }
      for (unsigned int i = 1; i < remainders.size(); ++i)
        AssertThrow(remainders[i] < remainders[i-1],
                    ExcMessage("O3 Taylor diagnostic did not show a decreasing first-order remainder."));
      const double final_order = std::log(remainders[remainders.size()-2]
                                          / remainders.back()) / std::log(2.0);
      AssertThrow(final_order >= 1.5,
                  ExcMessage("O3 Taylor diagnostic did not show second-order remainder convergence."));
      this->get_pcout() << "ASPECT_O3_TAYLOR_DIAGNOSTIC raw_directional_derivative="
                        << raw_directional_derivative
                        << " l2_directional_derivative=" << l2_directional_derivative
                        << " derivative_relative_difference=" << derivative_relative_difference
                        << " final_order=" << final_order << std::endl;
    }



    template <int dim>
    void
    Manager<dim>::run_initial_temperature_control_initialization_diagnostic(
      const unsigned int stage)
    {
      AssertThrow(stage >= 1 && stage <= 8,
                  ExcMessage("O3 initialization diagnostic stage must be in [1,8]."));
      const auto report_stage_memory = [this](const std::string &stage_name)
      {
        std::size_t rss_bytes = 0;
#if defined(__APPLE__)
        mach_task_basic_info_data_t task_info_data;
        mach_msg_type_number_t task_info_count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(),
                      MACH_TASK_BASIC_INFO,
                      reinterpret_cast<task_info_t>(&task_info_data),
                      &task_info_count) == KERN_SUCCESS)
          rss_bytes = task_info_data.resident_size;
#else
        dealii::Utilities::System::MemoryStats process_memory {};
        dealii::Utilities::System::get_memory_stats(process_memory);
        rss_bytes = process_memory.VmRSS * 1024ULL;
#endif
        const auto control_global = (initial_temperature_optimization.control_space == nullptr
                                     ? 0
                                     : initial_temperature_optimization.control_space->n_control_dofs());
        const auto control_owned = (initial_temperature_optimization.control_space == nullptr
                                    ? 0
                                    : initial_temperature_optimization.control_space->locally_owned_control_dofs()
                                    .n_elements());
        std::cerr << "ASPECT_O3_INIT_MEMORY rank="
                  << dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator())
                  << " stage=" << stage_name
                  << " rss_bytes=" << rss_bytes
                  << " solution_global=" << simulator->solution.size()
                  << " control_global=" << control_global
                  << " control_owned=" << control_owned
                  << " reverse_cache_size=" << trajectory_manager.reverse_state_cache.size()
                  << std::endl;
      };
      report_stage_memory("D0");
      this->get_pcout() << "ASPECT_O3_INIT_DIAGNOSTIC stage=0" << std::endl;
      restore_optimization_initial_state();
      report_stage_memory("D1");
      this->get_pcout() << "ASPECT_O3_INIT_DIAGNOSTIC stage=1 restore" << std::endl;
      if (stage == 1)
        return;

      simulator->compute_current_constraints();
      report_stage_memory("D2");
      this->get_pcout() << "ASPECT_O3_INIT_DIAGNOSTIC stage=2 constraints" << std::endl;
      if (stage == 2)
        return;

      const unsigned int temperature_block = this->introspection().block_indices.temperature;
      types::global_dof_index temperature_block_first_dof = 0;
      for (unsigned int block = 0; block < temperature_block; ++block)
        temperature_block_first_dof += this->introspection().system_dofs_per_block[block];
      initial_temperature_optimization.control_space = std::make_unique<InitialTemperatureControl<dim>>(
                                                         simulator->dof_handler, this->introspection().component_masks.temperature,
                                                         simulator->current_constraints, this->get_mpi_communicator(),
                                                         temperature_block_first_dof,
                                                         this->introspection().index_sets.system_partitioning[temperature_block],
                                                         this->introspection().index_sets.system_relevant_partitioning[temperature_block]);
      initial_temperature_optimization.control_space->assert_temperature_block_mapping();
      report_stage_memory("D3");
      std::cout << "ASPECT_O3_INIT_LAYOUT rank="
                << dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator())
                << " control_global=" << initial_temperature_optimization.control_space->n_control_dofs()
                << " control_owned="
                << initial_temperature_optimization.control_space->locally_owned_control_dofs().n_elements()
                << " temperature_global="
                << initial_temperature_optimization.control_space->temperature_block_owned_dofs().size()
                << " temperature_owned="
                << initial_temperature_optimization.control_space->temperature_block_owned_dofs().n_elements()
                << std::endl;
      this->get_pcout() << "ASPECT_O3_INIT_DIAGNOSTIC stage=3 control-space global="
                        << initial_temperature_optimization.control_space->n_control_dofs()
                        << " local="
                        << initial_temperature_optimization.control_space->locally_owned_control_dofs().n_elements()
                        << std::endl;
      if (stage == 3)
        return;

      LinearAlgebra::Vector control(initial_temperature_optimization.control_space->locally_owned_control_dofs(),
                                    this->get_mpi_communicator());
      control = 0.0;
      report_stage_memory("D4");
      this->get_pcout() << "ASPECT_O3_INIT_DIAGNOSTIC stage=4 control-vector"
                        << " expected_temperature_size="
                        << initial_temperature_optimization.control_space->temperature_block_owned_dofs().size()
                        << " expected_owned="
                        << initial_temperature_optimization.control_space->temperature_block_owned_dofs().n_elements()
                        << " solution_temperature_size="
                        << simulator->solution.block(temperature_block).size()
                        << " solution_temperature_owned="
                        << simulator->solution.block(temperature_block).locally_owned_elements().n_elements()
                        << " fixed_temperature_size=" << initial_temperature_optimization.fixed_temperature->size()
                        << " fixed_temperature_owned="
                        << initial_temperature_optimization.fixed_temperature->locally_owned_elements().n_elements()
                        << std::endl;
      if (stage == 4)
        return;

      LinearAlgebra::Vector controlled_temperature(
        this->introspection().index_sets.system_partitioning[temperature_block],
        this->get_mpi_communicator());
      initial_temperature_optimization.control_space->inject_physical(
        control, *initial_temperature_optimization.fixed_temperature, controlled_temperature);
      simulator->solution.block(temperature_block) = controlled_temperature;
      report_stage_memory("D5");
      this->get_pcout() << "ASPECT_O3_INIT_DIAGNOSTIC stage=5 physical-injection" << std::endl;
      if (stage == 5)
        return;

      simulator->old_solution.block(temperature_block) =
        simulator->solution.block(temperature_block);
      simulator->old_old_solution.block(temperature_block) =
        simulator->solution.block(temperature_block);
      simulator->current_linearization_point.block(temperature_block) =
        simulator->solution.block(temperature_block);
      report_stage_memory("D6");
      this->get_pcout() << "ASPECT_O3_INIT_DIAGNOSTIC stage=6 histories" << std::endl;
      if (stage == 6)
        return;

      simulator->compute_initial_pressure_field();
      report_stage_memory("D7");
      this->get_pcout() << "ASPECT_O3_INIT_DIAGNOSTIC stage=7 initial-pressure" << std::endl;
      if (stage == 7)
        return;

      simulator->initialize_current_linearization_point();
      simulator->rebuild_sparsity_and_matrices = true;
      simulator->rebuild_stokes_matrix = true;
      simulator->rebuild_stokes_preconditioner = true;
      report_stage_memory("D8");
      this->get_pcout() << "ASPECT_O3_INIT_DIAGNOSTIC stage=8 runtime-invalidation" << std::endl;
    }

  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::Manager<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
