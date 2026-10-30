/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/adjoint/core/instantaneous_stokes_workflow.h>
#include <aspect/adjoint/operators/stokes_adjoint_solver.h>
#include <aspect/postprocess/dynamic_topography.h>
#include <aspect/simulator.h>

#include <deal.II/base/table_handler.h>

#include <algorithm>
#include <iterator>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    InstantaneousStokesWorkflow<dim>::InstantaneousStokesWorkflow(
      Simulator<dim> &simulator_object,
      Objective::Manager<dim> &objectives)
      :
      simulator(simulator_object),
      objective_manager(objectives)
    {
      this->initialize_simulator(simulator_object);
    }



    template <int dim>
    void
    InstantaneousStokesWorkflow<dim>::validate_setup() const
    {
      using MassConservation =
        typename Parameters<dim>::Formulation::MassConservation;
      const std::vector<std::string> &objective_names =
        objective_manager.get_active_plugin_names();
      const bool uses_dynamic_topography =
        std::find(objective_names.begin(), objective_names.end(),
                  "dynamic topography") != objective_names.end();

      if (uses_dynamic_topography)
        AssertThrow(
          this->get_postprocess_manager().template has_matching_active_plugin<
          Postprocess::DynamicTopography<dim>>(),
          ExcMessage("The dynamic topography adjoint objective requires the "
                     "dynamic topography postprocessor."));

      AssertThrow(this->get_parameters().formulation_mass_conservation
                  == MassConservation::incompressible,
                  ExcMessage("The instantaneous Stokes adjoint currently supports "
                             "only incompressible mass conservation."));
      AssertThrow(!this->get_material_model().is_compressible(),
                  ExcMessage("The instantaneous Stokes adjoint currently supports "
                             "only incompressible material models."));
    }



    template <int dim>
    void
    InstantaneousStokesWorkflow<dim>::prepare_required_forward_postprocessors()
    {
      // Objectives declare data dependencies rather than invoking
      // postprocessors themselves. Execute the union once so dynamic
      // topography, boundary density, and geoid caches all correspond to the
      // same freshly accepted Stokes solution.
      const std::set<std::string> required =
        objective_manager.required_forward_postprocessors();
      if (required.empty())
        return;

      TableHandler temporary_statistics;
      simulator.postprocess_manager.execute_selected_postprocessors(required,
                                                                    temporary_statistics);
    }



    template <int dim>
    void
    InstantaneousStokesWorkflow<dim>::run(
      std::vector<std::unique_ptr<ObjectiveResult>> &objective_results,
      std::vector<std::unique_ptr<AdjointState>> &adjoint_states,
      KernelRepository<dim> &kernels)
    {
      // An observation-generating reference may deliberately use the
      // instantaneous forward solver without an adjoint objective. In that
      // case the ordinary postprocessing stage writes its native products
      // after this forward Stokes solve, and there is no RHS, transpose solve,
      // or kernel to assemble.
      if (objective_manager.get_active_plugins().empty())
        {
          simulator.assemble_and_solve_stokes();
          objective_results.clear();
          adjoint_states.clear();
          kernels = KernelRepository<dim>();
          return;
        }

      validate_setup();

      // The steady computation graph is
      //
      //   parameters -> Stokes state -> required postprocessors -> J,
      //
      // and its reverse is
      //
      //   J -> objective RHS -> Stokes transpose -> volume kernels,
      //                         \-> direct postprocessor kernels.
      //
      // Every derivative below is evaluated at this newly assembled forward
      // Stokes state. Required postprocessors run before objective evaluation
      // so their cached boundary fields refer to the same state.
      simulator.assemble_and_solve_stokes();
      prepare_required_forward_postprocessors();
      objective_results = objective_manager.evaluate_and_assemble_right_hand_sides();

      adjoint_states.clear();
      const auto &objectives = objective_manager.get_active_plugins();
      const auto &objective_names = objective_manager.get_active_plugin_names();
      AssertThrow(objectives.size() == objective_names.size(), ExcInternalError());
      StokesAdjointSolver<dim> stokes_solver(simulator);

      for (const auto &objective_result : objective_results)
        {
          // Match the result to the capability-bearing plugin by its stable
          // registered name. Each objective owns a distinct RHS and adjoint
          // state so kernels remain attributable when several functionals are
          // active in one run.
          const auto name = std::find(objective_names.begin(),
                                      objective_names.end(),
                                      objective_result->objective_name);
          AssertThrow(name != objective_names.end(), ExcInternalError());
          const auto objective = std::next(objectives.begin(),
                                           std::distance(objective_names.begin(), name));
          const auto *instantaneous =
            dynamic_cast<const Objective::InstantaneousInterface<dim> *>(
              objective->get());
          AssertThrow(instantaneous != nullptr, ExcInternalError());

          auto adjoint_state = std::make_unique<AdjointState>();
          adjoint_state->objective_name = objective_result->objective_name;
          if (instantaneous->requires_physical_stokes_adjoint())
            stokes_solver.solve(objective_result->right_hand_side, adjoint_state->solution);
          else
            {
              // A purely explicit functional still participates in the same
              // result/kernel pipeline. Its implicit-state derivative is
              // exactly zero, represented by a correctly partitioned vector;
              // direct kernels are added after the loop.
              adjoint_state->solution.reinit(
                this->introspection().index_sets.system_partitioning,
                this->get_mpi_communicator());
              adjoint_state->solution = 0.0;
            }
          adjoint_states.push_back(std::move(adjoint_state));
        }

      KernelCalculator<dim> kernel_calculator(simulator);
      // KernelCalculator contracts the physical adjoint with derivatives of
      // the Stokes residual. Objective-specific explicit paths (for example
      // density in a harmonic observation or CBF denominator) are additive
      // and are appended by the objective manager afterward.
      kernels = kernel_calculator.calculate(objective_results, adjoint_states);
      objective_manager.add_direct_kernel_contributions(kernels);
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template class Adjoint::InstantaneousStokesWorkflow<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
