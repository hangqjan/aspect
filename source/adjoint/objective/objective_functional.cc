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

#include <aspect/adjoint/objective/objective_functional.h>
#include <aspect/adjoint/core/kernel_calculator.h>
#include <aspect/utilities.h>

#include <deal.II/base/utilities.h>

#include <set>
#include <tuple>

namespace aspect
{
  namespace Adjoint
  {
    namespace Objective
    {
      namespace
      {
        std::tuple<aspect::internal::Plugins::UnusablePluginList,
            aspect::internal::Plugins::UnusablePluginList,
            aspect::internal::Plugins::PluginList<Interface<2>>,
            aspect::internal::Plugins::PluginList<Interface<3>>> registered_objectives;
      }



      template <int dim>
      std::set<std::string>
      Interface<dim>::required_forward_postprocessors() const
      {
        return {};
      }



      template <int dim>
      void
      Manager<dim>::register_objective_functional(const std::string &name,
                                                  const std::string &description,
                                                  void (*declare_parameters_function) (ParameterHandler &),
                                                  std::unique_ptr<Interface<dim>> (*factory_function) ())
      {
        std::get<dim>(registered_objectives).register_plugin(name,
                                                             description,
                                                             declare_parameters_function,
                                                             factory_function);
      }



      template <int dim>
      void
      Manager<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Adjoint");
        {
          prm.declare_entry("List of objectives", "",
                            Patterns::MultipleSelection(std::get<dim>(registered_objectives).get_pattern_of_names()),
                            "Comma separated list of adjoint objective functionals.\n\n"
                            "The following objective functionals are available:\n\n"
                            +
                            std::get<dim>(registered_objectives).get_description_string());
        }
        prm.leave_subsection();

        std::get<dim>(registered_objectives).declare_parameters(prm);
      }



      template <int dim>
      void
      Manager<dim>::parse_parameters(ParameterHandler &prm)
      {
        this->plugin_objects.clear();
        this->plugin_names.clear();

        prm.enter_subsection("Adjoint");
        {
          this->plugin_names =
            Utilities::split_string_list(prm.get("List of objectives"));
        }
        prm.leave_subsection();

        AssertThrow(Utilities::has_unique_entries(this->plugin_names),
                    ExcMessage("The list of strings for the parameter "
                               "'Adjoint/List of objectives' contains entries more than once. "
                               "This is not allowed. Please check your parameter file."));

        // Instantiate each selected objective through the ASPECT plugin
        // registry. The order is preserved so that objective results and
        // adjoint states can later be paired by index.
        for (const std::string &objective_name : this->plugin_names)
          {
            this->plugin_objects.emplace_back(std::get<dim>(registered_objectives)
                                              .create_plugin(objective_name,
                                                             "Adjoint::List of objectives"));
            this->plugin_objects.back()->initialize_simulator(this->get_simulator());
            this->plugin_objects.back()->parse_parameters(prm);
            this->plugin_objects.back()->initialize();
          }
      }



      template <int dim>
      std::set<std::string>
      Manager<dim>::required_forward_postprocessors() const
      {
        // Form a set union so shared dependencies run once even when several
        // objectives consume the same forward postprocessor cache.
        std::set<std::string> required_postprocessors;

        for (const auto &objective : this->get_active_plugins())
          {
            const std::set<std::string> objective_required_postprocessors =
              objective->required_forward_postprocessors();

            required_postprocessors.insert(objective_required_postprocessors.begin(),
                                           objective_required_postprocessors.end());
          }

        return required_postprocessors;
      }



      template <int dim>
      std::vector<std::unique_ptr<ObjectiveResult>>
      Manager<dim>::evaluate_and_assemble_right_hand_sides() const
      {
        const auto &objective_functionals = this->get_active_plugins();
        const auto &objective_names = this->get_active_plugin_names();

        AssertThrow(objective_functionals.size() == objective_names.size(),
                    ExcInternalError());

        std::vector<std::unique_ptr<ObjectiveResult>> objective_results;
        objective_results.reserve(objective_names.size());

        auto objective = objective_functionals.begin();
        for (unsigned int objective_index = 0;
             objective_index < objective_names.size();
             ++objective_index, ++objective)
          {
            // Capability dispatch is explicit: only InstantaneousInterface
            // plugins participate in this workflow. Keep J_k and bar(x)_k
            // together so the steady workflow solves one physical adjoint per
            // objective and preserves kernel provenance.
            const auto *instantaneous =
              dynamic_cast<const InstantaneousInterface<dim> *>(objective->get());
            if (instantaneous == nullptr)
              continue;

            auto result = std::make_unique<ObjectiveResult>();
            result->objective_name = objective_names[objective_index];
            result->value = (*objective)->evaluate();
            result->right_hand_side.reinit(
              this->introspection().index_sets.system_partitioning,
              this->get_mpi_communicator());
            result->right_hand_side = 0.0;

            instantaneous->assemble_adjoint_right_hand_side(result->right_hand_side);
            objective_results.push_back(std::move(result));
          }

        return objective_results;
      }

      template <int dim>
      std::map<std::string, double>
      Manager<dim>::evaluate_terminal_objectives() const
      {
        std::map<std::string, double> values;
        const auto &objective_functionals = this->get_active_plugins();
        const auto &objective_names = this->get_active_plugin_names();
        AssertThrow(objective_functionals.size() == objective_names.size(), ExcInternalError());
        auto objective = objective_functionals.begin();
        for (unsigned int objective_index = 0; objective_index < objective_names.size();
             ++objective_index, ++objective)
          if (dynamic_cast<const TerminalInterface<dim> *>(objective->get()) != nullptr)
            values[objective_names[objective_index]] = (*objective)->evaluate();
        return values;
      }

      template <int dim>
      void
      Manager<dim>::assemble_terminal_advection_field_right_hand_side(
        const AdvectionField &advection_field,
        LinearAlgebra::BlockVector &right_hand_side) const
      {
        for (const auto &objective : this->get_active_plugins())
          if (const auto *terminal = dynamic_cast<const TerminalInterface<dim> *>(objective.get()))
            terminal->assemble_terminal_advection_field_right_hand_side(advection_field,
                                                                        right_hand_side);
        right_hand_side.compress(VectorOperation::add);
      }

      template <int dim>
      std::map<std::string, double>
      Manager<dim>::evaluate_and_assemble_terminal_temperature_right_hand_side(
        LinearAlgebra::BlockVector &right_hand_side) const
      {
        return evaluate_and_assemble_terminal_advection_field_right_hand_side(AdvectionField::temperature(),
                                                                              right_hand_side);
      }

      template <int dim>
      std::map<std::string, double>
      Manager<dim>::evaluate_and_assemble_terminal_advection_field_right_hand_side(
        const AdvectionField &advection_field,
        LinearAlgebra::BlockVector &right_hand_side) const
      {
        const std::map<std::string, double> values = evaluate_terminal_objectives();
        assemble_terminal_advection_field_right_hand_side(advection_field, right_hand_side);
        return values;
      }

      template <int dim>
      std::map<std::string, double>
      Manager<dim>::evaluate_running_stokes_objectives(const unsigned int timestep) const
      {
        // Running objectives return the already time-discretized contribution
        // associated with this accepted step. TrajectoryManager accumulates
        // these scalars over the forward schedule.
        std::map<std::string, double> values;
        const auto &objective_functionals = this->get_active_plugins();
        const auto &objective_names = this->get_active_plugin_names();
        AssertThrow(objective_functionals.size() == objective_names.size(), ExcInternalError());
        auto objective = objective_functionals.begin();
        for (unsigned int objective_index = 0; objective_index < objective_names.size();
             ++objective_index, ++objective)
          if (const auto *running = dynamic_cast<const RunningStokesInterface<dim> *>(objective->get()))
            values[objective_names[objective_index]] =
              running->evaluate_running_stokes_objective(timestep);
        return values;
      }

      template <int dim>
      void
      Manager<dim>::assemble_running_stokes_adjoint_right_hand_side(
        const unsigned int timestep,
        LinearAlgebra::BlockVector &right_hand_side) const
      {
        // At reverse step n, add all derivatives of the running functional
        // with respect to the quasi-static Stokes state at n. The combined RHS
        // requires only one Stokes transpose solve at this time level.
        for (const auto &objective : this->get_active_plugins())
          if (const auto *running = dynamic_cast<const RunningStokesInterface<dim> *>(objective.get()))
            running->assemble_running_stokes_adjoint_rhs(timestep, right_hand_side);
        right_hand_side.compress(VectorOperation::add);
      }



      template <int dim>
      void
      Manager<dim>::add_direct_kernel_contributions(KernelRepository<dim> &kernels) const
      {
        const auto &objective_functionals = this->get_active_plugins();
        const auto &objective_names = this->get_active_plugin_names();

        AssertThrow(objective_functionals.size() == objective_names.size(),
                    ExcInternalError());

        // Direct kernels differentiate observation/postprocessor formulas at
        // fixed state. They are separate additive paths from the implicit
        // kernels obtained by contracting the physical Stokes adjoint.
        auto objective = objective_functionals.begin();
        for (unsigned int objective_index = 0;
             objective_index < objective_names.size();
             ++objective_index, ++objective)
          if (const auto *instantaneous =
                dynamic_cast<const InstantaneousInterface<dim> *>(objective->get()))
            instantaneous->add_direct_kernel_contributions(objective_names[objective_index],
                                                           kernels);
      }
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template std::set<std::string> Adjoint::Objective::Interface<dim>::required_forward_postprocessors() const; \
  template class Adjoint::Objective::Manager<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
