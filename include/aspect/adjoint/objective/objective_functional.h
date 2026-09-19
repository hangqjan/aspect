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

#ifndef _aspect_adjoint_objective_objective_functional_h
#define _aspect_adjoint_objective_objective_functional_h

#include <aspect/advection_field.h>
#include <aspect/adjoint/core/state.h>
#include <aspect/plugins.h>
#include <aspect/simulator_access.h>

#include <memory>
#include <map>
#include <set>
#include <string>

namespace aspect
{
  /**
   * A namespace for everything to do with adjoint methods.
   *
   * @ingroup Adjoint
   */
  namespace Adjoint
  {
    template <int dim> class KernelRepository;

    namespace Objective
    {

      /**
       * A base class for adjoint objective functionals. Implementations
       * evaluate an objective, assemble its contribution to the adjoint right
       * hand side, and may add direct physical-property kernel contributions.
       *
       * @ingroup Adjoint
       */
      template <int dim>
      class Interface : public Plugins::InterfaceBase,
        public SimulatorAccess<dim>
      {
        public:
          /**
           * Destructor.
           */
          virtual
          ~Interface() override = default;

          /**
           * Evaluate the objective functional for the current forward solution.
           *
           * @return The scalar value of the objective functional.
           */
          virtual
          double
          evaluate() const = 0;

          /**
           * Return the names of forward postprocessors that must be executed
           * before this objective functional is evaluated or assembled. The
           * names are the same strings used in parameter files. The default
           * implementation returns an empty set.
           *
           * @return A set of required forward postprocessor names.
           */
          virtual
          std::set<std::string>
          required_forward_postprocessors() const;
      };

      /** Capability implemented by objectives of the current Stokes state. */
      template <int dim>
      class InstantaneousInterface
      {
        public:
          virtual ~InstantaneousInterface() = default;

          virtual void
          assemble_adjoint_right_hand_side(LinearAlgebra::BlockVector &right_hand_side) const = 0;

          virtual void
          add_direct_kernel_contributions(const std::string &objective_name,
                                          KernelRepository<dim> &kernels) const = 0;

          virtual bool
          requires_physical_stokes_adjoint() const
          {
            return true;
          }
      };

      /** Capability implemented by objectives defined on the terminal state. */
      template <int dim>
      class TerminalInterface
      {
        public:
          virtual ~TerminalInterface() = default;

          virtual void
          assemble_terminal_advection_field_right_hand_side(
            const AdvectionField &advection_field,
            LinearAlgebra::BlockVector &right_hand_side) const = 0;
      };

      /**
       * Optional mixin for objective plugins that contribute at intermediate
       * Stokes states. Keeping this capability separate lets the manager opt in
       * without adding default no-op methods to the common objective interface.
       */
      template <int dim>
      class RunningStokesInterface
      {
        public:
          virtual ~RunningStokesInterface() = default;

          virtual double
          evaluate_running_stokes_objective(const unsigned int timestep) const = 0;

          virtual void
          assemble_running_stokes_adjoint_rhs(const unsigned int timestep,
                                              LinearAlgebra::BlockVector &rhs) const = 0;
      };



      /**
       * A class that manages all active adjoint objective functionals. It reads
       * the list of selected objectives from the parameter file and creates the
       * corresponding objective plugins. It also provides functions that call
       * all active objective plugins to collect required postprocessors,
       * assemble objective right hand sides, and add direct kernel
       * contributions.
       *
       * @ingroup Adjoint
       */
      template <int dim>
      class Manager : public Plugins::ManagerBase<Interface<dim>>,
        public SimulatorAccess<dim>
      {
        public:
          /**
           * Register an adjoint objective functional so that it can be selected
           * from the parameter file. This allows objective implementations to
           * register themselves without modifying the manager whenever a new
           * objective functional is added.
           *
           * @param name A string that identifies the objective functional.
           * @param description A text description of what this objective
           * functional does and that will be listed in the documentation of the
           * parameter file.
           * @param declare_parameters_function A pointer to a function that can
           * be used to declare the parameters that this objective functional
           * wants to read from input files.
           * @param factory_function A pointer to a function that can create an
           * object of this objective functional.
           */
          static
          void
          register_objective_functional(const std::string &name,
                                        const std::string &description,
                                        void (*declare_parameters_function) (ParameterHandler &),
                                        std::unique_ptr<Interface<dim>> (*factory_function) ());

          /**
           * Declare the parameters used by the objective manager and all
           * registered objective functional plugins.
           *
           * @param prm The parameter handler in which parameters are declared.
           */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /**
           * Read the list of objective functionals from the parameter file and
           * create the corresponding objective plugin objects.
           *
           * @param prm The parameter handler from which the objective list is
           * read.
           */
          void
          parse_parameters(ParameterHandler &prm) override;

          /**
           * Return the names of all forward postprocessors required by active
           * objective functionals.
           *
           * @return A set of required forward postprocessor names.
           */
          std::set<std::string>
          required_forward_postprocessors() const;

          /**
           * Evaluate all active objective functionals for the current forward
           * solution and assemble their adjoint right hand side contributions.
           *
           * @return Objective values and right hand sides for all active
           * objective functionals.
           */
          std::vector<std::unique_ptr<ObjectiveResult>>
          evaluate_and_assemble_right_hand_sides() const;

          /** Evaluate all terminal objectives once at the current state. */
          std::map<std::string, double>
          evaluate_terminal_objectives() const;

          /** Assemble terminal seeds for one selected advected field. */
          void
          assemble_terminal_advection_field_right_hand_side(
            const AdvectionField &advection_field,
            LinearAlgebra::BlockVector &right_hand_side) const;

          /**
           * Evaluate all active objective functionals and assemble their
           * terminal-temperature adjoint seed into a single right hand side.
           *
           * @param[in,out] right_hand_side The full-system vector that receives
           * the terminal temperature seed.
           *
           * @return A map from objective functional names to objective values.
           */
          std::map<std::string, double>
          evaluate_and_assemble_terminal_temperature_right_hand_side(
            LinearAlgebra::BlockVector &right_hand_side) const;

          /**
           * Evaluate all active objective functionals and assemble their
           * terminal adjoint seed for one advected field into a single right hand
           * side.
           *
           * @param advection_field The advected field whose terminal seed is
           * requested.
           * @param[in,out] right_hand_side The full-system vector that receives
           * the seed.
           *
           * @return A map from objective functional names to objective values.
           */
          std::map<std::string, double>
          evaluate_and_assemble_terminal_advection_field_right_hand_side(
            const AdvectionField &advection_field,
            LinearAlgebra::BlockVector &right_hand_side) const;

          /** Evaluate all time-distributed Stokes objective contributions. */
          std::map<std::string, double>
          evaluate_running_stokes_objectives(const unsigned int timestep) const;

          /** Assemble all time-distributed Stokes-adjoint seed contributions. */
          void
          assemble_running_stokes_adjoint_right_hand_side(
            const unsigned int timestep,
            LinearAlgebra::BlockVector &right_hand_side) const;


          /**
           * Add direct physical-property kernel contributions from all active
           * objective functionals for the current forward solution.
           *
           * @param[in,out] kernels The repository to which active objectives add
           * direct kernel contributions.
           */
          void
          add_direct_kernel_contributions(KernelRepository<dim> &kernels) const;
      };

    }



#define ASPECT_REGISTER_ADJOINT_OBJECTIVE(classname, name, description) \
  template class classname<2>; \
  template class classname<3>; \
  namespace ASPECT_REGISTER_ADJOINT_OBJECTIVE_ ## classname \
  { \
    aspect::internal::Plugins::RegisterHelper<aspect::Adjoint::Objective::Interface<2>,classname<2>> \
    dummy_ ## classname ## _2d (&aspect::Adjoint::Objective::Manager<2>::register_objective_functional, \
                                name, description); \
    aspect::internal::Plugins::RegisterHelper<aspect::Adjoint::Objective::Interface<3>,classname<3>> \
    dummy_ ## classname ## _3d (&aspect::Adjoint::Objective::Manager<3>::register_objective_functional, \
                                name, description); \
  }
  }
}

#endif
