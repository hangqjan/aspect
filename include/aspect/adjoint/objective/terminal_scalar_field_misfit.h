/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.
*/

#ifndef _aspect_adjoint_objective_terminal_scalar_field_misfit_h
#define _aspect_adjoint_objective_terminal_scalar_field_misfit_h

#include <aspect/adjoint/objective/objective_functional.h>
#include <aspect/adjoint/observation/full_field_observation.h>

#include <deal.II/base/parsed_function.h>

namespace aspect
{
  namespace Adjoint
  {
    /**
     * Shared implementation for terminal weighted volume-L2 scalar-field
     * misfits. This is deliberately not an objective plugin: concrete
     * temperature and composition plugins retain their own public parameters.
     */
    template <int dim>
    class TerminalScalarFieldMisfit : public Objective::Interface<dim>,
      public Objective::TerminalInterface<dim>
    {
      public:
        TerminalScalarFieldMisfit();

        void initialize_simulator(const Simulator<dim> &simulator) override;
        void initialize() override;

        double evaluate() const override;
        void assemble_terminal_advection_field_right_hand_side(
          const AdvectionField &advection_field,
          LinearAlgebra::BlockVector &right_hand_side) const override;

        bool observes_field(const AdvectionField &field) const;
        void set_discrete_terminal_observation(const LinearAlgebra::BlockVector &state);
        void clear_discrete_terminal_observation();

      protected:
        virtual AdvectionField observed_field() const = 0;

        static void declare_common_parameters(ParameterHandler &prm);
        void parse_common_parameters(ParameterHandler &prm);

        mutable Observation::FullFieldObservation<dim> observation;
        Functions::ParsedFunction<dim> weight_function;
        Utilities::Coordinates::CoordinateSystem coordinate_system;

      private:
        double weight(const Point<dim> &position) const;
    };
  }
}

#endif
