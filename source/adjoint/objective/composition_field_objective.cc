/*
  Copyright (C) 2011 - 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include <aspect/adjoint/objective/composition_field_objective.h>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    CompositionFieldObjective<dim>::CompositionFieldObjective()
      : field_name(""),
        compositional_index(numbers::invalid_unsigned_int)
    {}

    template <int dim>
    AdvectionField CompositionFieldObjective<dim>::observed_field() const
    {
      AssertThrow(compositional_index != numbers::invalid_unsigned_int,
                  ExcInternalError());
      return AdvectionField::composition(compositional_index);
    }

    template <int dim>
    void CompositionFieldObjective<dim>::set_discrete_terminal_observation(
      const LinearAlgebra::BlockVector &state)
    {
      TerminalScalarFieldMisfit<dim>::set_discrete_terminal_observation(state);
    }

    template <int dim>
    void CompositionFieldObjective<dim>::clear_discrete_terminal_observation()
    {
      TerminalScalarFieldMisfit<dim>::clear_discrete_terminal_observation();
    }

    template <int dim>
    void CompositionFieldObjective<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Adjoint");
      {
        prm.enter_subsection("Composition field");
        {
          prm.declare_entry("Field name", "", Patterns::Anything(),
                            "Name of the compositional field used by the terminal "
                            "composition misfit objective.");
          TerminalScalarFieldMisfit<dim>::declare_common_parameters(prm);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void CompositionFieldObjective<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Adjoint");
      {
        prm.enter_subsection("Composition field");
        {
          field_name = prm.get("Field name");
          AssertThrow(!field_name.empty(),
                      ExcMessage("Adjoint/Composition field/Field name must name one "
                                 "compositional field."));
          compositional_index = this->introspection().compositional_index_for_name(field_name);
          this->parse_common_parameters(prm);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    ASPECT_REGISTER_ADJOINT_OBJECTIVE(CompositionFieldObjective,
                                      "composition field",
                                      "Terminal composition-field misfit objective for "
                                      "the time-dependent temperature adjoint workflow.")
  }
}
