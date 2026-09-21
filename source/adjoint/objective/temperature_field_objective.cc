/*
  Copyright (C) 2011 - 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include <aspect/adjoint/objective/temperature_field_objective.h>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    TemperatureFieldObjective<dim>::TemperatureFieldObjective() = default;

    template <int dim>
    AdvectionField TemperatureFieldObjective<dim>::observed_field() const
    {
      return AdvectionField::temperature();
    }

    template <int dim>
    void TemperatureFieldObjective<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Adjoint");
      {
        prm.enter_subsection("Temperature field");
        {
          TerminalScalarFieldMisfit<dim>::declare_common_parameters(prm);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void TemperatureFieldObjective<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Adjoint");
      {
        prm.enter_subsection("Temperature field");
        {
          this->parse_common_parameters(prm);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    ASPECT_REGISTER_ADJOINT_OBJECTIVE(TemperatureFieldObjective,
                                      "temperature field",
                                      "Terminal temperature-field misfit objective for "
                                      "the time-dependent temperature adjoint workflow.")
  }
}
