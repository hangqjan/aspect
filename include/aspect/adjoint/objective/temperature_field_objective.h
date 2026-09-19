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

#ifndef _aspect_adjoint_objective_temperature_field_objective_h
#define _aspect_adjoint_objective_temperature_field_objective_h

#include <aspect/adjoint/objective/terminal_scalar_field_misfit.h>

namespace aspect
{
  namespace Adjoint
  {
    /**
     * Terminal temperature-field objective for the transient adjoint
     * workflow. The objective is
     * 1/2 int w(x) (T_final(x)-T_obs(x))^2 dx.
     *
     * @ingroup Adjoint
     */
    template <int dim>
    class TemperatureFieldObjective : public TerminalScalarFieldMisfit<dim>
    {
      public:
        TemperatureFieldObjective ();

        static
        void
        declare_parameters (ParameterHandler &prm);

        void
        parse_parameters (ParameterHandler &prm) override;

        /** Use a copied terminal state as the temperature observation. */
        void
        set_discrete_terminal_observation(const LinearAlgebra::BlockVector &observation);

        void
        clear_discrete_terminal_observation();

      private:
        AdvectionField observed_field() const override;
    };
  }
}

#endif
