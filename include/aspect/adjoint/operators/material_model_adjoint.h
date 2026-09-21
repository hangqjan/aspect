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

#ifndef _aspect_adjoint_operators_material_model_adjoint_h
#define _aspect_adjoint_operators_material_model_adjoint_h

#include <aspect/global.h>
#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/fe/fe_values.h>

#include <vector>

namespace aspect
{
  template <int dim> class Simulator;

  namespace Adjoint
  {
    /**
     * Quadrature-point adjoint seeds for material model outputs.
     */
    template <int dim>
    struct MaterialModelOutputAdjoints
    {
      MaterialModelOutputAdjoints(const unsigned int n_points = 0);

      std::vector<double> viscosities;
      std::vector<double> densities;
      std::vector<double> specific_heat;
      std::vector<double> thermal_conductivities;
    };



    /**
     * Quadrature-point adjoints for material model inputs.
     */
    template <int dim>
    struct MaterialModelInputAdjoints
    {
      MaterialModelInputAdjoints(const unsigned int n_points = 0,
                                 const unsigned int n_compositional_fields = 0);

      std::vector<double> temperature;
      std::vector<double> pressure;
      std::vector<std::vector<double>> composition;
    };



    /**
     * Reverse-mode wrapper for the local constitutive map represented by the
     * active MaterialModel::Interface.
     *
     * This class only maps material-output seeds at evaluation points to
     * material-input adjoints at the same points. It deliberately does not
     * scatter these values to finite-element DoFs.
     */
    template <int dim>
    class MaterialModelAdjoint : public SimulatorAccess<dim>
    {
      public:
        explicit
        MaterialModelAdjoint(const Simulator<dim> &simulator);

        /**
         * Apply the transpose of the material model derivative to @p bar_out
         * and accumulate the result into @p bar_in.
         */
        void
        apply_derivative_transpose(const MaterialModel::MaterialModelInputs<dim> &in,
                                   const MaterialModel::MaterialModelOutputs<dim> &out,
                                   const MaterialModelOutputAdjoints<dim> &bar_out,
                                   MaterialModelInputAdjoints<dim> &bar_in) const;

      private:
        double
        seeded_output_scalar(const MaterialModel::MaterialModelOutputs<dim> &out,
                             const MaterialModelOutputAdjoints<dim> &bar_out,
                             const unsigned int q) const;
    };



    /**
     * Scatter material-input adjoints at quadrature points back to the
     * temperature-advection linearization/history DoFs used to evaluate the
     * material model.
     */
    template <int dim>
    class MaterialModelInputAdjointScatter : public SimulatorAccess<dim>
    {
      public:
        explicit
        MaterialModelInputAdjointScatter(const Simulator<dim> &simulator);

        /**
         * Scatter @p bar_in to temperature and composition DoFs using the
         * supplied history weights. If @p previous_previous_temperature_adjoint
         * is null, its weight is accumulated into @p previous_temperature_adjoint;
         * this preserves the one-step midpoint behavior used by the
         * entropy-viscosity estimator.
         */
        void
        scatter_temperature_advection_inputs(
          const FEValuesBase<dim, dim> &fe_values,
          const std::vector<types::global_dof_index> &local_dof_indices,
          const MaterialModelInputAdjoints<dim> &bar_in,
          const double previous_weight,
          const double previous_previous_weight,
          LinearAlgebra::BlockVector &previous_temperature_adjoint,
          LinearAlgebra::BlockVector *previous_previous_temperature_adjoint,
          std::vector<LinearAlgebra::BlockVector> *previous_composition_adjoints,
          std::vector<LinearAlgebra::BlockVector> *previous_previous_composition_adjoints) const;
    };
  }
}

#endif
