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

#include <aspect/adjoint/operators/material_model_adjoint.h>
#include <aspect/advection_field.h>
#include <aspect/simulator.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace aspect
{
  namespace Adjoint
  {
    namespace
    {
      bool
      has_nonzero_entry(const std::vector<double> &values)
      {
        for (const double value : values)
          if (value != 0.0)
            return true;

        return false;
      }



      bool
      dependence_includes(const MaterialModel::NonlinearDependence::Dependence dependence,
                          const MaterialModel::NonlinearDependence::Dependence variable)
      {
        return (static_cast<int>(dependence) & static_cast<int>(variable)) != 0;
      }



      MaterialModel::MaterialProperties::Property
      add_property(const MaterialModel::MaterialProperties::Property properties,
                   const MaterialModel::MaterialProperties::Property property)
      {
        return static_cast<MaterialModel::MaterialProperties::Property>(
                 static_cast<int>(properties) | static_cast<int>(property));
      }



      double
      finite_difference_step(const double value)
      {
        return 1e-7 * std::max(1.0, std::abs(value));
      }
    }



    template <int dim>
    MaterialModelOutputAdjoints<dim>::MaterialModelOutputAdjoints(const unsigned int n_points)
      :
      viscosities(n_points, 0.0),
      densities(n_points, 0.0),
      specific_heat(n_points, 0.0),
      thermal_conductivities(n_points, 0.0)
    {}



    template <int dim>
    MaterialModelInputAdjoints<dim>::MaterialModelInputAdjoints(const unsigned int n_points,
                                                                const unsigned int n_compositional_fields)
      :
      temperature(n_points, 0.0),
      pressure(n_points, 0.0),
      composition(n_points, std::vector<double>(n_compositional_fields, 0.0))
    {}



    template <int dim>
    MaterialModelAdjoint<dim>::MaterialModelAdjoint(const Simulator<dim> &simulator)
    {
      this->initialize_simulator(simulator);
    }



    template <int dim>
    double
    MaterialModelAdjoint<dim>::seeded_output_scalar(const MaterialModel::MaterialModelOutputs<dim> &out,
                                                    const MaterialModelOutputAdjoints<dim> &bar_out,
                                                    const unsigned int q) const
    {
      // Contract the material outputs y=(eta,rho,c_p,k) with their adjoint
      // seeds. For one quadrature point this scalar is
      //   phi(x)=bar_eta*eta(x)+bar_rho*rho(x)
      //          +bar_c_p*c_p(x)+bar_k*k(x).
      // Differentiating phi with respect to a material input x is exactly the
      // vector-Jacobian product J_material(x)^T bar_y required by reverse mode.
      return bar_out.viscosities[q] * out.viscosities[q]
             + bar_out.densities[q] * out.densities[q]
             + bar_out.specific_heat[q] * out.specific_heat[q]
             + bar_out.thermal_conductivities[q] * out.thermal_conductivities[q];
    }



    template <int dim>
    void
    MaterialModelAdjoint<dim>::apply_derivative_transpose(const MaterialModel::MaterialModelInputs<dim> &in,
                                                          const MaterialModel::MaterialModelOutputs<dim> &out,
                                                          const MaterialModelOutputAdjoints<dim> &bar_out,
                                                          MaterialModelInputAdjoints<dim> &bar_in) const
    {
      // bar_in is an accumulation target, not an output to overwrite. Multiple
      // residual terms can seed the same material evaluation, so every
      // derivative below is added to the existing temperature, pressure, or
      // composition adjoint at its quadrature point.
      const unsigned int n_points = in.n_evaluation_points();
      AssertDimension(out.n_evaluation_points(), n_points);
      (void) out;
      AssertDimension(bar_out.viscosities.size(), n_points);
      AssertDimension(bar_out.densities.size(), n_points);
      AssertDimension(bar_out.specific_heat.size(), n_points);
      AssertDimension(bar_out.thermal_conductivities.size(), n_points);
      AssertDimension(bar_in.temperature.size(), n_points);
      AssertDimension(bar_in.pressure.size(), n_points);
      AssertDimension(bar_in.composition.size(), n_points);

      const unsigned int n_compositional_fields =
        n_points > 0 ? in.composition[0].size() : 0;
      for (unsigned int q = 0; q < n_points; ++q)
        {
          AssertDimension(bar_in.composition[q].size(), n_compositional_fields);
          AssertDimension(in.composition[q].size(), n_compositional_fields);
        }

      const bool seed_viscosity = has_nonzero_entry(bar_out.viscosities);
      const bool seed_density = has_nonzero_entry(bar_out.densities);
      const bool seed_specific_heat = has_nonzero_entry(bar_out.specific_heat);
      const bool seed_thermal_conductivity =
        has_nonzero_entry(bar_out.thermal_conductivities);

      // A zero output covector has a zero transpose action. Besides avoiding
      // material-model evaluations, this guard prevents rejecting unsupported
      // dependencies that are irrelevant to the active objective path.
      if (!seed_viscosity
          && !seed_density
          && !seed_specific_heat
          && !seed_thermal_conductivity)
        return;

      const MaterialModel::NonlinearDependence::ModelDependence model_dependence =
        this->get_material_model().get_model_dependence();

      // The finite-difference backend perturbs scalar T, p, and C inputs only.
      // Strain rate is tensor-valued and pressure-dependent viscosity is not
      // part of the audited v1 Stokes/material transpose. Fail before forming
      // a partial VJP whenever a nonzero viscosity seed would require either
      // omitted derivative.
      AssertThrow(!(seed_viscosity
                    && dependence_includes(model_dependence.viscosity,
                                           MaterialModel::NonlinearDependence::strain_rate)),
                  ExcMessage("The adjoint generated a non-zero viscosity seed, but the active material "
                             "model declares strain-rate dependence and the current material-adjoint "
                             "backend does not support strain-rate derivatives. Continuing would produce "
                             "an incomplete discrete derivative."));
      AssertThrow(!(seed_viscosity
                    && dependence_includes(model_dependence.viscosity,
                                           MaterialModel::NonlinearDependence::pressure)),
                  ExcMessage("The adjoint generated a non-zero viscosity seed, but the active material "
                             "model declares pressure-dependent viscosity. The v1 material-adjoint "
                             "layer supports only eta(T,C), so eta(p) must remain disabled."));

      auto any_seeded_output_depends_on =
        [&](const MaterialModel::NonlinearDependence::Dependence variable)
      {
        return (seed_viscosity
                && dependence_includes(model_dependence.viscosity, variable))
               || (seed_density
                   && dependence_includes(model_dependence.density, variable))
               || (seed_specific_heat
                   && dependence_includes(model_dependence.specific_heat, variable))
               || (seed_thermal_conductivity
                   && dependence_includes(model_dependence.thermal_conductivity, variable));
      };

      // Request only outputs that carry a nonzero seed. A material plugin may
      // allocate expensive additional outputs, so the transpose must preserve
      // the same requested-properties contract as an ordinary forward call.
      MaterialModel::MaterialProperties::Property requested_properties =
        MaterialModel::MaterialProperties::uninitialized;
      if (seed_viscosity)
        requested_properties = add_property(requested_properties,
                                            MaterialModel::MaterialProperties::viscosity);
      if (seed_density)
        requested_properties = add_property(requested_properties,
                                            MaterialModel::MaterialProperties::density);
      if (seed_specific_heat)
        requested_properties = add_property(requested_properties,
                                            MaterialModel::MaterialProperties::specific_heat);
      if (seed_thermal_conductivity)
        requested_properties = add_property(requested_properties,
                                            MaterialModel::MaterialProperties::thermal_conductivity);

      auto evaluate_seeded_scalar = [&](MaterialModel::MaterialModelInputs<dim> material_in,
                                        const unsigned int q)
      {
        // Evaluate the scalar contraction phi rather than finite-differencing
        // every material output separately. By linearity,
        //   d phi/dx = sum_j bar_y_j dy_j/dx = J^T bar_y,
        // which needs two material evaluations per active scalar input.
        MaterialModel::MaterialModelOutputs<dim> material_out(n_points,
                                                              n_compositional_fields);
        material_in.requested_properties = requested_properties;
        this->get_material_model().evaluate(material_in, material_out);

        return seeded_output_scalar(material_out, bar_out, q);
      };

      if (any_seeded_output_depends_on(MaterialModel::NonlinearDependence::temperature))
        // Central differences approximate the local VJP component by
        // [phi(T_q+eps)-phi(T_q-eps)]/(2 eps). Only point q is perturbed;
        // material plugins are assumed point-local under the standard ASPECT
        // material-model interface.
        for (unsigned int q = 0; q < n_points; ++q)
          {
            MaterialModel::MaterialModelInputs<dim> in_plus = in;
            MaterialModel::MaterialModelInputs<dim> in_minus = in;
            const double epsilon = finite_difference_step(in.temperature[q]);

            in_plus.temperature[q] += epsilon;
            in_minus.temperature[q] -= epsilon;

            bar_in.temperature[q] +=
              (evaluate_seeded_scalar(std::move(in_plus), q)
               - evaluate_seeded_scalar(std::move(in_minus), q))
              / (2.0 * epsilon);
          }

      if (any_seeded_output_depends_on(MaterialModel::NonlinearDependence::pressure))
        // Apply the identical scalar-contraction construction to pressure.
        // Callers that cannot scatter bar_p into a pressure space reject a
        // nonzero result explicitly rather than silently dropping it.
        for (unsigned int q = 0; q < n_points; ++q)
          {
            MaterialModel::MaterialModelInputs<dim> in_plus = in;
            MaterialModel::MaterialModelInputs<dim> in_minus = in;
            const double epsilon = finite_difference_step(in.pressure[q]);

            in_plus.pressure[q] += epsilon;
            in_minus.pressure[q] -= epsilon;

            bar_in.pressure[q] +=
              (evaluate_seeded_scalar(std::move(in_plus), q)
               - evaluate_seeded_scalar(std::move(in_minus), q))
              / (2.0 * epsilon);
          }

      if (any_seeded_output_depends_on(MaterialModel::NonlinearDependence::compositional_fields))
        // Each compositional component is an independent material input. The
        // nested q,c loop therefore forms every column of the local material
        // Jacobian that is touched by the output covector.
        for (unsigned int q = 0; q < n_points; ++q)
          for (unsigned int c = 0; c < n_compositional_fields; ++c)
            {
              MaterialModel::MaterialModelInputs<dim> in_plus = in;
              MaterialModel::MaterialModelInputs<dim> in_minus = in;
              const double epsilon = finite_difference_step(in.composition[q][c]);

              in_plus.composition[q][c] += epsilon;
              in_minus.composition[q][c] -= epsilon;

              bar_in.composition[q][c] +=
                (evaluate_seeded_scalar(std::move(in_plus), q)
                 - evaluate_seeded_scalar(std::move(in_minus), q))
                / (2.0 * epsilon);
            }
    }



    template <int dim>
    MaterialModelInputAdjointScatter<dim>::MaterialModelInputAdjointScatter(const Simulator<dim> &simulator)
    {
      this->initialize_simulator(simulator);
    }



    template <int dim>
    void
    MaterialModelInputAdjointScatter<dim>::scatter_temperature_advection_inputs(
      const FEValuesBase<dim, dim> &fe_values,
      const std::vector<types::global_dof_index> &local_dof_indices,
      const MaterialModelInputAdjoints<dim> &bar_in,
      const double previous_weight,
      const double previous_previous_weight,
      LinearAlgebra::BlockVector &previous_temperature_adjoint,
      LinearAlgebra::BlockVector *previous_previous_temperature_adjoint,
      std::vector<LinearAlgebra::BlockVector> *previous_composition_adjoints,
      std::vector<LinearAlgebra::BlockVector> *previous_previous_composition_adjoints) const
    {
      // Material properties in the advection residual are evaluated at the
      // extrapolated field
      //   x* = w_{n-1} x_{n-1} + w_{n-2} x_{n-2}.
      // Hence the reverse interpolation sends a quadrature seed bar_x* to
      // w_{n-1} P^T bar_x* and w_{n-2} P^T bar_x*, where P evaluates FE basis
      // functions at quadrature points. This routine performs that P^T scatter
      // for temperature and every compositional field.
      const unsigned int n_points = fe_values.n_quadrature_points;
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      AssertDimension(local_dof_indices.size(), dofs_per_cell);
      AssertDimension(bar_in.temperature.size(), n_points);
      AssertDimension(bar_in.pressure.size(), n_points);
      AssertDimension(bar_in.composition.size(), n_points);

      for (const double pressure_adjoint : bar_in.pressure)
        // The audited temperature-advection path has no pressure FE target for
        // a material input adjoint. A nonzero bar_p would therefore make the
        // derivative incomplete and must be rejected before assembly.
        AssertThrow(pressure_adjoint == 0.0,
                    ExcMessage("The material adjoint generated a non-zero pressure input adjoint, "
                               "but the v1 temperature-advection material-input scatter does not "
                               "support pressure DoFs. Continuing would silently drop part of the "
                               "discrete derivative."));

      const unsigned int n_compositional_fields =
        n_points > 0 ? bar_in.composition[0].size() : 0;

      Vector<double> local_previous_temperature(dofs_per_cell);
      Vector<double> local_previous_previous_temperature(dofs_per_cell);
      std::vector<Vector<double>> local_previous_composition(n_compositional_fields,
                                                              Vector<double>(dofs_per_cell));
      std::vector<Vector<double>> local_previous_previous_composition(n_compositional_fields,
                                                                       Vector<double>(dofs_per_cell));

      // Assemble local P^T actions. If BDF1 supplies no n-2 vector, fold its
      // (normally zero) weight into n-1; this keeps the routine valid for the
      // common two-level call shape without allocating a dummy global vector.
      for (unsigned int q = 0; q < n_points; ++q)
        {
          AssertDimension(bar_in.composition[q].size(), n_compositional_fields);

          if (bar_in.temperature[q] != 0.0)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              if (this->get_fe().system_to_component_index(i).first
                  == this->introspection().component_indices.temperature)
                {
                  const double contribution =
                    bar_in.temperature[q]
                    * fe_values[this->introspection().extractors.temperature].value(i, q);
                  local_previous_temperature(i) += previous_weight * contribution;
                  if (previous_previous_temperature_adjoint != nullptr)
                    local_previous_previous_temperature(i) += previous_previous_weight * contribution;
                  else
                    local_previous_temperature(i) += previous_previous_weight * contribution;
                }

          for (unsigned int c = 0; c < n_compositional_fields; ++c)
            if (bar_in.composition[q][c] != 0.0)
              {
                if (previous_composition_adjoints == nullptr)
                  continue;

                AssertDimension(previous_composition_adjoints->size(), n_compositional_fields);
                if (previous_previous_composition_adjoints != nullptr)
                  AssertDimension(previous_previous_composition_adjoints->size(),
                                  n_compositional_fields);

                const AdvectionField composition_field = AdvectionField::composition(c);
                const FEValuesExtractors::Scalar composition_extractor =
                  composition_field.scalar_extractor(this->introspection());
                const unsigned int composition_component =
                  composition_field.component_index(this->introspection());

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  if (this->get_fe().system_to_component_index(i).first
                      == composition_component)
                    {
                      const double contribution =
                        bar_in.composition[q][c]
                        * fe_values[composition_extractor].value(i, q);
                      local_previous_composition[c](i) += previous_weight * contribution;
                      if (previous_previous_composition_adjoints != nullptr)
                        local_previous_previous_composition[c](i) +=
                          previous_previous_weight * contribution;
                      else
                        local_previous_composition[c](i) +=
                          previous_previous_weight * contribution;
                    }
              }
        }

      this->get_current_constraints().distribute_local_to_global(
        local_previous_temperature,
        local_dof_indices,
        previous_temperature_adjoint);

      // Constraint distribution is part of the discrete transpose: local
      // constrained DoF contributions are mapped to their masters before the
      // caller compresses the distributed accumulation vectors.
      if (previous_previous_temperature_adjoint != nullptr)
        this->get_current_constraints().distribute_local_to_global(
          local_previous_previous_temperature,
          local_dof_indices,
          *previous_previous_temperature_adjoint);

      if (previous_composition_adjoints != nullptr)
        for (unsigned int c = 0; c < n_compositional_fields; ++c)
          {
            this->get_current_constraints().distribute_local_to_global(
              local_previous_composition[c],
              local_dof_indices,
              (*previous_composition_adjoints)[c]);

            if (previous_previous_composition_adjoints != nullptr)
              this->get_current_constraints().distribute_local_to_global(
                local_previous_previous_composition[c],
                local_dof_indices,
                (*previous_previous_composition_adjoints)[c]);
          }
    }
  }
}

#define INSTANTIATE(dim) \
  template struct aspect::Adjoint::MaterialModelOutputAdjoints<dim>; \
  template struct aspect::Adjoint::MaterialModelInputAdjoints<dim>; \
  template class aspect::Adjoint::MaterialModelAdjoint<dim>; \
  template class aspect::Adjoint::MaterialModelInputAdjointScatter<dim>;

ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
