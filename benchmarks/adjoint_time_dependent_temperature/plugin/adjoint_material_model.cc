/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/material_model/interface.h>

#include <cmath>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    class AdjointMaterialModel : public Interface<dim>
    {
      public:
        void
        evaluate(const MaterialModelInputs<dim> &in,
                 MaterialModelOutputs<dim> &out) const override;

        bool
        is_compressible() const override;

        static
        void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm) override;

      private:
        double reference_density;
        double reference_specific_heat;
        double reference_temperature;
        double thermal_expansion_coefficient;
        double specific_heat_temperature_coefficient;
        double density_differential;
        double viscosity;
        double viscosity_temperature_coefficient;
        double viscosity_composition_coefficient;
        double thermal_conductivity;
    };



    template <int dim>
    class AdjointPressureMaterialModel : public Interface<dim>
    {
      public:
        void
        evaluate(const MaterialModelInputs<dim> &in,
                 MaterialModelOutputs<dim> &out) const override;

        bool
        is_compressible() const override;

        static
        void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm) override;

      private:
        double reference_density;
        double reference_specific_heat;
        double pressure_density_coefficient;
        double viscosity;
        double thermal_conductivity;
    };



    template <int dim>
    void
    AdjointMaterialModel<dim>::evaluate(const MaterialModelInputs<dim> &in,
                                        MaterialModelOutputs<dim> &out) const
    {
      for (unsigned int q = 0; q < in.n_evaluation_points(); ++q)
        {
          const double composition =
            in.composition[q].empty() ? 0.0 : in.composition[q][0];
          const double delta_temperature =
            in.temperature[q] - reference_temperature;

          out.viscosities[q] =
            viscosity
            * std::exp(viscosity_temperature_coefficient * delta_temperature
                       + viscosity_composition_coefficient * composition);
          out.densities[q] =
            reference_density
            * (1.0 - thermal_expansion_coefficient * delta_temperature)
            + density_differential * composition;
          out.thermal_expansion_coefficients[q] = thermal_expansion_coefficient;
          out.specific_heat[q] =
            reference_specific_heat
            * (1.0 + specific_heat_temperature_coefficient * delta_temperature);
          out.thermal_conductivities[q] = thermal_conductivity;
          out.compressibilities[q] = 0.0;
          out.entropy_derivative_pressure[q] = 0.0;
          out.entropy_derivative_temperature[q] = 0.0;

          for (unsigned int c = 0; c < in.composition[q].size(); ++c)
            out.reaction_terms[q][c] = 0.0;
        }
    }



    template <int dim>
    bool
    AdjointMaterialModel<dim>::is_compressible() const
    {
      return false;
    }



    template <int dim>
    void
    AdjointMaterialModel<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Adjoint material model");
        {
          prm.declare_entry("Reference density", "1.0",
                            Patterns::Double(0.0),
                            "Reference density.");
          prm.declare_entry("Reference specific heat", "1.0",
                            Patterns::Double(0.0),
                            "Reference specific heat.");
          prm.declare_entry("Reference temperature", "0.0",
                            Patterns::Double(),
                            "Reference temperature.");
          prm.declare_entry("Thermal expansion coefficient", "1e-3",
                            Patterns::Double(0.0),
                            "Density temperature coefficient.");
          prm.declare_entry("Specific heat temperature coefficient", "2e-3",
                            Patterns::Double(),
                            "Linear temperature coefficient for specific heat.");
          prm.declare_entry("Density differential for compositional field 1", "0.2",
                            Patterns::Double(),
                            "Additive density contrast for the first compositional field.");
          prm.declare_entry("Viscosity", "1.0",
                            Patterns::Double(0.0),
                            "Reference viscosity.");
          prm.declare_entry("Viscosity temperature coefficient", "0.0",
                            Patterns::Double(),
                            "Exponential temperature coefficient for viscosity.");
          prm.declare_entry("Viscosity composition coefficient", "0.0",
                            Patterns::Double(),
                            "Exponential coefficient for the first compositional field in viscosity.");
          prm.declare_entry("Thermal conductivity", "1e-6",
                            Patterns::Double(0.0),
                            "Constant thermal conductivity.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    AdjointMaterialModel<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Adjoint material model");
        {
          reference_density =
            prm.get_double("Reference density");
          reference_specific_heat =
            prm.get_double("Reference specific heat");
          reference_temperature =
            prm.get_double("Reference temperature");
          thermal_expansion_coefficient =
            prm.get_double("Thermal expansion coefficient");
          specific_heat_temperature_coefficient =
            prm.get_double("Specific heat temperature coefficient");
          density_differential =
            prm.get_double("Density differential for compositional field 1");
          viscosity =
            prm.get_double("Viscosity");
          viscosity_temperature_coefficient =
            prm.get_double("Viscosity temperature coefficient");
          viscosity_composition_coefficient =
            prm.get_double("Viscosity composition coefficient");
          thermal_conductivity =
            prm.get_double("Thermal conductivity");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      this->model_dependence.viscosity = NonlinearDependence::none;
      if (viscosity_temperature_coefficient != 0.0)
        this->model_dependence.viscosity =
          this->model_dependence.viscosity | NonlinearDependence::temperature;
      if (viscosity_composition_coefficient != 0.0)
        this->model_dependence.viscosity =
          this->model_dependence.viscosity | NonlinearDependence::compositional_fields;
      this->model_dependence.density =
        NonlinearDependence::temperature | NonlinearDependence::compositional_fields;
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::temperature;
      this->model_dependence.thermal_conductivity = NonlinearDependence::none;
    }



    template <int dim>
    void
    AdjointPressureMaterialModel<dim>::evaluate(const MaterialModelInputs<dim> &in,
                                                MaterialModelOutputs<dim> &out) const
    {
      for (unsigned int q = 0; q < in.n_evaluation_points(); ++q)
        {
          out.viscosities[q] = viscosity;
          out.densities[q] =
            reference_density * (1.0 + pressure_density_coefficient * in.pressure[q]);
          out.thermal_expansion_coefficients[q] = 0.0;
          out.specific_heat[q] = reference_specific_heat;
          out.thermal_conductivities[q] = thermal_conductivity;
          out.compressibilities[q] = pressure_density_coefficient;
          out.entropy_derivative_pressure[q] = 0.0;
          out.entropy_derivative_temperature[q] = 0.0;

          for (unsigned int c = 0; c < in.composition[q].size(); ++c)
            out.reaction_terms[q][c] = 0.0;
        }
    }



    template <int dim>
    bool
    AdjointPressureMaterialModel<dim>::is_compressible() const
    {
      return false;
    }



    template <int dim>
    void
    AdjointPressureMaterialModel<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Adjoint pressure material model");
        {
          prm.declare_entry("Reference density", "1.0",
                            Patterns::Double(0.0),
                            "Reference density.");
          prm.declare_entry("Reference specific heat", "1.0",
                            Patterns::Double(0.0),
                            "Reference specific heat.");
          prm.declare_entry("Pressure density coefficient", "1e-6",
                            Patterns::Double(),
                            "Linear pressure coefficient for density.");
          prm.declare_entry("Viscosity", "1.0",
                            Patterns::Double(0.0),
                            "Constant viscosity.");
          prm.declare_entry("Thermal conductivity", "1e-6",
                            Patterns::Double(0.0),
                            "Constant thermal conductivity.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    AdjointPressureMaterialModel<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Adjoint pressure material model");
        {
          reference_density = prm.get_double("Reference density");
          reference_specific_heat = prm.get_double("Reference specific heat");
          pressure_density_coefficient = prm.get_double("Pressure density coefficient");
          viscosity = prm.get_double("Viscosity");
          thermal_conductivity = prm.get_double("Thermal conductivity");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      this->model_dependence.viscosity = NonlinearDependence::none;
      this->model_dependence.density = NonlinearDependence::pressure;
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::none;
      this->model_dependence.thermal_conductivity = NonlinearDependence::none;
    }
  }
}

namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(AdjointMaterialModel,
                                   "adjoint material model",
                                   "A small benchmark-only material model for "
                                   "validating the v1 material-property adjoint. "
                                   "It has smooth density dependence on temperature "
                                   "and the first composition, smooth specific-heat "
                                   "dependence on temperature, and constant viscosity "
                                   "and thermal conductivity.")

    ASPECT_REGISTER_MATERIAL_MODEL(AdjointPressureMaterialModel,
                                   "adjoint pressure material model",
                                   "A small benchmark-only material model that "
                                   "declares pressure-dependent density. It is "
                                   "used to verify that the v1 material-input "
                                   "adjoint scatter fails explicitly instead of "
                                   "dropping pressure adjoints.")
  }
}
