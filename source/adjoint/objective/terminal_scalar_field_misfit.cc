/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.
*/

#include <aspect/adjoint/objective/terminal_scalar_field_misfit.h>

#include <aspect/geometry_model/interface.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    TerminalScalarFieldMisfit<dim>::TerminalScalarFieldMisfit()
      : weight_function(1),
        coordinate_system(Utilities::Coordinates::cartesian)
    {}

    template <int dim>
    void TerminalScalarFieldMisfit<dim>::initialize_simulator(
      const Simulator<dim> &simulator)
    {
      Objective::Interface<dim>::initialize_simulator(simulator);
      observation.initialize_simulator(simulator);
    }

    template <int dim>
    void TerminalScalarFieldMisfit<dim>::initialize()
    {}

    template <int dim>
    void TerminalScalarFieldMisfit<dim>::declare_common_parameters(ParameterHandler &prm)
    {
      Observation::VolumeFieldObservation<dim>::declare_parameters(prm);
      prm.enter_subsection("Weight function");
      {
        Functions::ParsedFunction<dim>::declare_parameters(prm, 1);
        prm.declare_entry("Function expression", "1", Patterns::Anything(),
                          "Non-negative spatial weight for the terminal scalar-field misfit.");
      }
      prm.leave_subsection();
    }

    template <int dim>
    void TerminalScalarFieldMisfit<dim>::parse_common_parameters(ParameterHandler &prm)
    {
      observation.parse_parameters(prm);
      coordinate_system = Utilities::Coordinates::string_to_coordinate_system(
                            prm.get("Coordinate system"));
      prm.enter_subsection("Weight function");
      {
        try
          {
            weight_function.parse_parameters(prm);
          }
        catch (...)
          {
            std::cerr << "ERROR: FunctionParser failed to parse terminal weight function\n"
                      << "with expression\n\t'" << prm.get("Function expression") << "'\n";
            throw;
          }
      }
      prm.leave_subsection();
    }

    template <int dim>
    double TerminalScalarFieldMisfit<dim>::weight(const Point<dim> &position) const
    {
      const Utilities::NaturalCoordinate<dim> point =
        this->get_geometry_model().cartesian_to_other_coordinates(position,
                                                                   coordinate_system);
      return weight_function.value(
               Utilities::convert_array_to_point<dim>(point.get_coordinates()));
    }

    template <int dim>
    double TerminalScalarFieldMisfit<dim>::evaluate() const
    {
      const AdvectionField field = observed_field();
      observation.initialize(field);
      const QGauss<dim> quadrature(field.polynomial_degree(this->introspection()) + 1);
      FEValues<dim> fe_values(this->get_mapping(), this->get_fe(), quadrature,
                              update_values | update_quadrature_points | update_JxW_values);
      const FEValuesExtractors::Scalar extractor = field.scalar_extractor(this->introspection());
      std::vector<double> values(quadrature.size());
      std::vector<double> observed_values(quadrature.size());
      double local_value = 0.0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[extractor].get_function_values(this->get_solution(), values);
            if (observation.is_discrete())
              fe_values[extractor].get_function_values(observation.discrete_state(), observed_values);
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                const double observed = observation.is_discrete()
                                        ? observed_values[q]
                                        : observation.value(fe_values.quadrature_point(q));
                const double residual = values[q] - observed;
                local_value += 0.5 * weight(fe_values.quadrature_point(q))
                               * residual * residual * fe_values.JxW(q);
              }
          }
      return Utilities::MPI::sum(local_value, this->get_mpi_communicator());
    }

    template <int dim>
    void TerminalScalarFieldMisfit<dim>::assemble_terminal_advection_field_right_hand_side(
      const AdvectionField &field, LinearAlgebra::BlockVector &rhs) const
    {
      const AdvectionField selected_field = observed_field();
      if (field.field_index() != selected_field.field_index())
        return;

      observation.initialize(field);

      const QGauss<dim> quadrature(field.polynomial_degree(this->introspection()) + 1);
      FEValues<dim> fe_values(this->get_mapping(), this->get_fe(), quadrature,
                              update_values | update_quadrature_points | update_JxW_values);
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      const unsigned int component = field.component_index(this->introspection());
      const FEValuesExtractors::Scalar extractor = field.scalar_extractor(this->introspection());
      Vector<double> local_rhs(dofs_per_cell);
      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
      std::vector<double> values(quadrature.size());
      std::vector<double> observed_values(quadrature.size());

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[extractor].get_function_values(this->get_solution(), values);
            if (observation.is_discrete())
              fe_values[extractor].get_function_values(observation.discrete_state(), observed_values);
            local_rhs = 0.0;
            cell->get_dof_indices(dof_indices);
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                const double observed = observation.is_discrete()
                                        ? observed_values[q]
                                        : observation.value(fe_values.quadrature_point(q));
                const double seed = weight(fe_values.quadrature_point(q)) * (values[q] - observed);
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  if (this->get_fe().system_to_component_index(i).first == component)
                    local_rhs(i) += seed * fe_values[extractor].value(i, q) * fe_values.JxW(q);
              }
            this->get_current_constraints().distribute_local_to_global(local_rhs, dof_indices, rhs);
          }
    }

    template <int dim>
    bool TerminalScalarFieldMisfit<dim>::observes_field(const AdvectionField &field) const
    {
      return field.field_index() == observed_field().field_index();
    }

    template <int dim>
    void TerminalScalarFieldMisfit<dim>::set_discrete_terminal_observation(
      const LinearAlgebra::BlockVector &state)
    {
      observation.set_discrete_state(state);
    }

    template <int dim>
    void TerminalScalarFieldMisfit<dim>::clear_discrete_terminal_observation()
    {
      observation.clear_discrete_state();
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::TerminalScalarFieldMisfit<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
