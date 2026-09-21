/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.
*/

#include <aspect/adjoint/observation/volume_field_observation.h>

#include <aspect/geometry_model/interface.h>

#include <deal.II/numerics/vector_tools.h>

namespace aspect
{
  namespace Adjoint
  {
    namespace Observation
    {
      template <int dim>
      VolumeFieldObservation<dim>::VolumeFieldObservation()
        : source(Source::function),
          target_function(1),
          coordinate_system(Utilities::Coordinates::cartesian)
      {}

      template <int dim>
      void VolumeFieldObservation<dim>::declare_parameters(ParameterHandler &prm)
      {
        prm.declare_entry("Observation source", "function",
                          Patterns::Selection("function|file"),
                          "Reference-data source for this terminal scalar-field "
                          "misfit. A file is read once and interpolated into the "
                          "current ASPECT finite-element space.");

        prm.declare_entry("Coordinate system", "cartesian",
                          Patterns::Selection("cartesian|spherical|depth"),
                          "Coordinate system used to evaluate the terminal target "
                          "and weight functions.");

        prm.enter_subsection("Target function");
        {
          Functions::ParsedFunction<dim>::declare_parameters(prm, 1);
          prm.declare_entry("Function expression", "0", Patterns::Anything(),
                            "Target final scalar field.");
        }
        prm.leave_subsection();

        Utilities::AsciiDataInitial<dim>::declare_parameters(
          prm, ".", "observation.txt", "Observation file");
      }

      template <int dim>
      void VolumeFieldObservation<dim>::parse_parameters(ParameterHandler &prm)
      {
        const std::string source_name = prm.get("Observation source");
        source = source_name == "function" ? Source::function : Source::structured_file;
        coordinate_system = Utilities::Coordinates::string_to_coordinate_system(
                              prm.get("Coordinate system"));

        prm.enter_subsection("Target function");
        {
          try
            {
              target_function.parse_parameters(prm);
            }
          catch (...)
            {
              std::cerr << "ERROR: FunctionParser failed to parse terminal target function\n"
                        << "with expression\n\t'" << prm.get("Function expression") << "'\n";
              throw;
            }
        }
        prm.leave_subsection();

        if (source == Source::structured_file)
          {
            structured_data = std::make_unique<Utilities::AsciiDataInitial<dim>>();
            structured_data->initialize_simulator(this->get_simulator());
            structured_data->parse_parameters(prm, "Observation file");
          }
      }

      template <int dim>
      void VolumeFieldObservation<dim>::initialize(const AdvectionField &field)
      {
        if (file_observation != nullptr || in_memory_observation != nullptr)
          return;
        if (source != Source::structured_file)
          return;

        AssertThrow(field.is_discontinuous(this->introspection()) == false,
                    ExcMessage("Structured terminal observations currently require "
                               "a continuous scalar finite element."));
        AssertThrow(structured_data != nullptr, ExcInternalError());
        structured_data->initialize(1);

        LinearAlgebra::BlockVector materialized(
          this->introspection().index_sets.system_partitioning,
          this->get_mpi_communicator());
        materialized = 0.0;

        const unsigned int component = field.component_index(this->introspection());
        VectorFunctionFromScalarFunctionObject<dim> source_function(
          [this](const Point<dim> &position) -> double
        {
          return structured_data->get_data_component(position, 0);
        }, component, this->introspection().n_components);

        ComponentMask component_mask(this->introspection().n_components, false);
        component_mask.set(component, true);
        VectorTools::interpolate(this->get_mapping(), this->get_dof_handler(),
                                 source_function, materialized, component_mask);
        materialized.compress(VectorOperation::insert);
        this->get_current_constraints().distribute(materialized);
        file_observation = std::make_unique<LinearAlgebra::BlockVector>();
        file_observation->reinit(
          this->introspection().index_sets.system_partitioning,
          this->introspection().index_sets.system_relevant_partitioning,
          this->get_mpi_communicator());
        *file_observation = materialized;
        for (unsigned int block = 0; block < file_observation->n_blocks(); ++block)
          file_observation->block(block).update_ghost_values();
        structured_data.reset();
      }

      template <int dim>
      double VolumeFieldObservation<dim>::value(const Point<dim> &position) const
      {
        AssertThrow(source == Source::function,
                    ExcMessage("A materialized discrete observation must be evaluated "
                               "through FEValues, not pointwise."));
        const Utilities::NaturalCoordinate<dim> point =
          this->get_geometry_model().cartesian_to_other_coordinates(position,
                                                                    coordinate_system);
        return target_function.value(
                 Utilities::convert_array_to_point<dim>(point.get_coordinates()));
      }

      template <int dim>
      bool VolumeFieldObservation<dim>::is_discrete() const
      {
        return in_memory_observation != nullptr || file_observation != nullptr;
      }

      template <int dim>
      const LinearAlgebra::BlockVector &
      VolumeFieldObservation<dim>::discrete_state() const
      {
        AssertThrow(is_discrete(),
                    ExcMessage("No discrete terminal observation has been installed."));
        const LinearAlgebra::BlockVector &state =
          in_memory_observation != nullptr ? *in_memory_observation : *file_observation;
        assert_current_layout(state);
        return state;
      }

      template <int dim>
      void VolumeFieldObservation<dim>::assert_current_layout(
        const LinearAlgebra::BlockVector &state) const
      {
        const auto &partitioning = this->introspection().index_sets.system_partitioning;
        AssertThrow(state.n_blocks() == partitioning.size(),
                    ExcMessage("The terminal observation is incompatible with the current "
                               "DoF layout. Mesh changes after observation capture are not "
                               "supported; recapture or reload the observation."));
        for (unsigned int block = 0; block < state.n_blocks(); ++block)
          AssertThrow(state.block(block).locally_owned_elements() == partitioning[block],
                      ExcMessage("The terminal observation is incompatible with the current "
                                 "DoF layout. Mesh changes after observation capture are not "
                                 "supported; recapture or reload the observation."));
      }

      template <int dim>
      void VolumeFieldObservation<dim>::set_discrete_state(
        const LinearAlgebra::BlockVector &state)
      {
        assert_current_layout(state);
        in_memory_observation = std::make_unique<LinearAlgebra::BlockVector>();
        in_memory_observation->reinit(
          this->introspection().index_sets.system_partitioning,
          this->introspection().index_sets.system_relevant_partitioning,
          this->get_mpi_communicator());
        *in_memory_observation = state;
        for (unsigned int block = 0; block < in_memory_observation->n_blocks(); ++block)
          in_memory_observation->block(block).update_ghost_values();
      }

      template <int dim>
      void VolumeFieldObservation<dim>::clear_discrete_state()
      {
        in_memory_observation.reset();
      }
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::Observation::VolumeFieldObservation<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
