/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include <aspect/postprocess/forward_boundary_reduced_stress.h>

#include <aspect/postprocess/boundary_pressures.h>
#include <aspect/postprocess/dynamic_topography.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/material_model/interface.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <limits>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    BoundaryReducedStressHarmonics
    ForwardBoundaryReducedStressProvider<dim>::compute(
      const SimulatorAccess<dim> &access,
      const unsigned int minimum_degree,
      const unsigned int maximum_degree,
      const double outer_radius,
      const double inner_radius)
    {
      AssertThrow(dim == 3 && minimum_degree >= 2 && minimum_degree <= maximum_degree,
                  ExcMessage("Forward boundary reduced-stress harmonics require 3D and l_min >= 2."));
      AssertThrow(outer_radius > std::numeric_limits<double>::min()
                  && inner_radius > std::numeric_limits<double>::min(),
                  ExcMessage("Forward boundary reduced-stress harmonics require positive radii."));

      const DynamicTopography<dim> &dynamic_topography =
        access.get_postprocess_manager().template get_matching_active_plugin<DynamicTopography<dim>>();
      const BoundaryPressures<dim> &boundary_pressures =
        access.get_postprocess_manager().template get_matching_active_plugin<BoundaryPressures<dim>>();
      const LinearAlgebra::BlockVector &cbf_traction = dynamic_topography.topography_vector();
      AssertThrow(cbf_traction.size() != 0,
                  ExcMessage("The dynamic topography CBF traction is unavailable. "
                             "Run the dynamic topography postprocessor before this provider."));

      const types::boundary_id top_boundary_id =
        access.get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      const types::boundary_id bottom_boundary_id =
        access.get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");
      // This matches the Gauss face evaluation used by the historical
      // self-gravity branch. DynamicTopography itself uses GLL only to
      // diagonalize its CBF mass matrix.
      const unsigned int quadrature_degree =
        access.get_fe().base_element(access.introspection().base_elements.velocities).degree + 1;
      const QGauss<dim-1> quadrature(quadrature_degree);
      FEFaceValues<dim> face_values(access.get_mapping(), access.get_fe(), quadrature,
                                    update_values | update_normal_vectors |
                                    update_quadrature_points | update_JxW_values);

      BoundaryReducedStressHarmonics result;
      for (SphericalHarmonicCoefficients *field :
           {
             &result.surface, &result.cmb
           })
        field->reinit(minimum_degree, maximum_degree);
      std::vector<Tensor<1, dim>> traction_values(quadrature.size());

      // Project the reduced normal stress
      //
      //   tau = -t_CBF.n-p_mean
      //
      // into tau_lm=int_Gamma tau Y_lm dA/r_Gamma^2. The CBF vector stores a
      // traction representation in its velocity components; subtracting the
      // boundary mean pressure removes the pressure-gauge contribution used
      // by dynamic topography. Each real harmonic is accumulated separately
      // to preserve the established projection order.
      for (unsigned int degree = minimum_degree; degree <= maximum_degree; ++degree)
        for (unsigned int order = 0; order <= degree; ++order)
          {
            double top_cosine = 0.0, top_sine = 0.0;
            double bottom_cosine = 0.0, bottom_sine = 0.0;
            for (const auto &cell : access.get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned() && cell->at_boundary())
                for (const unsigned int face : cell->face_indices())
                  if (cell->at_boundary(face))
                    {
                      const types::boundary_id boundary_id = cell->face(face)->boundary_id();
                      if (boundary_id != top_boundary_id && boundary_id != bottom_boundary_id)
                        continue;
                      face_values.reinit(cell, face);
                      face_values[access.introspection().extractors.velocities].get_function_values(
                        cbf_traction, traction_values);
                      const double mean_pressure = boundary_id == top_boundary_id ?
                                                   boundary_pressures.pressure_at_top() :
                                                   boundary_pressures.pressure_at_bottom();
                      const double radius = boundary_id == top_boundary_id ? outer_radius : inner_radius;
                      for (unsigned int q = 0; q < quadrature.size(); ++q)
                        {
                          const std::array<double, dim> spherical_position =
                            Utilities::Coordinates::cartesian_to_spherical_coordinates(
                              face_values.quadrature_point(q));
                          const std::pair<double, double> harmonic =
                            Utilities::real_spherical_harmonic(degree, order,
                                                               spherical_position[2], spherical_position[1]);
                          // Use the outward FE normal at both boundaries.  The
                          // CMB sign is therefore inherited from the same CBF
                          // traction convention as DynamicTopography.
                          const double tau = -(traction_values[q] * face_values.normal_vector(q))
                                             - mean_pressure;
                          const double weight = face_values.JxW(q) / (radius * radius);
                          if (boundary_id == top_boundary_id)
                            {
                              top_cosine += tau * harmonic.first * weight;
                              top_sine += tau * harmonic.second * weight;
                            }
                          else
                            {
                              bottom_cosine += tau * harmonic.first * weight;
                              bottom_sine += tau * harmonic.second * weight;
                            }
                        }
                    }
            const unsigned int index = result.surface.coefficient_index(degree, order);
            result.surface.cosine(index) = top_cosine;
            result.surface.sine(index) = top_sine;
            result.cmb.cosine(index) = bottom_cosine;
            result.cmb.sine(index) = bottom_sine;
          }

      // The face loop is rank-local. Assemble the four coefficient arrays by
      // MPI sum so all ranks hold identical spectral operator inputs.
      std::vector<double> surface_cosine(result.surface.size());
      std::vector<double> surface_sine(result.surface.size());
      std::vector<double> cmb_cosine(result.cmb.size());
      std::vector<double> cmb_sine(result.cmb.size());
      Utilities::MPI::sum(result.surface.cosine_coefficients(),
                          access.get_mpi_communicator(), surface_cosine);
      Utilities::MPI::sum(result.surface.sine_coefficients(),
                          access.get_mpi_communicator(), surface_sine);
      Utilities::MPI::sum(result.cmb.cosine_coefficients(),
                          access.get_mpi_communicator(), cmb_cosine);
      Utilities::MPI::sum(result.cmb.sine_coefficients(),
                          access.get_mpi_communicator(), cmb_sine);
      result.surface.set_coefficients(surface_cosine, surface_sine);
      result.cmb.set_coefficients(cmb_cosine, cmb_sine);
      return result;
    }



    template <int dim>
    void
    assemble_boundary_reduced_stress_transpose(
      const SimulatorAccess<dim> &access,
      const SphericalHarmonicCoefficients &coefficient_seed,
      const DynamicTopographyBoundary boundary,
      const double boundary_radius,
      const double density_exterior,
      LinearAlgebra::BlockVector &topography_adjoint,
      double &boundary_area,
      double &pressure_weight)
    {
      AssertThrow(dim == 3 && coefficient_seed.minimum_degree() >= 2,
                  ExcMessage("Reduced-stress transpose requires 3D l >= 2 coefficients."));
      AssertThrow(boundary_radius > std::numeric_limits<double>::min(),
                  ExcMessage("Reduced-stress transpose requires a positive boundary radius."));
      const types::boundary_id boundary_id = access.get_geometry_model().
                                             translate_symbolic_boundary_name_to_id(boundary == DynamicTopographyBoundary::surface
                                                                                    ? "top" : "bottom");
      const unsigned int quadrature_degree =
        access.introspection().polynomial_degree.temperature;
      const QGauss<dim-1> quadrature(quadrature_degree);
      FEFaceValues<dim> face_values(access.get_mapping(), access.get_fe(), quadrature,
                                    update_values | update_gradients | update_quadrature_points | update_JxW_values);
      const unsigned int dofs_per_cell = access.get_fe().dofs_per_cell;
      Vector<double> local_topography_adjoint(dofs_per_cell);
      topography_adjoint = 0.0;
      boundary_area = 0.0;
      pressure_weight = 0.0;

      // Reverse the chain
      //
      //   tau_lm=int tau Y_lm dOmega,
      //   tau=Delta_rho |g| h,
      //
      // where h is the CBF topography field. Evaluating the coefficient seed
      // reconstructs bar(tau)(theta,phi); multiplication by
      // Delta_rho|g|/r^2 converts it to the physical-face dual seed bar(h).
      // pressure_weight separately tracks the transpose of -p_mean.
      for (const auto &cell : access.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int face : cell->face_indices())
            if (cell->at_boundary(face) && cell->face(face)->boundary_id() == boundary_id)
              {
                face_values.reinit(cell, face);
                local_topography_adjoint = 0.0;
                MaterialModel::MaterialModelInputs<dim> material_input(face_values, cell,
                                                                       access.introspection(), access.get_solution());
                MaterialModel::MaterialModelOutputs<dim> material_output(
                  face_values.n_quadrature_points, access.n_compositional_fields());
                material_input.requested_properties = MaterialModel::MaterialProperties::density;
                access.get_material_model().evaluate(material_input, material_output);
                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  {
                    const std::array<double, dim> spherical_position =
                      Utilities::Coordinates::cartesian_to_spherical_coordinates(
                        face_values.quadrature_point(q));
                    const double traction_seed = coefficient_seed.evaluate(spherical_position[2],
                                                                           spherical_position[1]);
                    const double gravity_norm = access.get_gravity_model().gravity_vector(
                                                  face_values.quadrature_point(q)).norm();
                    const double density_contrast = material_output.densities[q] - density_exterior;
                    AssertThrow(std::abs(density_contrast) > std::numeric_limits<double>::min()
                                && gravity_norm > std::numeric_limits<double>::min(),
                                ExcMessage("Reduced-stress transpose encountered zero density contrast or gravity."));
                    // q_tau dOmega is represented as q_tau/r^2 dA.  Convert
                    // it to the h=tau/(Delta-rho g) seed expected by CBF.
                    const double topography_seed = traction_seed / (boundary_radius * boundary_radius)
                                                   * density_contrast * gravity_norm;
                    boundary_area += face_values.JxW(q);
                    pressure_weight += traction_seed / (boundary_radius * boundary_radius)
                                       * face_values.JxW(q);
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      local_topography_adjoint(i) += topography_seed
                                                     * face_values[access.introspection().extractors.temperature].value(i, q)
                                                     * face_values.JxW(q);
                  }
                // The local vector is an FE covector. Add shared boundary DoF
                // contributions and complete them with compress(add) below.
                cell->distribute_local_to_global(local_topography_adjoint, topography_adjoint);
              }
      topography_adjoint.compress(VectorOperation::add);
      boundary_area = Utilities::MPI::sum(boundary_area, access.get_mpi_communicator());
      pressure_weight = Utilities::MPI::sum(pressure_weight, access.get_mpi_communicator());
      AssertThrow(boundary_area > std::numeric_limits<double>::min(),
                  ExcMessage("Reduced-stress transpose found zero selected-boundary area."));
    }

    template class ForwardBoundaryReducedStressProvider<2>;
    template class ForwardBoundaryReducedStressProvider<3>;
    template void assemble_boundary_reduced_stress_transpose<2>(const SimulatorAccess<2> &,
                                                                const SphericalHarmonicCoefficients &,
                                                                const DynamicTopographyBoundary,
                                                                const double, const double,
                                                                LinearAlgebra::BlockVector &, double &, double &);
    template void assemble_boundary_reduced_stress_transpose<3>(const SimulatorAccess<3> &,
                                                                const SphericalHarmonicCoefficients &,
                                                                const DynamicTopographyBoundary,
                                                                const double, const double,
                                                                LinearAlgebra::BlockVector &, double &, double &);
  }
}
