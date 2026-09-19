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

#include <aspect/postprocess/geoid_harmonics.h>

#include <aspect/geometry_model/interface.h>
#include <aspect/material_model/interface.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_values.h>

#include <cmath>
#include <limits>
#include <ostream>

namespace aspect
{
  namespace Postprocess
  {
    namespace
    {
      double
      density_geoid_radial_weight (const double radius,
                                   const double reference_radius,
                                   const unsigned int degree)
      {
        // A density anomaly at radius r contributes to the exterior degree-l
        // potential with r^l/R^(l+1). Writing this as
        // (1/r)(r/R)^(l+1) avoids separately forming large dimensional powers.
        return (1.0 / radius) * Utilities::pow(radius / reference_radius, degree + 1);
      }



      double
      density_geoid_degree_scaling (const unsigned int degree,
                                    const double surface_gravity)
      {
        // Poisson's equation maps a degree-l density moment to potential with
        // 4 pi G/(2l+1); division by surface gravity converts potential into
        // the geoid-height convention used by the objectives.
        return 4.0 * numbers::PI * constants::big_g
               / (surface_gravity * (2 * degree + 1));
      }
    }



    SphericalHarmonicCoefficients::SphericalHarmonicCoefficients()
      :
      SphericalHarmonicCoefficients(0, 0)
    {}



    SphericalHarmonicCoefficients::SphericalHarmonicCoefficients(
      const unsigned int minimum_degree,
      const unsigned int maximum_degree)
    {
      reinit(minimum_degree, maximum_degree);
    }



    unsigned int
    SphericalHarmonicCoefficients::coefficient_count(
      const unsigned int minimum_degree,
      const unsigned int maximum_degree)
    {
      AssertThrow(minimum_degree <= maximum_degree,
                  ExcMessage("The minimum harmonic degree must not exceed the maximum degree."));
      // Degrees are stored as contiguous triangular rows
      //   (l_min,0..l_min), ..., (l_max,0..l_max).
      // The difference of triangular numbers gives the exact allocation.
      return (maximum_degree * (maximum_degree + 1)
              - minimum_degree * (minimum_degree + 1)) / 2
             + maximum_degree + 1;
    }



    void
    SphericalHarmonicCoefficients::reinit(const unsigned int minimum_degree,
                                          const unsigned int maximum_degree)
    {
      stored_minimum_degree = minimum_degree;
      stored_maximum_degree = maximum_degree;
      const unsigned int count = coefficient_count(minimum_degree, maximum_degree);
      cosine_values.assign(count, 0.0);
      sine_values.assign(count, 0.0);
    }



    unsigned int
    SphericalHarmonicCoefficients::minimum_degree() const
    {
      return stored_minimum_degree;
    }



    unsigned int
    SphericalHarmonicCoefficients::maximum_degree() const
    {
      return stored_maximum_degree;
    }



    unsigned int
    SphericalHarmonicCoefficients::size() const
    {
      return cosine_values.size();
    }



    double &
    SphericalHarmonicCoefficients::cosine(const unsigned int index)
    {
      return cosine_values.at(index);
    }



    double
    SphericalHarmonicCoefficients::cosine(const unsigned int index) const
    {
      return cosine_values.at(index);
    }



    double &
    SphericalHarmonicCoefficients::sine(const unsigned int index)
    {
      return sine_values.at(index);
    }



    double
    SphericalHarmonicCoefficients::sine(const unsigned int index) const
    {
      return sine_values.at(index);
    }



    const std::vector<double> &
    SphericalHarmonicCoefficients::cosine_coefficients() const
    {
      return cosine_values;
    }



    const std::vector<double> &
    SphericalHarmonicCoefficients::sine_coefficients() const
    {
      return sine_values;
    }



    void
    SphericalHarmonicCoefficients::set_coefficients(
      const std::vector<double> &cosine,
      const std::vector<double> &sine)
    {
      AssertThrow(cosine.size() == size() && sine.size() == size(),
                  ExcMessage("Spherical-harmonic component vectors do not match the degree range."));
      cosine_values = cosine;
      sine_values = sine;
    }



    unsigned int
    SphericalHarmonicCoefficients::coefficient_index (const unsigned int degree,
                                                      const unsigned int order) const
    {
      AssertThrow(degree >= stored_minimum_degree && degree <= stored_maximum_degree,
                  ExcMessage("The requested harmonic degree is outside the stored range."));
      AssertThrow(order <= degree,
                  ExcMessage("The requested harmonic order exceeds its degree."));
      // Offset to the first entry of degree l, followed by order m. Cosine and
      // sine components share this mapping so their layouts cannot diverge.
      return (degree * (degree + 1)
              - stored_minimum_degree * (stored_minimum_degree + 1)) / 2 + order;
    }



    double
    SphericalHarmonicCoefficients::evaluate (const double theta,
                                             const double phi) const
    {
      // Reconstruct the real expansion
      //   f(theta,phi)=sum_lm [C_lm Y_lm^c + S_lm Y_lm^s].
      // Using ASPECT's common evaluator here and in every projection fixes the
      // normalization and phase convention for forward/transpose identities.
      double value = 0.0;
      for (unsigned int degree = stored_minimum_degree;
           degree <= stored_maximum_degree;
           ++degree)
        for (unsigned int order = 0; order <= degree; ++order)
          {
            const unsigned int index = coefficient_index(degree, order);
            const std::pair<double, double> harmonic =
              Utilities::real_spherical_harmonic(degree, order, theta, phi);
            value += cosine_values.at(index) * harmonic.first
                     + sine_values.at(index) * harmonic.second;
          }
      return value;
    }



    void
    write_spherical_harmonic_coefficients(
      std::ostream &output,
      const SphericalHarmonicCoefficients &coefficients)
    {
      for (unsigned int degree = coefficients.minimum_degree();
           degree <= coefficients.maximum_degree();
           ++degree)
        for (unsigned int order = 0; order <= degree; ++order)
          {
            const unsigned int index =
              coefficients.coefficient_index(degree, order);
            output << degree << ' '
                   << order << ' '
                   << coefficients.cosine(index) << ' '
                   << coefficients.sine(index) << std::endl;
          }
    }



    template <int dim>
    SphericalHarmonicCoefficients
    compute_density_geoid_harmonics (const SimulatorAccess<dim> &simulator_access,
                                     const unsigned int minimum_degree,
                                     const unsigned int maximum_degree,
                                     const double reference_radius,
                                     const double surface_gravity)
    {
      AssertThrow(dim == 3,
                  ExcMessage("Density geoid harmonics currently only work in 3D."));
      AssertThrow(minimum_degree <= maximum_degree,
                  ExcMessage("The minimum geoid degree must not exceed the maximum degree."));
      AssertThrow(reference_radius > std::numeric_limits<double>::min(),
                  ExcMessage("The geoid reference radius must be positive."));
      AssertThrow(surface_gravity > std::numeric_limits<double>::min(),
                  ExcMessage("The geoid surface gravity magnitude must be positive."));

      const unsigned int quadrature_degree =
        simulator_access.introspection().polynomial_degree.temperature;
      const QGauss<dim> quadrature_formula(quadrature_degree);
      FEValues<dim> fe_values(simulator_access.get_mapping(),
                              simulator_access.get_fe(),
                              quadrature_formula,
                              update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
      MaterialModel::MaterialModelInputs<dim> in(fe_values.n_quadrature_points,
                                                 simulator_access.n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points,
                                                   simulator_access.n_compositional_fields());
      in.requested_properties = MaterialModel::MaterialProperties::density;

      SphericalHarmonicCoefficients coefficients(minimum_degree, maximum_degree);

      // Project density into real geoid coefficients:
      //
      //   N_lm = Gamma_l integral_Omega rho(x) r^l/R^(l+1)
      //          Y_lm(theta,phi) dx,
      //   Gamma_l = 4 pi G/[g_s(2l+1)].
      //
      // Preserve the established degree/order, cell, and quadrature traversal
      // of Geoid::density_contribution. The material model is evaluated at
      // the accepted solution and each locally owned cell contributes once.
      for (unsigned int degree = minimum_degree; degree <= maximum_degree; ++degree)
        for (unsigned int order = 0; order <= degree; ++order)
          {
            double cosine_integral = 0.0;
            double sine_integral = 0.0;
            for (const auto &cell : simulator_access.get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  fe_values.reinit(cell);
                  in.reinit(fe_values, cell, simulator_access.introspection(),
                            simulator_access.get_solution());
                  simulator_access.get_material_model().evaluate(in, out);
                  for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
                    {
                      const std::array<double, dim> spherical_position =
                        Utilities::Coordinates::cartesian_to_spherical_coordinates(in.position[q]);
                      const std::pair<double, double> harmonic =
                        Utilities::real_spherical_harmonic(degree, order,
                                                           spherical_position[2], spherical_position[1]);
                      const double radius = in.position[q].norm();
                      const double radial_weight =
                        density_geoid_radial_weight(radius, reference_radius, degree);
                      cosine_integral += out.densities[q] * radial_weight * harmonic.first * fe_values.JxW(q);
                      sine_integral += out.densities[q] * radial_weight * harmonic.second * fe_values.JxW(q);
                    }
                }
            const unsigned int index = coefficients.coefficient_index(degree, order);
            coefficients.cosine(index) = cosine_integral;
            coefficients.sine(index) = sine_integral;
          }

      // These are rank-local volume integrals. MPI sum is the assembly
      // operation in coefficient space and leaves every rank with the same
      // complete coefficients for deterministic objective evaluation.
      std::vector<double> global_cosine(coefficients.size());
      std::vector<double> global_sine(coefficients.size());
      dealii::Utilities::MPI::sum(coefficients.cosine_coefficients(),
                                  simulator_access.get_mpi_communicator(),
                                  global_cosine);
      dealii::Utilities::MPI::sum(coefficients.sine_coefficients(),
                                  simulator_access.get_mpi_communicator(),
                                  global_sine);
      coefficients.set_coefficients(global_cosine, global_sine);

      // Apply the Poisson/geoid degree scaling after global assembly. Keeping
      // the cell loop unscaled makes its dual relation with the transpose
      // basis below explicit.
      unsigned int index = 0;
      for (unsigned int degree = minimum_degree; degree <= maximum_degree; ++degree)
        for (unsigned int order = 0; order <= degree; ++order, ++index)
          {
            const double scaling = density_geoid_degree_scaling(degree, surface_gravity);
            coefficients.cosine(index) *= scaling;
            coefficients.sine(index) *= scaling;
          }

      return coefficients;
    }



    template <int dim>
    Vector<double>
    compute_density_geoid_transpose (const SimulatorAccess<dim> &simulator_access,
                                     const SphericalHarmonicCoefficients &coefficient_seed,
                                     const double reference_radius,
                                     const double surface_gravity)
    {
      AssertThrow(dim == 3,
                  ExcMessage("Density geoid harmonics currently only work in 3D."));
      AssertThrow(reference_radius > std::numeric_limits<double>::min(),
                  ExcMessage("The geoid reference radius must be positive."));
      AssertThrow(surface_gravity > std::numeric_limits<double>::min(),
                  ExcMessage("The geoid surface gravity magnitude must be positive."));

      const unsigned int coefficient_count =
        (coefficient_seed.maximum_degree() * (coefficient_seed.maximum_degree() + 1)
         - coefficient_seed.minimum_degree() * (coefficient_seed.minimum_degree() + 1)) / 2
        + coefficient_seed.maximum_degree() + 1;
      AssertThrow(coefficient_seed.cosine_coefficients().size() == coefficient_count
                  && coefficient_seed.sine_coefficients().size() == coefficient_count,
                  ExcMessage("The density-geoid transpose seed has an invalid coefficient count."));

      const unsigned int quadrature_degree =
        simulator_access.introspection().polynomial_degree.temperature;
      const QGauss<dim> quadrature_formula(quadrature_degree);
      FEValues<dim> fe_values(simulator_access.get_mapping(),
                              simulator_access.get_fe(),
                              quadrature_formula,
                              update_quadrature_points | update_JxW_values);

      Vector<double> cellwise_kernel(simulator_access.get_triangulation().n_active_cells());

      // For coefficient seeds bar(C_lm),bar(S_lm), the density VJP is
      //
      //   bar(rho)(x)=sum_lm Gamma_l r^l/R^(l+1)
      //     [bar(C_lm)Y_lm^c+bar(S_lm)Y_lm^s].
      //
      // We return its volume average on each active cell because downstream
      // direct material kernels consume a piecewise-constant density seed.
      for (const auto &cell : simulator_access.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            double cell_volume = 0.0;
            double cell_integral = 0.0;
            for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
              {
                const Point<dim> &position = fe_values.quadrature_point(q);
                const std::array<double, dim> spherical_position =
                  Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
                const double radius = position.norm();
                double kernel_value = 0.0;
                for (unsigned int degree = coefficient_seed.minimum_degree();
                     degree <= coefficient_seed.maximum_degree();
                     ++degree)
                  for (unsigned int order = 0; order <= degree; ++order)
                    {
                      const unsigned int index = coefficient_seed.coefficient_index(degree, order);
                      const std::pair<double, double> harmonic =
                        Utilities::real_spherical_harmonic(degree, order,
                                                           spherical_position[2], spherical_position[1]);
                      const double basis_scale =
                        density_geoid_degree_scaling(degree, surface_gravity)
                        * density_geoid_radial_weight(radius, reference_radius, degree);
                      kernel_value += basis_scale * (coefficient_seed.cosine(index) * harmonic.first
                                                     + coefficient_seed.sine(index) * harmonic.second);
                    }
                cell_volume += fe_values.JxW(q);
                cell_integral += kernel_value * fe_values.JxW(q);
              }
            AssertThrow(cell_volume > std::numeric_limits<double>::min(),
                        ExcMessage("Density geoid transpose encountered a locally owned cell with zero volume."));
            // This division constructs the cellwise representative; it is not
            // an additional physical normalization of the geoid formula.
            cellwise_kernel(cell->active_cell_index()) = cell_integral / cell_volume;
          }

      return cellwise_kernel;
    }



    template <int dim>
    Vector<double>
    compute_self_gravitating_density_moment_transpose(
      const SimulatorAccess<dim> &simulator_access,
      const SphericalHarmonicCoefficients &coefficient_seed,
      const double outer_radius,
      const double inner_radius,
      const SelfGravitatingDensityMoment moment)
    {
      AssertThrow(dim == 3,
                  ExcMessage("Self-gravitating density moments currently only work in 3D."));
      AssertThrow(outer_radius > std::numeric_limits<double>::min()
                  && inner_radius > std::numeric_limits<double>::min()
                  && inner_radius < outer_radius,
                  ExcMessage("Self-gravitating density moments require positive ordered radii."));
      const unsigned int coefficient_count =
        (coefficient_seed.maximum_degree() * (coefficient_seed.maximum_degree() + 1)
         - coefficient_seed.minimum_degree() * (coefficient_seed.minimum_degree() + 1)) / 2
        + coefficient_seed.maximum_degree() + 1;
      AssertThrow(coefficient_seed.cosine_coefficients().size() == coefficient_count
                  && coefficient_seed.sine_coefficients().size() == coefficient_count,
                  ExcMessage("The self-gravitating density-moment transpose seed has an invalid coefficient count."));

      // Keep this quadrature and all harmonic conventions identical to the
      // forward moment projection in self_gravitating_geoid_operator.cc. The
      // two density moments are
      //
      //   I_lm^s = integral rho r^l/R_s^(l+1) Y_lm dx,
      //   I_lm^c = integral rho R_c^l/r^(l+1) Y_lm dx.
      //
      // Consequently their transposes differ only by the radial basis chosen
      // below; cosine and sine use the same basis.
      const QGauss<dim> quadrature(simulator_access.introspection().polynomial_degree.temperature);
      FEValues<dim> values(simulator_access.get_mapping(), simulator_access.get_fe(), quadrature,
                           update_quadrature_points | update_JxW_values);
      Vector<double> cellwise_kernel(simulator_access.get_triangulation().n_active_cells());
      for (const auto &cell : simulator_access.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            values.reinit(cell);
            double cell_volume = 0.0;
            double cell_integral = 0.0;
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                const Point<dim> &position = values.quadrature_point(q);
                const std::array<double, dim> spherical_position =
                  Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
                const double radius = position.norm();
                double kernel_value = 0.0;
                for (unsigned int degree = coefficient_seed.minimum_degree();
                     degree <= coefficient_seed.maximum_degree(); ++degree)
                  {
                    const double radial_weight =
                      moment == SelfGravitatingDensityMoment::surface
                      ? std::pow(radius, degree) / std::pow(outer_radius, degree + 1)
                      : std::pow(inner_radius, degree) / std::pow(radius, degree + 1);
                    for (unsigned int order = 0; order <= degree; ++order)
                      {
                        const unsigned int index = coefficient_seed.coefficient_index(degree, order);
                        const std::pair<double, double> harmonic =
                          Utilities::real_spherical_harmonic(degree, order,
                                                             spherical_position[2], spherical_position[1]);
                        kernel_value += radial_weight *
                                        (coefficient_seed.cosine(index) * harmonic.first
                                         + coefficient_seed.sine(index) * harmonic.second);
                      }
                  }
                cell_volume += values.JxW(q);
                cell_integral += kernel_value * values.JxW(q);
              }
            AssertThrow(cell_volume > std::numeric_limits<double>::min(),
                        ExcMessage("Self-gravitating density-moment transpose encountered a locally owned cell with zero volume."));
            cellwise_kernel(cell->active_cell_index()) = cell_integral / cell_volume;
          }
      return cellwise_kernel;
    }



    SphericalHarmonicCoefficients
    scale_surface_topography_geoid_harmonics(
      const SphericalHarmonicCoefficients &topography_coefficients,
      const double reference_radius,
      const double surface_gravity,
      const double surface_density_contrast)
    {
      // A displaced surface is a density sheet Delta_rho_s h_s. If
      // h_lm=int_S2 h_s Y_lm dOmega, then
      //   N_lm^s=Gamma_l Delta_rho_s R_s h_lm.
      SphericalHarmonicCoefficients result = topography_coefficients;
      for (unsigned int degree = result.minimum_degree();
           degree <= result.maximum_degree(); ++degree)
        for (unsigned int order = 0; order <= degree; ++order)
          {
            const unsigned int index = result.coefficient_index(degree, order);
            const double scaling = density_geoid_degree_scaling(degree, surface_gravity)
                                   * surface_density_contrast * reference_radius;
            result.cosine(index) *= scaling;
            result.sine(index) *= scaling;
          }
      return result;
    }



    SphericalHarmonicCoefficients
    scale_cmb_topography_geoid_harmonics(
      const SphericalHarmonicCoefficients &topography_coefficients,
      const double reference_radius,
      const double cmb_radius,
      const double surface_gravity,
      const double cmb_density_contrast)
    {
      // A CMB density sheet is continued to the surface reference sphere:
      //   N_lm^c=Gamma_l Delta_rho_c R_c
      //            (R_c/R_s)^(l+1) h_lm^c.
      SphericalHarmonicCoefficients result = topography_coefficients;
      for (unsigned int degree = result.minimum_degree();
           degree <= result.maximum_degree(); ++degree)
        for (unsigned int order = 0; order <= degree; ++order)
          {
            const unsigned int index = result.coefficient_index(degree, order);
            const double scaling = density_geoid_degree_scaling(degree, surface_gravity)
                                   * cmb_density_contrast * cmb_radius
                                   * Utilities::pow(cmb_radius / reference_radius, degree + 1);
            result.cosine(index) *= scaling;
            result.sine(index) *= scaling;
          }
      return result;
    }



    SphericalHarmonicCoefficients
    compute_surface_topography_geoid_transpose(
      const SphericalHarmonicCoefficients &coefficient_seed,
      const double reference_radius,
      const double surface_gravity,
      const double surface_density_contrast)
    {
      SphericalHarmonicCoefficients topography_seed = coefficient_seed;
      for (unsigned int degree = coefficient_seed.minimum_degree();
           degree <= coefficient_seed.maximum_degree(); ++degree)
        for (unsigned int order = 0; order <= degree; ++order)
          {
            const unsigned int index = coefficient_seed.coefficient_index(degree, order);
            // The forward operator integrates h Y_lm over dOmega, whereas
            // the finite-element surface seed is integrated over dA.  At the
            // reference surface dOmega = dA / R^2, so its L2 representative
            // carries one further factor 1/R^2 than the harmonic coefficient.
            const double scaling = density_geoid_degree_scaling(degree, surface_gravity)
                                   * surface_density_contrast / reference_radius;
            topography_seed.cosine(index) *= scaling;
            topography_seed.sine(index) *= scaling;
          }
      return topography_seed;
    }



    template <int dim>
    SphericalHarmonicCoefficients
    compute_surface_topography_geoid_harmonics(
      const SimulatorAccess<dim> &simulator_access,
      const LinearAlgebra::BlockVector &topography,
      const unsigned int minimum_degree,
      const unsigned int maximum_degree,
      const double reference_radius,
      const double surface_gravity,
      const double surface_density_contrast)
    {
      AssertThrow(dim == 3,
                  ExcMessage("Surface geoid harmonics currently only work in 3D."));
      const unsigned int quadrature_degree =
        simulator_access.introspection().polynomial_degree.temperature;
      const QGauss<dim-1> quadrature_formula(quadrature_degree);
      FEFaceValues<dim> fe_face_values(simulator_access.get_mapping(), simulator_access.get_fe(),
                                       quadrature_formula,
                                       update_values | update_quadrature_points | update_JxW_values);
      const types::boundary_id top_boundary_id =
        simulator_access.get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      SphericalHarmonicCoefficients coefficients(minimum_degree, maximum_degree);
      std::vector<double> topography_values(quadrature_formula.size());

      // Project surface displacement into angular coefficients:
      //   h_lm=int_surface h_s Y_lm dA/R_s^2.
      // The R_s^-2 factor converts physical face measure into dOmega. Only
      // locally owned boundary faces are visited; the reduction below
      // completes the global surface integral in coefficient space.
      for (unsigned int degree = minimum_degree; degree <= maximum_degree; ++degree)
        for (unsigned int order = 0; order <= degree; ++order)
          {
            double local_cosine = 0.0;
            double local_sine = 0.0;
            for (const auto &cell : simulator_access.get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                for (const unsigned int face : cell->face_indices())
                  if (cell->at_boundary(face) && cell->face(face)->boundary_id() == top_boundary_id)
                    {
                      fe_face_values.reinit(cell, face);
                      fe_face_values[simulator_access.introspection().extractors.temperature].get_function_values(
                        topography, topography_values);
                      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
                        {
                          const Point<dim> &position = fe_face_values.quadrature_point(q);
                          const std::array<double, dim> spherical_position =
                            Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
                          const std::pair<double, double> harmonic =
                            Utilities::real_spherical_harmonic(degree, order,
                                                               spherical_position[2], spherical_position[1]);
                          const double area_weight = fe_face_values.JxW(q)
                                                     / (reference_radius * reference_radius);
                          local_cosine += topography_values[q] * harmonic.first * area_weight;
                          local_sine += topography_values[q] * harmonic.second * area_weight;
                        }
                    }
            const unsigned int index = coefficients.coefficient_index(degree, order);
            coefficients.cosine(index) = local_cosine;
            coefficients.sine(index) = local_sine;
          }
      std::vector<double> global_cosine(coefficients.size());
      std::vector<double> global_sine(coefficients.size());
      dealii::Utilities::MPI::sum(coefficients.cosine_coefficients(),
                                  simulator_access.get_mpi_communicator(),
                                  global_cosine);
      dealii::Utilities::MPI::sum(coefficients.sine_coefficients(),
                                  simulator_access.get_mpi_communicator(),
                                  global_sine);
      coefficients.set_coefficients(global_cosine, global_sine);
      return scale_surface_topography_geoid_harmonics(coefficients,
                                                      reference_radius,
                                                      surface_gravity,
                                                      surface_density_contrast);
    }



    SphericalHarmonicCoefficients
    compute_cmb_topography_geoid_transpose(
      const SphericalHarmonicCoefficients &coefficient_seed,
      const double reference_radius,
      const double cmb_radius,
      const double surface_gravity,
      const double cmb_density_contrast)
    {
      SphericalHarmonicCoefficients topography_seed = coefficient_seed;
      for (unsigned int degree = coefficient_seed.minimum_degree();
           degree <= coefficient_seed.maximum_degree(); ++degree)
        for (unsigned int order = 0; order <= degree; ++order)
          {
            const unsigned int index = coefficient_seed.coefficient_index(degree, order);
            // dOmega at the CMB is dA/R_c^2. The coefficient-space CMB
            // factor is Gamma_l Delta-rho R_c (R_c/R)^(l+1), leaving this
            // physical-space dual seed after the surface-measure conversion.
            const double scaling = density_geoid_degree_scaling(degree, surface_gravity)
                                   * cmb_density_contrast / cmb_radius
                                   * Utilities::pow(cmb_radius / reference_radius, degree + 1);
            topography_seed.cosine(index) *= scaling;
            topography_seed.sine(index) *= scaling;
          }
      return topography_seed;
    }



    template <int dim>
    SphericalHarmonicCoefficients
    compute_cmb_topography_geoid_harmonics(
      const SimulatorAccess<dim> &simulator_access,
      const LinearAlgebra::BlockVector &topography,
      const unsigned int minimum_degree,
      const unsigned int maximum_degree,
      const double reference_radius,
      const double cmb_radius,
      const double surface_gravity,
      const double cmb_density_contrast)
    {
      AssertThrow(dim == 3,
                  ExcMessage("CMB geoid harmonics currently only work in 3D."));
      const unsigned int quadrature_degree =
        simulator_access.introspection().polynomial_degree.temperature;
      const QGauss<dim-1> quadrature_formula(quadrature_degree);
      FEFaceValues<dim> fe_face_values(simulator_access.get_mapping(), simulator_access.get_fe(),
                                       quadrature_formula,
                                       update_values | update_quadrature_points | update_JxW_values);
      const types::boundary_id bottom_boundary_id =
        simulator_access.get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");
      SphericalHarmonicCoefficients coefficients(minimum_degree, maximum_degree);
      std::vector<double> topography_values(quadrature_formula.size());

      // The CMB projection uses the same real harmonic convention:
      //   h_lm^c=int_CMB h_c Y_lm dA/R_c^2.
      // Density contrast and radial continuation are applied only after the
      // global projection, by scale_cmb_topography_geoid_harmonics().
      for (unsigned int degree = minimum_degree; degree <= maximum_degree; ++degree)
        for (unsigned int order = 0; order <= degree; ++order)
          {
            double local_cosine = 0.0;
            double local_sine = 0.0;
            for (const auto &cell : simulator_access.get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                for (const unsigned int face : cell->face_indices())
                  if (cell->at_boundary(face) && cell->face(face)->boundary_id() == bottom_boundary_id)
                    {
                      fe_face_values.reinit(cell, face);
                      fe_face_values[simulator_access.introspection().extractors.temperature].get_function_values(
                        topography, topography_values);
                      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
                        {
                          const Point<dim> &position = fe_face_values.quadrature_point(q);
                          const std::array<double, dim> spherical_position =
                            Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
                          const std::pair<double, double> harmonic =
                            Utilities::real_spherical_harmonic(degree, order,
                                                               spherical_position[2], spherical_position[1]);
                          const double area_weight = fe_face_values.JxW(q) / (cmb_radius * cmb_radius);
                          local_cosine += topography_values[q] * harmonic.first * area_weight;
                          local_sine += topography_values[q] * harmonic.second * area_weight;
                        }
                    }
            const unsigned int index = coefficients.coefficient_index(degree, order);
            coefficients.cosine(index) = local_cosine;
            coefficients.sine(index) = local_sine;
          }
      std::vector<double> global_cosine(coefficients.size());
      std::vector<double> global_sine(coefficients.size());
      dealii::Utilities::MPI::sum(coefficients.cosine_coefficients(),
                                  simulator_access.get_mpi_communicator(),
                                  global_cosine);
      dealii::Utilities::MPI::sum(coefficients.sine_coefficients(),
                                  simulator_access.get_mpi_communicator(),
                                  global_sine);
      coefficients.set_coefficients(global_cosine, global_sine);
      return scale_cmb_topography_geoid_harmonics(coefficients,
                                                  reference_radius,
                                                  cmb_radius,
                                                  surface_gravity,
                                                  cmb_density_contrast);
    }



    template SphericalHarmonicCoefficients
    compute_density_geoid_harmonics<2> (const SimulatorAccess<2> &,
                                        const unsigned int,
                                        const unsigned int,
                                        const double,
                                        const double);

    template SphericalHarmonicCoefficients
    compute_density_geoid_harmonics<3> (const SimulatorAccess<3> &,
                                        const unsigned int,
                                        const unsigned int,
                                        const double,
                                        const double);

    template Vector<double>
    compute_density_geoid_transpose<2> (const SimulatorAccess<2> &,
                                        const SphericalHarmonicCoefficients &,
                                        const double,
                                        const double);

    template Vector<double>
    compute_density_geoid_transpose<3> (const SimulatorAccess<3> &,
                                        const SphericalHarmonicCoefficients &,
                                        const double,
                                        const double);

    template Vector<double>
    compute_self_gravitating_density_moment_transpose<2>(const SimulatorAccess<2> &,
                                                         const SphericalHarmonicCoefficients &,
                                                         const double, const double,
                                                         const SelfGravitatingDensityMoment);

    template Vector<double>
    compute_self_gravitating_density_moment_transpose<3>(const SimulatorAccess<3> &,
                                                         const SphericalHarmonicCoefficients &,
                                                         const double, const double,
                                                         const SelfGravitatingDensityMoment);

    template SphericalHarmonicCoefficients
    compute_surface_topography_geoid_harmonics<2>(const SimulatorAccess<2> &,
                                                  const LinearAlgebra::BlockVector &,
                                                  const unsigned int, const unsigned int,
                                                  const double, const double, const double);

    template SphericalHarmonicCoefficients
    compute_surface_topography_geoid_harmonics<3>(const SimulatorAccess<3> &,
                                                  const LinearAlgebra::BlockVector &,
                                                  const unsigned int, const unsigned int,
                                                  const double, const double, const double);

    template SphericalHarmonicCoefficients
    compute_cmb_topography_geoid_harmonics<2>(const SimulatorAccess<2> &,
                                              const LinearAlgebra::BlockVector &,
                                              const unsigned int, const unsigned int,
                                              const double, const double, const double, const double);

    template SphericalHarmonicCoefficients
    compute_cmb_topography_geoid_harmonics<3>(const SimulatorAccess<3> &,
                                              const LinearAlgebra::BlockVector &,
                                              const unsigned int, const unsigned int,
                                              const double, const double, const double, const double);
  }
}
