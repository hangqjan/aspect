/*
  Copyright (C) 2011 - 2024 by the authors of ASPECT.
  This file is part of ASPECT.
*/

#include <aspect/postprocess/self_gravitating_geoid_operator.h>

#include <aspect/material_model/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/postprocess/boundary_densities.h>
#include <aspect/postprocess/dynamic_topography.h>
#include <aspect/postprocess/forward_boundary_reduced_stress.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <cmath>
#include <limits>

namespace aspect
{
  namespace Postprocess
  {
    namespace
    {
      void check_layout(const SphericalHarmonicCoefficients &c,
                        const unsigned int minimum_degree,
                        const unsigned int maximum_degree,
                        const std::string &name)
      {
        AssertThrow(c.minimum_degree() == minimum_degree && c.maximum_degree() == maximum_degree
                    && c.cosine_coefficients().size() == c.size()
                    && c.sine_coefficients().size() == c.size(),
                    ExcMessage("Self-gravitating geoid operator received an invalid " + name
                               + " harmonic layout."));
      }

      void check_operator_inputs(const SelfGravitatingGeoidOperatorInputs &in)
      {
        const unsigned int min_degree = in.density_moments.at_surface.minimum_degree();
        const unsigned int max_degree = in.density_moments.at_surface.maximum_degree();
        AssertThrow(min_degree >= 2,
                    ExcMessage("The coupled self-gravity operator requires l_min >= 2."));
        AssertThrow(in.outer_radius > 0.0 && in.inner_radius > 0.0 && in.inner_radius < in.outer_radius
                    && in.surface_gravity > 0.0,
                    ExcMessage("The coupled self-gravity operator requires positive ordered radii and gravity."));
        check_layout(in.density_moments.at_surface, min_degree, max_degree, "surface density moment");
        check_layout(in.density_moments.at_cmb, min_degree, max_degree, "CMB density moment");
        check_layout(in.surface_reduced_stress, min_degree, max_degree, "surface reduced stress");
        check_layout(in.cmb_reduced_stress, min_degree, max_degree, "CMB reduced stress");
        const double top_contrast = in.top_boundary_density - in.density_above;
        const double bottom_contrast = in.density_below - in.bottom_boundary_density;
        AssertThrow(std::abs(top_contrast) > std::numeric_limits<double>::min()
                    && std::abs(bottom_contrast) > std::numeric_limits<double>::min(),
                    ExcMessage("The coupled self-gravity operator encountered zero boundary density contrast."));
      }

      void set_layout(SphericalHarmonicCoefficients &coefficients,
                      const unsigned int minimum_degree,
                      const unsigned int maximum_degree)
      {
        coefficients.reinit(minimum_degree, maximum_degree);
      }
    }

    std::array<std::array<double, 4>, 4>
    SelfGravitatingGeoidOperator::matrix_for_degree(
      const SelfGravitatingGeoidOperatorInputs &in,
      const unsigned int degree)
    {
      // For each (l,m) cosine or sine coefficient we solve A_l x_lm=b_lm
      // with
      //   x = [h_s, h_c, N_s, N_c]^T,
      // where h denotes boundary deflection and N the geoid perturbation at
      // the surface (s) and CMB (c). The first two rows are the linearized
      // normal-stress conditions. The last two rows evaluate the potential of
      // both displaced density interfaces at the two observation radii.
      const double top_contrast = in.top_boundary_density - in.density_above;
      const double bottom_contrast = in.density_below - in.bottom_boundary_density;

      // A degree-l surface density harmonic produces a potential proportional
      // to 4*pi*G/(g_s*(2l+1)). Radial continuation contributes powers of
      // r_c/r_s. These factors are independent of m, so one matrix is reused
      // for every order and for both real-harmonic components of a degree.
      const double gamma = 4.0 * numbers::PI * constants::big_g
                           / (in.surface_gravity * (2.0 * degree + 1.0));
      const double radius_ratio = in.inner_radius / in.outer_radius;
      return {{
          {{top_contrast * in.surface_gravity, 0.0, -in.top_boundary_density * in.surface_gravity, 0.0}},
          {{0.0, bottom_contrast * in.surface_gravity, 0.0, -bottom_contrast * in.surface_gravity}},
          { {
              -gamma *in.outer_radius * top_contrast,
              -gamma *in.inner_radius *std::pow(radius_ratio, degree + 1) *bottom_contrast, 1.0, 0.0
            }
          },
          { {
              -gamma *in.outer_radius *std::pow(radius_ratio, degree) *top_contrast,
              -gamma *in.inner_radius * bottom_contrast, 0.0, 1.0
            }
          }
        }
      };
    }



    std::array<std::array<double, 4>, 4>
    SelfGravitatingGeoidOperator::matrix_density_derivative_for_degree(
      const SelfGravitatingGeoidOperatorInputs &in,
      const unsigned int degree,
      const SelfGravitatingMatrixDensityParameter parameter)
    {
      check_operator_inputs(in);
      // This routine differentiates A_l only. RHS density moments are handled
      // separately by their volume-integral transpose. Keeping the two paths
      // separate is essential for the implicit-system derivative
      //   dJ/dp = z^T db/dp - z^T (dA/dp) x,
      // with A_l^T z=q.
      const double gamma = 4.0 * numbers::PI * constants::big_g
                           / (in.surface_gravity * (2.0 * degree + 1.0));
      const double radius_ratio = in.inner_radius / in.outer_radius;
      if (parameter == SelfGravitatingMatrixDensityParameter::top_boundary_density)
        // d Delta_rho_t / d rho_bar_t = +1, while rho_bar_t also appears
        // explicitly in A(0,2).
        return {{
            {{in.surface_gravity, 0.0, -in.surface_gravity, 0.0}},
            {{0.0, 0.0, 0.0, 0.0}},
            {{-gamma * in.outer_radius, 0.0, 0.0, 0.0}},
            {{-gamma *in.outer_radius *std::pow(radius_ratio, degree), 0.0, 0.0, 0.0}}
          }
        };
      // Delta_rho_b = rho_below-rho_bar_b, so its derivative is -1.
      return {{
          {{0.0, 0.0, 0.0, 0.0}},
          {{0.0, -in.surface_gravity, 0.0, in.surface_gravity}},
          {{0.0, gamma *in.inner_radius *std::pow(radius_ratio, degree + 1), 0.0, 0.0}},
          {{0.0, gamma * in.inner_radius, 0.0, 0.0}}
        }
      };
    }

    std::array<std::array<double, 4>, 4>
    SelfGravitatingGeoidOperator::inverse(const std::array<std::array<double, 4>, 4> &matrix)
    {
      // Form [A|I] and use partial-pivoted Gauss-Jordan elimination. The
      // spectral block is only 4x4, so an explicit dense inverse keeps this
      // operator independent of Simulator and of a particular linear-algebra
      // backend. Partial pivoting is nevertheless required because stress and
      // potential rows have very different physical units and magnitudes.
      std::array<std::array<double, 8>, 4> augmented = {{}};
      for (unsigned int row = 0; row < 4; ++row)
        for (unsigned int column = 0; column < 4; ++column)
          {
            augmented[row][column] = matrix[row][column];
            augmented[row][column + 4] = (row == column ? 1.0 : 0.0);
          }
      for (unsigned int column = 0; column < 4; ++column)
        {
          // Select the largest remaining entry in this column before
          // normalizing the pivot row. This is the only row permutation; the
          // physical ordering of x is restored automatically in the inverse.
          unsigned int pivot = column;
          for (unsigned int row = column + 1; row < 4; ++row)
            if (std::abs(augmented[row][column]) > std::abs(augmented[pivot][column]))
              pivot = row;
          AssertThrow(std::abs(augmented[pivot][column]) > 1e-30,
                      ExcMessage("The coupled self-gravity 4x4 matrix is singular."));
          std::swap(augmented[column], augmented[pivot]);
          const double diagonal = augmented[column][column];
          for (double &entry : augmented[column]) entry /= diagonal;

          // Eliminate the pivot column from the other three rows. After the
          // fourth column, the left block is I and the right block is A^{-1}.
          for (unsigned int row = 0; row < 4; ++row)
            if (row != column)
              {
                const double factor = augmented[row][column];
                for (unsigned int entry = 0; entry < 8; ++entry)
                  augmented[row][entry] -= factor * augmented[column][entry];
              }
        }
      std::array<std::array<double, 4>, 4> result = {{}};
      for (unsigned int row = 0; row < 4; ++row)
        for (unsigned int column = 0; column < 4; ++column)
          result[row][column] = augmented[row][column + 4];
      return result;
    }

    SelfGravitatingGeoidCoefficients
    SelfGravitatingGeoidOperator::apply(const SelfGravitatingGeoidOperatorInputs &in) const
    {
      const unsigned int min_degree = in.density_moments.at_surface.minimum_degree();
      const unsigned int max_degree = in.density_moments.at_surface.maximum_degree();
      check_operator_inputs(in);

      // All output fields use exactly the density-moment harmonic layout. A
      // shared invariant layout prevents accidental mixing of degrees or of
      // triangular (l,m) indices between the four coupled unknowns.
      SelfGravitatingGeoidCoefficients out;
      for (SphericalHarmonicCoefficients *field :
           {
             &out.surface_topography, &out.cmb_topography,
             &out.surface_geoid, &out.cmb_geoid
           })
        set_layout(*field, min_degree, max_degree);
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
        {
          // Spherical symmetry decouples different degrees and orders. Build
          // and invert A_l once, then apply it to every m and to the cosine and
          // sine systems. No MPI communication is needed in this spectral
          // solve because its input coefficients have already been reduced.
          const double gamma = 4.0 * numbers::PI * constants::big_g
                               / (in.surface_gravity * (2.0 * degree + 1.0));
          const std::array<std::array<double, 4>, 4> matrix = matrix_for_degree(in, degree);
          const auto inverse_matrix = inverse(matrix); // One factorization/inverse per degree.
          for (unsigned int order = 0; order <= degree; ++order)
            {
              const unsigned int index = in.density_moments.at_surface.coefficient_index(degree, order);
              for (const bool sine :
              {
                false, true
              })
              {
                const auto component = [sine, index](const SphericalHarmonicCoefficients &c)
                {
                  return sine ? c.sine(index) : c.cosine(index);
                };
                const std::array<double, 4> rhs = {{
                    -component(in.surface_reduced_stress), component(in.cmb_reduced_stress),
                    gamma * component(in.density_moments.at_surface), gamma * component(in.density_moments.at_cmb)
                  }
                };

                // Compute x=A_l^{-1}b. The explicit residual is retained as a
                // scale-independent health metric; it catches singular or
                // badly scaled physical configurations without changing the
                // computed coefficients.
                std::array<double, 4> x = {{}};
                std::array<double, 4> residual = {{}};
                double rhs_norm = 0.0, residual_norm = 0.0;
                for (unsigned int row = 0; row < 4; ++row)
                  {
                    for (unsigned int col = 0; col < 4; ++col) x[row] += inverse_matrix[row][col] * rhs[col];
                    rhs_norm += rhs[row] * rhs[row];
                  }
                for (unsigned int row = 0; row < 4; ++row)
                  {
                    for (unsigned int col = 0; col < 4; ++col) residual[row] += matrix[row][col] * x[col];
                    residual[row] -= rhs[row];
                    residual_norm += residual[row] * residual[row];
                  }
                out.maximum_normalized_residual = std::max(out.maximum_normalized_residual,
                                                           std::sqrt(residual_norm) / std::max(std::sqrt(rhs_norm), 1e-30));
                const auto store = [sine, index](SphericalHarmonicCoefficients &c,
                                                 const double value)
                {
                  if (sine) c.sine(index) = value;
                  else c.cosine(index) = value;
                };

                // Store all four components at the same invariant (l,m)
                // index. The sine branch at m=0 is harmless and remains zero
                // for a consistent real-spherical-harmonic projection.
                store(out.surface_topography, x[0]);
                store(out.cmb_topography, x[1]);
                store(out.surface_geoid, x[2]);
                store(out.cmb_geoid, x[3]);
              }
            }
        }
      return out;
    }

    SelfGravitatingGeoidTransposeCoefficients
    SelfGravitatingGeoidOperator::apply_transpose(
      const SelfGravitatingGeoidOperatorInputs &in,
      const SelfGravitatingGeoidCoefficients &solution_seed) const
    {
      const unsigned int min_degree = in.density_moments.at_surface.minimum_degree();
      const unsigned int max_degree = in.density_moments.at_surface.maximum_degree();
      check_operator_inputs(in);
      // If the forward solve is x=A^{-1}b and q=dJ/dx, reverse mode first
      // solves
      //   A^T z = q.
      // It then sends z to the RHS dependencies, including the signs and
      // gamma factors used when b was assembled. The z coefficients are also
      // retained because parameter derivatives of A require -z^T(dA)x.
      for (const auto field :
      {
        std::make_pair(&solution_seed.surface_topography, "surface topography seed"),
              std::make_pair(&solution_seed.cmb_topography, "CMB topography seed"),
              std::make_pair(&solution_seed.surface_geoid, "surface geoid seed"),
              std::make_pair(&solution_seed.cmb_geoid, "CMB geoid seed")
        })
      check_layout(*field.first, min_degree, max_degree, field.second);

      SelfGravitatingGeoidTransposeCoefficients out;
      for (SphericalHarmonicCoefficients *field :
           {
             &out.surface_topography, &out.cmb_topography,
             &out.surface_geoid, &out.cmb_geoid,
             &out.rhs_seeds.surface_reduced_stress,
             &out.rhs_seeds.cmb_reduced_stress,
             &out.rhs_seeds.surface_density_moment,
             &out.rhs_seeds.cmb_density_moment
           })
        set_layout(*field, min_degree, max_degree);

      for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
        {
          const double gamma = 4.0 * numbers::PI * constants::big_g
                               / (in.surface_gravity * (2.0 * degree + 1.0));
          const std::array<std::array<double, 4>, 4> matrix = matrix_for_degree(in, degree);
          std::array<std::array<double, 4>, 4> transpose = {{}};
          for (unsigned int row = 0; row < 4; ++row)
            for (unsigned int column = 0; column < 4; ++column)
              transpose[row][column] = matrix[column][row];
          const auto inverse_transpose = inverse(transpose); // One factorization per degree.

          // As in the forward operator, one degree block serves every order
          // and both real components. This ordering must match apply() so the
          // discrete identity <q,A^{-1}b>=<A^{-T}q,b> holds coefficient-wise.
          for (unsigned int order = 0; order <= degree; ++order)
            {
              const unsigned int index = solution_seed.surface_topography.coefficient_index(degree, order);
              for (const bool sine :
              {
                false, true
              })
              {
                const auto component = [sine, index](const SphericalHarmonicCoefficients &c)
                {
                  return sine ? c.sine(index) : c.cosine(index);
                };
                const std::array<double, 4> q = {{
                    component(solution_seed.surface_topography),
                    component(solution_seed.cmb_topography),
                    component(solution_seed.surface_geoid),
                    component(solution_seed.cmb_geoid)
                  }
                };
                // z=A^{-T}q is both the solution adjoint and the unscaled RHS
                // adjoint. Compute a transpose residual independently of the
                // forward residual to diagnose errors in matrix orientation.
                std::array<double, 4> z = {{}};
                std::array<double, 4> residual = {{}};
                double q_norm = 0.0, residual_norm = 0.0;
                for (unsigned int row = 0; row < 4; ++row)
                  {
                    for (unsigned int column = 0; column < 4; ++column)
                      z[row] += inverse_transpose[row][column] * q[column];
                    q_norm += q[row] * q[row];
                  }
                for (unsigned int row = 0; row < 4; ++row)
                  {
                    for (unsigned int column = 0; column < 4; ++column)
                      residual[row] += transpose[row][column] * z[column];
                    residual[row] -= q[row];
                    residual_norm += residual[row] * residual[row];
                  }
                out.maximum_normalized_residual = std::max(out.maximum_normalized_residual,
                                                           std::sqrt(residual_norm) / std::max(std::sqrt(q_norm), 1e-30));
                const auto store = [sine, index](SphericalHarmonicCoefficients &c,
                                                 const double value)
                {
                  if (sine) c.sine(index) = value;
                  else c.cosine(index) = value;
                };
                store(out.surface_topography, z[0]);
                store(out.cmb_topography, z[1]);
                store(out.surface_geoid, z[2]);
                store(out.cmb_geoid, z[3]);

                // Reverse the exact forward RHS construction
                // b=[-tau_s, tau_c, gamma*M_s, gamma*M_c]^T.
                store(out.rhs_seeds.surface_reduced_stress, -z[0]);
                store(out.rhs_seeds.cmb_reduced_stress, z[1]);
                store(out.rhs_seeds.surface_density_moment, gamma * z[2]);
                store(out.rhs_seeds.cmb_density_moment, gamma * z[3]);
              }
            }
        }
      return out;
    }



    SelfGravitatingMatrixDensitySensitivity
    SelfGravitatingGeoidOperator::matrix_density_sensitivity(
      const SelfGravitatingGeoidOperatorInputs &in,
      const SelfGravitatingGeoidCoefficients &forward_solution,
      const SelfGravitatingGeoidTransposeCoefficients &transpose_solution) const
    {
      check_operator_inputs(in);
      const unsigned int min_degree = in.density_moments.at_surface.minimum_degree();
      const unsigned int max_degree = in.density_moments.at_surface.maximum_degree();
      for (const auto field :
      {
        std::make_pair(&forward_solution.surface_topography, "forward surface topography"),
              std::make_pair(&forward_solution.cmb_topography, "forward CMB topography"),
              std::make_pair(&forward_solution.surface_geoid, "forward surface geoid"),
              std::make_pair(&forward_solution.cmb_geoid, "forward CMB geoid"),
              std::make_pair(&transpose_solution.surface_topography, "transpose surface topography"),
              std::make_pair(&transpose_solution.cmb_topography, "transpose CMB topography"),
              std::make_pair(&transpose_solution.surface_geoid, "transpose surface geoid"),
              std::make_pair(&transpose_solution.cmb_geoid, "transpose CMB geoid")
        })
      check_layout(*field.first, min_degree, max_degree, field.second);

      SelfGravitatingMatrixDensitySensitivity result;
      // Boundary mean densities enter the implicit matrix as well as other
      // forward quantities. For a matrix-only parameter p, differentiating
      // A(p)x=b gives dx/dp=-A^{-1}A_p x and therefore
      //   dJ/dp = -z^T A_p x,  A^T z=dJ/dx.
      // Sum this scalar contraction over every independent harmonic system.
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
        {
          const auto top_derivative = matrix_density_derivative_for_degree(
                                        in, degree,
                                        SelfGravitatingMatrixDensityParameter::top_boundary_density);
          const auto bottom_derivative = matrix_density_derivative_for_degree(
                                           in, degree,
                                           SelfGravitatingMatrixDensityParameter::bottom_boundary_density);
          for (unsigned int order = 0; order <= degree; ++order)
            {
              const unsigned int index = forward_solution.surface_topography.coefficient_index(degree, order);
              for (const bool sine :
              {
                false, true
              })
              {
                const auto component = [sine, index](const SphericalHarmonicCoefficients &field)
                {
                  return sine ? field.sine(index) : field.cosine(index);
                };
                const std::array<double, 4> x = {{
                    component(forward_solution.surface_topography),
                    component(forward_solution.cmb_topography),
                    component(forward_solution.surface_geoid),
                    component(forward_solution.cmb_geoid)
                  }
                };
                const std::array<double, 4> z = {{
                    component(transpose_solution.surface_topography),
                    component(transpose_solution.cmb_topography),
                    component(transpose_solution.surface_geoid),
                    component(transpose_solution.cmb_geoid)
                  }
                };
                for (unsigned int row = 0; row < 4; ++row)
                  for (unsigned int column = 0; column < 4; ++column)
                    {
                      result.top_boundary_density -= z[row] * top_derivative[row][column] * x[column];
                      result.bottom_boundary_density -= z[row] * bottom_derivative[row][column] * x[column];
                    }
              }
            }
        }
      return result;
    }

    template <int dim>
    SelfGravitatingDensityMoments
    compute_self_gravitating_density_moments(const SimulatorAccess<dim> &access,
                                             const unsigned int min_degree, const unsigned int max_degree,
                                             const double outer_radius, const double inner_radius)
    {
      AssertThrow(dim == 3 && min_degree >= 2 && min_degree <= max_degree,
                  ExcMessage("Self-gravity density moments require 3D and l_min >= 2."));
      SelfGravitatingDensityMoments result;
      for (auto *c :
           {
             &result.at_surface, &result.at_cmb
           })
        c->reinit(min_degree, max_degree);
      const QGauss<dim> quadrature(access.introspection().polynomial_degree.temperature);
      FEValues<dim> values(access.get_mapping(), access.get_fe(), quadrature,
                           update_values | update_quadrature_points | update_JxW_values | update_gradients);
      MaterialModel::MaterialModelInputs<dim> material_input(quadrature.size(), access.n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> material_output(quadrature.size(), access.n_compositional_fields());
      material_input.requested_properties = MaterialModel::MaterialProperties::density;

      // Project rho onto two solid-harmonic kernels. At the surface,
      //   M^s_lm = integral rho r^l/r_s^(l+1) Y_lm dV,
      // while at the CMB,
      //   M^c_lm = integral rho r_c^l/r^(l+1) Y_lm dV.
      // The real and imaginary parts returned by the utility are stored as
      // cosine and sine coefficients. Cells are integrated only on their
      // owning rank, so each coefficient is a local partial integral here.
      for (unsigned int l = min_degree; l <= max_degree; ++l)
        for (unsigned int m = 0; m <= l; ++m)
          {
            double top_cos=0., top_sin=0., bottom_cos=0., bottom_sin=0.;
            for (const auto &cell : access.get_dof_handler().active_cell_iterators()) if (cell->is_locally_owned())
                {
                  values.reinit(cell);
                  material_input.reinit(values, cell, access.introspection(), access.get_solution());
                  access.get_material_model().evaluate(material_input, material_output);
                  for (unsigned int q=0; q<quadrature.size(); ++q)
                    {
                      const auto spherical=Utilities::Coordinates::cartesian_to_spherical_coordinates(material_input.position[q]);
                      const auto y=Utilities::real_spherical_harmonic(l,m,spherical[2],spherical[1]);
                      const double r=material_input.position[q].norm();
                      const double w=material_output.densities[q]*values.JxW(q);
                      top_cos += w*std::pow(r,l)/std::pow(outer_radius,l+1)*y.first;
                      top_sin += w*std::pow(r,l)/std::pow(outer_radius,l+1)*y.second;
                      bottom_cos += w*std::pow(inner_radius,l)/std::pow(r,l+1)*y.first;
                      bottom_sin += w*std::pow(inner_radius,l)/std::pow(r,l+1)*y.second;
                    }
                }
            const unsigned int index = result.at_surface.coefficient_index(l, m);
            result.at_surface.cosine(index) = top_cos;
            result.at_surface.sine(index) = top_sin;
            result.at_cmb.cosine(index) = bottom_cos;
            result.at_cmb.sine(index) = bottom_sin;
          }
      std::vector<double> surface_cosine(result.at_surface.size());
      std::vector<double> surface_sine(result.at_surface.size());
      std::vector<double> cmb_cosine(result.at_cmb.size());
      std::vector<double> cmb_sine(result.at_cmb.size());

      // Convert the rank-local cell integrals into replicated spectral input
      // for the small dense operator. All ranks then execute identical 4x4
      // solves, avoiding ownership-dependent coefficient values.
      Utilities::MPI::sum(result.at_surface.cosine_coefficients(),
                          access.get_mpi_communicator(), surface_cosine);
      Utilities::MPI::sum(result.at_surface.sine_coefficients(),
                          access.get_mpi_communicator(), surface_sine);
      Utilities::MPI::sum(result.at_cmb.cosine_coefficients(),
                          access.get_mpi_communicator(), cmb_cosine);
      Utilities::MPI::sum(result.at_cmb.sine_coefficients(),
                          access.get_mpi_communicator(), cmb_sine);
      result.at_surface.set_coefficients(surface_cosine, surface_sine);
      result.at_cmb.set_coefficients(cmb_cosine, cmb_sine);
      return result;
    }

    template <int dim>
    SelfGravitatingGeoidOperatorInputs
    compute_self_gravitating_geoid_inputs(const SimulatorAccess<dim> &access,
                                          const unsigned int minimum_degree,
                                          const unsigned int maximum_degree)
    {
      AssertThrow(dim == 3,
                  ExcMessage("The coupled self-gravity operator currently only works in 3D."));
      const auto *geometry = dynamic_cast<const GeometryModel::SphericalShell<dim> *>(
                               &access.get_geometry_model());
      AssertThrow(geometry != nullptr,
                  ExcMessage("The coupled self-gravity operator requires a spherical shell geometry."));
      const double outer_radius = geometry->outer_radius();
      const double inner_radius = geometry->inner_radius();
      Point<dim> surface_point;
      surface_point[0] = outer_radius;
      const double surface_gravity = access.get_gravity_model().gravity_vector(surface_point).norm();

      // Gather the three independent parts of the coupled model: mean density
      // contrasts from the boundary-density postprocessor, reduced normal
      // stresses from the forward Stokes state, and volume density moments.
      // This function performs Simulator-dependent data collection; the
      // spectral operator above remains a pure, independently testable map.
      const BoundaryDensities<dim> &boundary_densities =
        access.get_postprocess_manager().template get_matching_active_plugin<BoundaryDensities<dim>>();
      const DynamicTopography<dim> &dynamic_topography =
        access.get_postprocess_manager().template get_matching_active_plugin<DynamicTopography<dim>>();
      const BoundaryReducedStressHarmonics reduced_stress =
        ForwardBoundaryReducedStressProvider<dim>::compute(access, minimum_degree, maximum_degree,
                                                           outer_radius, inner_radius);
      return
      {
        compute_self_gravitating_density_moments(access, minimum_degree, maximum_degree,
        outer_radius, inner_radius),
        reduced_stress.surface, reduced_stress.cmb, outer_radius, inner_radius,
        surface_gravity, boundary_densities.density_at_top(), boundary_densities.density_at_bottom(),
        dynamic_topography.get_density_above(), dynamic_topography.get_density_below()
      };
    }

    template <int dim>
    SelfGravitatingGeoidCoefficients
    compute_self_gravitating_geoid_harmonics(const SimulatorAccess<dim> &access,
                                             const unsigned int minimum_degree,
                                             const unsigned int maximum_degree)
    {
      return SelfGravitatingGeoidOperator().apply(
               compute_self_gravitating_geoid_inputs(access, minimum_degree, maximum_degree));
    }
    template SelfGravitatingDensityMoments compute_self_gravitating_density_moments<2>(const SimulatorAccess<2> &, unsigned int, unsigned int, double, double);
    template SelfGravitatingDensityMoments compute_self_gravitating_density_moments<3>(const SimulatorAccess<3> &, unsigned int, unsigned int, double, double);
    template SelfGravitatingGeoidOperatorInputs compute_self_gravitating_geoid_inputs<2>(const SimulatorAccess<2> &, unsigned int, unsigned int);
    template SelfGravitatingGeoidOperatorInputs compute_self_gravitating_geoid_inputs<3>(const SimulatorAccess<3> &, unsigned int, unsigned int);
    template SelfGravitatingGeoidCoefficients compute_self_gravitating_geoid_harmonics<2>(const SimulatorAccess<2> &, unsigned int, unsigned int);
    template SelfGravitatingGeoidCoefficients compute_self_gravitating_geoid_harmonics<3>(const SimulatorAccess<3> &, unsigned int, unsigned int);
  }
}
