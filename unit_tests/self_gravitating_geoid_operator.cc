/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include "common.h"

#include <aspect/postprocess/self_gravitating_geoid_operator.h>
#include <aspect/utilities.h>

#include <cmath>
#include <sstream>

namespace
{
  aspect::Postprocess::SphericalHarmonicCoefficients
  coefficients()
  {
    return aspect::Postprocess::SphericalHarmonicCoefficients(2, 3);
  }

  aspect::Postprocess::SelfGravitatingGeoidOperatorInputs
  inputs()
  {
    using namespace aspect::Postprocess;
    SelfGravitatingGeoidOperatorInputs result
    {
      {coefficients(), coefficients()}, coefficients(), coefficients(),
      6.371e6, 3.480e6, 9.81, 3300., 5500., 0., 9900.
    };
    // Exercise l=2,m=0; l=2,m=1; and l=3,m=2 in both real components.
    for (const auto degree_order :
    {
      std::pair<unsigned int, unsigned int>(2, 0),
            std::pair<unsigned int, unsigned int>(2, 1),
            std::pair<unsigned int, unsigned int>(3, 2)
      })
    {
      const unsigned int i = result.surface_reduced_stress.coefficient_index(
                               degree_order.first, degree_order.second);
      result.surface_reduced_stress.cosine(i) = 1.3e5 + i;
      result.surface_reduced_stress.sine(i) = -2.1e5 - i;
      result.cmb_reduced_stress.cosine(i) = -0.7e5 - i;
      result.cmb_reduced_stress.sine(i) = 0.9e5 + i;
      result.density_moments.at_surface.cosine(i) = 4.0e15 + i;
      result.density_moments.at_surface.sine(i) = -5.0e15 - i;
      result.density_moments.at_cmb.cosine(i) = -6.0e15 - i;
      result.density_moments.at_cmb.sine(i) = 7.0e15 + i;
    }
    return result;
  }
}

TEST_CASE("Self-gravitating geoid 4x4 residual and real components")
{
  using namespace aspect;
  using namespace aspect::Postprocess;
  const SelfGravitatingGeoidOperatorInputs in = inputs();
  const SelfGravitatingGeoidCoefficients out = SelfGravitatingGeoidOperator().apply(in);
  REQUIRE(out.maximum_normalized_residual < 1e-10);

  // Independent audit of the documented l=2 and l=3 matrix, evaluated with
  // the returned x rather than with the operator's cached inverse.
  for (const unsigned int degree :
  {
    2u, 3u
  })
  {
    const double gamma = 4.0 * numbers::PI * constants::big_g /
                         (in.surface_gravity * (2.0 * degree + 1.0));
    const double ratio = in.inner_radius / in.outer_radius;
    const double top_contrast = in.top_boundary_density - in.density_above;
    const double bottom_contrast = in.density_below - in.bottom_boundary_density;
    const std::array<std::array<double, 4>, 4> a = {{
        {{top_contrast * in.surface_gravity, 0., -in.top_boundary_density * in.surface_gravity, 0.}},
        {{0., bottom_contrast * in.surface_gravity, 0., -bottom_contrast * in.surface_gravity}},
        { {
            -gamma *in.outer_radius * top_contrast,
            -gamma *in.inner_radius *std::pow(ratio, degree + 1) *bottom_contrast, 1., 0.
          }
        },
        { {
            -gamma *in.outer_radius *std::pow(ratio, degree) *top_contrast,
            -gamma *in.inner_radius * bottom_contrast, 0., 1.
          }
        }
      }
    };
    for (const unsigned int order :
    {
      0u, degree == 2 ? 1u : 2u
    })
    {
      const unsigned int i = out.surface_geoid.coefficient_index(degree, order);
      for (const bool sine :
      {
        false, true
      })
      {
        const auto value = [sine, i](const SphericalHarmonicCoefficients &c)
        {
          return sine ? c.sine(i) : c.cosine(i);
        };
        const std::array<double, 4> x = {{
            value(out.surface_topography), value(out.cmb_topography),
            value(out.surface_geoid), value(out.cmb_geoid)
          }
        };
        const std::array<double, 4> rhs = {{
            -value(in.surface_reduced_stress),
            value(in.cmb_reduced_stress),
            gamma * value(in.density_moments.at_surface),
            gamma * value(in.density_moments.at_cmb)
          }
        };
        for (unsigned int row = 0; row < 4; ++row)
          {
            double residual = -rhs[row];
            for (unsigned int column = 0; column < 4; ++column)
              residual += a[row][column] * x[column];
            REQUIRE(std::abs(residual) < 1e-10 * std::max(1.0, std::abs(rhs[row])));
          }
      }
    }
  }
}

TEST_CASE("Spherical-harmonic coefficient layout and evaluation")
{
  using aspect::Postprocess::SphericalHarmonicCoefficients;

  SphericalHarmonicCoefficients coefficients(2, 3);
  REQUIRE(coefficients.minimum_degree() == 2);
  REQUIRE(coefficients.maximum_degree() == 3);
  REQUIRE(coefficients.size() == 7);
  REQUIRE(coefficients.coefficient_index(2, 0) == 0);
  REQUIRE(coefficients.coefficient_index(2, 2) == 2);
  REQUIRE(coefficients.coefficient_index(3, 0) == 3);
  REQUIRE(coefficients.coefficient_index(3, 3) == 6);
  REQUIRE_THROWS(coefficients.coefficient_index(1, 0));
  REQUIRE_THROWS(coefficients.coefficient_index(3, 4));
  REQUIRE_THROWS(SphericalHarmonicCoefficients(3, 2));

  const unsigned int index = coefficients.coefficient_index(3, 2);
  coefficients.cosine(index) = 1.25;
  coefficients.sine(index) = -0.75;
  const double theta = 0.7;
  const double phi = 1.1;
  const auto harmonic =
    aspect::Utilities::real_spherical_harmonic(3, 2, theta, phi);
  REQUIRE(coefficients.evaluate(theta, phi)
          == Approx(1.25 * harmonic.first - 0.75 * harmonic.second));
}



TEST_CASE("Spherical-harmonic coefficient writer follows invariant layout")
{
  using namespace aspect::Postprocess;

  SphericalHarmonicCoefficients coefficients(1, 2);
  for (unsigned int index = 0; index < coefficients.size(); ++index)
    {
      coefficients.cosine(index) = 10.0 + index;
      coefficients.sine(index) = -20.0 - index;
    }

  std::ostringstream output;
  write_spherical_harmonic_coefficients(output, coefficients);
  CHECK(output.str() ==
        "1 0 10 -20\n"
        "1 1 11 -21\n"
        "2 0 12 -22\n"
        "2 1 13 -23\n"
        "2 2 14 -24\n");
}



TEST_CASE("Conventional topography geoid forward-transpose identity")
{
  using namespace aspect::Postprocess;

  SphericalHarmonicCoefficients topography(2, 4);
  SphericalHarmonicCoefficients coefficient_seed(2, 4);
  for (unsigned int index = 0; index < topography.size(); ++index)
    {
      topography.cosine(index) = 0.3 + 0.07 * index;
      topography.sine(index) = -0.2 + 0.03 * index;
      coefficient_seed.cosine(index) = -0.4 + 0.05 * index;
      coefficient_seed.sine(index) = 0.6 - 0.02 * index;
    }

  const double surface_radius = 6.371e6;
  const double cmb_radius = 3.480e6;
  const double gravity = 9.81;
  const double surface_density_contrast = 3300.0;
  const double cmb_density_contrast = 4400.0;

  const SphericalHarmonicCoefficients surface_forward =
    scale_surface_topography_geoid_harmonics(topography,
                                             surface_radius,
                                             gravity,
                                             surface_density_contrast);
  const SphericalHarmonicCoefficients surface_transpose =
    compute_surface_topography_geoid_transpose(coefficient_seed,
                                               surface_radius,
                                               gravity,
                                               surface_density_contrast);
  const SphericalHarmonicCoefficients cmb_forward =
    scale_cmb_topography_geoid_harmonics(topography,
                                         surface_radius,
                                         cmb_radius,
                                         gravity,
                                         cmb_density_contrast);
  const SphericalHarmonicCoefficients cmb_transpose =
    compute_cmb_topography_geoid_transpose(coefficient_seed,
                                           surface_radius,
                                           cmb_radius,
                                           gravity,
                                           cmb_density_contrast);

  const auto dot = [](const SphericalHarmonicCoefficients &left,
                      const SphericalHarmonicCoefficients &right)
  {
    double value = 0.0;
    for (unsigned int index = 0; index < left.size(); ++index)
      value += left.cosine(index) * right.cosine(index)
               + left.sine(index) * right.sine(index);
    return value;
  };

  // The transpose returns the physical boundary L2 seed. Harmonic projection
  // uses dOmega=dA/R_boundary^2, hence the boundary-area factors below.
  REQUIRE(dot(coefficient_seed, surface_forward)
          == Approx(surface_radius * surface_radius
                    * dot(surface_transpose, topography)).epsilon(1e-13));
  REQUIRE(dot(coefficient_seed, cmb_forward)
          == Approx(cmb_radius * cmb_radius
                    * dot(cmb_transpose, topography)).epsilon(1e-13));
}



TEST_CASE("Self-gravitating geoid limiting inputs")
{
  using namespace aspect::Postprocess;
  SelfGravitatingGeoidOperatorInputs in = inputs();
  in.density_moments = {coefficients(), coefficients()};
  in.surface_reduced_stress = coefficients();
  in.cmb_reduced_stress = coefficients();
  const SelfGravitatingGeoidCoefficients zero = SelfGravitatingGeoidOperator().apply(in);
  REQUIRE(zero.maximum_normalized_residual == Approx(0.0));
  for (const double value : zero.surface_geoid.cosine_coefficients())
    REQUIRE(value == Approx(0.0));

  const unsigned int i = in.surface_reduced_stress.coefficient_index(2, 1);
  in.surface_reduced_stress.cosine(i) = 1.0e5; // top-only loading
  const SelfGravitatingGeoidCoefficients top = SelfGravitatingGeoidOperator().apply(in);
  REQUIRE(std::abs(top.surface_topography.cosine(i)) > 0.0);
  in.surface_reduced_stress = coefficients();
  in.cmb_reduced_stress.sine(i) = 1.0e5; // bottom-only loading
  const SelfGravitatingGeoidCoefficients bottom = SelfGravitatingGeoidOperator().apply(in);
  REQUIRE(std::abs(bottom.cmb_topography.sine(i)) > 0.0);
  for (const double value : bottom.surface_topography.cosine_coefficients())
    REQUIRE(value == Approx(0.0));
  for (const double value : bottom.cmb_topography.cosine_coefficients())
    REQUIRE(value == Approx(0.0));
  for (const double value : bottom.surface_geoid.cosine_coefficients())
    REQUIRE(value == Approx(0.0));
  for (const double value : bottom.cmb_geoid.cosine_coefficients())
    REQUIRE(value == Approx(0.0));

  in.cmb_reduced_stress = coefficients();
  in.density_moments.at_surface.cosine(i) = 3.0e15; // density-only loading
  const SelfGravitatingGeoidCoefficients density = SelfGravitatingGeoidOperator().apply(in);
  REQUIRE(std::abs(density.surface_geoid.cosine(i)) > 0.0);
}

TEST_CASE("Self-gravitating geoid spectral transpose identity")
{
  using namespace aspect;
  using namespace aspect::Postprocess;
  const SelfGravitatingGeoidOperatorInputs physical_inputs = inputs();
  SelfGravitatingGeoidCoefficients q;
  for (auto *field :
       {
         &q.surface_topography, &q.cmb_topography,
         &q.surface_geoid, &q.cmb_geoid
       })
    *field = coefficients();

  SelfGravitatingGeoidOperatorInputs perturbation = physical_inputs;
  perturbation.density_moments = {coefficients(), coefficients()};
  perturbation.surface_reduced_stress = coefficients();
  perturbation.cmb_reduced_stress = coefficients();
  for (unsigned int degree = 2; degree <= 3; ++degree)
    for (unsigned int order = 0; order <= degree; ++order)
      {
        const unsigned int i = q.surface_geoid.coefficient_index(degree, order);
        for (const bool sine :
        {
          false, true
        })
        {
          const double sign = sine ? -1.0 : 1.0;
          const double scale = 1.0 + i;
          auto assign = [sine, i](SphericalHarmonicCoefficients &field, const double value)
          {
            if (sine) field.sine(i) = value;
            else field.cosine(i) = value;
          };
          assign(q.surface_topography, sign * 0.13 * scale);
          assign(q.cmb_topography, -sign * 0.17 * scale);
          assign(q.surface_geoid, sign * 0.19 * scale);
          assign(q.cmb_geoid, -sign * 0.23 * scale);
          assign(perturbation.surface_reduced_stress, sign * 1.1e5 * scale);
          assign(perturbation.cmb_reduced_stress, -sign * 0.7e5 * scale);
          assign(perturbation.density_moments.at_surface, sign * 2.3e15 * scale);
          assign(perturbation.density_moments.at_cmb, -sign * 1.7e15 * scale);
        }
      }

  const SelfGravitatingGeoidOperator operator_;
  const SelfGravitatingGeoidCoefficients delta_x = operator_.apply(perturbation);
  const SelfGravitatingGeoidTransposeCoefficients transpose =
    operator_.apply_transpose(physical_inputs, q);
  REQUIRE(transpose.maximum_normalized_residual < 1e-10);

  for (unsigned int degree = 2; degree <= 3; ++degree)
    {
      const double gamma = 4.0 * numbers::PI * constants::big_g /
                           (physical_inputs.surface_gravity * (2.0 * degree + 1.0));
      const double ratio = physical_inputs.inner_radius / physical_inputs.outer_radius;
      const double top_contrast = physical_inputs.top_boundary_density - physical_inputs.density_above;
      const double bottom_contrast = physical_inputs.density_below - physical_inputs.bottom_boundary_density;
      const std::array<std::array<double, 4>, 4> a = {{
          { {
              top_contrast * physical_inputs.surface_gravity, 0.,
              -physical_inputs.top_boundary_density * physical_inputs.surface_gravity, 0.
            }
          },
          { {
              0., bottom_contrast * physical_inputs.surface_gravity, 0.,
              -bottom_contrast *physical_inputs.surface_gravity
            }
          },
          { {
              -gamma *physical_inputs.outer_radius * top_contrast,
              -gamma *physical_inputs.inner_radius *std::pow(ratio, degree + 1) *bottom_contrast, 1., 0.
            }
          },
          { {
              -gamma *physical_inputs.outer_radius *std::pow(ratio, degree) *top_contrast,
              -gamma *physical_inputs.inner_radius * bottom_contrast, 0., 1.
            }
          }
        }
      };
      for (unsigned int order = 0; order <= degree; ++order)
        {
          const unsigned int i = q.surface_geoid.coefficient_index(degree, order);
          for (const bool sine :
          {
            false, true
          })
          {
            const auto value = [sine, i](const SphericalHarmonicCoefficients &field)
            {
              return sine ? field.sine(i) : field.cosine(i);
            };
            const std::array<double, 4> q_component = {{
                value(q.surface_topography),
                value(q.cmb_topography),
                value(q.surface_geoid),
                value(q.cmb_geoid)
              }
            };
            const std::array<double, 4> dx = {{
                value(delta_x.surface_topography),
                value(delta_x.cmb_topography),
                value(delta_x.surface_geoid),
                value(delta_x.cmb_geoid)
              }
            };
            const std::array<double, 4> z = {{
                value(transpose.surface_topography),
                value(transpose.cmb_topography),
                value(transpose.surface_geoid),
                value(transpose.cmb_geoid)
              }
            };
            double lhs = 0.0;
            double rhs = 0.0;
            for (unsigned int component = 0; component < 4; ++component)
              {
                lhs += q_component[component] * dx[component];
                double residual = -q_component[component];
                for (unsigned int row = 0; row < 4; ++row)
                  residual += a[row][component] * z[row];
                REQUIRE(std::abs(residual) < 1e-10 * std::max(1.0, std::abs(q_component[component])));
              }
            rhs += value(transpose.rhs_seeds.surface_reduced_stress)
                   * value(perturbation.surface_reduced_stress);
            rhs += value(transpose.rhs_seeds.cmb_reduced_stress)
                   * value(perturbation.cmb_reduced_stress);
            rhs += value(transpose.rhs_seeds.surface_density_moment)
                   * value(perturbation.density_moments.at_surface);
            rhs += value(transpose.rhs_seeds.cmb_density_moment)
                   * value(perturbation.density_moments.at_cmb);
            REQUIRE(std::abs(lhs - rhs) < 1e-10 * std::max({1.0, std::abs(lhs), std::abs(rhs)}));
            REQUIRE(value(transpose.rhs_seeds.surface_reduced_stress) == Approx(-z[0]));
            REQUIRE(value(transpose.rhs_seeds.cmb_reduced_stress) == Approx(z[1]));
            REQUIRE(value(transpose.rhs_seeds.surface_density_moment) == Approx(gamma * z[2]));
            REQUIRE(value(transpose.rhs_seeds.cmb_density_moment) == Approx(gamma * z[3]));
          }
        }
    }
}

TEST_CASE("Self-gravitating geoid matrix density derivative")
{
  using namespace aspect;
  using namespace aspect::Postprocess;
  const SelfGravitatingGeoidOperatorInputs physical_inputs = inputs();
  const SelfGravitatingGeoidOperator operator_;
  const SelfGravitatingGeoidCoefficients forward = operator_.apply(physical_inputs);
  const double ratio = physical_inputs.inner_radius / physical_inputs.outer_radius;

  for (const unsigned int degree :
  {
    2u, 3u
  })
  {
    const double gamma = 4.0 * numbers::PI * constants::big_g /
                         (physical_inputs.surface_gravity * (2.0 * degree + 1.0));
    const std::array<std::array<std::array<double, 4>, 4>, 2> expected = {{{{
            {{physical_inputs.surface_gravity, 0., -physical_inputs.surface_gravity, 0.}},
            {{0., 0., 0., 0.}},
            {{-gamma * physical_inputs.outer_radius, 0., 0., 0.}},
            {{-gamma *physical_inputs.outer_radius *std::pow(ratio, degree), 0., 0., 0.}}
          }
        }, {{
            {{0., 0., 0., 0.}},
            {{0., -physical_inputs.surface_gravity, 0., physical_inputs.surface_gravity}},
            {{0., gamma *physical_inputs.inner_radius *std::pow(ratio, degree + 1), 0., 0.}},
            {{0., gamma * physical_inputs.inner_radius, 0., 0.}}
          }
        }
      }
    };
    for (const auto parameter :
    {
      SelfGravitatingMatrixDensityParameter::top_boundary_density,
                                            SelfGravitatingMatrixDensityParameter::bottom_boundary_density
    })
    {
      const auto derivative = SelfGravitatingGeoidOperator::matrix_density_derivative_for_degree(
                                physical_inputs, degree, parameter);
      const unsigned int parameter_index = parameter == SelfGravitatingMatrixDensityParameter::top_boundary_density ? 0 : 1;
      for (unsigned int row = 0; row < 4; ++row)
        for (unsigned int column = 0; column < 4; ++column)
          REQUIRE(derivative[row][column] == Approx(expected[parameter_index][row][column]));
    }

    SelfGravitatingGeoidCoefficients q;
    for (auto *field :
         {
           &q.surface_topography, &q.cmb_topography,
           &q.surface_geoid, &q.cmb_geoid
         })
      *field = coefficients();
    for (unsigned int order = 0; order <= degree; ++order)
      {
        const unsigned int index = q.surface_geoid.coefficient_index(degree, order);
        const double scale = 1.0 + index;
        q.surface_topography.cosine(index) = 0.11 * scale;
        q.cmb_topography.cosine(index) = -0.13 * scale;
        q.surface_geoid.cosine(index) = 0.17 * scale;
        q.cmb_geoid.cosine(index) = -0.19 * scale;
        q.surface_topography.sine(index) = -0.23 * scale;
        q.cmb_topography.sine(index) = 0.29 * scale;
        q.surface_geoid.sine(index) = -0.31 * scale;
        q.cmb_geoid.sine(index) = 0.37 * scale;
      }
    const SelfGravitatingGeoidTransposeCoefficients transpose =
      operator_.apply_transpose(physical_inputs, q);
    const SelfGravitatingMatrixDensitySensitivity sensitivity =
      operator_.matrix_density_sensitivity(physical_inputs, forward, transpose);

    const auto objective = [&q](const SelfGravitatingGeoidCoefficients &solution)
    {
      double value = 0.0;
      for (unsigned int index = 0; index < q.surface_geoid.cosine_coefficients().size(); ++index)
        value += q.surface_topography.cosine(index) * solution.surface_topography.cosine(index)
                 + q.cmb_topography.cosine(index) * solution.cmb_topography.cosine(index)
                 + q.surface_geoid.cosine(index) * solution.surface_geoid.cosine(index)
                 + q.cmb_geoid.cosine(index) * solution.cmb_geoid.cosine(index)
                 + q.surface_topography.sine(index) * solution.surface_topography.sine(index)
                 + q.cmb_topography.sine(index) * solution.cmb_topography.sine(index)
                 + q.surface_geoid.sine(index) * solution.surface_geoid.sine(index)
                 + q.cmb_geoid.sine(index) * solution.cmb_geoid.sine(index);
      return value;
    };
    for (const auto parameter :
    {
      SelfGravitatingMatrixDensityParameter::top_boundary_density,
                                            SelfGravitatingMatrixDensityParameter::bottom_boundary_density
    })
    {
      const double epsilon = 1.e-2;
      SelfGravitatingGeoidOperatorInputs plus = physical_inputs;
      SelfGravitatingGeoidOperatorInputs minus = physical_inputs;
      if (parameter == SelfGravitatingMatrixDensityParameter::top_boundary_density)
        {
          plus.top_boundary_density += epsilon;
          minus.top_boundary_density -= epsilon;
        }
      else
        {
          plus.bottom_boundary_density += epsilon;
          minus.bottom_boundary_density -= epsilon;
        }
      const double finite_difference =
        (objective(operator_.apply(plus)) - objective(operator_.apply(minus))) / (2.0 * epsilon);
      const double analytical = parameter == SelfGravitatingMatrixDensityParameter::top_boundary_density
                                ? sensitivity.top_boundary_density : sensitivity.bottom_boundary_density;
      REQUIRE(std::abs(finite_difference - analytical)
              < 1.e-8 * std::max({1.0, std::abs(finite_difference), std::abs(analytical)}));
    }
  }
}

TEST_CASE("Self-gravitating geoid matrix-density superposition")
{
  using namespace aspect::Postprocess;
  const SelfGravitatingGeoidOperatorInputs in = inputs();
  const SelfGravitatingGeoidOperator operator_;
  const SelfGravitatingGeoidCoefficients forward = operator_.apply(in);
  SelfGravitatingGeoidCoefficients top_seed;
  SelfGravitatingGeoidCoefficients bottom_seed;
  for (auto *field :
       {
         &top_seed.surface_topography, &top_seed.cmb_topography,
         &top_seed.surface_geoid, &top_seed.cmb_geoid,
         &bottom_seed.surface_topography, &bottom_seed.cmb_topography,
         &bottom_seed.surface_geoid, &bottom_seed.cmb_geoid
       })
    *field = coefficients();
  const unsigned int index = top_seed.surface_geoid.coefficient_index(2, 1);
  top_seed.surface_topography.cosine(index) = 0.7;
  top_seed.surface_geoid.sine(index) = -0.4;
  bottom_seed.cmb_topography.cosine(index) = -0.6;
  bottom_seed.cmb_geoid.sine(index) = 0.9;
  SelfGravitatingGeoidCoefficients combined_seed = top_seed;
  for (const auto fields :
  {
    std::make_pair(&combined_seed.surface_topography, &bottom_seed.surface_topography),
          std::make_pair(&combined_seed.cmb_topography, &bottom_seed.cmb_topography),
          std::make_pair(&combined_seed.surface_geoid, &bottom_seed.surface_geoid),
          std::make_pair(&combined_seed.cmb_geoid, &bottom_seed.cmb_geoid)
    })
  for (unsigned int i = 0; i < fields.first->size(); ++i)
    {
      fields.first->cosine(i) += fields.second->cosine(i);
      fields.first->sine(i) += fields.second->sine(i);
    }
  const SelfGravitatingMatrixDensitySensitivity top = operator_.matrix_density_sensitivity(
                                                        in, forward, operator_.apply_transpose(in, top_seed));
  const SelfGravitatingMatrixDensitySensitivity bottom = operator_.matrix_density_sensitivity(
                                                           in, forward, operator_.apply_transpose(in, bottom_seed));
  const SelfGravitatingMatrixDensitySensitivity combined = operator_.matrix_density_sensitivity(
                                                             in, forward, operator_.apply_transpose(in, combined_seed));
  REQUIRE(combined.top_boundary_density == Approx(top.top_boundary_density + bottom.top_boundary_density));
  REQUIRE(combined.bottom_boundary_density == Approx(top.bottom_boundary_density + bottom.bottom_boundary_density));
}
