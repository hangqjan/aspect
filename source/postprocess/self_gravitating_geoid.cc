/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include <aspect/postprocess/self_gravitating_geoid.h>

#include <sstream>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string, std::string>
    SelfGravitatingGeoid<dim>::execute(TableHandler &statistics)
    {
      // Keep Simulator data collection outside the spectral solve. inputs()
      // projects reduced stresses and density moments and supplies the scalar
      // background model; apply() then solves an independent 4x4 system for
      // every (l,m), making the mathematical operator unit-testable.
      const SelfGravitatingGeoidOperatorInputs inputs =
        compute_self_gravitating_geoid_inputs(*this, minimum_degree, maximum_degree);
      result = SelfGravitatingGeoidOperator().apply(inputs);
      statistics.add_value("Self-gravitating geoid maximum normalized residual",
                           result.maximum_normalized_residual);
      statistics.set_scientific("Self-gravitating geoid maximum normalized residual", true);
      statistics.set_precision("Self-gravitating geoid maximum normalized residual", 16);
      const std::array<std::pair<const char *, const SphericalHarmonicCoefficients *>, 4> fields = {{
          {"surface topography", &result.surface_topography},
          {"CMB topography", &result.cmb_topography},
          {"surface geoid", &result.surface_geoid},
          {"CMB geoid", &result.cmb_geoid}
        }
      };
      // Publish each coupled unknown separately. The four output fields are
      // [surface topography, CMB topography, surface geoid, CMB geoid]; cosine
      // and sine components share the same degree-dependent matrix but remain
      // distinct observations for non-axisymmetric models.
      for (const auto &field : fields)
        for (unsigned int degree = minimum_degree; degree <= maximum_degree; ++degree)
          for (unsigned int order = 0; order <= degree; ++order)
            {
              const unsigned int index = field.second->coefficient_index(degree, order);
              for (const auto component :
              {
                std::make_pair("cosine", field.second->cosine(index)),
                      std::make_pair("sine", field.second->sine(index))
                })
              {
                const std::string name = "Self-gravitating geoid " + std::string(field.first)
                                         + " l=" + std::to_string(degree)
                                         + " m=" + std::to_string(order) + " " + component.first;
                statistics.add_value(name, component.second);
                statistics.set_scientific(name, true);
                statistics.set_precision(name, 16);
              }
            }
      const std::array<std::pair<const char *, const SphericalHarmonicCoefficients *>, 4> input_fields = {{
          {"surface reduced stress", &inputs.surface_reduced_stress},
          {"CMB reduced stress", &inputs.cmb_reduced_stress},
          {"surface density moment", &inputs.density_moments.at_surface},
          {"CMB density moment", &inputs.density_moments.at_cmb}
        }
      };
      // Record the projected right-hand-side data as well as the solution.
      // These values make a run reproducible and allow the spectral residual
      // to be diagnosed without re-evaluating Simulator fields.
      for (const auto &field : input_fields)
        for (unsigned int degree = minimum_degree; degree <= maximum_degree; ++degree)
          for (unsigned int order = 0; order <= degree; ++order)
            {
              const unsigned int index = field.second->coefficient_index(degree, order);
              for (const auto component :
              {
                std::make_pair("cosine", field.second->cosine(index)),
                      std::make_pair("sine", field.second->sine(index))
                })
              {
                const std::string name = "Self-gravitating geoid " + std::string(field.first)
                                         + " l=" + std::to_string(degree)
                                         + " m=" + std::to_string(order) + " " + component.first;
                statistics.add_value(name, component.second);
                statistics.set_scientific(name, true);
                statistics.set_precision(name, 16);
              }
            }
      for (const auto scalar :
      {
        std::make_pair("outer radius", inputs.outer_radius),
              std::make_pair("CMB radius", inputs.inner_radius),
              std::make_pair("surface gravity", inputs.surface_gravity),
              std::make_pair("top boundary density", inputs.top_boundary_density),
              std::make_pair("bottom boundary density", inputs.bottom_boundary_density),
              std::make_pair("density above", inputs.density_above),
              std::make_pair("density below", inputs.density_below)
        })
      {
        // Background radii, gravity, and density jumps determine the 4x4
        // matrices A_l. Reporting them closes the provenance chain from the
        // Simulator-dependent input preparation to each coefficient solve.
        const std::string name = "Self-gravitating geoid " + std::string(scalar.first);
        statistics.add_value(name, scalar.second);
        statistics.set_scientific(name, true);
        statistics.set_precision(name, 16);
      }
      std::ostringstream output;
      output.precision(3);
      output << result.maximum_normalized_residual;
      return {"Coupled self-gravitating geoid maximum normalized residual:", output.str()};
    }

    template <int dim>
    std::list<std::string>
    SelfGravitatingGeoid<dim>::required_other_postprocessors() const
    {
      return {"dynamic topography", "boundary densities"};
    }

    template <int dim>
    void SelfGravitatingGeoid<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Self-gravitating geoid");
        {
          prm.declare_entry("Minimum degree", "2", Patterns::Integer(2),
                            "Minimum spherical-harmonic degree.");
          prm.declare_entry("Maximum degree", "20", Patterns::Integer(2),
                            "Maximum spherical-harmonic degree.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void SelfGravitatingGeoid<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Self-gravitating geoid");
        {
          minimum_degree = prm.get_integer("Minimum degree");
          maximum_degree = prm.get_integer("Maximum degree");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
      AssertThrow(minimum_degree <= maximum_degree,
                  ExcMessage("The self-gravitating geoid minimum degree exceeds its maximum degree."));
    }

    template <int dim>
    const SelfGravitatingGeoidCoefficients &SelfGravitatingGeoid<dim>::get_result() const
    {
      return result;
    }
  }
}

namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(SelfGravitatingGeoid,
                                  "self-gravitating geoid",
                                  "Forward-only coupled self-gravitating geoid and dynamic-topography "
                                  "harmonics for spherical-shell models.")
  }
}
