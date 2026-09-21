/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include <aspect/adjoint/objective/self_gravitating_geoid_objective.h>

#include <aspect/adjoint/operators/dynamic_topography_boundary_operator.h>
#include <aspect/adjoint/core/kernel_calculator.h>
#include <aspect/postprocess/boundary_densities.h>
#include <aspect/postprocess/dynamic_topography.h>
#include <aspect/postprocess/forward_boundary_reduced_stress.h>
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
      bool nonzero(const Postprocess::SphericalHarmonicCoefficients &coefficients)
      {
        return std::any_of(coefficients.cosine_coefficients().begin(),
                           coefficients.cosine_coefficients().end(),
                           [](const double value)
        {
          return value != 0.0;
        })
        || std::any_of(coefficients.sine_coefficients().begin(),
                       coefficients.sine_coefficients().end(),
                       [](const double value)
        {
          return value != 0.0;
        });
      }
    }

    template <int dim>
    Postprocess::SelfGravitatingGeoidOperatorInputs
    SelfGravitatingGeoidObjective<dim>::inputs() const
    {
      // Assemble all Simulator-dependent data once into the value-type input
      // of the pure spectral operator: density moments, reduced stresses,
      // radii, gravity, and boundary mean densities. The external densities
      // also enter the CBF topography conversion, so accepting inconsistent
      // values here would break forward/transpose duality.
      const Postprocess::SelfGravitatingGeoidOperatorInputs result =
        Postprocess::compute_self_gravitating_geoid_inputs(*this, minimum_degree, maximum_degree);
      const double scale = std::max({1.0, std::abs(density_above), std::abs(density_below),
                                     std::abs(result.density_above), std::abs(result.density_below)
                                    });
      AssertThrow(std::abs(result.density_above - density_above) <= 1.e-12 * scale
                  && std::abs(result.density_below - density_below) <= 1.e-12 * scale,
                  ExcMessage("Self-gravitating geoid external densities must match the required "
                             "Postprocess/Dynamic topography Density above and Density below settings."));
      return result;
    }

    template <int dim>
    Postprocess::SelfGravitatingGeoidCoefficients
    SelfGravitatingGeoidObjective<dim>::forward_coefficients() const
    {
      return Postprocess::SelfGravitatingGeoidOperator().apply(inputs());
    }

    template <int dim>
    Postprocess::SelfGravitatingGeoidCoefficients
    SelfGravitatingGeoidObjective<dim>::coefficient_seed() const
    {
      // For the four-field least-squares functional
      //
      //   J=1/2 sum_k w_k ||x_k-x_k^obs||_2^2,
      //
      // the coefficient seed is bar(x_k)=w_k(x_k-x_k^obs). The same weights
      // apply to cosine and sine components; the four fields remain distinct
      // because the coupled 4x4 operator mixes them in its transpose.
      Postprocess::SelfGravitatingGeoidCoefficients seed = forward_coefficients();
      const std::array<Postprocess::SphericalHarmonicCoefficients *, 4> fields = {{
          &seed.surface_topography, &seed.cmb_topography,
          &seed.surface_geoid, &seed.cmb_geoid
        }
      };
      for (unsigned int field = 0; field < fields.size(); ++field)
        for (unsigned int index = 0; index < fields[field]->size(); ++index)
          {
            fields[field]->cosine(index) = weights[index]
                                           * (fields[field]->cosine(index)
                                              - observed_coefficients[field].cosine(index));
            fields[field]->sine(index) = weights[index]
                                         * (fields[field]->sine(index)
                                            - observed_coefficients[field].sine(index));
          }
      return seed;
    }

    template <int dim>
    bool
    SelfGravitatingGeoidObjective<dim>::has_traction_seed(
      const Postprocess::SelfGravitatingGeoidRHSSeeds &seeds) const
    {
      return nonzero(seeds.surface_reduced_stress) || nonzero(seeds.cmb_reduced_stress);
    }

    template <int dim>
    double
    SelfGravitatingGeoidObjective<dim>::evaluate() const
    {
      const Postprocess::SelfGravitatingGeoidCoefficients forward = forward_coefficients();
      const std::array<const Postprocess::SphericalHarmonicCoefficients *, 4> fields = {{
          &forward.surface_topography, &forward.cmb_topography,
          &forward.surface_geoid, &forward.cmb_geoid
        }
      };
      double objective = 0.0;
      // Accumulate in coefficient space rather than physical space. With the
      // shared real-harmonic normalization this treats each stored (l,m)
      // cosine/sine datum exactly as prescribed by its user weight.
      for (unsigned int field = 0; field < fields.size(); ++field)
        for (unsigned int index = 0; index < fields[field]->size(); ++index)
          {
            const double cosine_residual =
              fields[field]->cosine(index)
              - observed_coefficients[field].cosine(index);
            const double sine_residual =
              fields[field]->sine(index)
              - observed_coefficients[field].sine(index);
            objective += 0.5 * weights[index]
                         * (cosine_residual * cosine_residual + sine_residual * sine_residual);
          }
      return objective;
    }

    template <int dim>
    void
    SelfGravitatingGeoidObjective<dim>::assemble_adjoint_right_hand_side(
      LinearAlgebra::BlockVector &rhs) const
    {
      const Postprocess::SelfGravitatingGeoidOperatorInputs forward_inputs = inputs();
      const Postprocess::SelfGravitatingGeoidTransposeCoefficients transpose =
        Postprocess::SelfGravitatingGeoidOperator().apply_transpose(forward_inputs, coefficient_seed());
      const auto &dynamic_topography = this->get_postprocess_manager().template
                                       get_matching_active_plugin<Postprocess::DynamicTopography<dim>>();
      const auto add_boundary = [&](const Postprocess::SphericalHarmonicCoefficients &traction_seed,
                                    const Postprocess::DynamicTopographyBoundary boundary,
                                    const double radius,
                                    const double density_exterior)
      {
        if (!nonzero(traction_seed))
          return;

        // The spectral transpose first produces a seed for the boundary
        // reduced-stress coefficients. Projection transpose converts it into
        // an FE dual vector for CBF topography. The boundary operator then
        // applies the transpose of CBF and reduced traction to obtain the
        // velocity-pressure right-hand side of the physical Stokes adjoint.
        LinearAlgebra::BlockVector topography_adjoint(
          this->introspection().index_sets.system_partitioning,
          this->introspection().index_sets.system_relevant_partitioning,
          this->get_mpi_communicator());
        double area = 0.0, pressure_weight = 0.0;
        Postprocess::assemble_boundary_reduced_stress_transpose(
          *this, traction_seed, boundary, radius, density_exterior,
          topography_adjoint, area, pressure_weight);
        DynamicTopographyBoundaryOperator<dim>(this->get_simulator()).add_state_transpose(
          topography_adjoint, area, pressure_weight, density_exterior, rhs, boundary);
      };
      // Surface and CMB are two additive observation paths into the same
      // Stokes state. Their transpose seeds therefore add to one RHS, after
      // which the objective manager performs exactly one physical adjoint
      // solve rather than solving separately for each boundary.
      add_boundary(transpose.rhs_seeds.surface_reduced_stress,
                   Postprocess::DynamicTopographyBoundary::surface,
                   forward_inputs.outer_radius, dynamic_topography.get_density_above());
      add_boundary(transpose.rhs_seeds.cmb_reduced_stress,
                   Postprocess::DynamicTopographyBoundary::cmb,
                   forward_inputs.inner_radius, dynamic_topography.get_density_below());
    }

    template <int dim>
    bool
    SelfGravitatingGeoidObjective<dim>::requires_physical_stokes_adjoint() const
    {
      const Postprocess::SelfGravitatingGeoidOperatorInputs forward_inputs = inputs();
      const Postprocess::SelfGravitatingGeoidTransposeCoefficients transpose =
        Postprocess::SelfGravitatingGeoidOperator().apply_transpose(forward_inputs, coefficient_seed());
      return has_traction_seed(transpose.rhs_seeds);
    }

    template <int dim>
    std::set<std::string>
    SelfGravitatingGeoidObjective<dim>::required_forward_postprocessors() const
    {
      return {"dynamic topography", "boundary densities"};
    }

    template <int dim>
    void
    SelfGravitatingGeoidObjective<dim>::add_direct_kernel_contributions(
      const std::string &objective_name, KernelRepository<dim> &kernels) const
    {
      const Postprocess::SelfGravitatingGeoidOperatorInputs forward_inputs = inputs();
      const Postprocess::SelfGravitatingGeoidCoefficients forward_solution =
        Postprocess::SelfGravitatingGeoidOperator().apply(forward_inputs);
      const Postprocess::SelfGravitatingGeoidTransposeCoefficients transpose =
        Postprocess::SelfGravitatingGeoidOperator().apply_transpose(forward_inputs, coefficient_seed());
      // The coupled spectral output depends on volume density by three
      // mathematically different paths: two explicit forcing moments and the
      // boundary mean densities inside A_l. The first two are the transpose of
      // b_l(rho); keep them separate from -z_l^T(dA_l/drho)x_l below.
      kernels.add_contribution({objective_name, "self-gravitating surface density moment",
                                PhysicalProperty::density
                               },
                               Postprocess::compute_self_gravitating_density_moment_transpose(
                                 *this, transpose.rhs_seeds.surface_density_moment,
                                 forward_inputs.outer_radius, forward_inputs.inner_radius,
                                 Postprocess::SelfGravitatingDensityMoment::surface));
      kernels.add_contribution({objective_name, "self-gravitating CMB density moment",
                                PhysicalProperty::density
                               },
                               Postprocess::compute_self_gravitating_density_moment_transpose(
                                 *this, transpose.rhs_seeds.cmb_density_moment,
                                 forward_inputs.outer_radius, forward_inputs.inner_radius,
                                 Postprocess::SelfGravitatingDensityMoment::cmb));

      // For A_l(rho_boundary)x_l=b_l, implicit differentiation contributes
      //
      //   dJ/drho_boundary = -z_l^T (dA_l/drho_boundary) x_l,
      //   A_l^T z_l = bar(x_l).
      //
      // Only the two area-weighted mantle means that construct A_l are chained
      // here. Explicit moments, CBF, and implicit-Stokes terms are independent
      // paths and must not be re-added through this contraction.
      const Postprocess::SelfGravitatingMatrixDensitySensitivity matrix_sensitivity =
        Postprocess::SelfGravitatingGeoidOperator().matrix_density_sensitivity(
          forward_inputs, forward_solution, transpose);
      Vector<double> matrix_density_kernel =
        Postprocess::compute_boundary_mean_density_transpose(
          *this, Postprocess::BoundaryMeanDensityBoundary::top,
          matrix_sensitivity.top_boundary_density);
      matrix_density_kernel += Postprocess::compute_boundary_mean_density_transpose(
                                 *this, Postprocess::BoundaryMeanDensityBoundary::bottom,
                                 matrix_sensitivity.bottom_boundary_density);
      kernels.add_contribution({objective_name, "self-gravity matrix density term",
                                PhysicalProperty::density
                               }, matrix_density_kernel);

      const auto &dynamic_topography = this->get_postprocess_manager().template
                                       get_matching_active_plugin<Postprocess::DynamicTopography<dim>>();
      const auto add_boundary = [&](const Postprocess::SphericalHarmonicCoefficients &traction_seed,
                                    const Postprocess::DynamicTopographyBoundary boundary,
                                    const double radius,
                                    const double density_exterior)
      {
        if (!nonzero(traction_seed))
          return;

        // Reuse the same reduced-stress projection transpose as for the
        // physical RHS, but now differentiate CBF with respect to density and
        // viscosity while holding the Stokes state fixed. These are direct
        // material kernels and are added to, not substituted for, the kernel
        // obtained through the physical Stokes adjoint.
        LinearAlgebra::BlockVector topography_adjoint(
          this->introspection().index_sets.system_partitioning,
          this->introspection().index_sets.system_relevant_partitioning,
          this->get_mpi_communicator());
        double area = 0.0, pressure_weight = 0.0;
        Postprocess::assemble_boundary_reduced_stress_transpose(
          *this, traction_seed, boundary, radius, density_exterior,
          topography_adjoint, area, pressure_weight);
        DynamicTopographyBoundaryOperator<dim>(this->get_simulator()).add_direct_kernel_contributions(
          topography_adjoint, dynamic_topography.topography_vector(), density_exterior,
          objective_name, kernels, boundary);
      };
      add_boundary(transpose.rhs_seeds.surface_reduced_stress,
                   Postprocess::DynamicTopographyBoundary::surface,
                   forward_inputs.outer_radius, dynamic_topography.get_density_above());
      add_boundary(transpose.rhs_seeds.cmb_reduced_stress,
                   Postprocess::DynamicTopographyBoundary::cmb,
                   forward_inputs.inner_radius, dynamic_topography.get_density_below());
    }

    template <int dim>
    void
    SelfGravitatingGeoidObjective<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Adjoint");
      {
        prm.enter_subsection("Self-gravitating geoid");
        {
          prm.declare_entry("Minimum spherical harmonic degree", "2", Patterns::Integer(2),
                            "Minimum coupled self-gravity harmonic degree. The coupled operator is defined only for l >= 2.");
          prm.declare_entry("Maximum spherical harmonic degree", "2", Patterns::Integer(2),
                            "Maximum coupled self-gravity harmonic degree.");
          const std::array<const char *, 4> names = {{"surface topography", "CMB topography",
              "surface geoid", "CMB geoid"
            }
          };
          for (const char *name : names)
            {
              prm.declare_entry("Observed " + std::string(name) + " cosine coefficients", "", Patterns::Anything(),
                                "Observed coupled coefficient values in degree-major order.");
              prm.declare_entry("Observed " + std::string(name) + " sine coefficients", "", Patterns::Anything(),
                                "Observed coupled coefficient values in degree-major order.");
            }
          prm.declare_entry("Coefficient weights", "", Patterns::Anything(),
                            "Optional nonnegative degree/order weights shared by all four coupled fields and by cosine/sine.");
          prm.declare_entry("Surface density above", "0", Patterns::Double(0.0),
                            "Fixed exterior density above the surface. It must match Postprocess/Dynamic topography/Density above.");
          prm.declare_entry("CMB density below", "9900", Patterns::Double(0.0),
                            "Fixed exterior density below the CMB. It must match Postprocess/Dynamic topography/Density below.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    SelfGravitatingGeoidObjective<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Adjoint");
      {
        prm.enter_subsection("Self-gravitating geoid");
        {
          minimum_degree = prm.get_integer("Minimum spherical harmonic degree");
          maximum_degree = prm.get_integer("Maximum spherical harmonic degree");
          density_above = prm.get_double("Surface density above");
          density_below = prm.get_double("CMB density below");
          AssertThrow(minimum_degree >= 2 && minimum_degree <= maximum_degree,
                      ExcMessage("The self-gravitating geoid requires 2 <= minimum degree <= maximum degree."));
          const Postprocess::SphericalHarmonicCoefficients layout(minimum_degree,
                                                                  maximum_degree);
          const unsigned int count = layout.size();
          const std::array<const char *, 4> names = {{"surface topography", "CMB topography",
              "surface geoid", "CMB geoid"
            }
          };
          for (unsigned int field = 0; field < names.size(); ++field)
            {
              const std::vector<double> observed_cosine =
                Utilities::string_to_double(Utilities::split_string_list(
                                              prm.get("Observed " + std::string(names[field])
                                                      + " cosine coefficients")));
              const std::vector<double> observed_sine =
                Utilities::string_to_double(Utilities::split_string_list(
                                              prm.get("Observed " + std::string(names[field])
                                                      + " sine coefficients")));
              AssertThrow(observed_cosine.size() == count
                          && observed_sine.size() == count,
                          ExcMessage("Every coupled self-gravity observed coefficient list must contain exactly "
                                     + std::to_string(count) + " entries."));
              observed_coefficients[field].reinit(minimum_degree, maximum_degree);
              observed_coefficients[field].set_coefficients(observed_cosine,
                                                            observed_sine);
            }
          const std::string weight_string = prm.get("Coefficient weights");
          weights = weight_string.empty() ? std::vector<double>(count, 1.0)
                    : Utilities::string_to_double(Utilities::split_string_list(weight_string));
          AssertThrow(weights.size() == count,
                      ExcMessage("The coupled self-gravity coefficient weight list has an invalid length."));
          for (const double weight : weights)
            AssertThrow(weight >= 0.0, ExcMessage("Coupled self-gravity coefficient weights must be nonnegative."));
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}

namespace aspect
{
  namespace Adjoint
  {
    ASPECT_REGISTER_ADJOINT_OBJECTIVE(SelfGravitatingGeoidObjective,
                                      "self-gravitating geoid",
                                      "Coupled self-gravitating geoid objective with top/CMB CBF stress, "
                                      "density-moment, physical Stokes, and boundary-mean density matrix terms. "
                                      "It excludes gravity, geometry, and external-density matrix derivatives.")
  }
}
