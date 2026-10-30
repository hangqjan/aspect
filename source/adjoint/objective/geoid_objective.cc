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

#include <aspect/adjoint/objective/geoid_objective.h>
#include <aspect/adjoint/operators/dynamic_topography_boundary_operator.h>
#include <aspect/adjoint/core/kernel_calculator.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/postprocess/boundary_densities.h>
#include <aspect/postprocess/dynamic_topography.h>
#include <aspect/simulator.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <cmath>
#include <limits>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    void GeoidObjective<dim>::initialize_simulator(const Simulator<dim> &simulator)
    {
      Objective::Interface<dim>::initialize_simulator(simulator);
      observation.initialize_simulator(simulator);
    }



    template <int dim>
    std::pair<double, double>
    GeoidObjective<dim>::operator_context() const
    {
      AssertThrow(dim == 3,
                  ExcMessage("The non-self-gravitating geoid objective currently only works in 3D."));
      const auto *geometry_model = dynamic_cast<const GeometryModel::SphericalShell<dim> *>(
                                     &this->get_geometry_model());
      AssertThrow(geometry_model != nullptr,
                  ExcMessage("The non-self-gravitating geoid objective requires a spherical shell geometry model."));
      const double outer_radius = geometry_model->outer_radius();
      Point<dim> surface_point;
      surface_point[0] = outer_radius;
      const double surface_gravity =
        this->get_gravity_model().gravity_vector(surface_point).norm();
      AssertThrow(surface_gravity > std::numeric_limits<double>::min(),
                  ExcMessage("The non-self-gravitating geoid objective requires nonzero surface gravity."));
      return {outer_radius, surface_gravity};
    }



    template <int dim>
    Postprocess::SphericalHarmonicCoefficients
    GeoidObjective<dim>::forward_coefficients() const
    {
      const std::pair<double, double> context = operator_context();
      // The conventional model is an additive observation operator
      //   N_lm = N_lm^rho + I_s N_lm^{h_s} + I_c N_lm^{h_c}.
      // Compute each physical contribution in the same invariant harmonic
      // layout and add coefficient-wise. No self-consistent feedback of N onto
      // boundary deflection is present in this conventional formulation.
      Postprocess::SphericalHarmonicCoefficients coefficients =
        Postprocess::compute_density_geoid_harmonics(*this, minimum_degree,
                                                     maximum_degree, context.first,
                                                     context.second);
      if (include_surface_topography)
        {
          // A surface height coefficient carries sheet density
          // Delta rho_s h_lm. The harmonic helper applies the degree-dependent
          // Newtonian-potential factor and divides by surface gravity to
          // convert potential to geoid height.
          const auto &dynamic_topography = this->get_postprocess_manager().template
                                           get_matching_active_plugin<Postprocess::DynamicTopography<dim>>();
          const auto &boundary_densities = this->get_postprocess_manager().template
                                           get_matching_active_plugin<Postprocess::BoundaryDensities<dim>>();
          const Postprocess::SphericalHarmonicCoefficients surface_coefficients =
            Postprocess::compute_surface_topography_geoid_harmonics(
              *this, dynamic_topography.topography_vector(), minimum_degree, maximum_degree,
              context.first, context.second, boundary_densities.density_at_top() - density_above);
          for (unsigned int index = 0; index < coefficients.cosine_coefficients().size(); ++index)
            {
              coefficients.cosine(index) += surface_coefficients.cosine(index);
              coefficients.sine(index) += surface_coefficients.sine(index);
            }
        }
      if (include_cmb_topography)
        {
          // CMB relief is radially continued from r_c to r_s and uses the CMB
          // density jump with the orientation encoded by the shared helper.
          // Keeping this contribution separate makes its transpose sign and
          // radius power independently testable.
          const auto &dynamic_topography = this->get_postprocess_manager().template
                                           get_matching_active_plugin<Postprocess::DynamicTopography<dim>>();
          const auto &boundary_densities = this->get_postprocess_manager().template
                                           get_matching_active_plugin<Postprocess::BoundaryDensities<dim>>();
          const auto *geometry_model = dynamic_cast<const GeometryModel::SphericalShell<dim> *>(
                                         &this->get_geometry_model());
          AssertThrow(geometry_model != nullptr, ExcInternalError());
          const Postprocess::SphericalHarmonicCoefficients cmb_coefficients =
            Postprocess::compute_cmb_topography_geoid_harmonics(
              *this, dynamic_topography.topography_vector(), minimum_degree, maximum_degree,
              context.first, geometry_model->inner_radius(), context.second,
              density_below - boundary_densities.density_at_bottom());
          for (unsigned int index = 0; index < coefficients.cosine_coefficients().size(); ++index)
            {
              coefficients.cosine(index) += cmb_coefficients.cosine(index);
              coefficients.sine(index) += cmb_coefficients.sine(index);
            }
        }
      return coefficients;
    }



    template <int dim>
    void
    GeoidObjective<dim>::assemble_surface_topography_seed(
      LinearAlgebra::BlockVector &topography_adjoint, double &top_area,
      double &surface_pressure_weight) const
    {
      const auto &boundary_densities = this->get_postprocess_manager().template
                                       get_matching_active_plugin<Postprocess::BoundaryDensities<dim>>();
      const std::pair<double, double> context = operator_context();
      const Postprocess::SphericalHarmonicCoefficients topography_coefficients =
        Postprocess::compute_surface_topography_geoid_transpose(
          coefficient_seed(), context.first, context.second,
          boundary_densities.density_at_top() - density_above);
      // coefficient_seed is dJ/dN_lm. The harmonic transpose returns the
      // scalar field seed dJ/dh(theta,phi). We next assemble the FE dual vector
      //   bar h_i = integral_Gamma bar h phi_i dS,
      // using exactly the quadrature used for forward harmonic analysis.
      const types::boundary_id top_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      // The topography dual vector must use the forward harmonic analysis
      // quadrature, independently of the quadrature used internally by CBF.
      const unsigned int quadrature_degree =
        this->introspection().polynomial_degree.temperature;
      const QGauss<dim-1> quadrature_formula_face(quadrature_degree);
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      FEFaceValues<dim> fe_face_values(this->get_mapping(), this->get_fe(), quadrature_formula_face,
                                       update_values | update_gradients | update_quadrature_points |
                                       update_normal_vectors | update_JxW_values);
      Vector<double> local_topography_adjoint(dofs_per_cell);
      topography_adjoint = 0.0;
      top_area = 0.0;
      surface_pressure_weight = 0.0;
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int face : cell->face_indices())
            if (cell->at_boundary(face) && cell->face(face)->boundary_id() == top_boundary_id)
              {
                fe_face_values.reinit(cell, face);
                local_topography_adjoint = 0.0;
                MaterialModel::MaterialModelInputs<dim> in(fe_face_values, cell,
                                                           this->introspection(), this->get_solution());
                MaterialModel::MaterialModelOutputs<dim> out(fe_face_values.n_quadrature_points,
                                                             this->n_compositional_fields());
                in.requested_properties = MaterialModel::MaterialProperties::density;
                this->get_material_model().evaluate(in, out);
                for (unsigned int q = 0; q < quadrature_formula_face.size(); ++q)
                  {
                    const std::array<double, dim> spherical_position =
                      Utilities::Coordinates::cartesian_to_spherical_coordinates(
                        fe_face_values.quadrature_point(q));
                    const double seed_value = topography_coefficients.evaluate(spherical_position[2],
                                                                               spherical_position[1]);
                    const Tensor<1, dim> gravity = this->get_gravity_model().gravity_vector(
                                                     fe_face_values.quadrature_point(q));
                    const double gravity_norm = gravity.norm();
                    const double density_contrast = out.densities[q] - density_above;
                    AssertThrow(std::abs(density_contrast) > std::numeric_limits<double>::min(),
                                ExcMessage("GeoidObjective encountered zero surface density contrast."));
                    AssertThrow(gravity_norm > std::numeric_limits<double>::min(),
                                ExcMessage("GeoidObjective encountered zero surface gravity."));
                    top_area += fe_face_values.JxW(q);
                    // The CBF topography definition includes removal of mean
                    // pressure and division by Delta rho |g|. Accumulate the
                    // scalar seed of that boundary mean alongside the FE
                    // topography seed so the boundary operator can transpose
                    // both dependencies in one pass.
                    surface_pressure_weight += seed_value / (density_contrast * gravity_norm)
                                               * fe_face_values.JxW(q);
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      local_topography_adjoint(i) += seed_value
                                                     * fe_face_values[this->introspection().extractors.temperature].value(i, q)
                                                     * fe_face_values.JxW(q);
                  }
                cell->distribute_local_to_global(local_topography_adjoint, topography_adjoint);
              }
      topography_adjoint.compress(VectorOperation::add);
      // Face ownership partitions both the dual-vector assembly and the two
      // scalar boundary integrals. The vector compress and MPI reductions
      // therefore complete different pieces of the same global transpose.
      top_area = Utilities::MPI::sum(top_area, this->get_mpi_communicator());
      surface_pressure_weight = Utilities::MPI::sum(surface_pressure_weight, this->get_mpi_communicator());
    }



    template <int dim>
    void
    GeoidObjective<dim>::assemble_cmb_topography_seed(
      LinearAlgebra::BlockVector &topography_adjoint, double &bottom_area,
      double &bottom_pressure_weight) const
    {
      // This is the CMB analogue of assemble_surface_topography_seed(). The
      // harmonic transpose already contains radial continuation; this routine
      // only maps its physical boundary-field seed into the discrete FE dual
      // vector and records the pressure-mean scalar seed.
      const auto &boundary_densities = this->get_postprocess_manager().template
                                       get_matching_active_plugin<Postprocess::BoundaryDensities<dim>>();
      const auto *geometry_model = dynamic_cast<const GeometryModel::SphericalShell<dim> *>(
                                     &this->get_geometry_model());
      AssertThrow(geometry_model != nullptr, ExcInternalError());
      const std::pair<double, double> context = operator_context();
      const Postprocess::SphericalHarmonicCoefficients topography_coefficients =
        Postprocess::compute_cmb_topography_geoid_transpose(
          coefficient_seed(), context.first, geometry_model->inner_radius(), context.second,
          density_below - boundary_densities.density_at_bottom());
      const types::boundary_id bottom_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");
      const unsigned int quadrature_degree =
        this->introspection().polynomial_degree.temperature;
      const QGauss<dim-1> quadrature_formula_face(quadrature_degree);
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      FEFaceValues<dim> fe_face_values(this->get_mapping(), this->get_fe(), quadrature_formula_face,
                                       update_values | update_gradients | update_quadrature_points |
                                       update_normal_vectors | update_JxW_values);
      Vector<double> local_topography_adjoint(dofs_per_cell);
      topography_adjoint = 0.0;
      bottom_area = 0.0;
      bottom_pressure_weight = 0.0;
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int face : cell->face_indices())
            if (cell->at_boundary(face) && cell->face(face)->boundary_id() == bottom_boundary_id)
              {
                fe_face_values.reinit(cell, face);
                local_topography_adjoint = 0.0;
                MaterialModel::MaterialModelInputs<dim> in(fe_face_values, cell,
                                                           this->introspection(), this->get_solution());
                MaterialModel::MaterialModelOutputs<dim> out(fe_face_values.n_quadrature_points,
                                                             this->n_compositional_fields());
                in.requested_properties = MaterialModel::MaterialProperties::density;
                this->get_material_model().evaluate(in, out);
                for (unsigned int q = 0; q < quadrature_formula_face.size(); ++q)
                  {
                    const std::array<double, dim> spherical_position =
                      Utilities::Coordinates::cartesian_to_spherical_coordinates(
                        fe_face_values.quadrature_point(q));
                    const double seed_value = topography_coefficients.evaluate(spherical_position[2],
                                                                               spherical_position[1]);
                    const double gravity_norm = this->get_gravity_model().gravity_vector(
                                                  fe_face_values.quadrature_point(q)).norm();
                    const double density_contrast = out.densities[q] - density_below;
                    AssertThrow(std::abs(density_contrast) > std::numeric_limits<double>::min(),
                                ExcMessage("GeoidObjective encountered zero CMB density contrast."));
                    bottom_area += fe_face_values.JxW(q);
                    bottom_pressure_weight += seed_value / (density_contrast * gravity_norm)
                                              * fe_face_values.JxW(q);
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      local_topography_adjoint(i) += seed_value
                                                     * fe_face_values[this->introspection().extractors.temperature].value(i, q)
                                                     * fe_face_values.JxW(q);
                  }
                cell->distribute_local_to_global(local_topography_adjoint, topography_adjoint);
              }
      topography_adjoint.compress(VectorOperation::add);
      // bottom_area and bottom_pressure_weight are local face integrals until
      // this collective reduction. They must be identical on all ranks before
      // the subsequent state-transpose assembly.
      bottom_area = Utilities::MPI::sum(bottom_area, this->get_mpi_communicator());
      bottom_pressure_weight = Utilities::MPI::sum(bottom_pressure_weight, this->get_mpi_communicator());
    }



    template <int dim>
    Postprocess::SphericalHarmonicCoefficients
    GeoidObjective<dim>::coefficient_seed() const
    {
      Postprocess::SphericalHarmonicCoefficients coefficients = forward_coefficients();
      if (use_file_observation)
        {
          observation.initialize(minimum_degree, maximum_degree);
          observed_coefficients = observation.coefficients();
        }
      // For
      //   J = 1/2 sum_lm w_lm[(C-Cobs)^2+(S-Sobs)^2],
      // the spectral covector is w_lm times the coefficient residual. Reusing
      // the coefficient container preserves the exact (l,m) layout throughout
      // every downstream transpose.
      for (unsigned int index = 0; index < coefficients.cosine_coefficients().size(); ++index)
        {
          coefficients.cosine(index) = weights[index]
                                       * (coefficients.cosine(index)
                                          - observed_coefficients.cosine(index));
          coefficients.sine(index) = weights[index]
                                     * (coefficients.sine(index)
                                        - observed_coefficients.sine(index));
        }
      return coefficients;
    }



    template <int dim>
    double
    GeoidObjective<dim>::evaluate() const
    {
      const Postprocess::SphericalHarmonicCoefficients coefficients = forward_coefficients();
      if (use_file_observation)
        {
          observation.initialize(minimum_degree, maximum_degree);
          observed_coefficients = observation.coefficients();
        }
      double objective = 0.0;
      // Cosine and sine coefficients are independent real degrees of freedom
      // under this normalization. Accumulate their weighted Euclidean
      // misfit; the factor 1/2 cancels the factor 2 in its derivative.
      for (unsigned int index = 0; index < coefficients.cosine_coefficients().size(); ++index)
        {
          const double cosine_residual =
            coefficients.cosine(index) - observed_coefficients.cosine(index);
          const double sine_residual =
            coefficients.sine(index) - observed_coefficients.sine(index);
          objective += 0.5 * weights[index]
                       * (cosine_residual * cosine_residual + sine_residual * sine_residual);
        }
      return objective;
    }



    template <int dim>
    void
    GeoidObjective<dim>::assemble_adjoint_right_hand_side(
      LinearAlgebra::BlockVector &rhs) const
    {
      if (include_surface_topography)
        {
          // Reverse N <- h_s first, obtaining bar h_s, and then reverse the
          // CBF map h_s <- (u,p,rho,eta). The boundary operator adds only the
          // state part here; direct rho/eta terms are accumulated separately
          // in add_direct_kernel_contributions().
          LinearAlgebra::BlockVector topography_adjoint(
            this->introspection().index_sets.system_partitioning,
            this->introspection().index_sets.system_relevant_partitioning,
            this->get_mpi_communicator());
          double top_area = 0.0;
          double surface_pressure_weight = 0.0;
          assemble_surface_topography_seed(topography_adjoint, top_area, surface_pressure_weight);
          DynamicTopographyBoundaryOperator<dim>(this->get_simulator()).add_state_transpose(
            topography_adjoint, top_area, surface_pressure_weight, density_above, rhs,
            Postprocess::DynamicTopographyBoundary::surface);
        }
      if (include_cmb_topography)
        {
          // The surface and CMB chains share the same implementation but use
          // distinct density contrasts, normals, boundary measures, and
          // pressure-mean corrections.
          LinearAlgebra::BlockVector topography_adjoint(
            this->introspection().index_sets.system_partitioning,
            this->introspection().index_sets.system_relevant_partitioning,
            this->get_mpi_communicator());
          double bottom_area = 0.0;
          double bottom_pressure_weight = 0.0;
          assemble_cmb_topography_seed(topography_adjoint, bottom_area, bottom_pressure_weight);
          DynamicTopographyBoundaryOperator<dim>(this->get_simulator()).add_state_transpose(
            topography_adjoint, bottom_area, bottom_pressure_weight, density_below, rhs,
            Postprocess::DynamicTopographyBoundary::cmb);
        }
      // J depends only on the explicit material-density observation operator.
      // requires_physical_stokes_adjoint() consequently lets the manager
      // supply an exactly zero state without a physical Stokes solve.
      (void)rhs;
    }



    template <int dim>
    bool
    GeoidObjective<dim>::requires_physical_stokes_adjoint() const
    {
      return include_surface_topography || include_cmb_topography;
    }



    template <int dim>
    std::set<std::string>
    GeoidObjective<dim>::required_forward_postprocessors() const
    {
      if (include_surface_topography || include_cmb_topography)
        return {"dynamic topography", "boundary densities"};
      return {};
    }



    template <int dim>
    void
    GeoidObjective<dim>::add_direct_kernel_contributions(
      const std::string &objective_name, KernelRepository<dim> &kernels) const
    {
      const std::pair<double, double> context = operator_context();
      const Postprocess::SphericalHarmonicCoefficients seed = coefficient_seed();
      const Vector<double> density_kernel =
        Postprocess::compute_density_geoid_transpose(*this, seed, context.first, context.second);
      // The volume-density observation is explicit: transpose its solid-
      // harmonic projection directly into a cellwise density kernel. It does
      // not require solving the physical Stokes adjoint.
      kernels.add_contribution({objective_name, "mantle-density geoid harmonic",
                                PhysicalProperty::density
                               }, density_kernel);
      if (include_surface_topography)
        {
          // Add the denominator and CBF-residual material derivatives of the
          // surface topography chain. These are deliberately separate from
          // the state RHS assembled above to avoid double-counting eta/rho.
          const auto &dynamic_topography = this->get_postprocess_manager().template
                                           get_matching_active_plugin<Postprocess::DynamicTopography<dim>>();
          LinearAlgebra::BlockVector topography_adjoint(
            this->introspection().index_sets.system_partitioning,
            this->introspection().index_sets.system_relevant_partitioning,
            this->get_mpi_communicator());
          double top_area = 0.0;
          double surface_pressure_weight = 0.0;
          assemble_surface_topography_seed(topography_adjoint, top_area, surface_pressure_weight);
          DynamicTopographyBoundaryOperator<dim>(this->get_simulator()).add_direct_kernel_contributions(
            topography_adjoint, dynamic_topography.topography_vector(), density_above,
            objective_name, kernels, Postprocess::DynamicTopographyBoundary::surface);
        }
      if (include_cmb_topography)
        {
          const auto &dynamic_topography = this->get_postprocess_manager().template
                                           get_matching_active_plugin<Postprocess::DynamicTopography<dim>>();
          LinearAlgebra::BlockVector topography_adjoint(
            this->introspection().index_sets.system_partitioning,
            this->introspection().index_sets.system_relevant_partitioning,
            this->get_mpi_communicator());
          double bottom_area = 0.0;
          double bottom_pressure_weight = 0.0;
          assemble_cmb_topography_seed(topography_adjoint, bottom_area, bottom_pressure_weight);
          DynamicTopographyBoundaryOperator<dim>(this->get_simulator()).add_direct_kernel_contributions(
            topography_adjoint, dynamic_topography.topography_vector(), density_below,
            objective_name, kernels, Postprocess::DynamicTopographyBoundary::cmb);
        }
    }



    template <int dim>
    void
    GeoidObjective<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Adjoint");
      {
        prm.enter_subsection("Geoid");
        {
          prm.declare_entry("Minimum spherical harmonic degree", "0", Patterns::Integer(0),
                            "Minimum degree l_min in the shared degree-major geoid coefficient layout.");
          prm.declare_entry("Maximum spherical harmonic degree", "0", Patterns::Integer(0),
                            "Maximum degree l_max in the shared degree-major geoid coefficient layout.");
          prm.declare_entry("Observed cosine coefficients", "", Patterns::Anything(),
                            "Comma-separated observed cosine coefficients in shared degree-major order.");
          prm.declare_entry("Observed sine coefficients", "", Patterns::Anything(),
                            "Comma-separated observed sine coefficients in shared degree-major order.");
          prm.declare_entry("Observation source", "inline", Patterns::Selection("inline|file"),
                            "Use inline coefficient lists or a cached coefficient-file observation.");
          prm.enter_subsection("Observation");
          {
            Observation::SphericalHarmonicObservation<dim>::declare_parameters(prm);
          }
          prm.leave_subsection();
          prm.declare_entry("Coefficient weights", "", Patterns::Anything(),
                            "Optional comma-separated nonnegative diagonal weights. One weight applies to both "
                            "cosine and sine components of each degree/order; an empty list means all weights are one.");
          prm.declare_entry("Include surface dynamic topography contribution", "false", Patterns::Bool(),
                            "Include the surface dynamic-topography geoid contribution through the validated top CBF transpose.");
          prm.declare_entry("Surface density above", "0", Patterns::Double(0.0),
                            "Density above the surface used by the dynamic-topography and surface-geoid operators.");
          prm.declare_entry("Include CMB dynamic topography contribution", "false", Patterns::Bool(),
                            "Include the CMB dynamic-topography geoid contribution through the bottom CBF transpose.");
          prm.declare_entry("CMB density below", "9900", Patterns::Double(0.0),
                            "Density below the CMB used by the CMB dynamic-topography and geoid operators.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    GeoidObjective<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Adjoint");
      {
        prm.enter_subsection("Geoid");
        {
          minimum_degree = prm.get_integer("Minimum spherical harmonic degree");
          maximum_degree = prm.get_integer("Maximum spherical harmonic degree");
          use_file_observation = prm.get("Observation source") == "file";
          include_surface_topography = prm.get_bool("Include surface dynamic topography contribution");
          density_above = prm.get_double("Surface density above");
          include_cmb_topography = prm.get_bool("Include CMB dynamic topography contribution");
          density_below = prm.get_double("CMB density below");
          AssertThrow(minimum_degree <= maximum_degree,
                      ExcMessage("The minimum geoid degree must not exceed the maximum degree."));

          const Postprocess::SphericalHarmonicCoefficients layout(minimum_degree,
                                                                  maximum_degree);
          const unsigned int coefficient_count = layout.size();

          if (use_file_observation)
            {
              prm.enter_subsection("Observation");
              {
                observation.parse_parameters(prm);
              }
              prm.leave_subsection();
              observed_coefficients.reinit(minimum_degree, maximum_degree);
            }
          else
            {
              const std::vector<double> observed_cosine = Utilities::string_to_double(
                                                            Utilities::split_string_list(
                                                              prm.get("Observed cosine coefficients")));
              const std::vector<double> observed_sine = Utilities::string_to_double(
                                                          Utilities::split_string_list(
                                                            prm.get("Observed sine coefficients")));
              AssertThrow(observed_cosine.size() == coefficient_count
                          && observed_sine.size() == coefficient_count,
                          ExcMessage("Observed geoid cosine and sine coefficient lists must each contain exactly "
                                     + std::to_string(coefficient_count) + " entries."));
              observed_coefficients.reinit(minimum_degree, maximum_degree);
              observed_coefficients.set_coefficients(observed_cosine, observed_sine);
            }

          const std::string weight_string = prm.get("Coefficient weights");
          if (weight_string.empty())
            weights.assign(coefficient_count, 1.0);
          else
            weights = Utilities::string_to_double(Utilities::split_string_list(weight_string));
          AssertThrow(weights.size() == coefficient_count,
                      ExcMessage("The geoid coefficient weight list must contain exactly "
                                 + std::to_string(coefficient_count) + " entries."));
          for (const double weight : weights)
            AssertThrow(weight >= 0.0,
                        ExcMessage("Geoid coefficient weights must be nonnegative."));
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
    ASPECT_REGISTER_ADJOINT_OBJECTIVE(GeoidObjective,
                                      "geoid",
                                      "Coefficient-space non-self-gravitating geoid objective. It evaluates one half "
                                      "of the diagonally weighted spherical-harmonic coefficient misfit for mantle "
                                      "density and optional surface/CMB dynamic topography. Enabled topography "
                                      "branches share one physical Stokes adjoint; the mantle-density contribution "
                                      "is the direct transpose of the shared harmonic operator.")
  }
}
