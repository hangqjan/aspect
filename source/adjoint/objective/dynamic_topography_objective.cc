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

#include <aspect/adjoint/objective/dynamic_topography_objective.h>
#include <aspect/adjoint/operators/dynamic_topography_boundary_operator.h>
#include <aspect/geometry_model/interface.h>
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
    DynamicTopographyObjective<dim>::DynamicTopographyObjective ()
      : target_topography_model ("none"),
        target_topography_function (1),
        target_coordinate_system (Utilities::Coordinates::spherical)
    {}

    template <int dim>
    void DynamicTopographyObjective<dim>::initialize_simulator(const Simulator<dim> &simulator)
    {
      Objective::Interface<dim>::initialize_simulator(simulator);
      observation.initialize_simulator(simulator);
      observation.set_native_quantity(Observation::SurfaceFieldObservation<dim>::NativeQuantity::dynamic_topography);
    }

    template <int dim>
    double DynamicTopographyObjective<dim>::target_topography (const Point<dim> &position) const
    {
      if (target_topography_model == "file")
        return observation.value(position);
      if (target_topography_model == "function")
        {
          const Utilities::NaturalCoordinate<dim> point =
            this->get_geometry_model().cartesian_to_other_coordinates(position, target_coordinate_system);
          return target_topography_function.value(
                   Utilities::convert_array_to_point<dim>(point.get_coordinates()));
        }
      AssertThrow(false, ExcMessage("Dynamic topography objective requires a target topography model. "
                                    "Set Adjoint/Dynamic topography/Target topography model."));
      return 0.0;
    }

    template <int dim>
    std::set<std::string> DynamicTopographyObjective<dim>::required_forward_postprocessors() const
    {
      return {"dynamic topography"};
    }

    template <int dim>
    double DynamicTopographyObjective<dim>::evaluate() const
    {
      if (target_topography_model == "file")
        observation.initialize();
      const auto &dynamic_topography = this->get_postprocess_manager().template
                                       get_matching_active_plugin<Postprocess::DynamicTopography<dim>>();
      AssertThrow(dynamic_topography.topography_vector().size() != 0,
                  ExcMessage("DynamicTopographyObjective requires the dynamic topography "
                             "postprocessor to have run before objective evaluation."));
      const types::boundary_id top_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      const unsigned int quadrature_degree = this->get_fe().base_element(
                                               this->introspection().base_elements.velocities).degree + 1;
      const QGauss<dim-1> quadrature_formula_face(quadrature_degree);
      FEFaceValues<dim> fe_face_values(this->get_mapping(), this->get_fe(), quadrature_formula_face,
                                       update_values | update_quadrature_points | update_JxW_values);
      double local_objective = 0.0;
      std::vector<double> topo_values(quadrature_formula_face.size());

      // Evaluate the boundary L2 misfit
      //   J(h)=1/2 integral_Gamma_top [h(x)-h_obs(x)]^2 dA.
      // Only locally owned faces contribute, so the final MPI sum is the
      // distributed assembly of this scalar surface integral.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int f : cell->face_indices())
            if (cell->at_boundary(f) && cell->face(f)->boundary_id() == top_boundary_id)
              {
                fe_face_values.reinit(cell, f);
                fe_face_values[this->introspection().extractors.temperature].get_function_values(
                  dynamic_topography.topography_vector(), topo_values);
                for (unsigned int q = 0; q < quadrature_formula_face.size(); ++q)
                  {
                    const double residual = topo_values[q] - target_topography(fe_face_values.quadrature_point(q));
                    local_objective += 0.5 * residual * residual * fe_face_values.JxW(q);
                  }
              }
      return Utilities::MPI::sum(local_objective, this->get_mpi_communicator());
    }

    template <int dim>
    void DynamicTopographyObjective<dim>::assemble_surface_topography_seed(
      LinearAlgebra::BlockVector &topography_adjoint, double &top_area,
      double &surface_pressure_weight) const
    {
      if (target_topography_model == "file")
        observation.initialize();
      const auto &dynamic_topography = this->get_postprocess_manager().template
                                       get_matching_active_plugin<Postprocess::DynamicTopography<dim>>();
      AssertThrow(dynamic_topography.topography_vector().size() != 0,
                  ExcMessage("DynamicTopographyObjective requires the dynamic topography "
                             "postprocessor to have run before assembling the adjoint RHS."));
      const types::boundary_id top_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      const unsigned int quadrature_degree = this->get_fe().base_element(
                                               this->introspection().base_elements.velocities).degree + 1;
      const QGauss<dim-1> quadrature_formula_face(quadrature_degree);
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      FEFaceValues<dim> fe_face_values(this->get_mapping(), this->get_fe(), quadrature_formula_face,
                                       update_values | update_gradients | update_quadrature_points |
                                       update_normal_vectors | update_JxW_values);
      Vector<double> local_topography_adjoint(dofs_per_cell);
      std::vector<double> topo_values(quadrature_formula_face.size());
      topography_adjoint = 0.0;
      top_area = 0.0;
      surface_pressure_weight = 0.0;

      // The functional derivative with respect to the CBF topography is
      // bar(h)=h-h_obs. Its FE dual vector has entries
      //   b_i^h=int_Gamma bar(h) phi_i dA.
      //
      // CBF computes h from reduced normal stress divided by
      // Delta_rho |g|. The scalar pressure_weight accumulated here is the
      // companion seed required to transpose the mean-pressure correction in
      // the shared boundary operator. It uses the same quadrature density and
      // gravity values as the forward CBF evaluation.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int f : cell->face_indices())
            if (cell->at_boundary(f) && cell->face(f)->boundary_id() == top_boundary_id)
              {
                fe_face_values.reinit(cell, f);
                local_topography_adjoint = 0.0;
                MaterialModel::MaterialModelInputs<dim> in(fe_face_values, cell,
                                                           this->introspection(), this->get_solution());
                MaterialModel::MaterialModelOutputs<dim> out(fe_face_values.n_quadrature_points,
                                                             this->n_compositional_fields());
                in.requested_properties = MaterialModel::MaterialProperties::density;
                this->get_material_model().evaluate(in, out);
                fe_face_values[this->introspection().extractors.temperature].get_function_values(
                  dynamic_topography.topography_vector(), topo_values);
                for (unsigned int q = 0; q < quadrature_formula_face.size(); ++q)
                  {
                    const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector(
                                                    fe_face_values.quadrature_point(q));
                    const double gravity_norm = gravity.norm();
                    const double density_contrast = out.densities[q] - dynamic_topography.get_density_above();
                    AssertThrow(std::abs(density_contrast) > std::numeric_limits<double>::min(),
                                ExcMessage("DynamicTopographyObjective encountered zero density contrast at the top boundary."));
                    AssertThrow(gravity_norm > std::numeric_limits<double>::min(),
                                ExcMessage("DynamicTopographyObjective encountered zero gravity magnitude at the top boundary."));
                    top_area += fe_face_values.JxW(q);
                    const double residual = topo_values[q] - target_topography(fe_face_values.quadrature_point(q));
                    surface_pressure_weight += residual / (density_contrast * gravity_norm) * fe_face_values.JxW(q);
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      local_topography_adjoint(i) += residual *
                                                     fe_face_values[this->introspection().extractors.temperature].value(i, q) *
                                                     fe_face_values.JxW(q);
                  }
                // This vector represents a boundary functional, not a primal
                // field interpolation. Direct local-to-global addition is
                // therefore followed by compress(add) across MPI ranks.
                cell->distribute_local_to_global(local_topography_adjoint, topography_adjoint);
              }
      topography_adjoint.compress(VectorOperation::add);
      top_area = Utilities::MPI::sum(top_area, this->get_mpi_communicator());
      surface_pressure_weight = Utilities::MPI::sum(surface_pressure_weight, this->get_mpi_communicator());
      AssertThrow(top_area > std::numeric_limits<double>::min(),
                  ExcMessage("DynamicTopographyObjective found zero top boundary area."));
    }

    template <int dim>
    void DynamicTopographyObjective<dim>::assemble_adjoint_right_hand_side(
      LinearAlgebra::BlockVector &rhs) const
    {
      const auto &dynamic_topography = this->get_postprocess_manager().template
                                       get_matching_active_plugin<Postprocess::DynamicTopography<dim>>();
      LinearAlgebra::BlockVector topography_adjoint(this->introspection().index_sets.system_partitioning,
                                                    this->introspection().index_sets.system_relevant_partitioning,
                                                    this->get_mpi_communicator());
      double top_area = 0.0, surface_pressure_weight = 0.0;
      assemble_surface_topography_seed(topography_adjoint, top_area, surface_pressure_weight);

      // Apply (partial h/partial [u,p])^T bar(h). This contributes only the
      // state-dependent path of the objective; density and viscosity
      // derivatives of the same CBF expression are handled as direct kernels
      // below to avoid mixing state and parameter derivatives.
      DynamicTopographyBoundaryOperator<dim>(this->get_simulator()).add_state_transpose(
        topography_adjoint, top_area, surface_pressure_weight,
        dynamic_topography.get_density_above(), rhs,
        Postprocess::DynamicTopographyBoundary::surface);
    }

    template <int dim>
    void DynamicTopographyObjective<dim>::add_direct_kernel_contributions(
      const std::string &objective_name, KernelRepository<dim> &kernels) const
    {
      const auto &dynamic_topography = this->get_postprocess_manager().template
                                       get_matching_active_plugin<Postprocess::DynamicTopography<dim>>();
      LinearAlgebra::BlockVector topography_adjoint(this->introspection().index_sets.system_partitioning,
                                                    this->introspection().index_sets.system_relevant_partitioning,
                                                    this->get_mpi_communicator());
      double top_area = 0.0, surface_pressure_weight = 0.0;
      assemble_surface_topography_seed(topography_adjoint, top_area, surface_pressure_weight);

      // Holding the Stokes solution fixed, transpose h=sigma_nn/(Delta_rho|g|)
      // through Delta_rho and the viscosity-dependent reduced stress. These
      // terms complement the implicit-state kernel produced by the Stokes
      // adjoint and are stored under the same objective name.
      DynamicTopographyBoundaryOperator<dim>(this->get_simulator()).add_direct_kernel_contributions(
        topography_adjoint, dynamic_topography.topography_vector(), dynamic_topography.get_density_above(),
        objective_name, kernels, Postprocess::DynamicTopographyBoundary::surface);
    }

    template <int dim>
    void DynamicTopographyObjective<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Adjoint");
      {
        prm.enter_subsection ("Dynamic topography");
        {
          prm.declare_entry ("Target topography model", "none", Patterns::Selection ("none|function|file"),
                             "The model used to prescribe the target dynamic topography. The dynamic topography "
                             "objective evaluates one half of the squared misfit between the computed and target "
                             "topography. A target topography model is required when this objective is used.");
          prm.enter_subsection ("Function");
          {
            prm.declare_entry ("Coordinate system", "spherical", Patterns::Selection ("cartesian|spherical|depth"),
                               "A selection that determines the assumed coordinate system for the target function "
                               "variables. Allowed values are `cartesian', `spherical', and `depth'. `spherical' "
                               "coordinates are interpreted as r,phi or r,phi,theta in 2d/3d respectively with theta "
                               "being the polar angle.");
            Functions::ParsedFunction<dim>::declare_parameters (prm, 1);
            prm.declare_entry ("Variable names", dim == 2 ? "r,phi" : "r,phi,theta", Patterns::Anything(),
                               "Names of the coordinates used in the target topography function.");
            prm.declare_entry ("Function expression", "", Patterns::Anything(), "Target dynamic topography in meters.");
          }
          prm.leave_subsection();
          prm.enter_subsection("Observation");
          {
            Observation::SurfaceFieldObservation<dim>::declare_parameters(prm);
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void DynamicTopographyObjective<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Adjoint");
      {
        prm.enter_subsection ("Dynamic topography");
        {
          target_topography_model = prm.get ("Target topography model");
          AssertThrow(target_topography_model != "none",
                      ExcMessage("Dynamic topography objective requires a target topography model. "
                                 "Set Adjoint/Dynamic topography/Target topography model."));
          if (target_topography_model == "function")
            {
              prm.enter_subsection ("Function");
              {
                target_coordinate_system = Utilities::Coordinates::string_to_coordinate_system(
                                             prm.get ("Coordinate system"));
              }
              try
                {
                  target_topography_function.parse_parameters (prm);
                }
              catch (...)
                {
                  std::cerr << "ERROR: FunctionParser failed to parse\n\t'Adjoint.Dynamic topography.Function'\n"
                            << "with expression\n\t'" << prm.get ("Function expression") << "'"
                            << "More information about the cause of the parse error \n"
                            << "is shown below.\n";
                  throw;
                }
              prm.leave_subsection();
            }
          else if (target_topography_model == "file")
            {
              prm.enter_subsection("Observation");
              {
                observation.parse_parameters(prm);
              }
              prm.leave_subsection();
              AssertThrow(observation.uses_file_source(),
                          ExcMessage("Dynamic-topography Target topography model=file requires "
                                     "Observation source=file."));
            }
          else
            AssertThrow (false, ExcNotImplemented());
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
    ASPECT_REGISTER_ADJOINT_OBJECTIVE(DynamicTopographyObjective,
                                      "dynamic topography",
                                      "Objective functional for instantaneous dynamic-topography inversion. The objective "
                                      "evaluates one half of the squared surface dynamic topography misfit and assembles the "
                                      "corresponding top-boundary adjoint right hand side.")
  }
}
