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

#include <aspect/adjoint/operators/dynamic_topography_boundary_operator.h>
#include <aspect/simulator.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    DynamicTopographyBoundaryOperator<dim>::DynamicTopographyBoundaryOperator(const Simulator<dim> &simulator)
    {
      this->initialize_simulator(simulator);
    }



    template <int dim>
    void
    DynamicTopographyBoundaryOperator<dim>::add_state_transpose(
      const LinearAlgebra::BlockVector &topography_adjoint,
      const double boundary_area,
      const double pressure_weight,
      const double density_exterior,
      LinearAlgebra::BlockVector &rhs,
      const Postprocess::DynamicTopographyBoundary boundary) const
    {
      AssertThrow(boundary_area > std::numeric_limits<double>::min(),
                  ExcMessage("DynamicTopographyBoundaryOperator found zero selected-boundary area."));

      // The forward CBF postprocessor computes nodal topography from reduced
      // normal stress as
      //   h_a = (M_a)^{-1} r_a / (Delta rho_a |g_a|),
      // where M_a is the diagonal GLL face mass and r_a is the boundary-normal
      // projection of the discrete Stokes residual. This routine applies the
      // exact reverse chain
      //   bar r_a += (M_a)^{-T} bar h_a/(Delta rho_a |g_a|)
      // followed by the transpose of the volume Stokes bilinear form.
      const types::boundary_id boundary_id = this->get_geometry_model().
                                             translate_symbolic_boundary_name_to_id(boundary == Postprocess::DynamicTopographyBoundary::surface ? "top" : "bottom");
      const unsigned int quadrature_degree =
        this->get_fe().base_element(this->introspection().base_elements.velocities).degree + 1;
      const QGauss<dim> quadrature_formula(quadrature_degree);
      const QGauss<dim-1> quadrature_formula_face(quadrature_degree);
      const QGaussLobatto<dim-1> cbf_face_quadrature_formula(quadrature_degree);
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;

      FEValues<dim> fe_values(this->get_mapping(), this->get_fe(), quadrature_formula,
                              update_values | update_gradients | update_quadrature_points | update_JxW_values);
      FEFaceValues<dim> fe_face_values(this->get_mapping(), this->get_fe(), quadrature_formula_face,
                                       update_values | update_gradients | update_normal_vectors |
                                       update_quadrature_points | update_JxW_values);
      FEFaceValues<dim> fe_cbf_face_values(this->get_mapping(), this->get_fe(), cbf_face_quadrature_formula,
                                           update_values | update_JxW_values);
      const std::vector<Point<dim-1>> face_support_points =
        this->get_fe().base_element(this->introspection().base_elements.temperature).get_unit_face_support_points();
      const Quadrature<dim-1> support_quadrature(face_support_points);
      FEFaceValues<dim> fe_support_values(this->get_mapping(), this->get_fe(), support_quadrature,
                                          update_values | update_gradients | update_normal_vectors |
                                          update_quadrature_points);

      Vector<double> local_rhs(dofs_per_cell);
      Vector<double> local_cbf_mass_matrix(dofs_per_cell);
      Vector<double> local_cbf_state_adjoint(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      std::vector<types::global_dof_index> face_dof_indices(this->get_fe().dofs_per_face);
      LinearAlgebra::BlockVector cbf_mass_matrix(this->introspection().index_sets.system_partitioning,
                                                 this->introspection().index_sets.system_relevant_partitioning,
                                                 this->get_mpi_communicator());
      cbf_mass_matrix = 0.0;

      // Reassemble precisely the diagonal GLL face mass used by the forward
      // CBF calculation. It is a global nodal quantity: neighboring boundary
      // cells contribute to the same face DoF and compress(add) completes the
      // distributed sum. Its construction remains independent of bar h.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int f : cell->face_indices())
            if (cell->at_boundary(f) && cell->face(f)->boundary_id() == boundary_id)
              {
                fe_cbf_face_values.reinit(cell, f);
                local_cbf_mass_matrix = 0.0;
                for (unsigned int q = 0; q < cbf_face_quadrature_formula.size(); ++q)
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      const Tensor<1, dim> phi_u =
                        fe_cbf_face_values[this->introspection().extractors.velocities].value(i, q);
                      local_cbf_mass_matrix(i) += phi_u * phi_u * fe_cbf_face_values.JxW(q);
                    }
                cell->distribute_local_to_global(local_cbf_mass_matrix, cbf_mass_matrix);
              }
      cbf_mass_matrix.compress(VectorOperation::add);

      // Convert each topography seed into an adjoint of the CBF residual. The
      // temperature support points are also the locations where the forward
      // code samples density and gravity, so using any other quadrature here
      // would transpose a different discrete operator.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int f : cell->face_indices())
            if (cell->at_boundary(f) && cell->face(f)->boundary_id() == boundary_id)
              {
                fe_values.reinit(cell);
                fe_support_values.reinit(cell, f);
                local_rhs = 0.0;
                local_cbf_state_adjoint = 0.0;

                MaterialModel::MaterialModelInputs<dim> support_in(fe_support_values, cell,
                                                                   this->introspection(), this->get_solution());
                MaterialModel::MaterialModelOutputs<dim> support_out(fe_support_values.n_quadrature_points,
                                                                     this->n_compositional_fields());
                support_in.requested_properties = MaterialModel::MaterialProperties::density;
                this->get_material_model().evaluate(support_in, support_out);
                cell->get_dof_indices(local_dof_indices);
                cell->face(f)->get_dof_indices(face_dof_indices);

                for (unsigned int face_i = 0; face_i < face_dof_indices.size(); ++face_i)
                  {
                    const std::pair<unsigned int, unsigned int> component_index =
                      this->get_fe().face_system_to_component_index(face_i);
                    const unsigned int component = component_index.first;
                    const unsigned int support_index = component_index.second;
                    if (component == this->introspection().component_indices.temperature)
                      {
                        const Tensor<1,dim> normal = fe_support_values.normal_vector(support_index);
                        const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector(
                                                        fe_support_values.quadrature_point(support_index));
                        const double gravity_norm = gravity.norm();
                        const double density_contrast = support_out.densities[support_index] - density_exterior;
                        AssertThrow(std::abs(density_contrast) > std::numeric_limits<double>::min(),
                                    ExcMessage("DynamicTopographyBoundaryOperator encountered zero density contrast at the selected boundary."));
                        AssertThrow(gravity_norm > std::numeric_limits<double>::min(),
                                    ExcMessage("DynamicTopographyBoundaryOperator encountered zero gravity magnitude at the selected boundary."));
                        const double coefficient = topography_adjoint(face_dof_indices[face_i]) /
                                                   (density_contrast * gravity_norm);
                        // M is diagonal in the GLL nodal basis. Distribute the
                        // scalar normal-stress seed over velocity components
                        // through n dot phi_u and divide by the assembled nodal
                        // mass. This is M^{-T}=M^{-1} for the diagonal mass.
                        for (unsigned int velocity_face_i = 0; velocity_face_i < face_dof_indices.size(); ++velocity_face_i)
                          {
                            const std::pair<unsigned int, unsigned int> velocity_component_index =
                              this->get_fe().face_system_to_component_index(velocity_face_i);
                            if (this->introspection().component_masks.velocities[velocity_component_index.first] == false)
                              continue;
                            const auto local_dof = std::find(local_dof_indices.begin(), local_dof_indices.end(),
                                                             face_dof_indices[velocity_face_i]);
                            AssertThrow(local_dof != local_dof_indices.end(), ExcInternalError());
                            const double mass = cbf_mass_matrix(face_dof_indices[velocity_face_i]);
                            if (mass <= std::numeric_limits<double>::min())
                              continue;
                            const unsigned int local_index = local_dof - local_dof_indices.begin();
                            const Tensor<1, dim> phi_u = fe_support_values[
                                                           this->introspection().extractors.velocities].value(local_index, support_index);
                            local_cbf_state_adjoint(local_index) += coefficient * (phi_u * normal) / mass;
                          }
                      }
                  }

                MaterialModel::MaterialModelInputs<dim> volume_in(fe_values, cell,
                                                                  this->introspection(), this->get_solution());
                MaterialModel::MaterialModelOutputs<dim> volume_out(fe_values.n_quadrature_points,
                                                                    this->n_compositional_fields());
                volume_in.requested_properties = MaterialModel::MaterialProperties::viscosity;
                this->get_material_model().evaluate(volume_in, volume_out);
                // The CBF residual uses the Stokes bilinear form
                //   a(u,v)=int 2 eta eps(u):eps(v)-p div(v) dV.
                // With bar r represented by local_cbf_state_adjoint, applying
                // a(.,.)^T produces the velocity/pressure RHS below. The loop
                // order is algebraic only; it does not introduce a new
                // continuous boundary formulation.
                for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    if (local_cbf_state_adjoint(j) != 0.0)
                      {
                        const SymmetricTensor<2,dim> epsilon_phi_j =
                          fe_values[this->introspection().extractors.velocities].symmetric_gradient(j, q);
                        const double div_phi_j =
                          fe_values[this->introspection().extractors.velocities].divergence(j, q);
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            const SymmetricTensor<2,dim> epsilon_phi_i =
                              fe_values[this->introspection().extractors.velocities].symmetric_gradient(i, q);
                            const double pressure_phi_i =
                              fe_values[this->introspection().extractors.pressure].value(i, q);
                            local_rhs(i) += local_cbf_state_adjoint(j) *
                                            (2.0 * volume_out.viscosities[q] * (epsilon_phi_j * epsilon_phi_i)
                                             - this->get_pressure_scaling() * div_phi_j * pressure_phi_i) * fe_values.JxW(q);
                          }
                      }

                fe_face_values.reinit(cell, f);
                // Dynamic topography subtracts the boundary mean pressure.
                // Reverse that scalar mean by spreading pressure_weight/A
                // uniformly over the selected boundary pressure test space.
                for (unsigned int q = 0; q < quadrature_formula_face.size(); ++q)
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    local_rhs(i) += this->get_pressure_scaling() * pressure_weight / boundary_area
                                    * fe_face_values[this->introspection().extractors.pressure].value(i, q)
                                    * fe_face_values.JxW(q);
                this->get_current_constraints().distribute_local_to_global(local_rhs, local_dof_indices, rhs);
              }
      rhs.compress(VectorOperation::add);

      if (this->get_parameters().pressure_normalization != "no")
        {
          // ASPECT's pressure normalization quotients out the constant
          // pressure mode. Project the assembled adjoint RHS onto the same
          // quotient space: first measure its component along the scaled
          // constant-pressure vector, then subtract the corresponding
          // boundary-average functional. The MPI sum is required because the
          // pressure vector is distributed by ownership.
          const unsigned int pressure_block_index = this->introspection().block_indices.pressure;
          const double inverse_pressure_scaling = 1.0 / this->get_pressure_scaling();
          const IndexSet local_pressure_dofs = rhs.block(pressure_block_index).locally_owned_elements();
          double local_pressure_constant_rhs_dot = 0.0;
          for (unsigned int k = 0; k < local_pressure_dofs.n_elements(); ++k)
            {
              const types::global_dof_index local_index = local_pressure_dofs.nth_index_in_set(k);
              local_pressure_constant_rhs_dot += rhs.block(pressure_block_index)[local_index] * inverse_pressure_scaling;
            }
          const double pressure_constant_rhs_dot = Utilities::MPI::sum(local_pressure_constant_rhs_dot,
                                                                       this->get_mpi_communicator());
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              for (const unsigned int f : cell->face_indices())
                if (cell->at_boundary(f) && cell->face(f)->boundary_id() == boundary_id)
                  {
                    fe_face_values.reinit(cell, f);
                    local_rhs = 0.0;
                    cell->get_dof_indices(local_dof_indices);
                    for (unsigned int q = 0; q < quadrature_formula_face.size(); ++q)
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        local_rhs(i) += -pressure_constant_rhs_dot * this->get_pressure_scaling() / boundary_area
                                        * fe_face_values[this->introspection().extractors.pressure].value(i, q)
                                        * fe_face_values.JxW(q);
                    this->get_current_constraints().distribute_local_to_global(local_rhs, local_dof_indices, rhs);
                  }
          rhs.compress(VectorOperation::add);
        }
    }



    template <int dim>
    void
    DynamicTopographyBoundaryOperator<dim>::add_direct_kernel_contributions(
      const LinearAlgebra::BlockVector &topography_adjoint,
      const LinearAlgebra::BlockVector &topography_vector,
      const double density_exterior,
      const std::string &objective_name,
      KernelRepository<dim> &kernels,
      const Postprocess::DynamicTopographyBoundary boundary) const
    {
      // Besides its dependence through the Stokes state, h=r/(Delta rho |g|)
      // has direct material derivatives. This routine accumulates cellwise
      // density and viscosity kernels for
      //   bar rho   = -bar h * h / Delta rho
      // and
      //   bar eta   = bar h/(Delta rho |g|) * dr/deta.
      // The second term includes the eta-dependence of the CBF volume residual;
      // the first includes both the denominator and buoyancy-force paths.
      const unsigned int n_active_cells = this->get_triangulation().n_active_cells();
      const unsigned int quadrature_degree =
        this->get_fe().base_element(this->introspection().base_elements.velocities).degree + 1;
      const QGauss<dim> quadrature_formula(quadrature_degree);
      const QGauss<dim-1> face_quadrature_formula(quadrature_degree);
      const QGaussLobatto<dim-1> cbf_face_quadrature_formula(quadrature_degree);
      const types::boundary_id boundary_id = this->get_geometry_model().
                                             translate_symbolic_boundary_name_to_id(boundary == Postprocess::DynamicTopographyBoundary::surface ? "top" : "bottom");
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      FEValues<dim> fe_values(this->get_mapping(), this->get_fe(), quadrature_formula,
                              update_values | update_gradients | update_quadrature_points | update_JxW_values);
      FEFaceValues<dim> fe_cbf_face_values(this->get_mapping(), this->get_fe(), cbf_face_quadrature_formula,
                                           update_values | update_JxW_values);
      const std::vector<Point<dim-1>> face_support_points =
        this->get_fe().base_element(this->introspection().base_elements.temperature).get_unit_face_support_points();
      const Quadrature<dim-1> support_quadrature(face_support_points);
      FEFaceValues<dim> fe_support_values(this->get_mapping(), this->get_fe(), support_quadrature,
                                          update_values | update_gradients | update_normal_vectors |
                                          update_quadrature_points);
      Vector<double> cell_volumes(n_active_cells);
      Vector<double> density_surface(n_active_cells);
      Vector<double> viscosity_surface(n_active_cells);
      Vector<double> local_cbf_mass_matrix(dofs_per_cell);
      Vector<double> local_density_cbf_rhs(dofs_per_cell);
      Vector<double> local_viscosity_cbf_rhs(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      std::vector<types::global_dof_index> face_dof_indices(this->get_fe().dofs_per_face);
      std::vector<double> support_topo_values(support_quadrature.size());
      LinearAlgebra::BlockVector cbf_mass_matrix(this->introspection().index_sets.system_partitioning,
                                                 this->introspection().index_sets.system_relevant_partitioning,
                                                 this->get_mpi_communicator());
      cbf_mass_matrix = 0.0;

      // Kernels are stored as cell averages. Accumulate physical cell volumes
      // now so the integrated sensitivities formed below can be normalized
      // once, after all boundary faces and shared topography DoFs are handled.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            const unsigned int cell_index = cell->active_cell_index();
            for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
              cell_volumes(cell_index) += fe_values.JxW(q);
          }

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int f : cell->face_indices())
            if (cell->at_boundary(f) && cell->face(f)->boundary_id() == boundary_id)
              {
                fe_cbf_face_values.reinit(cell, f);
                local_cbf_mass_matrix = 0.0;
                for (unsigned int q = 0; q < cbf_face_quadrature_formula.size(); ++q)
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      const Tensor<1, dim> phi_u =
                        fe_cbf_face_values[this->introspection().extractors.velocities].value(i, q);
                      local_cbf_mass_matrix(i) += phi_u * phi_u * fe_cbf_face_values.JxW(q);
                    }
                cell->distribute_local_to_global(local_cbf_mass_matrix, cbf_mass_matrix);
              }
      cbf_mass_matrix.compress(VectorOperation::add);

      // Different boundary cells can see the same continuous topography DoF.
      // Record its explicit denominator derivative once by global DoF index,
      // rather than once per adjacent face. The associated active cell selects
      // where this surface sensitivity is represented in the cellwise kernel.
      std::map<types::global_dof_index, std::pair<unsigned int, double>> density_surface_by_topography_dof;
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int f : cell->face_indices())
            if (cell->at_boundary(f) && cell->face(f)->boundary_id() == boundary_id)
              {
                fe_support_values.reinit(cell, f);
                MaterialModel::MaterialModelInputs<dim> support_in(fe_support_values, cell,
                                                                   this->introspection(), this->get_solution());
                MaterialModel::MaterialModelOutputs<dim> support_out(fe_support_values.n_quadrature_points,
                                                                     this->n_compositional_fields());
                support_in.requested_properties = MaterialModel::MaterialProperties::density;
                this->get_material_model().evaluate(support_in, support_out);
                fe_support_values[this->introspection().extractors.temperature].get_function_values(
                  topography_vector, support_topo_values);
                cell->face(f)->get_dof_indices(face_dof_indices);
                for (unsigned int face_i = 0; face_i < face_dof_indices.size(); ++face_i)
                  {
                    const std::pair<unsigned int, unsigned int> component_index =
                      this->get_fe().face_system_to_component_index(face_i);
                    const unsigned int support_index = component_index.second;
                    if (component_index.first == this->introspection().component_indices.temperature)
                      {
                        const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector(
                                                        fe_support_values.quadrature_point(support_index));
                        const double gravity_norm = gravity.norm();
                        const double density_contrast = support_out.densities[support_index] - density_exterior;
                        AssertThrow(std::abs(density_contrast) > std::numeric_limits<double>::min(),
                                    ExcMessage("DynamicTopographyBoundaryOperator encountered zero density contrast at the selected boundary."));
                        AssertThrow(gravity_norm > std::numeric_limits<double>::min(),
                                    ExcMessage("DynamicTopographyBoundaryOperator encountered zero gravity magnitude at the selected boundary."));
                        (void)gravity_norm;
                        density_surface_by_topography_dof[face_dof_indices[face_i]] =
                          std::make_pair(cell->active_cell_index(),
                                         -topography_adjoint(face_dof_indices[face_i]) *
                                         support_topo_values[support_index] / density_contrast);
                      }
                  }
              }

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int f : cell->face_indices())
            if (cell->at_boundary(f) && cell->face(f)->boundary_id() == boundary_id)
              {
                fe_values.reinit(cell);
                fe_support_values.reinit(cell, f);
                local_density_cbf_rhs = 0.0;
                local_viscosity_cbf_rhs = 0.0;
                MaterialModel::MaterialModelInputs<dim> volume_in(fe_values, cell,
                                                                  this->introspection(), this->get_solution());
                MaterialModel::MaterialModelOutputs<dim> volume_out(fe_values.n_quadrature_points,
                                                                    this->n_compositional_fields());
                volume_in.requested_properties = MaterialModel::MaterialProperties::viscosity;
                this->get_material_model().evaluate(volume_in, volume_out);
                // Differentiate the two material coefficients in the forward
                // Stokes residual. For density, dr/drho contains -g dot phi_u;
                // for viscosity, dr/deta contains 2 eps(phi_u):eps(u). These
                // local vectors are subsequently projected through the same
                // diagonal CBF mass and boundary normal used by the state path.
                for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
                  {
                    const Tensor<1, dim> gravity = this->get_gravity_model().gravity_vector(
                                                     fe_values.quadrature_point(q));
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        const Tensor<1, dim> phi_u =
                          fe_values[this->introspection().extractors.velocities].value(i, q);
                        const SymmetricTensor<2, dim> epsilon_phi_u =
                          fe_values[this->introspection().extractors.velocities].symmetric_gradient(i, q);
                        local_density_cbf_rhs(i) -= (gravity * phi_u) * fe_values.JxW(q);
                        local_viscosity_cbf_rhs(i) += 2.0 * (epsilon_phi_u * volume_in.strain_rate[q]) * fe_values.JxW(q);
                      }
                  }
                MaterialModel::MaterialModelInputs<dim> support_in(fe_support_values, cell,
                                                                   this->introspection(), this->get_solution());
                MaterialModel::MaterialModelOutputs<dim> support_out(fe_support_values.n_quadrature_points,
                                                                     this->n_compositional_fields());
                support_in.requested_properties = MaterialModel::MaterialProperties::density;
                this->get_material_model().evaluate(support_in, support_out);
                cell->get_dof_indices(local_dof_indices);
                cell->face(f)->get_dof_indices(face_dof_indices);
                for (unsigned int face_i = 0; face_i < face_dof_indices.size(); ++face_i)
                  {
                    const std::pair<unsigned int, unsigned int> component_index =
                      this->get_fe().face_system_to_component_index(face_i);
                    const unsigned int support_index = component_index.second;
                    if (component_index.first == this->introspection().component_indices.temperature)
                      {
                        const Tensor<1,dim> normal = fe_support_values.normal_vector(support_index);
                        const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector(
                                                        fe_support_values.quadrature_point(support_index));
                        const double gravity_norm = gravity.norm();
                        const double density_contrast = support_out.densities[support_index] - density_exterior;
                        double density_force_normal_stress_derivative = 0.0;
                        double normal_stress_derivative = 0.0;
                        AssertThrow(std::abs(density_contrast) > std::numeric_limits<double>::min(),
                                    ExcMessage("DynamicTopographyBoundaryOperator encountered zero density contrast at the selected boundary."));
                        AssertThrow(gravity_norm > std::numeric_limits<double>::min(),
                                    ExcMessage("DynamicTopographyBoundaryOperator encountered zero gravity magnitude at the selected boundary."));
                        for (unsigned int velocity_face_i = 0; velocity_face_i < face_dof_indices.size(); ++velocity_face_i)
                          {
                            const std::pair<unsigned int, unsigned int> velocity_component_index =
                              this->get_fe().face_system_to_component_index(velocity_face_i);
                            if (this->introspection().component_masks.velocities[velocity_component_index.first] == false)
                              continue;
                            const auto local_dof = std::find(local_dof_indices.begin(), local_dof_indices.end(),
                                                             face_dof_indices[velocity_face_i]);
                            AssertThrow(local_dof != local_dof_indices.end(), ExcInternalError());
                            const double mass = cbf_mass_matrix(face_dof_indices[velocity_face_i]);
                            if (mass <= std::numeric_limits<double>::min())
                              continue;
                            const unsigned int local_index = local_dof - local_dof_indices.begin();
                            const Tensor<1, dim> phi_u = fe_support_values[
                                                           this->introspection().extractors.velocities].value(local_index, support_index);
                            density_force_normal_stress_derivative +=
                              (phi_u * normal) * local_density_cbf_rhs(local_index) / mass;
                            normal_stress_derivative += (phi_u * normal) * local_viscosity_cbf_rhs(local_index) / mass;
                          }
                        density_surface(cell->active_cell_index()) -= topography_adjoint(face_dof_indices[face_i]) *
                                                                      density_force_normal_stress_derivative / (density_contrast * gravity_norm);
                        viscosity_surface(cell->active_cell_index()) -= topography_adjoint(face_dof_indices[face_i]) *
                                                                        normal_stress_derivative / (density_contrast * gravity_norm);
                      }
                  }
              }
      // Add the explicit 1/Delta-rho derivative after the residual-mediated
      // term. Then convert integrated cell sensitivities to the cell-average
      // convention used by KernelRepository.
      for (const auto &entry : density_surface_by_topography_dof)
        density_surface(entry.second.first) += entry.second.second;
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int cell_index = cell->active_cell_index();
            AssertThrow(cell_volumes(cell_index) > std::numeric_limits<double>::min(),
                        ExcMessage("DynamicTopographyBoundaryOperator encountered a locally owned cell with zero volume."));
            density_surface(cell_index) /= cell_volumes(cell_index);
            viscosity_surface(cell_index) /= cell_volumes(cell_index);
          }
      const std::string contribution_name = boundary == Postprocess::DynamicTopographyBoundary::surface
                                            ? "dynamic topography surface objective"
                                            : "dynamic topography CMB objective";
      kernels.add_contribution({objective_name, contribution_name, PhysicalProperty::density},
                               density_surface);
      kernels.add_contribution({objective_name, contribution_name, PhysicalProperty::viscosity},
                               viscosity_surface);
    }



    template class DynamicTopographyBoundaryOperator<2>;
    template class DynamicTopographyBoundaryOperator<3>;
  }
}
