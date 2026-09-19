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

#include <aspect/adjoint/operators/advection_adjoint_calculator.h>
#include <aspect/adjoint/operators/boundary_conditions.h>
#include <aspect/adjoint/operators/artificial_viscosity_adjoint.h>
#include <aspect/adjoint/operators/material_model_adjoint.h>
#include <aspect/adjoint/core/simulator_state.h>
#include <aspect/advection_field.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/simulator.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include <Epetra_CrsMatrix.h>
#include <EpetraExt_Transpose_RowMatrix.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <set>

namespace aspect
{
  namespace Adjoint
  {
    namespace
    {
      /**
       * Minimal matrix wrapper used so SolverGMRES can apply the transpose of
       * ASPECT's already assembled advection block without explicitly forming
       * a second sparse matrix. The forward assembly still owns the matrix;
       * this class only redirects vmult() to Tvmult().
       */
      class TransposedSparseMatrix
      {
        public:
          explicit
          TransposedSparseMatrix(const LinearAlgebra::SparseMatrix &matrix)
            :
            matrix(matrix)
          {}

          void
          vmult(LinearAlgebra::Vector &dst,
                const LinearAlgebra::Vector &src) const
          {
            matrix.Tvmult(dst, src);
          }

        private:
          const LinearAlgebra::SparseMatrix &matrix;
      };
    }



    template <int dim>
    AdvectionAdjointCalculator<dim>::AdvectionAdjointCalculator(Simulator<dim> &simulator_object)
      :
      simulator(&simulator_object)
    {
      this->initialize_simulator(simulator_object);
    }



    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::set_preconditioner(const Preconditioner new_preconditioner)
    {
      preconditioner = new_preconditioner;
    }



    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::set_run_preconditioner_comparison(const bool run_comparison)
    {
      run_preconditioner_comparison = run_comparison;
    }



    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::solve_transposed_advection_system(const AdvectionField &advection_field,
                                                                       const LinearAlgebra::BlockVector &rhs,
                                                                       LinearAlgebra::BlockVector &solution) const
    {
      // The accepted forward step solves A_n q_n=b_n. Given a seed bar(q_n),
      // the implicit-function transpose first solves
      //
      //                         A_n^T z_n = bar(q_n).
      //
      // Subsequent routines contract z_n with derivatives of b_n-A_n q_n
      // with respect to history fields, velocity, and material coefficients.
      // Reusing the assembled forward block is important: it preserves the
      // exact quadrature, stabilization, constraint rows, and linearization
      // used by the primal time step.
      const unsigned int field_block_index =
        advection_field.block_index(this->introspection());

      // The transposed solution is a constraint-distribution target below;
      // keep it owned-only and writable. Replay snapshots, not adjoint
      // accumulation vectors, carry ghosted FE views.
      solution.reinit(this->introspection().index_sets.system_partitioning,
                      this->get_mpi_communicator());
      solution = 0.0;

      const double advection_solver_tolerance =
        advection_field.is_temperature()
        ? this->get_parameters().temperature_solver_tolerance
        : this->get_parameters().composition_solver_tolerance;
      const double tolerance =
        std::max(1e-50,
                 advection_solver_tolerance
                 * rhs.block(field_block_index).l2_norm());

      SolverControl solver_control(1000, tolerance);
      SolverGMRES<LinearAlgebra::Vector> solver(
        solver_control,
        SolverGMRES<LinearAlgebra::Vector>::AdditionalData(
          this->get_parameters().advection_gmres_restart_length,
          true));

      if (rhs.block(field_block_index).l2_norm() <= std::numeric_limits<double>::min())
        return;

      // Constraint rows in the forward matrix encode prescribed primal
      // values. Their adjoint right-hand-side entries must be zero; otherwise
      // a terminal functional could reintroduce fixed boundary data into z.
      LinearAlgebra::BlockVector homogeneous_rhs(rhs);
      this->get_current_constraints().set_zero(homogeneous_rhs);

      this->get_pcout() << "   Solving transposed "
                        << advection_field.name(this->introspection())
                        << " system... " << std::flush;

      const LinearAlgebra::SparseMatrix &advection_matrix =
        simulator->system_matrix.block(field_block_index, field_block_index);
      TransposedSparseMatrix transposed_advection_matrix(advection_matrix);

      LinearAlgebra::SparseMatrix explicit_transposed_advection_matrix;
      LinearAlgebra::Vector distributed_field(
        this->introspection().index_sets.system_partitioning[field_block_index],
        this->get_mpi_communicator());

      bool &comparison_completed = advection_field.is_temperature()
                                   ? temperature_preconditioner_comparison_completed
                                   : composition_preconditioner_comparison_completed;
      if (run_preconditioner_comparison && !comparison_completed)
        {
          // This diagnostic solves the same A^T system twice. It checks only
          // preconditioning quality: the identity-preconditioned result and
          // ILU(A^T)-preconditioned result must agree to solver tolerance.
          // Forming A^T explicitly is necessary for ILU construction, not for
          // the matrix-vector product used by GMRES.
          LinearAlgebra::Vector z_none(
            this->introspection().index_sets.system_partitioning[field_block_index],
            this->get_mpi_communicator());
          LinearAlgebra::Vector z_ilu(z_none);
          z_none = 0.0;
          z_ilu = 0.0;

          SolverControl none_solver_control(1000, tolerance);
          SolverGMRES<LinearAlgebra::Vector> none_solver(
            none_solver_control,
            SolverGMRES<LinearAlgebra::Vector>::AdditionalData(
              this->get_parameters().advection_gmres_restart_length, true));
          PreconditionIdentity identity_preconditioner;
          none_solver.solve(transposed_advection_matrix,
                            z_none,
                            homogeneous_rhs.block(field_block_index),
                            identity_preconditioner);

          EpetraExt::RowMatrix_Transpose transpose_builder;
          const Epetra_CrsMatrix &epetra_transposed_matrix =
            dynamic_cast<const Epetra_CrsMatrix &>(
              transpose_builder(const_cast<Epetra_CrsMatrix &>(
                                  advection_matrix.trilinos_matrix())));
          explicit_transposed_advection_matrix.reinit(epetra_transposed_matrix);
          LinearAlgebra::PreconditionILU ilu_preconditioner;
          ilu_preconditioner.initialize(explicit_transposed_advection_matrix);

          SolverControl ilu_solver_control(1000, tolerance);
          SolverGMRES<LinearAlgebra::Vector> ilu_solver(
            ilu_solver_control,
            SolverGMRES<LinearAlgebra::Vector>::AdditionalData(
              this->get_parameters().advection_gmres_restart_length, true));
          ilu_solver.solve(transposed_advection_matrix,
                           z_ilu,
                           homogeneous_rhs.block(field_block_index),
                           ilu_preconditioner);

          LinearAlgebra::Vector difference(z_ilu);
          difference.add(-1.0, z_none);
          LinearAlgebra::Vector none_residual(z_none);
          LinearAlgebra::Vector ilu_residual(z_ilu);
          transposed_advection_matrix.vmult(none_residual, z_none);
          transposed_advection_matrix.vmult(ilu_residual, z_ilu);
          none_residual.add(-1.0, homogeneous_rhs.block(field_block_index));
          ilu_residual.add(-1.0, homogeneous_rhs.block(field_block_index));

          const double relative_difference =
            difference.l2_norm() / std::max(z_none.l2_norm(), 1e-50);
          this->get_pcout() << "   Adjoint "
                            << advection_field.name(this->introspection())
                            << " preconditioner comparison: relative ||z_ILU - z_none||_2 = "
                            << relative_difference
                            << ", none residual = " << none_residual.l2_norm()
                            << " (" << none_solver_control.last_step() << " iterations)"
                            << ", ILU(A^T) residual = " << ilu_residual.l2_norm()
                            << " (" << ilu_solver_control.last_step() << " iterations)."
                            << std::endl;
          comparison_completed = true;
        }

      if (preconditioner == Preconditioner::none)
        {
          PreconditionIdentity identity_preconditioner;
          TimerOutput::Scope solve_timer(
            this->get_computing_timer(),
            advection_field.is_temperature()
            ? "Solve adjoint temperature system"
            : "Solve adjoint composition system");
          solver.solve(transposed_advection_matrix,
                       distributed_field,
                       homogeneous_rhs.block(field_block_index),
                       identity_preconditioner);
        }
      else
        {
          LinearAlgebra::PreconditionILU ilu_preconditioner;
          LinearAlgebra::PreconditionILU::AdditionalData preconditioner_data;
          {
            TimerOutput::Scope setup_timer(
              this->get_computing_timer(),
              advection_field.is_temperature()
              ? "Build adjoint temperature preconditioner"
              : "Build adjoint composition preconditioner");
            if (preconditioner == Preconditioner::transpose_ilu)
              {
                // Keep the validated adjoint action above, but explicitly
                // transpose a short-lived matrix for ILU(A^T). The EpetraExt
                // result is copied before the transposer is destroyed.
                EpetraExt::RowMatrix_Transpose transpose_builder;
                const Epetra_CrsMatrix &epetra_transposed_matrix =
                  dynamic_cast<const Epetra_CrsMatrix &>(
                    transpose_builder(const_cast<Epetra_CrsMatrix &>(
                                        advection_matrix.trilinos_matrix())));
                explicit_transposed_advection_matrix.reinit(epetra_transposed_matrix);
                ilu_preconditioner.initialize(explicit_transposed_advection_matrix,
                                              preconditioner_data);
              }
            else
              // This experimental option intentionally applies the forward
              // ILU(A) to the transpose solve. It can be cheaper to build but
              // is not algebraically equivalent to ILU(A^T), so it remains a
              // named opt-in preconditioner rather than the default.
              simulator->build_advection_preconditioner(advection_field,
                                                        ilu_preconditioner,
                                                        0.0);
          }

          try
            {
              TimerOutput::Scope solve_timer(
                this->get_computing_timer(),
                advection_field.is_temperature()
                ? "Solve adjoint temperature system"
                : "Solve adjoint composition system");
              solver.solve(transposed_advection_matrix,
                           distributed_field,
                           homogeneous_rhs.block(field_block_index),
                           ilu_preconditioner);
            }
          catch (const std::exception &)
            {
              // Match the forward recovery policy. This changes only the
              // factorization, never the adjoint operator or constraints.
              this->get_pcout() << "retrying transposed solve with different "
                                << "preconditioner... " << std::flush;
              if (preconditioner == Preconditioner::transpose_ilu)
                {
                  preconditioner_data.ilu_atol = 1e-5;
                  TimerOutput::Scope setup_timer(
                    this->get_computing_timer(),
                    advection_field.is_temperature()
                    ? "Build adjoint temperature preconditioner"
                    : "Build adjoint composition preconditioner");
                  ilu_preconditioner.initialize(explicit_transposed_advection_matrix,
                                                preconditioner_data);
                }
              else
                {
                  TimerOutput::Scope setup_timer(
                    this->get_computing_timer(),
                    advection_field.is_temperature()
                    ? "Build adjoint temperature preconditioner"
                    : "Build adjoint composition preconditioner");
                  simulator->build_advection_preconditioner(advection_field,
                                                            ilu_preconditioner,
                                                            1e-5);
                }
              TimerOutput::Scope solve_timer(
                this->get_computing_timer(),
                advection_field.is_temperature()
                ? "Solve adjoint temperature system"
                : "Solve adjoint composition system");
              solver.solve(transposed_advection_matrix,
                           distributed_field,
                           homogeneous_rhs.block(field_block_index),
                           ilu_preconditioner);
            }
        }

      solution.block(field_block_index) = distributed_field;
      // Reconstruct dependent DoFs from the homogeneous constraint graph.
      // This is distinct from set_zero(): the latter destroys the affine
      // component relations used for curved free-slip boundaries.
      this->get_current_constraints().distribute(solution);

      const char *preconditioner_name =
        (preconditioner == Preconditioner::none ? "none" :
         preconditioner == Preconditioner::forward_ilu_experimental ? "ILU(A), experimental" :
         "ILU(A^T)");
      this->get_pcout() << solver_control.last_step()
                        << " iterations (" << preconditioner_name
                        << " preconditioner)." << std::endl;
    }



    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::add_history_transpose(const AdvectionField &advection_field,
                                                           const ForwardStateView<dim> &snapshot,
                                                           const LinearAlgebra::BlockVector &z,
                                                           LinearAlgebra::BlockVector &previous_field_adjoint,
                                                           LinearAlgebra::BlockVector *previous_previous_field_adjoint) const
    {
      // z is solver output and deliberately owned-only/writable. FEValues
      // needs locally relevant entries, so materialize a short-lived ghosted
      // read view without changing the solver vector itself.
      LinearAlgebra::BlockVector z_read;
      z_read.reinit(this->introspection().index_sets.system_partitioning,
                    this->introspection().index_sets.system_relevant_partitioning,
                    this->get_mpi_communicator());
      z_read = z;

      const unsigned int quadrature_degree =
        this->get_fe().base_element(advection_field.base_element(this->introspection())).degree + 1;
      const QGauss<dim> quadrature_formula(quadrature_degree);
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      const FEValuesExtractors::Scalar field_extractor =
        advection_field.scalar_extractor(this->introspection());

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_values |
                              update_gradients |
                              update_quadrature_points |
                              update_JxW_values);

      std::vector<double> z_values(quadrature_formula.size());
      Vector<double> local_previous(dofs_per_cell);
      Vector<double> local_previous_previous(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      const bool use_bdf2_scheme = (snapshot.timestep_number > 1);
      const double previous_coefficient =
        use_bdf2_scheme ? (1.0 + snapshot.time_step / snapshot.old_time_step) : 1.0;
      const double previous_previous_coefficient =
        use_bdf2_scheme
        ? -(snapshot.time_step * snapshot.time_step)
        / (snapshot.old_time_step * (snapshot.time_step + snapshot.old_time_step))
        : 0.0;

      // The history portion of b_n is mass-like. In the variable-step BDF2
      // convention used by the forward assembly it contains
      //
      //   integral rho*C_p [a_1 q_{n-1}+a_2 q_{n-2}] phi_i dx,
      //   a_1 = 1+dt/dt_old,
      //   a_2 = -dt^2/[dt_old(dt+dt_old)].
      //
      // Its transpose therefore integrates z phi_j with the same coefficients
      // and scatters to the two history slots. For composition rho*C_p is
      // replaced by one. Constraints are applied during local-to-global
      // distribution, making this the transpose on the constrained FE space.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[field_extractor]
            .get_function_values(z_read, z_values);

            local_previous = 0.0;
            local_previous_previous = 0.0;
            cell->get_dof_indices(local_dof_indices);

            MaterialModel::MaterialModelInputs<dim> in(fe_values,
                                                       cell,
                                                       this->introspection(),
                                                       simulator->current_linearization_point);
            MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points,
                                                         this->n_compositional_fields());
            in.requested_properties =
              MaterialModel::MaterialProperties::equation_of_state_properties;
            this->get_material_model().evaluate(in, out);

            if (this->get_parameters().formulation_temperature_equation ==
                Parameters<dim>::Formulation::TemperatureEquation::reference_density_profile)
              for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                out.densities[q] =
                  this->get_adiabatic_conditions().density(fe_values.quadrature_point(q));

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double density_c_p =
                  advection_field.is_temperature()
                  ? out.densities[q] * out.specific_heat[q]
                  : 1.0;
                const double JxW = fe_values.JxW(q);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const double phi_i =
                      fe_values[field_extractor].value(i, q);
                    const double contribution = z_values[q] * density_c_p * phi_i * JxW;

                    local_previous(i) += previous_coefficient * contribution;
                    if (use_bdf2_scheme)
                      local_previous_previous(i) += previous_previous_coefficient * contribution;
                  }
              }

            this->get_current_constraints().distribute_local_to_global(local_previous,
                                                                       local_dof_indices,
                                                                       previous_field_adjoint);

            if (use_bdf2_scheme)
              {
                AssertThrow(previous_previous_field_adjoint != nullptr,
                            ExcInternalError());
                this->get_current_constraints().distribute_local_to_global(local_previous_previous,
                                                                           local_dof_indices,
                                                                           *previous_previous_field_adjoint);
              }
          }

      previous_field_adjoint.compress(VectorOperation::add);
      if (use_bdf2_scheme)
        previous_previous_field_adjoint->compress(VectorOperation::add);
    }



    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::add_velocity_transpose(const AdvectionField &advection_field,
                                                            const ForwardStateView<dim> &snapshot,
                                                            const LinearAlgebra::BlockVector &z,
                                                            LinearAlgebra::BlockVector &previous_stokes_adjoint_rhs,
                                                            LinearAlgebra::BlockVector *previous_previous_stokes_adjoint_rhs) const
    {
      // z is an owned-only solver output. Materialize a local ghosted view
      // solely for the FE evaluation below.
      LinearAlgebra::BlockVector z_read;
      z_read.reinit(this->introspection().index_sets.system_partitioning,
                    this->introspection().index_sets.system_relevant_partitioning,
                    this->get_mpi_communicator());
      z_read = z;

      const unsigned int quadrature_degree =
        this->get_fe().base_element(advection_field.base_element(this->introspection())).degree + 1;
      const QGauss<dim> quadrature_formula(quadrature_degree);
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      const FEValuesExtractors::Scalar field_extractor =
        advection_field.scalar_extractor(this->introspection());

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_values |
                              update_gradients |
                              update_quadrature_points |
                              update_JxW_values);

      std::vector<double> z_values(quadrature_formula.size());
      std::vector<Tensor<1,dim>> field_gradients(quadrature_formula.size());
      Vector<double> local_previous(dofs_per_cell);
      Vector<double> local_previous_previous(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      const bool use_bdf2_scheme = (snapshot.timestep_number > 1);
      const double gamma1 =
        use_bdf2_scheme ? (1.0 + snapshot.time_step / snapshot.old_time_step) : 1.0;
      const double gamma2 =
        use_bdf2_scheme ? -snapshot.time_step / snapshot.old_time_step : 0.0;

      // The forward transport contribution has the form
      //
      //   dt integral rho*C_p (u*.grad(q_n)) phi_i dx,
      //   u* = gamma1 u_{n-1}+gamma2 u_{n-2}.
      //
      // Contracting with z produces, for a velocity basis function psi_j,
      //
      //   -dt integral rho*C_p z (psi_j.grad(q_n)) dx.
      //
      // The minus sign follows the b-Aq residual convention. Multiplication by
      // gamma1/gamma2 then sends this seed to the two quasi-static Stokes
      // states from which u* was extrapolated.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[field_extractor]
            .get_function_values(z_read, z_values);
            fe_values[field_extractor]
            .get_function_gradients(simulator->solution, field_gradients);

            local_previous = 0.0;
            local_previous_previous = 0.0;
            cell->get_dof_indices(local_dof_indices);

            MaterialModel::MaterialModelInputs<dim> in(fe_values,
                                                       cell,
                                                       this->introspection(),
                                                       simulator->current_linearization_point);
            MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points,
                                                         this->n_compositional_fields());
            in.requested_properties =
              MaterialModel::MaterialProperties::equation_of_state_properties;
            this->get_material_model().evaluate(in, out);

            if (this->get_parameters().formulation_temperature_equation ==
                Parameters<dim>::Formulation::TemperatureEquation::reference_density_profile)
              for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                out.densities[q] =
                  this->get_adiabatic_conditions().density(fe_values.quadrature_point(q));

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double density_c_p =
                  advection_field.is_temperature()
                  ? out.densities[q] * out.specific_heat[q]
                  : 1.0;
                const double scale =
                  -snapshot.time_step * density_c_p * z_values[q] * fe_values.JxW(q);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  if (this->introspection().component_masks.velocities[
                        this->get_fe().system_to_component_index(i).first])
                    {
                      const Tensor<1,dim> phi_u =
                        fe_values[this->introspection().extractors.velocities].value(i, q);
                      const double contribution =
                        scale * (phi_u * field_gradients[q]);

                      local_previous(i) += gamma1 * contribution;
                      if (use_bdf2_scheme)
                        local_previous_previous(i) += gamma2 * contribution;
                    }
              }

            this->get_current_constraints().distribute_local_to_global(local_previous,
                                                                       local_dof_indices,
                                                                       previous_stokes_adjoint_rhs);

            if (use_bdf2_scheme)
              {
                AssertThrow(previous_previous_stokes_adjoint_rhs != nullptr,
                            ExcInternalError());
                this->get_current_constraints().distribute_local_to_global(local_previous_previous,
                                                                           local_dof_indices,
                                                                           *previous_previous_stokes_adjoint_rhs);
              }
          }

      previous_stokes_adjoint_rhs.compress(VectorOperation::add);
      if (use_bdf2_scheme)
        previous_previous_stokes_adjoint_rhs->compress(VectorOperation::add);
    }



    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::add_temperature_material_prefactor_transpose(
      const ForwardStateView<dim> &snapshot,
      const LinearAlgebra::BlockVector &z,
      LinearAlgebra::BlockVector &previous_temperature_adjoint,
      LinearAlgebra::BlockVector *previous_previous_temperature_adjoint,
      std::vector<LinearAlgebra::BlockVector> *previous_composition_adjoints,
      std::vector<LinearAlgebra::BlockVector> *previous_previous_composition_adjoints) const
    {
      if (this->get_parameters().formulation_temperature_equation ==
          Parameters<dim>::Formulation::TemperatureEquation::reference_density_profile)
        return;

      // z is the owned-only output of the transposed advection solve. This
      // material-prefactor transpose evaluates it with FEValues, so create a
      // short-lived locally-relevant read view at the FE boundary.
      LinearAlgebra::BlockVector z_read;
      z_read.reinit(this->introspection().index_sets.system_partitioning,
                    this->introspection().index_sets.system_relevant_partitioning,
                    this->get_mpi_communicator());
      z_read = z;

      const unsigned int quadrature_degree =
        this->get_fe().base_element(this->introspection().base_elements.temperature).degree + 1;
      const QGauss<dim> quadrature_formula(quadrature_degree);
      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_values |
                              update_gradients |
                              update_quadrature_points |
                              update_JxW_values);

      std::vector<double> z_values(quadrature_formula.size());
      std::vector<double> temperature_values(quadrature_formula.size());
      std::vector<Tensor<1, dim>> temperature_gradients(quadrature_formula.size());
      std::vector<double> old_temperature_values(quadrature_formula.size());
      std::vector<double> old_old_temperature_values(quadrature_formula.size());
      std::vector<Tensor<1, dim>> current_velocity_values(quadrature_formula.size());
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      const bool use_bdf2_scheme = (snapshot.timestep_number > 1);
      const double bdf2_factor = use_bdf2_scheme
                                 ? (2.0 * snapshot.time_step + snapshot.old_time_step)
                                 / (snapshot.time_step + snapshot.old_time_step)
                                 : 1.0;
      const double gamma1 =
        use_bdf2_scheme ? (1.0 + snapshot.time_step / snapshot.old_time_step) : 1.0;
      const double gamma2 =
        use_bdf2_scheme ? -snapshot.time_step / snapshot.old_time_step : 0.0;

      // Differentiate the temperature equation through a=rho(T,C) C_p(T,C).
      // With the forward residual rearranged as b-A T_n, the local seed is
      //
      //   bar(a)_q = z_q [H_q-dt u_q.grad(T_n)-bdf T_n] JxW_q,
      //
      // where H_q is the BDF history combination. The product rule gives
      // bar(rho)=C_p bar(a) and bar(C_p)=rho bar(a). The material-model VJP
      // maps these output seeds to temperature and composition inputs, and
      // the scatter applies the same extrapolation weights gamma1/gamma2 used
      // when those inputs were evaluated in the forward assembly.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);

            fe_values[this->introspection().extractors.temperature]
            .get_function_values(z_read, z_values);
            fe_values[this->introspection().extractors.temperature]
            .get_function_values(simulator->solution, temperature_values);
            fe_values[this->introspection().extractors.temperature]
            .get_function_gradients(simulator->solution, temperature_gradients);
            fe_values[this->introspection().extractors.temperature]
            .get_function_values(simulator->old_solution, old_temperature_values);
            fe_values[this->introspection().extractors.temperature]
            .get_function_values(simulator->old_old_solution, old_old_temperature_values);
            fe_values[this->introspection().extractors.velocities]
            .get_function_values(simulator->current_linearization_point, current_velocity_values);

            MaterialModel::MaterialModelInputs<dim> in(fe_values,
                                                       cell,
                                                       this->introspection(),
                                                       simulator->current_linearization_point);
            MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points,
                                                         this->n_compositional_fields());
            in.requested_properties =
              MaterialModel::MaterialProperties::equation_of_state_properties;
            this->get_material_model().evaluate(in, out);

            MaterialModelOutputAdjoints<dim> bar_out(fe_values.n_quadrature_points);

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                const double old_field_term =
                  use_bdf2_scheme
                  ? (old_temperature_values[q] * (1.0 + snapshot.time_step / snapshot.old_time_step)
                     - old_old_temperature_values[q] * (snapshot.time_step * snapshot.time_step)
                     / (snapshot.old_time_step * (snapshot.time_step + snapshot.old_time_step)))
                  : old_temperature_values[q];
                const double operator_field_term =
                  snapshot.time_step * (current_velocity_values[q] * temperature_gradients[q])
                  + bdf2_factor * temperature_values[q];
                const double coefficient_seed =
                  z_values[q] * (old_field_term - operator_field_term) * fe_values.JxW(q);

                if (coefficient_seed == 0.0)
                  continue;

                bar_out.densities[q] += out.specific_heat[q] * coefficient_seed;
                bar_out.specific_heat[q] += out.densities[q] * coefficient_seed;
              }

            MaterialModelInputAdjoints<dim> bar_in(fe_values.n_quadrature_points,
                                                   this->n_compositional_fields());
            const MaterialModelAdjoint<dim> material_model_adjoint(*simulator);
            material_model_adjoint.apply_derivative_transpose(in,
                                                              out,
                                                              bar_out,
                                                              bar_in);

            const MaterialModelInputAdjointScatter<dim> material_input_scatter(*simulator);
            material_input_scatter.scatter_temperature_advection_inputs(
              fe_values,
              local_dof_indices,
              bar_in,
              gamma1,
              gamma2,
              previous_temperature_adjoint,
              previous_previous_temperature_adjoint,
              previous_composition_adjoints,
              previous_previous_composition_adjoints);
          }

      previous_temperature_adjoint.compress(VectorOperation::add);
      if (use_bdf2_scheme)
        {
          AssertThrow(previous_previous_temperature_adjoint != nullptr,
                      ExcInternalError());
          previous_previous_temperature_adjoint->compress(VectorOperation::add);
        }
      for (unsigned int c = 0; c < this->introspection().n_compositional_fields; ++c)
        if (previous_composition_adjoints != nullptr)
          {
            (*previous_composition_adjoints)[c].compress(VectorOperation::add);
            if (use_bdf2_scheme)
              {
                AssertThrow(previous_previous_composition_adjoints != nullptr,
                            ExcInternalError());
                (*previous_previous_composition_adjoints)[c].compress(VectorOperation::add);
              }
          }
    }































    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::propagate_advection_field_adjoint(const AdvectionField &advection_field,
                                                                       const ForwardStateView<dim> &snapshot,
                                                                       const LinearAlgebra::BlockVector &current_field_adjoint,
                                                                       LinearAlgebra::BlockVector &previous_field_adjoint,
                                                                       LinearAlgebra::BlockVector *previous_previous_field_adjoint,
                                                                       LinearAlgebra::BlockVector &previous_stokes_adjoint_rhs,
                                                                       LinearAlgebra::BlockVector *previous_previous_stokes_adjoint_rhs)
    {
      // Each propagation call realizes one discrete reverse-time edge:
      // restore state n, reassemble A_n, solve A_n^T z_n=bar(q_n), and apply
      // (partial R_n/partial x)^T z_n to every predecessor x. The scoped state
      // guard makes this transaction exception-safe and prevents replay of an
      // old time level from leaking time, constraints, matrices, or pressure
      // normalization into the enclosing reverse sweep.
      ScopedSimulatorState<dim> simulator_state(*simulator);

      simulator->solution = snapshot.solution.get();
      simulator->old_solution = snapshot.old_solution.get();
      simulator->old_old_solution = snapshot.old_old_solution.get();
      simulator->current_linearization_point = snapshot.advection_linearization_point.get();
      simulator->time = snapshot.time;
      simulator->time_step = snapshot.time_step;
      simulator->old_time_step = snapshot.old_time_step;
      simulator->timestep_number = snapshot.timestep_number;
      simulator->last_pressure_normalization_adjustment =
        snapshot.pressure_normalization_adjustment;

      BoundaryConditions<dim> boundary_conditions;
      boundary_conditions.build_temperature_adjoint_constraints(snapshot,
                                                                simulator->current_constraints);

      simulator->assemble_advection_system(advection_field);

      LinearAlgebra::BlockVector z;
      solve_transposed_advection_system(advection_field, current_field_adjoint, z);

      if (this->get_parameters().advection_stabilization_method
          == Parameters<dim>::AdvectionStabilizationMethod::entropy_viscosity)
        {
          ArtificialViscosityAdjoint<dim> artificial_viscosity_adjoint(*simulator);
          // The diffusion part of the assembled residual depends on the
          // cellwise artificial-viscosity coefficient. First collect the seed
          // for that coefficient from the residual transpose; then push it
          // further back through the entropy-viscosity estimator to old field
          // and old velocity states.
          const std::vector<double> artificial_viscosity_seed =
            artificial_viscosity_adjoint.assemble_coefficient_seed(advection_field, snapshot, z);

          double local_seed_norm_square = 0.0;
          for (const double value : artificial_viscosity_seed)
            local_seed_norm_square += value * value;

          this->get_pcout() << "   Artificial-viscosity coefficient adjoint seed norm: "
                            << std::sqrt(local_seed_norm_square) << std::endl;

          artificial_viscosity_adjoint.add_estimator_transpose(
            advection_field,
            snapshot,
            artificial_viscosity_seed,
            previous_field_adjoint,
            previous_previous_field_adjoint,
            previous_stokes_adjoint_rhs,
            previous_previous_stokes_adjoint_rhs);
        }

      // Reverse the computation graph in dependency order. Stabilization is
      // handled first because its coefficient was used while forming A_n;
      // the ordinary BDF history and transport-velocity dependencies are then
      // accumulated into the same n-1/n-2 adjoint slots. Addition is valid
      // because all three are independent summands of the forward residual.
      add_history_transpose(advection_field,
                            snapshot,
                            z,
                            previous_field_adjoint,
                            previous_previous_field_adjoint);
      // The remaining explicit velocity dependence of the advection residual
      // is handled outside the artificial-viscosity estimator, because it
      // comes from the ordinary transport term rather than stabilization.
      add_velocity_transpose(advection_field,
                             snapshot,
                             z,
                             previous_stokes_adjoint_rhs,
                             previous_previous_stokes_adjoint_rhs);

    }



    template <int dim>
    void
    AdvectionAdjointCalculator<dim>::propagate_temperature_adjoint(const ForwardStateView<dim> &snapshot,
                                                                   const LinearAlgebra::BlockVector &current_temperature_adjoint,
                                                                   LinearAlgebra::BlockVector &previous_temperature_adjoint,
                                                                   LinearAlgebra::BlockVector *previous_previous_temperature_adjoint,
                                                                   LinearAlgebra::BlockVector &previous_stokes_adjoint_rhs,
                                                                   LinearAlgebra::BlockVector *previous_previous_stokes_adjoint_rhs,
                                                                   std::vector<LinearAlgebra::BlockVector> *previous_composition_adjoints,
                                                                   std::vector<LinearAlgebra::BlockVector> *previous_previous_composition_adjoints)
    {
      // Temperature propagation is intentionally not just a call to the
      // generic field propagation: rho*C_p can depend on temperature and
      // composition, so the transpose must pass through the material model
      // before applying the common BDF history and velocity terms.
      ScopedSimulatorState<dim> simulator_state(*simulator);

      simulator->solution = snapshot.solution.get();
      simulator->old_solution = snapshot.old_solution.get();
      simulator->old_old_solution = snapshot.old_old_solution.get();
      simulator->current_linearization_point = snapshot.advection_linearization_point.get();
      simulator->time = snapshot.time;
      simulator->time_step = snapshot.time_step;
      simulator->old_time_step = snapshot.old_time_step;
      simulator->timestep_number = snapshot.timestep_number;
      simulator->last_pressure_normalization_adjustment =
        snapshot.pressure_normalization_adjustment;

      BoundaryConditions<dim> boundary_conditions;
      boundary_conditions.build_temperature_adjoint_constraints(snapshot,
                                                                simulator->current_constraints);

      simulator->assemble_advection_system(AdvectionField::temperature());

      LinearAlgebra::BlockVector z;
      solve_transposed_advection_system(AdvectionField::temperature(),
                                        current_temperature_adjoint,
                                        z);

      add_temperature_material_prefactor_transpose(
        snapshot,
        z,
        previous_temperature_adjoint,
        previous_previous_temperature_adjoint,
        previous_composition_adjoints,
        previous_previous_composition_adjoints);

      if (this->get_parameters().advection_stabilization_method
          == Parameters<dim>::AdvectionStabilizationMethod::entropy_viscosity)
        {
          ArtificialViscosityAdjoint<dim> artificial_viscosity_adjoint(*simulator);
          // The temperature artificial-viscosity estimator can depend on
          // temperature, composition, and velocity through residuals and
          // material prefactors. The optional composition slices allow those
          // dependencies to be accumulated alongside the temperature history.
          const std::vector<double> artificial_viscosity_seed =
            artificial_viscosity_adjoint.assemble_coefficient_seed(AdvectionField::temperature(),
                                                                   snapshot,
                                                                   z);

          double local_seed_norm_square = 0.0;
          for (const double value : artificial_viscosity_seed)
            local_seed_norm_square += value * value;

          this->get_pcout() << "   Artificial-viscosity coefficient adjoint seed norm: "
                            << std::sqrt(local_seed_norm_square) << std::endl;

          artificial_viscosity_adjoint.add_estimator_transpose(
            AdvectionField::temperature(),
            snapshot,
            artificial_viscosity_seed,
            previous_temperature_adjoint,
            previous_previous_temperature_adjoint,
            previous_stokes_adjoint_rhs,
            previous_previous_stokes_adjoint_rhs,
            previous_composition_adjoints,
            previous_previous_composition_adjoints);
        }

      add_history_transpose(AdvectionField::temperature(),
                            snapshot,
                            z,
                            previous_temperature_adjoint,
                            previous_previous_temperature_adjoint);
      add_velocity_transpose(AdvectionField::temperature(),
                             snapshot,
                             z,
                             previous_stokes_adjoint_rhs,
                             previous_previous_stokes_adjoint_rhs);

    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template class Adjoint::AdvectionAdjointCalculator<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
