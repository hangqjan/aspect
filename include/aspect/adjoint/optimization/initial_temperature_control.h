/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#ifndef _aspect_adjoint_optimization_initial_temperature_control_h
#define _aspect_adjoint_optimization_initial_temperature_control_h

#include <aspect/global.h>
#include <aspect/linear_algebra_types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_control.h>

#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace aspect
{
  namespace Adjoint
  {
    /**
     * Fixed initial-temperature control space for a single ASPECT temperature
     * component.  Its coordinates are global DoF numbers in the simulator's
     * component-wise DoFHandler numbering, not a separate discretisation.
     *
     * The authoritative convention is
     * \f[T=T_{fixed}+Pm,\quad \delta T=P\delta m,\quad
     * g_c=P^Tg_{raw},\quad M_c=P^TM_TP.\f]
     * Thus the ROL gradient is the primal L2 Riesz representative satisfying
     * \f[M_cg_{L2}=g_c\f].
     */
    template <int dim>
    class InitialTemperatureControl
    {
      public:
        using Expansion = std::vector<std::pair<types::global_dof_index, double>>;

        InitialTemperatureControl(const dealii::DoFHandler<dim> &dof_handler,
                                  const dealii::ComponentMask &temperature_mask,
                                  const dealii::AffineConstraints<double> &constraints,
                                  const MPI_Comm communicator);

        InitialTemperatureControl(const dealii::DoFHandler<dim> &dof_handler,
                                  const dealii::ComponentMask &temperature_mask,
                                  const dealii::AffineConstraints<double> &constraints,
                                  const MPI_Comm communicator,
                                  const types::global_dof_index temperature_block_first_dof,
                                  const dealii::IndexSet &temperature_block_owned,
                                  const dealii::IndexSet &temperature_block_relevant);

        const dealii::IndexSet &locally_owned_control_dofs() const;
        const dealii::IndexSet &locally_relevant_control_dofs() const;
        const dealii::IndexSet &temperature_block_owned_dofs() const;
        const dealii::IndexSet &temperature_block_relevant_dofs() const;
        void assert_temperature_block_mapping() const;
        types::global_dof_index n_control_dofs() const;

        /**
         * Inject the homogeneous perturbation Pm into an owned temperature
         * block. P contains free rows and hanging-node interpolation rows;
         * Dirichlet rows are zero so an optimization step cannot alter a
         * prescribed temperature boundary.
         */
        void inject_perturbation(const LinearAlgebra::Vector &control,
                                 LinearAlgebra::Vector &temperature) const;

        /**
         * Inject the physical state T_fixed + Pm. The fixed field carries
         * inhomogeneous boundary values, while Pm is strictly homogeneous.
         * Keeping these two pieces separate makes bounds and adjoints refer
         * to the same control convention.
         */
        void inject_physical(const LinearAlgebra::Vector &control,
                             const LinearAlgebra::Vector &fixed_temperature,
                             LinearAlgebra::Vector &temperature) const;

        /**
         * Restrict a temperature coefficient covector exactly as g_c=P^T
         * g_raw. This is a transpose operation, not interpolation: each
         * hanging-node contribution is accumulated into its free masters.
         */
        void restrict_covector(const LinearAlgebra::Vector &raw_covector,
                               LinearAlgebra::Vector &control_covector) const;

        /**
         * Assemble and cache M_c=P^T M_T P. This sparse distributed matrix
         * defines the L2 inner product in reduced coordinates and must never
         * be replaced by a dense control-space matrix.
         */
        void assemble_mass_matrix(const dealii::Mapping<dim> &mapping,
                                  const dealii::Quadrature<dim> &quadrature);

        std::shared_ptr<const LinearAlgebra::SparseMatrix> mass_matrix() const;

        /** Solve M_c result=rhs using the cached Jacobi preconditioner. */
        void solve_l2_riesz(const LinearAlgebra::Vector &rhs,
                            LinearAlgebra::Vector &result,
                            const double relative_tolerance = 1e-12) const;

        /** The homogeneous row of P belonging to a temperature DoF. */
        Expansion expansion(const types::global_dof_index temperature_dof) const;

      private:
        void assert_control_layout(const LinearAlgebra::Vector &control) const;
        void assert_temperature_layout(const LinearAlgebra::Vector &temperature) const;
        void build_homogeneous_constraints();
        types::global_dof_index temperature_block_index(const types::global_dof_index dof) const;
        Expansion expand(const types::global_dof_index dof) const;

        const dealii::DoFHandler<dim> &dof_handler;
        const dealii::ComponentMask temperature_mask;
        const dealii::AffineConstraints<double> &constraints;
        const MPI_Comm communicator;
        const types::global_dof_index temperature_block_first_dof;
        dealii::IndexSet locally_owned_temperature_dofs;
        dealii::IndexSet locally_relevant_temperature_dofs;
        dealii::IndexSet locally_owned_temperature_block_dofs;
        dealii::IndexSet locally_relevant_temperature_block_dofs;
        dealii::IndexSet locally_owned_controls;
        dealii::IndexSet locally_relevant_controls;
        std::map<types::global_dof_index, types::global_dof_index> full_to_reduced;
        dealii::AffineConstraints<double> homogeneous_constraints;
        dealii::AffineConstraints<double> physical_constraints;
        std::shared_ptr<LinearAlgebra::SparseMatrix> reduced_mass_matrix;
        std::shared_ptr<LinearAlgebra::PreconditionJacobi> mass_preconditioner;
    };
  }
}

#endif
