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

#ifndef _aspect_adjoint_operators_advection_adjoint_calculator_h
#define _aspect_adjoint_operators_advection_adjoint_calculator_h

#include <aspect/adjoint/core/state.h>
#include <aspect/advection_field.h>
#include <aspect/global.h>
#include <aspect/simulator_access.h>

#include <vector>

namespace aspect
{
  template <int dim> class Simulator;

  namespace Adjoint
  {
    namespace Diagnostics
    {
      template <int dim> class AdvectionOperatorChecks;
    }

    /**
     * Transpose actions for finite-element advection equations in the
     * time-dependent adjoint workflow.
     *
     * In the forward run, one accepted advection step can be viewed
     * schematically as
     *
     * @code
     *   A_n(q_n, u_n, ...) q_n = b_n(q_{n-1}, q_{n-2}, u^*_n, ...)
     * @endcode
     *
     * where @p A_n is the assembled advection matrix at the restored
     * linearization point and the right hand side contains BDF history,
     * extrapolated velocity, material prefactors, and stabilization terms.
     * Given a seed \f$\bar q_n\f$ from the later adjoint calculation, this
     * helper solves \f$A_n^T z_n = \bar q_n\f$ and then applies the transpose
     * of each right-hand-side dependence to the older time levels
     * \f$q_{n-1}\f$ and, for BDF2, \f$q_{n-2}\f$.
     *
     * The vectors passed to the public propagation functions are full-system
     * block vectors. Only the requested field or Stokes blocks are touched,
     * but full vectors are used so that constraints, material-model queries,
     * and ASPECT's existing assembly utilities can be reused without building
     * a separate reduced adjoint data structure.
     *
     * @ingroup Adjoint
     */
    template <int dim>
    class AdvectionAdjointCalculator : public SimulatorAccess<dim>
    {
      public:
        /** Preconditioners for the transposed advection solve. */
        enum class Preconditioner
        {
          none,
          forward_ilu_experimental,
          transpose_ilu
        };

        explicit
        AdvectionAdjointCalculator(Simulator<dim> &simulator);

        /** Select the preconditioner used by subsequent transposed solves. */
        void
        set_preconditioner(const Preconditioner preconditioner);

        /** Enable one none-versus-ILU(A^T) diagnostic per field type. */
        void
        set_run_preconditioner_comparison(const bool run_comparison);

        /**
         * Restore @p snapshot, assemble the forward advection operator for
         * @p advection_field, solve \f$A_n^T z_n = \bar q_n\f$, and add all
         * supported transposed right-hand-side contributions to older
         * adjoints.
         *
         * @p current_field_adjoint is the seed \f$\bar q_n\f$ accumulated at
         * the current accepted state. @p previous_field_adjoint receives
         * contributions for \f$q_{n-1}\f$ and
         * @p previous_previous_field_adjoint receives contributions for
         * \f$q_{n-2}\f$ when the restored step used BDF2. The Stokes right
         * hand sides collect the transpose of velocity dependence; they are
         * solved later by Manager because velocity is constrained by the
         * Stokes equation, not by the advection equation itself.
         */
        void
        propagate_advection_field_adjoint(const AdvectionField &advection_field,
                                          const ForwardStateView<dim> &snapshot,
                                          const LinearAlgebra::BlockVector &current_field_adjoint,
                                          LinearAlgebra::BlockVector &previous_field_adjoint,
                                          LinearAlgebra::BlockVector *previous_previous_field_adjoint,
                                          LinearAlgebra::BlockVector &previous_stokes_adjoint_rhs,
                                          LinearAlgebra::BlockVector *previous_previous_stokes_adjoint_rhs);

        /**
         * Temperature-specific wrapper around propagate_advection_field_adjoint().
         *
         * The temperature equation contains material prefactors such as
         * \f$\rho C_p\f$. When these depend on temperature or composition,
         * the transpose of the temperature advection residual also feeds
         * adjoints of older temperature and composition states. The optional
         * composition-vector arguments provide those slices of the reverse
         * time history.
         */
        void
        propagate_temperature_adjoint(const ForwardStateView<dim> &snapshot,
                                      const LinearAlgebra::BlockVector &current_temperature_adjoint,
                                      LinearAlgebra::BlockVector &previous_temperature_adjoint,
                                      LinearAlgebra::BlockVector *previous_previous_temperature_adjoint,
                                      LinearAlgebra::BlockVector &previous_stokes_adjoint_rhs,
                                      LinearAlgebra::BlockVector *previous_previous_stokes_adjoint_rhs,
                                      std::vector<LinearAlgebra::BlockVector> *previous_composition_adjoints = nullptr,
                                      std::vector<LinearAlgebra::BlockVector> *previous_previous_composition_adjoints = nullptr);

      private:
        friend class Diagnostics::AdvectionOperatorChecks<dim>;

        /**
         * Test-driver implementation of a finite-difference dot-product check
         * for the velocity dependence of one restored advection solve.
         */
        void
        run_velocity_transpose_diagnostic(const AdvectionField &advection_field,
                                          const ForwardStateView<dim> &snapshot,
                                          const LinearAlgebra::BlockVector &current_field_adjoint,
                                          const double epsilon) const;

        /**
         * Run the transposed advection solve and verify that constrained
         * adjoint DoFs on @p advection_field are exactly homogeneous.
         */
        void
        run_transposed_advection_constraint_diagnostic(const AdvectionField &advection_field,
                                                       const ForwardStateView<dim> &snapshot,
                                                       const LinearAlgebra::BlockVector &current_field_adjoint) const;

        /**
         * Run the isolated R_q -> nu_art coefficient-seed diagnostic for one
         * restored advection solve.
         */
        void
        run_artificial_viscosity_coefficient_seed_diagnostic(const AdvectionField &advection_field,
                                                             const ForwardStateView<dim> &snapshot,
                                                             const LinearAlgebra::BlockVector &current_field_adjoint,
                                                             const double epsilon) const;

        /**
         * Run the standalone entropy-viscosity estimator transpose diagnostic
         * for one restored advection solve.
         */
        void
        run_artificial_viscosity_estimator_transpose_diagnostic(const AdvectionField &advection_field,
                                                                const ForwardStateView<dim> &snapshot,
                                                                const double epsilon) const;

        /**
         * Run the frozen-denominator local entropy-residual transpose
         * diagnostic for one restored advection solve.
         */
        void
        run_local_entropy_residual_transpose_diagnostic(const AdvectionField &advection_field,
                                                        const ForwardStateView<dim> &snapshot,
                                                        const double epsilon) const;

        /**
         * Run the scalar denominator diagnostic for global_entropy_variation.
         */
        void
        run_global_entropy_denominator_scalar_diagnostic(const AdvectionField &advection_field,
                                                         const ForwardStateView<dim> &snapshot,
                                                         const double epsilon) const;

        /**
         * Run the standalone frozen-average_field transpose diagnostic for
         * global_entropy_variation(field_history).
         */
        void
        run_global_entropy_variation_transpose_diagnostic(const AdvectionField &advection_field,
                                                          const ForwardStateView<dim> &snapshot,
                                                          const double epsilon) const;

        Preconditioner preconditioner = Preconditioner::transpose_ilu;
        bool run_preconditioner_comparison = false;
        mutable bool temperature_preconditioner_comparison_completed = false;
        mutable bool composition_preconditioner_comparison_completed = false;

        /**
         * Solve the transposed field block for one restored advection system.
         *
         * This is the local algebraic step
         * \f$z=A_n^{-T}\bar q_n\f$. The returned vector is full-system shaped,
         * but only the block belonging to @p advection_field is populated.
         * Constrained degrees of freedom are set to zero because adjoint
         * variables satisfy homogeneous versions of the primal constraints.
         */
        void
        solve_transposed_advection_system(const AdvectionField &advection_field,
                                          const LinearAlgebra::BlockVector &rhs,
                                          LinearAlgebra::BlockVector &solution) const;

        /**
         * Add the transpose of BDF history mass terms to older field states.
         *
         * For the first time step this contributes only to q_{n-1}. For BDF2
         * it splits the seed between q_{n-1} and q_{n-2} using the same
         * variable-time-step coefficients as the forward advection assembly.
         */
        void
        add_history_transpose(const AdvectionField &advection_field,
                              const ForwardStateView<dim> &snapshot,
                              const LinearAlgebra::BlockVector &z,
                              LinearAlgebra::BlockVector &previous_field_adjoint,
                              LinearAlgebra::BlockVector *previous_previous_field_adjoint) const;

        /**
         * Add the transpose of the extrapolated velocity term in the
         * advection residual.
         *
         * The forward residual uses a velocity linearization/extrapolation
         * assembled from earlier Stokes states. This action maps the
         * advection transpose solution @p z into Stokes adjoint right hand
         * sides at q_{n-1} and, for BDF2, q_{n-2}.
         */
        void
        add_velocity_transpose(const AdvectionField &advection_field,
                               const ForwardStateView<dim> &snapshot,
                               const LinearAlgebra::BlockVector &z,
                               LinearAlgebra::BlockVector &previous_stokes_adjoint_rhs,
                               LinearAlgebra::BlockVector *previous_previous_stokes_adjoint_rhs) const;

        /**
         * Add the transpose of the temperature advection prefactor
         * \f$\rho C_p\f$.
         *
         * This term is separate from add_history_transpose() because its
         * derivative passes through the material model. The material-model
         * VJP scatters back to temperature and composition inputs at the old
         * time levels used by the prefactor.
         */
        void
        add_temperature_material_prefactor_transpose(const ForwardStateView<dim> &snapshot,
                                                     const LinearAlgebra::BlockVector &z,
                                                     LinearAlgebra::BlockVector &previous_temperature_adjoint,
                                                     LinearAlgebra::BlockVector *previous_previous_temperature_adjoint,
                                                     std::vector<LinearAlgebra::BlockVector> *previous_composition_adjoints,
                                                     std::vector<LinearAlgebra::BlockVector> *previous_previous_composition_adjoints) const;

        Simulator<dim> *simulator;
    };
  }
}

#endif
