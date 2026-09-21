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

#ifndef _aspect_adjoint_core_state_h
#define _aspect_adjoint_core_state_h

#include <aspect/global.h>

#include <deal.II/lac/affine_constraints.h>

#include <string>

namespace aspect
{
  namespace Adjoint
  {
    /**
     * A non-owning full-system-vector reference used by a transient replay
     * view. Existing adjoint helpers accept block vectors by const reference,
     * so this conversion preserves those interfaces without making another
     * deep copy of q_n, q_(n-1), or q_(n-2).
     */
    struct TrajectoryVectorReference
    {
        explicit TrajectoryVectorReference(const LinearAlgebra::BlockVector &reference)
          : vector(&reference)
        {}

        operator const LinearAlgebra::BlockVector &() const
        {
          return get();
        }

        const LinearAlgebra::BlockVector &
        get() const
        {
          Assert(vector != nullptr, ExcInternalError());
          return *vector;
        }

      private:
        const LinearAlgebra::BlockVector *vector;
    };

    /**
     * A data structure that stores the value and right hand side contribution
     * of one objective before the adjoint linear solve.
     *
     * @ingroup Adjoint
     */
    struct ObjectiveResult
    {
      /**
       * Objective functional name used in output and repositories.
       */
      std::string objective_name;

      /**
       * Scalar objective value.
       */
      double value = 0.0;

      /**
       * Adjoint right hand side contribution associated with this objective.
       */
      LinearAlgebra::BlockVector right_hand_side;
    };

    /**
     * A temporary reconstructed view of one accepted forward time step used
     * by the time-dependent adjoint workflow.
     *
     * The reverse sweep replays, in reverse order, the linearized algebra
     * that was assembled during the forward run. ASPECT's assembly routines
     * read most state through the mutable Simulator object, so the reverse
     * code temporarily materializes full-system vectors here before applying
     * a transpose action. This object is never the persistent trajectory
     * format: TrajectoryStore contains only unique physical fields,
     * and this view obtains BDF history by record index.
     *
     * The temporal meaning of the vectors is the usual ASPECT convention for
     * an accepted time step n:
     *
     * - solution is the converged state q_n.
     * - old_solution is q_{n-1}, i.e. the state used in BDF history terms.
     * - old_old_solution is q_{n-2}; it is only active for BDF2 steps.
     * - advection_linearization_point is reconstructed from q_{n-1}.
     * - stokes_linearization_point is reconstructed from q_n under the v1
     *   T,C-dependent Picard Stokes restrictions.
     *
     * @ingroup Adjoint
     */
    template <int dim>
    class ForwardStateView
    {
      public:
        /**
         * Construct a complete non-owning view of one reconstructed forward
         * state. All vector references must outlive this object.
         */
        ForwardStateView(const LinearAlgebra::BlockVector &solution,
                         const LinearAlgebra::BlockVector &old_solution,
                         const LinearAlgebra::BlockVector &old_old_solution,
                         const LinearAlgebra::BlockVector &advection_linearization_point,
                         const LinearAlgebra::BlockVector &stokes_linearization_point,
                         const double time,
                         const double time_step,
                         const double old_time_step,
                         const unsigned int timestep_number)
          :
          solution(solution),
          old_solution(old_solution),
          old_old_solution(old_old_solution),
          advection_linearization_point(advection_linearization_point),
          stokes_linearization_point(stokes_linearization_point),
          time(time),
          time_step(time_step),
          old_time_step(old_time_step),
          timestep_number(timestep_number)
        {}

        TrajectoryVectorReference solution;
        TrajectoryVectorReference old_solution;
        TrajectoryVectorReference old_old_solution;
        TrajectoryVectorReference advection_linearization_point;
        TrajectoryVectorReference stokes_linearization_point;

        /**
         * The constraints reconstructed from the fixed mesh, recorded time,
         * and configured boundary managers. This copy exists only while the
         * transient view is used to build homogeneous adjoint constraints; it
         * is never written to the trajectory archive.
         */
        AffineConstraints<double> constraints;

        double time = 0.0;
        double time_step = 0.0;
        double old_time_step = 0.0;
        double pressure_normalization_adjustment = 0.0;
        unsigned int timestep_number = numbers::invalid_unsigned_int;
    };

    /**
     * A data structure that stores the adjoint solution associated with a
     * single objective.
     *
     * @ingroup Adjoint
     */
    struct AdjointState
    {
      /**
       * Objective functional name associated with this adjoint solution.
       */
      std::string objective_name;

      /**
       * Solution of the adjoint Stokes system.
       */
      LinearAlgebra::BlockVector solution;
    };

  }
}

#endif
