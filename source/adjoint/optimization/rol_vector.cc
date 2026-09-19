/*
  Copyright (C) 2026 by the authors of the ASPECT code.

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

#include <aspect/adjoint/optimization/rol_vector.h>

#include <algorithm>
#include <climits>
#include <cstdint>

namespace aspect
{
  namespace Adjoint
  {
    namespace
    {
      bool
      communicators_are_compatible(const MPI_Comm first,
                                   const MPI_Comm second)
      {
        // Congruent communicators have identical process groups and rank
        // ordering even when they are different MPI handles. That is exactly
        // the compatibility needed for entrywise distributed vector algebra.
        int comparison = MPI_UNEQUAL;
        const int error = MPI_Comm_compare(first, second, &comparison);
        AssertThrow(error == MPI_SUCCESS,
                    ExcMessage("Unable to compare MPI communicators for ROL vectors."));
        return comparison == MPI_IDENT || comparison == MPI_CONGRUENT;
      }


      void
      finish_owned_write(LinearAlgebra::Vector &vector)
      {
        vector.compress(dealii::VectorOperation::insert);
      }
    }


    ROLDistributedVector::ROLDistributedVector(const dealii::IndexSet &locally_owned_elements,
                                               const MPI_Comm communicator)
    {
      vector.reinit(locally_owned_elements, communicator);
      AssertThrow(!vector.has_ghost_elements(),
                  ExcMessage("ROLDistributedVector requires owned-only storage."));
    }


    ROLDistributedVector::ROLDistributedVector(const LinearAlgebra::Vector &input_vector)
      : vector(input_vector)
    {
      AssertThrow(!vector.has_ghost_elements(),
                  ExcMessage("ROLDistributedVector requires owned-only storage."));
    }


    void
    ROLDistributedVector::plus(const ROL::Vector<double> &input_vector)
    {
      const ROLDistributedVector &other = get_vector(input_vector);
      assert_compatible(other);
      vector += other.vector;
    }


    void
    ROLDistributedVector::scale(const double alpha)
    {
      vector *= alpha;
    }


    double
    ROLDistributedVector::dot(const ROL::Vector<double> &input_vector) const
    {
      const ROLDistributedVector &other = get_vector(input_vector);
      assert_compatible(other);

      // Each global entry is owned exactly once. Sum owned products locally,
      // then one all-reduction realizes the global Euclidean inner product;
      // ghosted vectors are forbidden by construction to prevent duplicates.
      double local_dot = 0.0;
      for (const auto index : vector.locally_owned_elements())
        local_dot += vector[index] * other.vector[index];

      return dealii::Utilities::MPI::sum(local_dot, vector.get_mpi_communicator());
    }


    double
    ROLDistributedVector::norm() const
    {
      return std::sqrt(dot(*this));
    }


    ROL::Ptr<ROL::Vector<double>>
    ROLDistributedVector::clone() const
    {
      return ROL::makePtr<ROLDistributedVector>(vector.locally_owned_elements(),
                                                vector.get_mpi_communicator());
    }


    void
    ROLDistributedVector::axpy(const double alpha,
                               const ROL::Vector<double> &input_vector)
    {
      const ROLDistributedVector &other = get_vector(input_vector);
      assert_compatible(other);
      vector.add(alpha, other.vector);
    }


    void
    ROLDistributedVector::set(const ROL::Vector<double> &input_vector)
    {
      const ROLDistributedVector &other = get_vector(input_vector);
      assert_compatible(other);
      vector = other.vector;
    }


    ROL::Ptr<ROL::Vector<double>>
    ROLDistributedVector::basis(const int index) const
    {
      AssertThrow(index >= 0 && static_cast<types::global_dof_index>(index) < vector.size(),
                  ExcMessage("The requested ROL basis index is outside the vector range."));

      // Only the owning rank writes e_i=1. compress(insert) completes the
      // Trilinos write protocol without communicating a duplicated value.
      ROL::Ptr<ROL::Vector<double>> result = clone();
      ROLDistributedVector &distributed_result = get_vector(*result);
      if (distributed_result.vector.locally_owned_elements().is_element(index))
        distributed_result.vector[index] = 1.0;
      finish_owned_write(distributed_result.vector);
      return result;
    }


    int
    ROLDistributedVector::dimension() const
    {
      AssertThrow(vector.size() <= static_cast<types::global_dof_index>(INT_MAX),
                  ExcMessage("ROL::Vector::dimension() cannot represent this global vector size."));
      return static_cast<int>(vector.size());
    }


    void
    ROLDistributedVector::applyUnary(const ROL::Elementwise::UnaryFunction<double> &function)
    {
      for (const auto index : vector.locally_owned_elements())
        vector[index] = function.apply(vector[index]);
      finish_owned_write(vector);
    }


    void
    ROLDistributedVector::applyBinary(const ROL::Elementwise::BinaryFunction<double> &function,
                                      const ROL::Vector<double> &input_vector)
    {
      const ROLDistributedVector &other = get_vector(input_vector);
      assert_compatible(other);
      for (const auto index : vector.locally_owned_elements())
        vector[index] = function.apply(vector[index], other.vector[index]);
      finish_owned_write(vector);
    }


    double
    ROLDistributedVector::reduce(const ROL::Elementwise::ReductionOp<double> &reduction) const
    {
      // ROL's elementwise reduction first folds each ownership partition and
      // then requires the matching MPI monoid globally. Logical AND and
      // bitwise OR need integral MPI datatypes rather than applying floating-
      // point SUM semantics to ROL's encoded values.
      double local_value = reduction.initialValue();
      for (const auto index : vector.locally_owned_elements())
        reduction.reduce(vector[index], local_value);

      double global_value = reduction.initialValue();
      switch (reduction.reductionType())
        {
          case ROL::Elementwise::REDUCE_SUM:
            MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_SUM,
                          vector.get_mpi_communicator());
            break;
          case ROL::Elementwise::REDUCE_MIN:
            MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MIN,
                          vector.get_mpi_communicator());
            break;
          case ROL::Elementwise::REDUCE_MAX:
            MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX,
                          vector.get_mpi_communicator());
            break;
          case ROL::Elementwise::REDUCE_AND:
          {
            int local_flag = local_value != 0.0 ? 1 : 0;
            int global_flag = 0;
            MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LAND,
                          vector.get_mpi_communicator());
            global_value = static_cast<double>(global_flag);
            break;
          }
          case ROL::Elementwise::REDUCE_BOR:
          {
            std::uint64_t local_bits = static_cast<std::uint64_t>(local_value);
            std::uint64_t global_bits = 0;
            MPI_Allreduce(&local_bits, &global_bits, 1, MPI_UINT64_T, MPI_BOR,
                          vector.get_mpi_communicator());
            global_value = static_cast<double>(global_bits);
            break;
          }
        }
      return global_value;
    }


    void
    ROLDistributedVector::setScalar(const double value)
    {
      for (const auto index : vector.locally_owned_elements())
        vector[index] = value;
      finish_owned_write(vector);
    }


    LinearAlgebra::Vector &
    ROLDistributedVector::get_vector()
    {
      return vector;
    }


    const LinearAlgebra::Vector &
    ROLDistributedVector::get_vector() const
    {
      return vector;
    }


    ROLDistributedVector &
    ROLDistributedVector::get_vector(ROL::Vector<double> &input_vector)
    {
      ROLDistributedVector *distributed_vector =
        dynamic_cast<ROLDistributedVector *>(&input_vector);
      AssertThrow(distributed_vector != nullptr,
                  ExcMessage("ROLDistributedVector requires another ROLDistributedVector."));
      return *distributed_vector;
    }


    const ROLDistributedVector &
    ROLDistributedVector::get_vector(const ROL::Vector<double> &input_vector)
    {
      const ROLDistributedVector *distributed_vector =
        dynamic_cast<const ROLDistributedVector *>(&input_vector);
      AssertThrow(distributed_vector != nullptr,
                  ExcMessage("ROLDistributedVector requires another ROLDistributedVector."));
      return *distributed_vector;
    }


    bool
    ROLDistributedVector::has_same_layout(const ROLDistributedVector &other) const
    {
      return vector.size() == other.vector.size()
             && vector.locally_owned_elements() == other.vector.locally_owned_elements()
             && communicators_are_compatible(vector.get_mpi_communicator(),
                                             other.vector.get_mpi_communicator());
    }


    void
    ROLDistributedVector::assert_compatible(const ROLDistributedVector &other) const
    {
      AssertThrow(has_same_layout(other),
                  ExcMessage("ROLDistributedVector operands must have the same global size, "
                             "owned IndexSet, and MPI communicator."));
    }


    ROLReducedL2Vector::ROLReducedL2Vector(
      const dealii::IndexSet &locally_owned_elements,
      const MPI_Comm communicator,
      const std::shared_ptr<const LinearAlgebra::SparseMatrix> &mass_matrix)
      : ROLDistributedVector(locally_owned_elements, communicator)
      , metric_mass_matrix(mass_matrix)
    {
      AssertThrow(metric_mass_matrix != nullptr,
                  ExcMessage("ROLReducedL2Vector requires a mass-metric matrix."));
      AssertThrow(metric_mass_matrix->m() == get_vector().size()
                  && metric_mass_matrix->n() == get_vector().size(),
                  ExcMessage("The ROL L2 metric has an incompatible global size."));
    }


    ROLReducedL2Vector::ROLReducedL2Vector(
      const LinearAlgebra::Vector &input_vector,
      const std::shared_ptr<const LinearAlgebra::SparseMatrix> &mass_matrix)
      : ROLDistributedVector(input_vector)
      , metric_mass_matrix(mass_matrix)
    {
      AssertThrow(metric_mass_matrix != nullptr,
                  ExcMessage("ROLReducedL2Vector requires a mass-metric matrix."));
      AssertThrow(metric_mass_matrix->m() == get_vector().size()
                  && metric_mass_matrix->n() == get_vector().size(),
                  ExcMessage("The ROL L2 metric has an incompatible global size."));
    }


    double
    ROLReducedL2Vector::dot(const ROL::Vector<double> &input_vector) const
    {
      const ROLReducedL2Vector *other = dynamic_cast<const ROLReducedL2Vector *>(&input_vector);
      AssertThrow(other != nullptr,
                  ExcMessage("An L2 ROL vector can only be paired with another L2 ROL vector."));
      assert_metric_compatible(*other);

      // Reduced coefficients represent FE temperature perturbations. Their
      // physical inner product is
      //   <x,y>_L2 = integral (P x)(P y) dx = x^T M_c y,
      // where M_c=P^T M_T P is assembled by InitialTemperatureControl.
      LinearAlgebra::Vector mass_times_other(other->get_vector());
      metric_mass_matrix->vmult(mass_times_other, other->get_vector());
      return get_vector() * mass_times_other;
    }


    ROL::Ptr<ROL::Vector<double>>
    ROLReducedL2Vector::clone() const
    {
      return ROL::makePtr<ROLReducedL2Vector>(get_vector().locally_owned_elements(),
                                              get_vector().get_mpi_communicator(),
                                              metric_mass_matrix);
    }


    const std::shared_ptr<const LinearAlgebra::SparseMatrix> &
    ROLReducedL2Vector::mass_matrix() const
    {
      return metric_mass_matrix;
    }


    void
    ROLReducedL2Vector::assert_metric_compatible(const ROLReducedL2Vector &other) const
    {
      AssertThrow(has_same_layout(other) && metric_mass_matrix == other.metric_mass_matrix,
                  ExcMessage("L2 ROL vector operands must share layout and mass metric."));
    }


    ROLDistributedBoundConstraint::ROLDistributedBoundConstraint(
      const LinearAlgebra::Vector &lower_bound,
      const LinearAlgebra::Vector &upper_bound,
      const double                 scale,
      const double                 feasibility_tolerance)
      : scale(scale)
      , feasibility_tolerance(feasibility_tolerance)
      , minimum_bound_width(std::numeric_limits<double>::infinity())
    {
      AssertThrow(!lower_bound.has_ghost_elements() && !upper_bound.has_ghost_elements(),
                  ExcMessage("ROLDistributedBoundConstraint requires owned-only bounds."));
      ROLDistributedVector lower_vector(lower_bound);
      ROLDistributedVector upper_vector(upper_bound);
      AssertThrow(lower_vector.has_same_layout(upper_vector),
                  ExcMessage("The lower and upper ROL bounds have incompatible layouts."));
      AssertThrow(scale > 0.0 && feasibility_tolerance > 0.0,
                  ExcMessage("ROL bound scale and feasibility tolerance must be positive."));

      for (const auto index : lower_bound.locally_owned_elements())
        {
          AssertThrow(lower_bound[index] <= upper_bound[index],
                      ExcMessage("A ROL lower bound exceeds its upper bound."));
          minimum_bound_width = std::min(minimum_bound_width,
                                         upper_bound[index] - lower_bound[index]);
        }
      // Active-set tolerances must be consistent across ranks. Use half of the
      // narrowest global interval as a hard cap so upper and lower active
      // neighborhoods cannot overlap.
      minimum_bound_width = dealii::Utilities::MPI::min(
                              minimum_bound_width,
                              lower_bound.get_mpi_communicator());
      minimum_bound_width *= 0.5;

      this->lower_ = ROL::makePtr<ROLDistributedVector>(lower_bound);
      this->upper_ = ROL::makePtr<ROLDistributedVector>(upper_bound);
      this->activate();
    }


    void
    ROLDistributedBoundConstraint::project(ROL::Vector<double> &input_vector)
    {
      ROLDistributedVector &distributed_vector = ROLDistributedVector::get_vector(input_vector);
      assert_compatible(distributed_vector);
      const LinearAlgebra::Vector &lower = ROLDistributedVector::get_vector(*this->lower_).get_vector();
      const LinearAlgebra::Vector &upper = ROLDistributedVector::get_vector(*this->upper_).get_vector();
      LinearAlgebra::Vector &values = distributed_vector.get_vector();

      // Orthogonal projection for box constraints is componentwise clipping.
      // Ownership partitions are independent, so no reduction is required.
      for (const auto index : values.locally_owned_elements())
        values[index] = std::min(static_cast<double>(upper[index]),
                                 std::max(static_cast<double>(lower[index]),
                                          static_cast<double>(values[index])));
      finish_owned_write(values);
    }


    void
    ROLDistributedBoundConstraint::projectInterior(ROL::Vector<double> &input_vector)
    {
      ROLDistributedVector &distributed_vector = ROLDistributedVector::get_vector(input_vector);
      assert_compatible(distributed_vector);
      const LinearAlgebra::Vector &lower = ROLDistributedVector::get_vector(*this->lower_).get_vector();
      const LinearAlgebra::Vector &upper = ROLDistributedVector::get_vector(*this->upper_).get_vector();
      LinearAlgebra::Vector &values = distributed_vector.get_vector();
      const double tolerance = 100.0 * std::numeric_limits<double>::epsilon();

      // Construct a strictly feasible point. The additive margin based on the
      // narrowest interval protects zero/small bounds; the relative margin
      // protects large signed bounds from losing separation to roundoff.
      for (const auto index : values.locally_owned_elements())
        {
          const double lower_interior =
            std::min(lower[index] + feasibility_tolerance * minimum_bound_width,
                     lower[index] < -tolerance ? (1.0 - feasibility_tolerance) * lower[index] :
                     lower[index] > tolerance ? (1.0 + feasibility_tolerance) * lower[index] :
                     lower[index] + feasibility_tolerance);
          const double upper_interior =
            std::max(upper[index] - feasibility_tolerance * minimum_bound_width,
                     upper[index] < -tolerance ? (1.0 + feasibility_tolerance) * upper[index] :
                     upper[index] > tolerance ? (1.0 - feasibility_tolerance) * upper[index] :
                     upper[index] - feasibility_tolerance);
          values[index] = std::min(upper_interior,
                                   std::max(lower_interior,
                                            static_cast<double>(values[index])));
        }
      finish_owned_write(values);
    }


    void
    ROLDistributedBoundConstraint::pruneUpperActive(ROL::Vector<double> &input_vector,
                                                    const ROL::Vector<double> &x,
                                                    const double epsilon)
    {
      ROLDistributedVector &values = ROLDistributedVector::get_vector(input_vector);
      const ROLDistributedVector &state = ROLDistributedVector::get_vector(x);
      assert_compatible(values);
      assert_compatible(state);
      const LinearAlgebra::Vector &upper = ROLDistributedVector::get_vector(*this->upper_).get_vector();
      const double threshold = active_tolerance(epsilon);

      // Remove components whose state lies in the epsilon-neighborhood of the
      // upper active set. This overload ignores direction and is used when ROL
      // requests the geometric active-set projector.
      for (const auto index : values.get_vector().locally_owned_elements())
        if (state.get_vector()[index] >= upper[index] - threshold)
          values.get_vector()[index] = 0.0;
      finish_owned_write(values.get_vector());
    }


    void
    ROLDistributedBoundConstraint::pruneUpperActive(ROL::Vector<double> &input_vector,
                                                    const ROL::Vector<double> &gradient,
                                                    const ROL::Vector<double> &x,
                                                    const double x_epsilon,
                                                    const double gradient_epsilon)
    {
      ROLDistributedVector &values = ROLDistributedVector::get_vector(input_vector);
      const ROLDistributedVector &direction = ROLDistributedVector::get_vector(gradient);
      const ROLDistributedVector &state = ROLDistributedVector::get_vector(x);
      assert_compatible(values);
      assert_compatible(direction);
      assert_compatible(state);
      const LinearAlgebra::Vector &upper = ROLDistributedVector::get_vector(*this->upper_).get_vector();
      const double threshold = active_tolerance(x_epsilon);

      // The directional overload prunes only a component that is both near
      // the upper bound and points outward according to ROL's gradient-sign
      // convention. Feasible inward directions remain available.
      for (const auto index : values.get_vector().locally_owned_elements())
        if (state.get_vector()[index] >= upper[index] - threshold
            && direction.get_vector()[index] < -gradient_epsilon)
          values.get_vector()[index] = 0.0;
      finish_owned_write(values.get_vector());
    }


    void
    ROLDistributedBoundConstraint::pruneLowerActive(ROL::Vector<double> &input_vector,
                                                    const ROL::Vector<double> &x,
                                                    const double epsilon)
    {
      ROLDistributedVector &values = ROLDistributedVector::get_vector(input_vector);
      const ROLDistributedVector &state = ROLDistributedVector::get_vector(x);
      assert_compatible(values);
      assert_compatible(state);
      const LinearAlgebra::Vector &lower = ROLDistributedVector::get_vector(*this->lower_).get_vector();
      const double threshold = active_tolerance(epsilon);

      // Lower-bound counterpart of the geometric upper active-set projector.
      for (const auto index : values.get_vector().locally_owned_elements())
        if (state.get_vector()[index] <= lower[index] + threshold)
          values.get_vector()[index] = 0.0;
      finish_owned_write(values.get_vector());
    }


    void
    ROLDistributedBoundConstraint::pruneLowerActive(ROL::Vector<double> &input_vector,
                                                    const ROL::Vector<double> &gradient,
                                                    const ROL::Vector<double> &x,
                                                    const double x_epsilon,
                                                    const double gradient_epsilon)
    {
      ROLDistributedVector &values = ROLDistributedVector::get_vector(input_vector);
      const ROLDistributedVector &direction = ROLDistributedVector::get_vector(gradient);
      const ROLDistributedVector &state = ROLDistributedVector::get_vector(x);
      assert_compatible(values);
      assert_compatible(direction);
      assert_compatible(state);
      const LinearAlgebra::Vector &lower = ROLDistributedVector::get_vector(*this->lower_).get_vector();
      const double threshold = active_tolerance(x_epsilon);

      // At a lower bound, a positive gradient component is outward under the
      // sign convention used by the bound-constrained trust-region step.
      for (const auto index : values.get_vector().locally_owned_elements())
        if (state.get_vector()[index] <= lower[index] + threshold
            && direction.get_vector()[index] > gradient_epsilon)
          values.get_vector()[index] = 0.0;
      finish_owned_write(values.get_vector());
    }


    bool
    ROLDistributedBoundConstraint::isFeasible(const ROL::Vector<double> &input_vector)
    {
      const ROLDistributedVector &distributed_vector = ROLDistributedVector::get_vector(input_vector);
      assert_compatible(distributed_vector);
      const LinearAlgebra::Vector &lower = ROLDistributedVector::get_vector(*this->lower_).get_vector();
      const LinearAlgebra::Vector &upper = ROLDistributedVector::get_vector(*this->upper_).get_vector();

      bool locally_feasible = true;
      for (const auto index : distributed_vector.get_vector().locally_owned_elements())
        if (distributed_vector.get_vector()[index] < lower[index]
            || distributed_vector.get_vector()[index] > upper[index])
          {
            locally_feasible = false;
            break;
          }

      // Feasibility is global: a violation on one rank invalidates the ROL
      // iterate everywhere. MPI logical AND supplies the collective answer.
      const int local_flag = locally_feasible ? 1 : 0;
      int global_flag = 0;
      MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LAND,
                    distributed_vector.get_vector().get_mpi_communicator());
      return global_flag != 0;
    }


    void
    ROLDistributedBoundConstraint::applyInverseScalingFunction(
      ROL::Vector<double> &scaled_vector,
      const ROL::Vector<double> &input_vector,
      const ROL::Vector<double> &x,
      const ROL::Vector<double> &gradient) const
    {
      ROLDistributedVector &scaled = ROLDistributedVector::get_vector(scaled_vector);
      const ROLDistributedVector &values = ROLDistributedVector::get_vector(input_vector);
      const ROLDistributedVector &state = ROLDistributedVector::get_vector(x);
      const ROLDistributedVector &direction = ROLDistributedVector::get_vector(gradient);
      assert_compatible(scaled);
      assert_compatible(values);
      assert_compatible(state);
      assert_compatible(direction);
      build_scaling_function(scaled, state, direction);
      // Apply D(x,g)^-1 componentwise. build_scaling_function enforces a
      // strictly positive cap for feasible states, so division is well-defined.
      for (const auto index : scaled.get_vector().locally_owned_elements())
        scaled.get_vector()[index] = values.get_vector()[index] / scaled.get_vector()[index];
      finish_owned_write(scaled.get_vector());
    }


    void
    ROLDistributedBoundConstraint::applyScalingFunctionJacobian(
      ROL::Vector<double> &scaled_vector,
      const ROL::Vector<double> &input_vector,
      const ROL::Vector<double> &x,
      const ROL::Vector<double> &gradient) const
    {
      ROLDistributedVector &scaled = ROLDistributedVector::get_vector(scaled_vector);
      const ROLDistributedVector &values = ROLDistributedVector::get_vector(input_vector);
      const ROLDistributedVector &state = ROLDistributedVector::get_vector(x);
      const ROLDistributedVector &direction = ROLDistributedVector::get_vector(gradient);
      assert_compatible(scaled);
      assert_compatible(values);
      assert_compatible(state);
      assert_compatible(direction);
      const LinearAlgebra::Vector &lower = ROLDistributedVector::get_vector(*this->lower_).get_vector();
      const LinearAlgebra::Vector &upper = ROLDistributedVector::get_vector(*this->upper_).get_vector();
      build_scaling_function(scaled, state, direction);

      // Differentiate the selected piece of D(x,g). If the global cap is the
      // active minimum, D is locally constant and its Jacobian is zero;
      // otherwise the sign identifies the selected distance-to-bound branch.
      for (const auto index : scaled.get_vector().locally_owned_elements())
        {
          const double indicator = scaled.get_vector()[index] < scaling_cap(index) ? 1.0 : 0.0;
          if (indicator == 0.0)
            scaled.get_vector()[index] = 0.0;
          else
            {
              double derivative = direction.get_vector()[index] > 0.0 ? 1.0 :
                                  direction.get_vector()[index] < 0.0 ? -1.0 : 1.0;
              if (direction.get_vector()[index] == 0.0
                  && upper[index] - state.get_vector()[index]
                  < state.get_vector()[index] - lower[index])
                derivative = -1.0;
              scaled.get_vector()[index] = derivative * direction.get_vector()[index]
                                           * values.get_vector()[index];
            }
        }
      finish_owned_write(scaled.get_vector());
    }


    void
    ROLDistributedBoundConstraint::assert_compatible(const ROLDistributedVector &input_vector) const
    {
      AssertThrow(input_vector.has_same_layout(ROLDistributedVector::get_vector(*this->lower_))
                  && input_vector.has_same_layout(ROLDistributedVector::get_vector(*this->upper_)),
                  ExcMessage("ROL vector and bound layouts are incompatible."));
    }


    double
    ROLDistributedBoundConstraint::active_tolerance(const double epsilon) const
    {
      return std::min(scale * epsilon, minimum_bound_width);
    }


    double
    ROLDistributedBoundConstraint::scaling_cap(const types::global_dof_index index) const
    {
      const LinearAlgebra::Vector &lower = ROLDistributedVector::get_vector(*this->lower_).get_vector();
      const LinearAlgebra::Vector &upper = ROLDistributedVector::get_vector(*this->upper_).get_vector();
      return std::min(0.5 * (upper[index] - lower[index]), 1.0);
    }


    void
    ROLDistributedBoundConstraint::build_scaling_function(
      ROLDistributedVector &scaling,
      const ROLDistributedVector &state,
      const ROLDistributedVector &gradient) const
    {
      const LinearAlgebra::Vector &lower = ROLDistributedVector::get_vector(*this->lower_).get_vector();
      const LinearAlgebra::Vector &upper = ROLDistributedVector::get_vector(*this->upper_).get_vector();
      // Coleman-Li style diagonal scaling chooses the distance to the bound
      // toward which the descent direction points, capped for conditioning.
      // If neither bound is threatened, use the nearest feasible distance.
      for (const auto index : scaling.get_vector().locally_owned_elements())
        {
          const double gradient_value = gradient.get_vector()[index];
          const double lower_distance = state.get_vector()[index] - lower[index];
          const double upper_distance = upper[index] - state.get_vector()[index];
          const double cap = scaling_cap(index);
          if (-gradient_value > lower_distance && lower_distance <= upper_distance)
            scaling.get_vector()[index] = std::min(std::abs(gradient_value), cap);
          else if (gradient_value > upper_distance && upper_distance <= lower_distance)
            scaling.get_vector()[index] = std::min(std::abs(gradient_value), cap);
          else
            scaling.get_vector()[index] = std::min({lower_distance, upper_distance, cap});
        }
      finish_owned_write(scaling.get_vector());
    }
  }
}
