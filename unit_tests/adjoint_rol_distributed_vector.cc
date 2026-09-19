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

#include "common.h"

#include <aspect/adjoint/optimization/rol_vector.h>

#include <ROL_Elementwise_Reduce.hpp>
#include <ROL_StdVector.hpp>

#include <cmath>
#include <vector>

namespace adjoint_rol_distributed_vector_tests
{
  constexpr dealii::types::global_dof_index global_size = 11;

  dealii::IndexSet
  owned_index_set()
  {
    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    dealii::IndexSet result(global_size);
    result.add_range(global_size * rank / n_ranks,
                     global_size * (rank + 1) / n_ranks);
    return result;
  }


  void
  finish_owned_write(aspect::LinearAlgebra::Vector &vector)
  {
    vector.compress(dealii::VectorOperation::insert);
  }


  template <typename Function>
  void
  assign_owned(aspect::Adjoint::ROLDistributedVector &vector,
               const Function &function)
  {
    for (const auto index : vector.get_vector().locally_owned_elements())
      vector.get_vector()[index] = function(index);
    finish_owned_write(vector.get_vector());
  }


  void
  require_owned_values(const aspect::Adjoint::ROLDistributedVector &vector,
                       const std::function<double(dealii::types::global_dof_index)> &expected)
  {
    for (const auto index : vector.get_vector().locally_owned_elements())
      {
        INFO("global index " << index);
        REQUIRE(vector.get_vector()[index] == Approx(expected(index)).margin(1.e-13));
      }
  }


  class AddOne : public ROL::Elementwise::UnaryFunction<double>
  {
    public:
      double apply(const double &value) const override
      {
        return value + 1.0;
      }
  };


  class TakeSecond : public ROL::Elementwise::BinaryFunction<double>
  {
    public:
      double apply(const double &, const double &second) const override
      {
        return second;
      }
  };


  TEST_CASE("distributed ROL vector and bounds use owned ASPECT coefficients")
  {
    using aspect::Adjoint::ROLDistributedBoundConstraint;
    using aspect::Adjoint::ROLDistributedVector;

    const dealii::IndexSet owned = owned_index_set();
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    REQUIRE(dealii::Utilities::MPI::sum(owned.n_elements(), MPI_COMM_WORLD) == global_size);

    ROLDistributedVector x(owned, MPI_COMM_WORLD);
    ROLDistributedVector y(owned, MPI_COMM_WORLD);
    assign_owned(x, [](const auto index)
    {
      return static_cast<double>(index + 1);
    });
    assign_owned(y, [](const auto index)
    {
      return 2.0 * static_cast<double>(index + 1);
    });

    REQUIRE(x.dimension() == static_cast<int>(global_size));
    REQUIRE(y.dimension() == static_cast<int>(global_size));
    REQUIRE(!x.get_vector().has_ghost_elements());

    const double expected_dot = 2.0 * global_size * (global_size + 1)
                                * (2.0 * global_size + 1) / 6.0;
    REQUIRE(x.dot(y) == Approx(expected_dot));
    REQUIRE(x.dot(y) == Approx(x.get_vector() * y.get_vector()));
    REQUIRE(x.norm() == Approx(std::sqrt(expected_dot / 2.0)));

    auto clone_base = x.clone();
    auto &clone = ROLDistributedVector::get_vector(*clone_base);
    REQUIRE(clone.has_same_layout(x));
    clone.setScalar(-3.0);
    require_owned_values(x, [](const auto index)
    {
      return static_cast<double>(index + 1);
    });
    require_owned_values(clone, [](const auto)
    {
      return -3.0;
    });
    auto clone_of_clone = clone.clone();
    REQUIRE(ROLDistributedVector::get_vector(*clone_of_clone).has_same_layout(clone));

    ROLDistributedVector z(owned, MPI_COMM_WORLD);
    z.set(x);
    z.scale(2.0);
    require_owned_values(z, [](const auto index)
    {
      return 2.0 * static_cast<double>(index + 1);
    });
    z.set(x);
    z.plus(y);
    require_owned_values(z, [](const auto index)
    {
      return 3.0 * static_cast<double>(index + 1);
    });
    z.set(x);
    z.axpy(-0.5, y);
    require_owned_values(z, [](const auto index)
    {
      return 0.0 * static_cast<double>(index + 1);
    });

    AddOne add_one;
    z.set(x);
    z.applyUnary(add_one);
    require_owned_values(z, [](const auto index)
    {
      return static_cast<double>(index + 2);
    });
    TakeSecond take_second;
    z.applyBinary(take_second, y);
    require_owned_values(z, [](const auto index)
    {
      return 2.0 * static_cast<double>(index + 1);
    });
    ROL::Elementwise::ReductionSum<double> sum;
    REQUIRE(x.reduce(sum) == Approx(global_size * (global_size + 1) / 2.0));

    auto basis_zero = x.basis(0);
    auto basis_last = x.basis(static_cast<int>(global_size - 1));
    require_owned_values(ROLDistributedVector::get_vector(*basis_zero),
                         [](const auto index)
    {
      return index == 0 ? 1.0 : 0.0;
    });
    require_owned_values(ROLDistributedVector::get_vector(*basis_last),
                         [](const auto index)
    {
      return index == global_size - 1 ? 1.0 : 0.0;
    });

    ROL::StdVector<double> wrong_type(3, 0.0);
    REQUIRE_THROWS(x.plus(wrong_type));

    aspect::LinearAlgebra::Vector lower(owned, MPI_COMM_WORLD);
    aspect::LinearAlgebra::Vector upper(owned, MPI_COMM_WORLD);
    lower = 0.0;
    for (const auto index : upper.locally_owned_elements())
      upper[index] = 1.0;
    finish_owned_write(upper);
    ROLDistributedBoundConstraint bounds(lower, upper);

    ROLDistributedVector projected(owned, MPI_COMM_WORLD);
    assign_owned(projected, [](const auto index)
    {
      return index % 3 == 0 ? -0.25 : index % 3 == 1 ? 0.25 : 1.25;
    });
    bounds.project(projected);
    require_owned_values(projected, [](const auto index)
    {
      return index % 3 == 0 ? 0.0 : index % 3 == 1 ? 0.25 : 1.0;
    });
    REQUIRE(bounds.isFeasible(projected));

    ROLDistributedVector feasible(owned, MPI_COMM_WORLD);
    feasible.setScalar(0.5);
    REQUIRE(bounds.isFeasible(feasible));
    for (const auto index : feasible.get_vector().locally_owned_elements())
      if (index == 0)
        feasible.get_vector()[index] = -0.1;
    finish_owned_write(feasible.get_vector());
    REQUIRE_FALSE(bounds.isFeasible(feasible));

    feasible.setScalar(0.5);
    const dealii::types::global_dof_index remote_index =
      n_ranks > 1 ? global_size / n_ranks : 0;
    for (const auto index : feasible.get_vector().locally_owned_elements())
      if (index == remote_index)
        feasible.get_vector()[index] = 1.1;
    finish_owned_write(feasible.get_vector());
    REQUIRE_FALSE(bounds.isFeasible(feasible));
    if (n_ranks > 1)
      REQUIRE((rank == 0 || remote_index >= global_size * rank / n_ranks));

    ROLDistributedVector state(owned, MPI_COMM_WORLD);
    ROLDistributedVector direction(owned, MPI_COMM_WORLD);
    ROLDistributedVector pruned(owned, MPI_COMM_WORLD);
    assign_owned(state, [](const auto index)
    {
      return index == 0 ? 0.0 : index == global_size - 1 ? 1.0 :
             index == global_size - 2 ? 0.95 : 0.5;
    });
    assign_owned(direction, [](const auto index)
    {
      return index == 0 ? 1.0 : index == global_size - 1 ? -1.0 : 0.0;
    });

    pruned.setScalar(2.0);
    bounds.pruneLowerActive(pruned, state);
    require_owned_values(pruned, [](const auto index)
    {
      return index == 0 ? 0.0 : 2.0;
    });
    pruned.setScalar(2.0);
    bounds.pruneUpperActive(pruned, state);
    require_owned_values(pruned, [](const auto index)
    {
      return index == global_size - 1 ? 0.0 : 2.0;
    });
    pruned.setScalar(2.0);
    bounds.pruneLowerActive(pruned, direction, state);
    require_owned_values(pruned, [](const auto index)
    {
      return index == 0 ? 0.0 : 2.0;
    });
    pruned.setScalar(2.0);
    bounds.pruneUpperActive(pruned, direction, state);
    require_owned_values(pruned, [](const auto index)
    {
      return index == global_size - 1 ? 0.0 : 2.0;
    });
    pruned.setScalar(2.0);
    bounds.pruneUpperActive(pruned, state, 0.1);
    require_owned_values(pruned, [](const auto index)
    {
      return index == global_size - 1 || index == global_size - 2 ? 0.0 : 2.0;
    });

    ROLDistributedVector interior(owned, MPI_COMM_WORLD);
    interior.set(state);
    bounds.projectInterior(interior);
    for (const auto index : interior.get_vector().locally_owned_elements())
      {
        REQUIRE(interior.get_vector()[index] > 0.0);
        REQUIRE(interior.get_vector()[index] < 1.0);
      }

    const auto lower_bound = bounds.getLowerBound();
    const auto upper_bound = bounds.getUpperBound();
    REQUIRE(lower_bound != ROL::nullPtr);
    REQUIRE(upper_bound != ROL::nullPtr);
    REQUIRE(ROLDistributedVector::get_vector(*lower_bound).has_same_layout(x));
    REQUIRE(ROLDistributedVector::get_vector(*upper_bound).has_same_layout(x));
    require_owned_values(ROLDistributedVector::get_vector(*lower_bound), [](const auto)
    {
      return 0.0;
    });
    require_owned_values(ROLDistributedVector::get_vector(*upper_bound), [](const auto)
    {
      return 1.0;
    });

    aspect::LinearAlgebra::Vector invalid_lower(owned, MPI_COMM_WORLD);
    aspect::LinearAlgebra::Vector invalid_upper(owned, MPI_COMM_WORLD);
    for (const auto index : invalid_lower.locally_owned_elements())
      {
        invalid_lower[index] = 1.0;
        invalid_upper[index] = 0.0;
      }
    finish_owned_write(invalid_lower);
    finish_owned_write(invalid_upper);
    REQUIRE_THROWS(ROLDistributedBoundConstraint(invalid_lower, invalid_upper));
  }
}
