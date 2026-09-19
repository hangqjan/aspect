/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include "common.h"

#include <aspect/adjoint/optimization/rol_vector.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <cmath>

namespace adjoint_rol_l2_metric_tests
{
  constexpr dealii::types::global_dof_index global_size = 9;

  dealii::IndexSet owned_index_set()
  {
    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    dealii::IndexSet result(global_size);
    result.add_range(global_size * rank / n_ranks, global_size * (rank + 1) / n_ranks);
    return result;
  }

  std::shared_ptr<const aspect::LinearAlgebra::SparseMatrix>
  diagonal_mass(const dealii::IndexSet &owned)
  {
    dealii::DynamicSparsityPattern dsp(owned);
    for (const auto index : owned)
      dsp.add(index, index);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, owned, MPI_COMM_WORLD, owned);
    auto matrix = std::make_shared<aspect::LinearAlgebra::SparseMatrix>();
    matrix->reinit(owned, dsp, MPI_COMM_WORLD);
    for (const auto index : owned)
      matrix->set(index, index, 1.0 + static_cast<double>(index));
    matrix->compress(dealii::VectorOperation::insert);
    return matrix;
  }

  void assign(aspect::Adjoint::ROLReducedL2Vector &vector, const double offset)
  {
    for (const auto index : vector.get_vector().locally_owned_elements())
      vector.get_vector()[index] = offset + static_cast<double>(index);
    vector.get_vector().compress(dealii::VectorOperation::insert);
  }


  TEST_CASE("reduced ROL control vector uses its cached distributed L2 metric")
  {
    using aspect::Adjoint::ROLReducedL2Vector;
    const auto owned = owned_index_set();
    const auto mass = diagonal_mass(owned);
    ROLReducedL2Vector u(owned, MPI_COMM_WORLD, mass);
    ROLReducedL2Vector v(owned, MPI_COMM_WORLD, mass);
    assign(u, 1.0);
    assign(v, -0.5);

    double expected_dot = 0.0;
    double expected_norm2 = 0.0;
    for (const auto index : owned)
      {
        const double weight = 1.0 + static_cast<double>(index);
        expected_dot += weight * (1.0 + index) * (-0.5 + index);
        expected_norm2 += weight * (1.0 + index) * (1.0 + index);
      }
    expected_dot = dealii::Utilities::MPI::sum(expected_dot, MPI_COMM_WORLD);
    expected_norm2 = dealii::Utilities::MPI::sum(expected_norm2, MPI_COMM_WORLD);

    REQUIRE(u.dot(v) == Approx(expected_dot).margin(1e-12));
    REQUIRE(u.norm() * u.norm() == Approx(expected_norm2).margin(1e-12));

    auto clone = u.clone();
    const auto &l2_clone = dynamic_cast<const ROLReducedL2Vector &>(*clone);
    REQUIRE(l2_clone.mass_matrix() == mass);
  }
}
