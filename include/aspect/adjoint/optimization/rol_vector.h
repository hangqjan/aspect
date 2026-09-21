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

#ifndef _aspect_adjoint_optimization_rol_vector_h
#define _aspect_adjoint_optimization_rol_vector_h

#include <aspect/global.h>
#include <aspect/linear_algebra_types.h>

#include <ROL_BoundConstraint.hpp>
#include <ROL_Vector.hpp>

#include <cmath>
#include <limits>
#include <memory>

namespace aspect
{
  namespace Adjoint
  {
    /**
     * An owned-only distributed ROL vector backed by ASPECT's scalar linear
     * algebra vector. O1 deliberately uses the coefficient-space Euclidean
     * metric; a later control-space layer supplies the physical Riesz metric.
     */
    class ROLDistributedVector : public ROL::Vector<double>
    {
      public:
        ROLDistributedVector(const dealii::IndexSet &locally_owned_elements,
                             const MPI_Comm          communicator);

        explicit ROLDistributedVector(const LinearAlgebra::Vector &vector);

        void plus(const ROL::Vector<double> &vector) override;
        void scale(const double alpha) override;
        double dot(const ROL::Vector<double> &vector) const override;
        double norm() const override;
        ROL::Ptr<ROL::Vector<double>> clone() const override;

        void axpy(const double alpha, const ROL::Vector<double> &vector) override;
        void set(const ROL::Vector<double> &vector) override;
        ROL::Ptr<ROL::Vector<double>> basis(const int index) const override;
        int dimension() const override;
        void applyUnary(const ROL::Elementwise::UnaryFunction<double> &function) override;
        void applyBinary(const ROL::Elementwise::BinaryFunction<double> &function,
                         const ROL::Vector<double> &vector) override;
        double reduce(const ROL::Elementwise::ReductionOp<double> &reduction) const override;
        void setScalar(const double value) override;

        LinearAlgebra::Vector &get_vector();
        const LinearAlgebra::Vector &get_vector() const;

        static ROLDistributedVector &get_vector(ROL::Vector<double> &vector);
        static const ROLDistributedVector &get_vector(const ROL::Vector<double> &vector);

        bool has_same_layout(const ROLDistributedVector &other) const;

      private:
        void assert_compatible(const ROLDistributedVector &other) const;

        LinearAlgebra::Vector vector;
    };


    /**
     * A reduced-control vector whose primal inner product is the cached
     * consistent-temperature mass matrix.  The vector remains a primal vector:
     * O4 supplies the Riesz representative as its gradient, so ROL's default
     * self-dual representation is the intended one.
     */
    class ROLReducedL2Vector : public ROLDistributedVector
    {
      public:
        ROLReducedL2Vector(const dealii::IndexSet &locally_owned_elements,
                           const MPI_Comm communicator,
                           const std::shared_ptr<const LinearAlgebra::SparseMatrix> &mass_matrix);

        ROLReducedL2Vector(const LinearAlgebra::Vector &vector,
                           const std::shared_ptr<const LinearAlgebra::SparseMatrix> &mass_matrix);

        double dot(const ROL::Vector<double> &vector) const override;
        ROL::Ptr<ROL::Vector<double>> clone() const override;

        const std::shared_ptr<const LinearAlgebra::SparseMatrix> &mass_matrix() const;

      private:
        void assert_metric_compatible(const ROLReducedL2Vector &other) const;

        std::shared_ptr<const LinearAlgebra::SparseMatrix> metric_mass_matrix;
    };


    /**
     * Componentwise finite bounds for ROLDistributedVector. All bound data
     * have the same owned-only partition as the optimized vector.
     */
    class ROLDistributedBoundConstraint : public ROL::BoundConstraint<double>
    {
      public:
        ROLDistributedBoundConstraint(const LinearAlgebra::Vector &lower_bound,
                                      const LinearAlgebra::Vector &upper_bound,
                                      const double                 scale = 1.0,
                                      const double                 feasibility_tolerance =
                                        std::sqrt(std::numeric_limits<double>::epsilon()));

        void project(ROL::Vector<double> &vector) override;
        void projectInterior(ROL::Vector<double> &vector) override;
        void pruneUpperActive(ROL::Vector<double> &vector,
                              const ROL::Vector<double> &x,
                              const double epsilon = 0.0) override;
        void pruneUpperActive(ROL::Vector<double> &vector,
                              const ROL::Vector<double> &gradient,
                              const ROL::Vector<double> &x,
                              const double x_epsilon = 0.0,
                              const double gradient_epsilon = 0.0) override;
        void pruneLowerActive(ROL::Vector<double> &vector,
                              const ROL::Vector<double> &x,
                              const double epsilon = 0.0) override;
        void pruneLowerActive(ROL::Vector<double> &vector,
                              const ROL::Vector<double> &gradient,
                              const ROL::Vector<double> &x,
                              const double x_epsilon = 0.0,
                              const double gradient_epsilon = 0.0) override;
        bool isFeasible(const ROL::Vector<double> &vector) override;

        void applyInverseScalingFunction(ROL::Vector<double> &scaled_vector,
                                         const ROL::Vector<double> &vector,
                                         const ROL::Vector<double> &x,
                                         const ROL::Vector<double> &gradient) const override;
        void applyScalingFunctionJacobian(ROL::Vector<double> &scaled_vector,
                                          const ROL::Vector<double> &vector,
                                          const ROL::Vector<double> &x,
                                          const ROL::Vector<double> &gradient) const override;

      private:
        void assert_compatible(const ROLDistributedVector &vector) const;
        double active_tolerance(const double epsilon) const;
        double scaling_cap(const types::global_dof_index index) const;
        void build_scaling_function(ROLDistributedVector &scaling,
                                    const ROLDistributedVector &x,
                                    const ROLDistributedVector &gradient) const;

        const double scale;
        const double feasibility_tolerance;
        double minimum_bound_width;
    };
  }
}

#endif
