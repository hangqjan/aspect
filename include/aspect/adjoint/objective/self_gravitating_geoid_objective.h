/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#ifndef _aspect_adjoint_objective_self_gravitating_geoid_objective_h
#define _aspect_adjoint_objective_self_gravitating_geoid_objective_h

#include <aspect/adjoint/objective/objective_functional.h>
#include <aspect/postprocess/self_gravitating_geoid_operator.h>

namespace aspect
{
  namespace Adjoint
  {
    /**
     * Coupled self-gravitating geoid objective. It propagates spectral seeds
     * through CBF and volume-density transposes, and chains the boundary-mean
     * raw-density dependence of the 4-by-4 coefficient matrix.
     */
    template <int dim>
    class SelfGravitatingGeoidObjective : public Objective::Interface<dim>,
      public Objective::InstantaneousInterface<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;
        double evaluate() const override;
        void assemble_adjoint_right_hand_side(LinearAlgebra::BlockVector &right_hand_side) const override;
        bool requires_physical_stokes_adjoint() const override;
        std::set<std::string> required_forward_postprocessors() const override;
        void add_direct_kernel_contributions(const std::string &objective_name,
                                             KernelRepository<dim> &kernels) const override;

      private:
        Postprocess::SelfGravitatingGeoidOperatorInputs inputs() const;
        Postprocess::SelfGravitatingGeoidCoefficients forward_coefficients() const;
        Postprocess::SelfGravitatingGeoidCoefficients coefficient_seed() const;
        bool has_traction_seed(const Postprocess::SelfGravitatingGeoidRHSSeeds &seeds) const;

        unsigned int minimum_degree = 2;
        unsigned int maximum_degree = 2;
        std::array<Postprocess::SphericalHarmonicCoefficients, 4> observed_coefficients;
        std::vector<double> weights;
        std::array<double, 4> field_weights = {{1.0, 1.0, 1.0, 1.0}};
        double density_above = 0.0;
        double density_below = 9900.0;
    };
  }
}

#endif
