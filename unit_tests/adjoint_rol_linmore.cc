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

#include <ROL_Objective.hpp>
#include <ROL_StdBoundConstraint.hpp>
#include <ROL_StdVector.hpp>
#include <ROL_TypeB_LinMoreAlgorithm.hpp>
#include <ROL_lBFGS.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
  /**
   * O0-only serial objective. Production ASPECT controls will use a custom
   * distributed vector and a control-space metric in O1/O2.
   */
  class BoundedQuadraticObjective : public ROL::Objective<double>
  {
    public:
      unsigned int value_calls = 0;
      unsigned int gradient_calls = 0;
      unsigned int hessvec_calls = 0;
      unsigned int inverse_hessvec_calls = 0;

      double
      value(const ROL::Vector<double> &x,
            double & /* tolerance */) override
      {
        ++value_calls;
        const auto &entries =
          *static_cast<const ROL::StdVector<double> &>(x).getVector();

        double result = 0.0;
        for (unsigned int i = 0; i < entries.size(); ++i)
          result += 0.5 * coefficients[i]
                    * (entries[i] - unconstrained_minimizer[i])
                    * (entries[i] - unconstrained_minimizer[i]);
        return result;
      }

      void
      gradient(ROL::Vector<double> &g,
               const ROL::Vector<double> &x,
               double & /* tolerance */) override
      {
        ++gradient_calls;
        const auto &x_entries =
          *static_cast<const ROL::StdVector<double> &>(x).getVector();
        auto &g_entries =
          *static_cast<ROL::StdVector<double> &>(g).getVector();

        for (unsigned int i = 0; i < x_entries.size(); ++i)
          g_entries[i] = coefficients[i]
                         * (x_entries[i] - unconstrained_minimizer[i]);
      }

      void
      hessVec(ROL::Vector<double> &,
              const ROL::Vector<double> &,
              const ROL::Vector<double> &,
              double &) override
      {
        ++hessvec_calls;
        throw std::runtime_error(
          "O0 failure: Objective::hessVec was invoked. Lin-More is not using "
          "the intended external L-BFGS curvature model.");
      }

      void
      invHessVec(ROL::Vector<double> &,
                 const ROL::Vector<double> &,
                 const ROL::Vector<double> &,
                 double &) override
      {
        ++inverse_hessvec_calls;
        throw std::runtime_error(
          "O0 failure: Objective::invHessVec was invoked.");
      }

      static constexpr std::array<double, 3> coefficients = {{1.0, 2.0, 4.0}};
      static constexpr std::array<double, 3> unconstrained_minimizer =
      {{-0.5, 0.4, 1.5}};
  };

  constexpr std::array<double, 3> BoundedQuadraticObjective::coefficients;
  constexpr std::array<double, 3>
  BoundedQuadraticObjective::unconstrained_minimizer;

  double
  quadratic_value(const std::array<double, 3> &x)
  {
    double value = 0.0;
    for (unsigned int i = 0; i < x.size(); ++i)
      value += 0.5 * BoundedQuadraticObjective::coefficients[i]
               * (x[i] - BoundedQuadraticObjective::unconstrained_minimizer[i])
               * (x[i] - BoundedQuadraticObjective::unconstrained_minimizer[i]);
    return value;
  }

  struct OptimizationResult
  {
    std::array<double, 3> final_x;
    double initial_value;
    double final_value;
    double max_abs_solution_error;
    int termination_status;
    int outer_iterations;
    int function_evaluations;
    int gradient_evaluations;
    int secant_storage;
    int secant_pairs;
    unsigned int value_calls;
    unsigned int gradient_calls;
    unsigned int hessvec_calls;
    unsigned int inverse_hessvec_calls;
  };

  OptimizationResult
  run_bounded_quadratic()
  {
    constexpr unsigned int history_size = 5;
    const std::array<double, 3> initial_x = {{0.8, 0.8, 0.2}};
    const std::array<double, 3> exact_solution = {{0.0, 0.4, 1.0}};

    auto x_entries = ROL::makePtr<std::vector<double>>(initial_x.begin(),
                                                        initial_x.end());
    ROL::StdVector<double> x(x_entries);
    ROL::StdVector<double> g(3, 0.0);

    std::vector<double> lower(3, 0.0);
    std::vector<double> upper(3, 1.0);
    ROL::StdBoundConstraint<double> bounds(lower, upper);

    ROL::ParameterList parameters;
    parameters.sublist("Status Test").set("Iteration Limit", 100);
    parameters.sublist("Status Test").set("Gradient Tolerance", 1.e-10);
    parameters.sublist("Status Test").set("Step Tolerance", 1.e-14);
    parameters.sublist("Step").sublist("Trust Region")
    .set("Initial Radius", 1.0);
    parameters.sublist("General").sublist("Secant")
    .set("Type", std::string("Limited-Memory BFGS"));
    parameters.sublist("General").sublist("Secant")
    .set("Maximum Storage", static_cast<int>(history_size));
    parameters.sublist("General").sublist("Secant")
    .set("Use as Hessian", true);
    parameters.sublist("General").sublist("Secant")
    .set("Use as Preconditioner", false);

    auto lbfgs = ROL::makePtr<ROL::lBFGS<double>>(history_size,
                                                   true,
                                                   1.0);
    const ROL::Ptr<ROL::Secant<double>> secant = lbfgs;
    ROL::TypeB::LinMoreAlgorithm<double> algorithm(parameters, secant);

    BoundedQuadraticObjective objective;
    std::ostringstream rol_output;
    algorithm.run(x, g, objective, bounds, rol_output);

    OptimizationResult result;
    result.initial_value = quadratic_value(initial_x);
    result.final_value = quadratic_value({{
        (*x_entries)[0],
        (*x_entries)[1],
        (*x_entries)[2]
      }
    });
    for (unsigned int i = 0; i < result.final_x.size(); ++i)
      {
        result.final_x[i] = (*x_entries)[i];
        result.max_abs_solution_error =
          std::max(result.max_abs_solution_error,
                   std::abs(result.final_x[i] - exact_solution[i]));
      }

    const auto algorithm_state = algorithm.getState();
    result.termination_status = static_cast<int>(algorithm_state->statusFlag);
    result.outer_iterations = algorithm_state->iter;
    result.function_evaluations = algorithm_state->nfval;
    result.gradient_evaluations = algorithm_state->ngrad;
    const ROL::lBFGS<double> &const_lbfgs = *lbfgs;
    result.secant_storage = const_lbfgs.get_state()->storage;
    result.secant_pairs = const_lbfgs.get_state()->current + 1;
    result.value_calls = objective.value_calls;
    result.gradient_calls = objective.gradient_calls;
    result.hessvec_calls = objective.hessvec_calls;
    result.inverse_hessvec_calls = objective.inverse_hessvec_calls;
    return result;
  }
}

TEST_CASE("ROL Type-B Lin-More uses explicit L-BFGS curvature without objective Hessians")
{
  const OptimizationResult first_run = run_bounded_quadratic();
  const OptimizationResult second_run = run_bounded_quadratic();

  const std::array<double, 3> lower = {{0.0, 0.0, 0.0}};
  const std::array<double, 3> upper = {{1.0, 1.0, 1.0}};
  const std::array<double, 3> expected = {{0.0, 0.4, 1.0}};
  constexpr double solution_tolerance = 1.e-7;
  constexpr double feasibility_tolerance = 1.e-12;

  INFO("ROL termination status: " << first_run.termination_status);
  INFO("ROL outer iterations: " << first_run.outer_iterations);
  INFO("ROL secant pairs: " << first_run.secant_pairs);

  REQUIRE(first_run.final_value < first_run.initial_value);
  REQUIRE(first_run.max_abs_solution_error < solution_tolerance);
  REQUIRE(first_run.value_calls > 0);
  REQUIRE(first_run.gradient_calls > 0);
  REQUIRE(first_run.hessvec_calls == 0);
  REQUIRE(first_run.inverse_hessvec_calls == 0);

  // This is stable public SecantState evidence that the explicit lBFGS object
  // received at least one accepted secant update.
  REQUIRE(first_run.secant_storage == 5);
  REQUIRE(first_run.secant_pairs > 0);

  for (unsigned int i = 0; i < first_run.final_x.size(); ++i)
    {
      INFO("component " << i);
      REQUIRE(first_run.final_x[i] >= lower[i] - feasibility_tolerance);
      REQUIRE(first_run.final_x[i] <= upper[i] + feasibility_tolerance);
      REQUIRE(std::abs(first_run.final_x[i] - expected[i]) < solution_tolerance);
      REQUIRE(first_run.final_x[i] == Approx(second_run.final_x[i]).margin(1.e-14));
    }

  REQUIRE(first_run.final_value == Approx(second_run.final_value).margin(1.e-14));
  REQUIRE(first_run.termination_status == second_run.termination_status);
  REQUIRE(first_run.outer_iterations == second_run.outer_iterations);
}
