/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/adjoint/optimization/initial_temperature_optimization.h>

#include <aspect/adjoint/core/manager.h>
#include <aspect/adjoint/optimization/optimization_parameters.h>
#include <aspect/adjoint/optimization/rol_reduced_objective.h>
#include <aspect/adjoint/optimization/rol_vector.h>

#include <ROL_TypeB_LinMoreAlgorithm.hpp>
#include <ROL_lBFGS.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    void
    InitialTemperatureOptimization<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Adjoint");
      {
        prm.declare_entry("Solve mode", "kernel", Patterns::Selection("kernel|optimize"),
                          "Run the ordinary adjoint kernel or the opt-in reduced "
                          "initial scalar-field ROL optimization.");
        prm.enter_subsection("Optimization");
        {
          prm.declare_entry("Control field", "temperature", Patterns::Anything(),
                            "The scalar advection field optimized at the initial "
                            "time. Use the value temperature for the historical control, or "
                            "the name of one continuous chemical composition field.");
          prm.declare_entry("Maximum iterations", "1", Patterns::Integer(0));
          prm.declare_entry("Gradient tolerance", "1e-8", Patterns::Double(0.0));
          prm.declare_entry("Step tolerance", "1e-12", Patterns::Double(0.0));
          prm.declare_entry("Initial trust-region radius", "1e-3", Patterns::Double(0.0));
          prm.declare_entry("L-BFGS history size", "3", Patterns::Integer(1));
          prm.declare_entry("Lower bound", "-1e-2", Patterns::Double());
          prm.declare_entry("Upper bound", "1e-2", Patterns::Double());
          prm.declare_entry("Output verbosity", "0", Patterns::Integer(0));
          prm.declare_entry("Output accepted controls", "false", Patterns::Bool(),
                            "Write the zero control and each accepted initial control as "
                            "VTK diagnostics under output/diagnostics/accepted_controls.");
          prm.declare_entry("Use physical temperature bounds", "false", Patterns::Bool(),
                            "Use the per-control bounds induced by 0 <= q_fixed + Pm <= 1 "
                            "for the selected scalar field instead of uniform increment bounds. "
                            "The parameter name is retained for backward compatibility.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    InitialTemperatureOptimization<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Adjoint");
      {
        const std::string solve_mode = prm.get("Solve mode");
        optimization_enabled = (solve_mode == "optimize");
        AssertThrow(solve_mode == "kernel" || solve_mode == "optimize",
                    ExcMessage("Unknown adjoint solve mode."));

        prm.enter_subsection("Optimization");
        {
          maximum_iterations = prm.get_integer("Maximum iterations");
          gradient_tolerance = prm.get_double("Gradient tolerance");
          step_tolerance = prm.get_double("Step tolerance");
          initial_trust_region_radius = prm.get_double("Initial trust-region radius");
          lbfgs_history_size = prm.get_integer("L-BFGS history size");
          lower_bound = prm.get_double("Lower bound");
          upper_bound = prm.get_double("Upper bound");
          output_verbosity = prm.get_integer("Output verbosity");
          output_accepted_controls = prm.get_bool("Output accepted controls");
          use_physical_temperature_bounds = prm.get_bool("Use physical temperature bounds");
          control_field_name = prm.get("Control field");
          AssertThrow(!control_field_name.empty(),
                      ExcMessage("Adjoint/Optimization/Control field may not be empty."));
          validate_initial_temperature_optimization_parameters(
            lbfgs_history_size,
            gradient_tolerance,
            step_tolerance,
            initial_trust_region_radius,
            lower_bound,
            upper_bound);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    bool
    InitialTemperatureOptimization<dim>::enabled() const
    {
      return optimization_enabled;
    }



    template <int dim>
    void
    InitialTemperatureOptimization<dim>::run(Manager<dim> &adjoint_manager) const
    {
      AssertThrow(optimization_enabled
                  && adjoint_manager.has_optimization_initial_state(),
                  ExcMessage("Optimization requires the captured initial-field baseline."));

      // Optimize j(c)=J(q_0=q_fixed+P c), where q is the selected scalar
      // advection field. The vector space uses <x,y>=x^T(P^T M_q P)y, so ROL
      // norms, gradients, and trust-region radii have a physical L2 meaning.
      const auto &space = adjoint_manager.get_initial_temperature_control_space();
      const auto mass = space.mass_matrix();
      ROLReducedL2Vector x(space.locally_owned_control_dofs(),
                           this->get_mpi_communicator(), mass);
      ROLReducedL2Vector gradient(space.locally_owned_control_dofs(),
                                  this->get_mpi_communicator(), mass);
      x.setScalar(0.0);
      if (output_accepted_controls)
        adjoint_manager.write_accepted_optimization_control(x.get_vector(), 0);

      LinearAlgebra::Vector lower(x.get_vector()), upper(x.get_vector());
      if (use_physical_temperature_bounds)
        {
          // Convert 0<=q_fixed+P c<=1 into per-free-coordinate bounds. With
          // uniform increment bounds, the user instead bounds c directly.
          const auto physical_bounds =
            adjoint_manager.initial_temperature_control_physical_bounds();
          lower = physical_bounds.first;
          upper = physical_bounds.second;
        }
      else
        {
          lower = lower_bound;
          upper = upper_bound;
        }

      ROLDistributedBoundConstraint bounds(lower, upper);
      AssertThrow(bounds.isFeasible(x),
                  ExcMessage("The initial reduced control violates optimization bounds."));

      ROL::ParameterList parameters;
      // Lin-More is a bound-constrained trust-region method. The reduced
      // objective intentionally supplies first derivatives only; L-BFGS builds
      // a secant Hessian approximation from accepted (s,y) pairs and is not
      // also installed as a preconditioner.
      parameters.sublist("Status Test").set("Iteration Limit",
                                            static_cast<int>(maximum_iterations));
      parameters.sublist("Status Test").set("Gradient Tolerance", gradient_tolerance);
      parameters.sublist("Status Test").set("Step Tolerance", step_tolerance);
      parameters.sublist("Step").sublist("Trust Region").set(
        "Initial Radius", initial_trust_region_radius);
      parameters.sublist("General").sublist("Secant").set(
        "Type", std::string("Limited-Memory BFGS"));
      parameters.sublist("General").sublist("Secant").set(
        "Maximum Storage", static_cast<int>(lbfgs_history_size));
      parameters.sublist("General").sublist("Secant").set("Use as Hessian", true);
      parameters.sublist("General").sublist("Secant").set("Use as Preconditioner", false);

      auto lbfgs = ROL::makePtr<ROL::lBFGS<double>>(lbfgs_history_size, true, 1.0);
      ROL::Ptr<ROL::Secant<double>> secant = lbfgs;
      ROL::TypeB::LinMoreAlgorithm<double> algorithm(parameters, secant);
      ROLReducedObjective<dim> objective(adjoint_manager);
      std::ostringstream output;
      algorithm.run(x, gradient, objective, bounds, output);

      if (output_accepted_controls
          && Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
        {
          const std::string filename = this->get_output_directory()
                                       + "diagnostics/accepted_controls/objective_history.txt";
          std::ofstream history(filename, std::ios::trunc);
          AssertThrow(history, ExcMessage("Could not write accepted optimization objective history: " + filename));
          history << "# accepted_iteration objective\n" << std::setprecision(17);
          for (const auto &[iteration, value] : objective.accepted_objective_history())
            history << iteration << ' ' << value << '\n';
        }

      // Report callback counts as lifecycle invariants: a cached value/gradient
      // pair should share one forward generation, reverted trials must not
      // silently consume another trajectory, and hessVec must remain unused.
      const auto state = algorithm.getState();
      const unsigned int lbfgs_pairs =
        static_cast<unsigned int>(std::max(0,
                                           static_cast<const ROL::Secant<double> &>(*lbfgs)
                                           .get_state()->current + 1));
      this->get_pcout() << "ASPECT_O6_OPTIMIZE iter=" << state->iter
                        << " value=" << state->value
                        << " gnorm=" << state->gnorm
                        << " forward=" << objective.forward_evaluations()
                        << " reverse=" << objective.reverse_evaluations()
                        << " lbfgs_pairs=" << lbfgs_pairs
                        << " update_initial=" << objective.initial_updates()
                        << " update_trial=" << objective.trial_updates()
                        << " update_revert=" << objective.revert_updates()
                        << " update_accept=" << objective.accept_updates()
                        << " precond=" << objective.preconditioner_calls()
                        << " hessvec=" << objective.hessvec_calls()
                        << " inv_hessvec=" << objective.inverse_hessvec_calls()
                        << " cache_hits=" << objective.cache_hits()
                        << " cache_misses=" << objective.cache_misses() << std::endl;

      if (output_verbosity == 0)
        return;

      // A new adapter deliberately cannot reuse the optimizer callback cache.
      // Its value/gradient pair therefore qualifies the final accepted state
      // through an independent forward and reverse evaluation.
      ROLReducedObjective<dim> fresh_objective(adjoint_manager);
      ROLReducedL2Vector fresh_gradient(x.get_vector(), mass);
      double tolerance = 1e-14;
      const double fresh_value = fresh_objective.value(x, tolerance);
      fresh_objective.gradient(fresh_gradient, x, tolerance);

      const auto *final_gradient =
        dynamic_cast<const ROLReducedL2Vector *>(state->gradientVec.get());
      AssertThrow(final_gradient != nullptr,
                  ExcMessage("Lin-More did not retain an L2 final gradient."));
      ROLReducedL2Vector gradient_difference(final_gradient->get_vector(), mass);
      gradient_difference.axpy(-1.0, fresh_gradient);
      double local_max_coefficient_difference = 0.0;
      for (const auto index : gradient_difference.get_vector().locally_owned_elements())
        local_max_coefficient_difference =
          std::max(local_max_coefficient_difference,
                   std::abs(gradient_difference.get_vector()[index]));
      const double max_coefficient_difference =
        Utilities::MPI::max(local_max_coefficient_difference,
                            this->get_mpi_communicator());
      const double gradient_relative_difference =
        gradient_difference.norm()
        / std::max(fresh_gradient.norm(), std::numeric_limits<double>::min());
      AssertThrow(gradient_relative_difference <= 1e-11,
                  ExcMessage("The final optimizer gradient differs from a fresh ROL evaluation."));

      ROLReducedL2Vector direction(x.get_vector(), mass);
      for (const auto index : direction.get_vector().locally_owned_elements())
        direction.get_vector()[index] = 1.0 + static_cast<double>(index % 3);
      direction.get_vector().compress(VectorOperation::insert);
      direction.scale(1.0 / direction.norm());

      double local_maximum_feasible_step = std::numeric_limits<double>::max();
      for (const auto index : direction.get_vector().locally_owned_elements())
        if (direction.get_vector()[index] > 0.0)
          local_maximum_feasible_step =
            std::min(local_maximum_feasible_step,
                     (upper[index] - x.get_vector()[index]) / direction.get_vector()[index]);
        else if (direction.get_vector()[index] < 0.0)
          local_maximum_feasible_step =
            std::min(local_maximum_feasible_step,
                     (lower[index] - x.get_vector()[index]) / direction.get_vector()[index]);
      const double maximum_feasible_step =
        Utilities::MPI::min(local_maximum_feasible_step,
                            this->get_mpi_communicator());
      const double initial_taylor_step =
        std::min(1e-3, 0.25 * maximum_feasible_step);
      AssertThrow(initial_taylor_step > 0.0,
                  ExcMessage("The final optimizer point admits no feasible Taylor perturbation."));

      this->get_pcout() << "ASPECT_O6_FRESH_GRADIENT value=" << fresh_value
                        << " relative_l2_difference=" << gradient_relative_difference
                        << " max_coefficient_difference=" << max_coefficient_difference
                        << std::endl;

      // The L2 vector dot product gives j'(c)[d]=<grad_L2 j,d>_L2. For a correct
      // discrete adjoint, the Taylor remainder
      //   |j(c+h d)-j(c)-h j'(c)[d]|
      // decreases as O(h^2) while the active bound set remains unchanged.
      const double directional_derivative = fresh_gradient.dot(direction);
      std::vector<double> remainders;
      remainders.reserve(5);
      for (unsigned int level = 0; level < 5; ++level)
        {
          const double step = std::ldexp(initial_taylor_step,
                                         -static_cast<int>(level));
          ROLReducedL2Vector perturbed(x.get_vector(), mass);
          perturbed.axpy(step, direction);
          const double perturbed_value = fresh_objective.value(perturbed, tolerance);
          const double delta = perturbed_value - fresh_value;
          const double linear = step * directional_derivative;
          const double remainder = std::abs(delta - linear);
          remainders.push_back(remainder);
          const double observed_order =
            (level == 0 ? 0.0
             : std::log(remainders[level-1] / remainder) / std::log(2.0));
          this->get_pcout() << "ASPECT_O6_TAYLOR h=" << step
                            << " delta=" << delta
                            << " linear=" << linear
                            << " remainder=" << remainder
                            << " order=" << observed_order << std::endl;
        }
      AssertThrow(remainders[3] > remainders[4]
                  && std::log(remainders[3] / remainders[4]) / std::log(2.0) >= 1.5,
                  ExcMessage("The final production ROL Taylor check is not second order."));
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::InitialTemperatureOptimization<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
