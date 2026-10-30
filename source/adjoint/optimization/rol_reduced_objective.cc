/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include <aspect/adjoint/optimization/rol_reduced_objective.h>
#include <aspect/adjoint/core/manager.h>

#include <deal.II/base/utilities.h>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    ROLReducedObjective<dim>::ROLReducedObjective(Manager<dim> &manager)
      : manager(manager)
      , cached_objective(0.0)
      , cached_value_is_valid(false)
      , cached_gradient_is_valid(false)
      , forward_trajectory_is_live(false)
      , current_generation_id(0)
      , next_generation_id(0)
      , accepted_objective(0.0)
      , accepted_value_is_valid(false)
      , accepted_gradient_is_valid(false)
      , accepted_generation_id(0)
      , n_cache_hits(0)
      , n_cache_misses(0)
      , n_value_calls(0)
      , n_gradient_calls(0)
      , n_forward_evaluations(0)
      , n_reverse_evaluations(0)
    {}



    template <int dim>
    const ROLReducedL2Vector &
    ROLReducedObjective<dim>::as_l2_control(const ROL::Vector<double> &vector)
    {
      const auto *result = dynamic_cast<const ROLReducedL2Vector *>(&vector);
      AssertThrow(result != nullptr,
                  ExcMessage("ROLReducedObjective requires an L2-metric reduced control vector."));
      return *result;
    }



    template <int dim>
    ROLReducedL2Vector &
    ROLReducedObjective<dim>::as_l2_control(ROL::Vector<double> &vector)
    {
      auto *result = dynamic_cast<ROLReducedL2Vector *>(&vector);
      AssertThrow(result != nullptr,
                  ExcMessage("ROLReducedObjective requires an L2-metric reduced gradient vector."));
      return *result;
    }



    template <int dim>
    double
    ROLReducedObjective<dim>::extract_scalar_objective(
      const std::map<std::string, double> &objectives)
    {
      AssertThrow(!objectives.empty(),
                  ExcMessage("ROLReducedObjective requires at least one active scalar objective."));
      double total = 0.0;
      for (const auto &objective : objectives)
        total += objective.second;
      return total;
    }



    template <int dim>
    bool
    ROLReducedObjective<dim>::has_cached_control(const ROLReducedL2Vector &control) const
    {
      if (cached_control == nullptr
          || cached_control->size() != control.get_vector().size()
          || cached_control->locally_owned_elements()
          != control.get_vector().locally_owned_elements())
        return false;

      // Cache identity is exact rather than tolerance-based. A trajectory is
      // the discrete forward solution for one concrete coefficient vector;
      // reusing it for a merely close trust-region trial would return a
      // gradient of a different reduced functional. MPI sum makes the answer
      // identical on every rank.
      unsigned int local_difference = 0;
      for (const auto index : control.get_vector().locally_owned_elements())
        if ((*cached_control)[index] != control.get_vector()[index])
          {
            local_difference = 1;
            break;
          }

      return dealii::Utilities::MPI::sum(
               local_difference, control.get_vector().get_mpi_communicator()) == 0;
    }



    template <int dim>
    void
    ROLReducedObjective<dim>::invalidate_live_forward_if_control_changed(
      const ROLReducedL2Vector &control)
    {
      if (!has_cached_control(control))
        forward_trajectory_is_live = false;
    }



    template <int dim>
    void
    ROLReducedObjective<dim>::begin_generation(const ROLReducedL2Vector &control,
                                               const double objective)
    {
      // A generation is the complete forward trajectory associated with one
      // control. ROL may ask for value and gradient separately; preserving
      // this identity lets gradient() run the reverse sweep on this exact
      // trajectory instead of silently evaluating a second forward solve.
      if (cached_control == nullptr)
        cached_control = std::make_unique<LinearAlgebra::Vector>(control.get_vector());
      else
        *cached_control = control.get_vector();

      cached_objective = objective;
      cached_value_is_valid = true;
      cached_gradient_is_valid = false;
      cached_l2_gradient.reset();
      forward_trajectory_is_live = true;
      current_generation_id = ++next_generation_id;
    }



    template <int dim>
    void
    ROLReducedObjective<dim>::save_accepted_generation()
    {
      // A trust-region trial can be reverted. Preserve the last accepted
      // value/gradient pair before invalidating the live trajectory, so the
      // Revert event restores a mathematically consistent callback cache.
      if (accepted_value_is_valid || !cached_value_is_valid)
        return;

      accepted_control = std::make_unique<LinearAlgebra::Vector>(*cached_control);
      accepted_l2_gradient = cached_l2_gradient == nullptr
                             ? nullptr
                             : std::make_unique<LinearAlgebra::Vector>(*cached_l2_gradient);
      accepted_objective = cached_objective;
      accepted_value_is_valid = true;
      accepted_gradient_is_valid = cached_gradient_is_valid;
      accepted_generation_id = current_generation_id;
    }



    template <int dim>
    void
    ROLReducedObjective<dim>::restore_accepted_generation()
    {
      // The restored trajectory itself is not live: a later gradient request
      // must rebuild it unless it already has a cached L2 representative.
      if (!accepted_value_is_valid)
        return;

      cached_control = std::make_unique<LinearAlgebra::Vector>(*accepted_control);
      cached_l2_gradient = accepted_l2_gradient == nullptr
                           ? nullptr
                           : std::make_unique<LinearAlgebra::Vector>(*accepted_l2_gradient);
      cached_objective = accepted_objective;
      cached_value_is_valid = true;
      cached_gradient_is_valid = accepted_gradient_is_valid;
      forward_trajectory_is_live = false;
      current_generation_id = accepted_generation_id;
      discard_accepted_generation();
    }



    template <int dim>
    void
    ROLReducedObjective<dim>::discard_accepted_generation()
    {
      accepted_control.reset();
      accepted_l2_gradient.reset();
      accepted_value_is_valid = false;
      accepted_gradient_is_valid = false;
      accepted_generation_id = 0;
    }



    template <int dim>
    void
    ROLReducedObjective<dim>::update(const ROL::Vector<double> &x,
                                     const ROL::UpdateType type,
                                     const int)
    {
      const ROLReducedL2Vector &control = as_l2_control(x);
      switch (type)
        {
          case ROL::UpdateType::Initial:
            ++n_initial_updates;
            cached_control.reset();
            cached_l2_gradient.reset();
            cached_value_is_valid = false;
            cached_gradient_is_valid = false;
            forward_trajectory_is_live = false;
            current_generation_id = 0;
            discard_accepted_generation();
            accepted_objectives.clear();
            break;

          case ROL::UpdateType::Trial:
            ++n_trial_updates;
          case ROL::UpdateType::Temp:
            // ROL uses both Trial and Temp while probing a trust region. Both
            // are provisional and therefore preserve the accepted cache.
            save_accepted_generation();
            invalidate_live_forward_if_control_changed(control);
            break;

          case ROL::UpdateType::Revert:
            ++n_revert_updates;
            restore_accepted_generation();
            invalidate_live_forward_if_control_changed(control);
            break;

          case ROL::UpdateType::Accept:
            ++n_accept_updates;
            // Acceptance changes the optimizer's reference point. The next
            // callback may reuse the matching cache, but any old accepted
            // backup is now obsolete.
            AssertThrow(cached_value_is_valid && has_cached_control(control),
                        ExcMessage("ROL accepted a control without its matching reduced-objective value."));
            const unsigned int accepted_iteration = ++n_accepted_control_outputs;
            accepted_objectives.emplace_back(accepted_iteration, cached_objective);
            invalidate_live_forward_if_control_changed(control);
            discard_accepted_generation();
            manager.write_accepted_optimization_control(control.get_vector(),
                                                        accepted_iteration);
            break;
        }
    }



    template <int dim>
    void
    ROLReducedObjective<dim>::update(const ROL::Vector<double> &x,
                                     const bool flag,
                                     const int)
    {
      if (flag)
        invalidate_live_forward_if_control_changed(as_l2_control(x));
    }



    template <int dim>
    double
    ROLReducedObjective<dim>::value(const ROL::Vector<double> &x, double & /* tolerance */)
    {
      const ROLReducedL2Vector &control = as_l2_control(x);
      ++n_value_calls;
      if (cached_value_is_valid && has_cached_control(control))
        {
          ++n_cache_hits;
          return cached_objective;
        }

      // A miss performs the forward half of the reduced map
      // c -> T_0(c) -> trajectory -> J. Manager intentionally leaves that
      // trajectory live so an immediately following gradient callback can
      // execute only the reverse half.
      ++n_cache_misses;
      ++n_forward_evaluations;
      const double objective = extract_scalar_objective(
                                 manager.evaluate_initial_temperature_control(control.get_vector()));
      begin_generation(control, objective);
      if (accepted_objectives.empty() && n_initial_updates == 1
          && n_accepted_control_outputs == 0)
        accepted_objectives.emplace_back(0, objective);
      return objective;
    }



    template <int dim>
    void
    ROLReducedObjective<dim>::gradient(ROL::Vector<double> &g,
                                       const ROL::Vector<double> &x,
                                       double & /* tolerance */)
    {
      const ROLReducedL2Vector &control = as_l2_control(x);
      ROLReducedL2Vector &gradient = as_l2_control(g);
      AssertThrow(gradient.has_same_layout(control)
                  && gradient.mass_matrix() == control.mass_matrix(),
                  ExcMessage("ROLReducedObjective requires control and gradient vectors "
                             "with the same reduced L2 metric."));
      if (cached_gradient_is_valid && has_cached_control(control))
        {
          gradient.get_vector() = *cached_l2_gradient;
          ++n_cache_hits;
          ++n_gradient_calls;
          return;
        }

      LinearAlgebra::Vector l2_gradient;
      if (cached_value_is_valid && forward_trajectory_is_live && has_cached_control(control))
        {
          // Reuse the exact forward generation created by value(). The manager
          // returns the L2 Riesz representative g satisfying
          // dJ(c)[s]=s^T M_c g, not the raw Euclidean covector P^T dJ/dT_0.
          ++n_cache_hits;
          manager.complete_initial_temperature_control_gradient(l2_gradient);
          forward_trajectory_is_live = false;
        }
      else
        {
          // The live trajectory was consumed, reverted, or belongs to another
          // control. Evaluate forward and reverse as one transaction; this is
          // more expensive but preserves the reduced-gradient identity.
          ++n_cache_misses;
          ++n_forward_evaluations;
          const double objective = extract_scalar_objective(
                                     manager.evaluate_initial_temperature_control(control.get_vector(), &l2_gradient));
          begin_generation(control, objective);
          forward_trajectory_is_live = false;
        }
      AssertThrow(l2_gradient.size() == gradient.get_vector().size()
                  && l2_gradient.locally_owned_elements()
                  == gradient.get_vector().locally_owned_elements(),
                  ExcMessage("The O3 L2 gradient does not match the ROL reduced layout."));
      gradient.get_vector() = l2_gradient;
      // The gradient vector is tied to the same generation as cached_objective
      // and can be reused even after its on-disk trajectory is no longer live.
      cached_l2_gradient = std::make_unique<LinearAlgebra::Vector>(l2_gradient);
      cached_gradient_is_valid = true;
      ++n_gradient_calls;
      ++n_reverse_evaluations;
    }



    template <int dim>
    void
    ROLReducedObjective<dim>::hessVec(ROL::Vector<double> &,
                                      const ROL::Vector<double> &,
                                      const ROL::Vector<double> &,
                                      double &)
    {
      ++n_hessvec_calls;
      AssertThrow(false,
                  ExcMessage("ROLReducedObjective::hessVec is intentionally unsupported; "
                             "Lin-More must use its external L-BFGS secant model."));
    }



    template <int dim>
    void
    ROLReducedObjective<dim>::invHessVec(ROL::Vector<double> &,
                                         const ROL::Vector<double> &,
                                         const ROL::Vector<double> &,
                                         double &)
    {
      ++n_inverse_hessvec_calls;
      AssertThrow(false,
                  ExcMessage("ROLReducedObjective::invHessVec is intentionally unsupported."));
    }



    template <int dim>
    void
    ROLReducedObjective<dim>::precond(ROL::Vector<double> &pv,
                                      const ROL::Vector<double> &v,
                                      const ROL::Vector<double> &,
                                      double &)
    {
      // ROL vectors already encode the reduced L2 metric through dot() and
      // dual(). At this objective layer the natural preconditioner is thus the
      // identity in that Hilbert space; applying another mass inverse would
      // double-count the Riesz map.
      ++n_preconditioner_calls;
      as_l2_control(pv).set(as_l2_control(v));
    }



    template <int dim>
    unsigned int ROLReducedObjective<dim>::value_calls() const
    {
      return n_value_calls;
    }

    template <int dim>
    unsigned int ROLReducedObjective<dim>::gradient_calls() const
    {
      return n_gradient_calls;
    }

    template <int dim>
    unsigned int ROLReducedObjective<dim>::forward_evaluations() const
    {
      return n_forward_evaluations;
    }

    template <int dim>
    unsigned int ROLReducedObjective<dim>::reverse_evaluations() const
    {
      return n_reverse_evaluations;
    }

    template <int dim>
    unsigned int ROLReducedObjective<dim>::cache_hits() const
    {
      return n_cache_hits;
    }

    template <int dim>
    unsigned int ROLReducedObjective<dim>::cache_misses() const
    {
      return n_cache_misses;
    }

    template <int dim>
    unsigned int ROLReducedObjective<dim>::generation_id() const
    {
      return current_generation_id;
    }

    template <int dim> unsigned int ROLReducedObjective<dim>::initial_updates() const
    {
      return n_initial_updates;
    }
    template <int dim> unsigned int ROLReducedObjective<dim>::trial_updates() const
    {
      return n_trial_updates;
    }
    template <int dim> unsigned int ROLReducedObjective<dim>::revert_updates() const
    {
      return n_revert_updates;
    }
    template <int dim> unsigned int ROLReducedObjective<dim>::accept_updates() const
    {
      return n_accept_updates;
    }
    template <int dim> unsigned int ROLReducedObjective<dim>::preconditioner_calls() const
    {
      return n_preconditioner_calls;
    }
    template <int dim> unsigned int ROLReducedObjective<dim>::hessvec_calls() const
    {
      return n_hessvec_calls;
    }
    template <int dim> unsigned int ROLReducedObjective<dim>::inverse_hessvec_calls() const
    {
      return n_inverse_hessvec_calls;
    }

    template <int dim>
    const std::vector<std::pair<unsigned int, double>> &
    ROLReducedObjective<dim>::accepted_objective_history() const
    {
      return accepted_objectives;
    }

  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::ROLReducedObjective<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
