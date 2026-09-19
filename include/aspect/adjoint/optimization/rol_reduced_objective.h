/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#ifndef _aspect_adjoint_optimization_rol_reduced_objective_h
#define _aspect_adjoint_optimization_rol_reduced_objective_h

#include <aspect/adjoint/optimization/rol_vector.h>

#include <ROL_Objective.hpp>

#include <map>
#include <memory>
#include <string>

namespace aspect
{
  namespace Adjoint
  {
    template <int dim> class Manager;

    /**
     * Minimal native ROL adapter for the O3 reusable initial-temperature
     * reduced functional. The simulator owns all PDE, P^T, and Riesz work;
     * this class only validates the L2 control representation and transfers
     * callback results across the ROL boundary. In particular, it caches a
     * forward generation so ROL's separate value() and gradient() callbacks
     * share one trajectory. The accepted-generation save/restore state is
     * generic trust-region bookkeeping: trial/revert events must not leak a
     * rejected forward state into the next callback.
     */
    template <int dim>
    class ROLReducedObjective : public ROL::Objective<double>
    {
      public:
        explicit ROLReducedObjective(Manager<dim> &manager);

        void update(const ROL::Vector<double> &x,
                    ROL::UpdateType type,
                    int iteration = -1) override;
        void update(const ROL::Vector<double> &x,
                    bool flag = true,
                    int iteration = -1) override;
        double value(const ROL::Vector<double> &x, double &tolerance) override;
        void gradient(ROL::Vector<double> &g,
                      const ROL::Vector<double> &x,
                      double &tolerance) override;
        void hessVec(ROL::Vector<double> &hv,
                     const ROL::Vector<double> &v,
                     const ROL::Vector<double> &x,
                     double &tolerance) override;
        void invHessVec(ROL::Vector<double> &hv,
                        const ROL::Vector<double> &v,
                        const ROL::Vector<double> &x,
                        double &tolerance) override;
        void precond(ROL::Vector<double> &pv,
                     const ROL::Vector<double> &v,
                     const ROL::Vector<double> &x,
                     double &tolerance) override;

        unsigned int value_calls() const;
        unsigned int gradient_calls() const;
        unsigned int forward_evaluations() const;
        unsigned int reverse_evaluations() const;
        unsigned int cache_hits() const;
        unsigned int cache_misses() const;
        unsigned int generation_id() const;
        unsigned int initial_updates() const;
        unsigned int trial_updates() const;
        unsigned int revert_updates() const;
        unsigned int accept_updates() const;
        unsigned int preconditioner_calls() const;
        unsigned int hessvec_calls() const;
        unsigned int inverse_hessvec_calls() const;

      private:
        static const ROLReducedL2Vector &as_l2_control(const ROL::Vector<double> &vector);
        static ROLReducedL2Vector &as_l2_control(ROL::Vector<double> &vector);
        static double extract_scalar_objective(const std::map<std::string, double> &objectives);
        bool has_cached_control(const ROLReducedL2Vector &control) const;
        void invalidate_live_forward_if_control_changed(const ROLReducedL2Vector &control);
        void begin_generation(const ROLReducedL2Vector &control, const double objective);
        void save_accepted_generation();
        void restore_accepted_generation();
        void discard_accepted_generation();

        Manager<dim> &manager;
        std::unique_ptr<LinearAlgebra::Vector> cached_control;
        std::unique_ptr<LinearAlgebra::Vector> cached_l2_gradient;
        double cached_objective;
        bool cached_value_is_valid;
        bool cached_gradient_is_valid;
        bool forward_trajectory_is_live;
        unsigned int current_generation_id;
        unsigned int next_generation_id;
        std::unique_ptr<LinearAlgebra::Vector> accepted_control;
        std::unique_ptr<LinearAlgebra::Vector> accepted_l2_gradient;
        double accepted_objective;
        bool accepted_value_is_valid;
        bool accepted_gradient_is_valid;
        unsigned int accepted_generation_id;
        unsigned int n_cache_hits;
        unsigned int n_cache_misses;
        unsigned int n_initial_updates = 0;
        unsigned int n_trial_updates = 0;
        unsigned int n_revert_updates = 0;
        unsigned int n_accept_updates = 0;
        unsigned int n_preconditioner_calls = 0;
        unsigned int n_hessvec_calls = 0;
        unsigned int n_inverse_hessvec_calls = 0;
        unsigned int n_value_calls;
        unsigned int n_gradient_calls;
        unsigned int n_forward_evaluations;
        unsigned int n_reverse_evaluations;
    };
  }
}

#endif
