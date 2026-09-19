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

#include <aspect/adjoint/core/manager.h>
#include <aspect/adjoint/core/instantaneous_stokes_workflow.h>
#include <aspect/adjoint/core/simulator_state.h>
#include <aspect/adjoint/operators/stokes_adjoint_solver.h>
#include <aspect/adjoint/objective/temperature_field_objective.h>

#include <aspect/geometry_model/box.h>
#include <aspect/gravity_model/vertical.h>
#include <aspect/material_model/simple.h>
#include <aspect/adjoint/operators/advection_adjoint_calculator.h>
#include <aspect/adjoint/operators/boundary_conditions.h>
#include <aspect/adjoint/operators/material_model_adjoint.h>
#include <aspect/postprocess/dynamic_topography.h>
#include <aspect/simulator.h>
#include <aspect/simulator/solver/stokes_matrix_free.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <string>

#if defined(__APPLE__)
#  include <mach/mach.h>
#endif

namespace aspect
{
  namespace Adjoint
  {
    template <int dim>
    Manager<dim>::Manager()
      :
      simulator(nullptr),
      stop_after_adjoint_checkpoint(numbers::invalid_unsigned_int),
      run_trajectory_stokes_recovery_check(false),
      run_trajectory_replay_check(false),
      run_sparse_trajectory_replay_check(false),
      run_matrix_free_cold_rebuild_diagnostic(false),
      trajectory_stokes_recovery_tolerance(1e-8),
      trajectory_replay_tolerance(1e-11),
      lifecycle_failure_injection(Diagnostics::LifecycleFailurePoint::none),
      checkpoint_transaction_failure_injection(Diagnostics::CheckpointFailurePoint::none),
      checkpoint_transaction_failure_rank(-1),
      checkpoint_transaction_failure_next_reverse_step(numbers::invalid_unsigned_int),
      trajectory_cleanup_failure_injection(Diagnostics::CleanupFailurePoint::none),
      trajectory_cleanup_failure_rank(-1),
      trajectory_cleanup_failure_next_reverse_step(numbers::invalid_unsigned_int),
      trajectory_append_failure_injection(Diagnostics::AppendFailurePoint::none),
      trajectory_append_failure_rank(-1),
      trajectory_append_failure_step(numbers::invalid_unsigned_int),
      material_model_name(""),
      advection_preconditioner("transpose ilu"),
      run_advection_preconditioner_comparison(false),
      run_material_model_adjoint_vjp_diagnostic(false),
      material_model_adjoint_vjp_diagnostic_epsilon(1e-7),
      run_composition_stokes_rhs_transpose_diagnostic(false),
      composition_stokes_rhs_transpose_diagnostic_epsilon(1e-7),
      run_direct_stokes_viscosity_derivative_diagnostic(false),
      direct_stokes_viscosity_derivative_diagnostic_epsilon(1e-7),
      run_transposed_advection_constraint_diagnostic(false),
      run_artificial_viscosity_coefficient_seed_diagnostic(false),
      artificial_viscosity_coefficient_seed_diagnostic_epsilon(1e-7),
      run_artificial_viscosity_estimator_transpose_diagnostic(false),
      artificial_viscosity_estimator_transpose_diagnostic_epsilon(1e-7),
      run_local_entropy_residual_transpose_diagnostic(false),
      local_entropy_residual_transpose_diagnostic_epsilon(1e-7),
      run_global_entropy_denominator_scalar_diagnostic(false),
      global_entropy_denominator_scalar_diagnostic_epsilon(1e-7),
      run_global_entropy_variation_transpose_diagnostic(false),
      global_entropy_variation_transpose_diagnostic_epsilon(1e-7)
    {}



    template <int dim>
    void
    Manager<dim>::declare_parameters (ParameterHandler &prm)
    {
      // Keep capability-owned parameter declarations outside the facade. The
      // objective registry and optimization driver define their own public
      // interfaces; this file declares only cross-workflow trajectory and
      // diagnostic controls that the top-level manager dispatches.
      Objective::Manager<dim>::declare_parameters (prm);
      InitialTemperatureOptimization<dim>::declare_parameters(prm);

      prm.enter_subsection("Adjoint");
      {
        prm.enter_subsection("Time dependent");
        {
          // Storage mode changes only which accepted fields are persisted.
          // Both modes retain the same dense logical time schedule, and both
          // reconstruct the same ForwardStateView consumed by the reverse
          // operators.
          prm.declare_entry("Trajectory storage mode", "minimal",
                            Patterns::Selection("minimal|stokes cached"),
                            "Storage policy for the time-dependent adjoint "
                            "trajectory. 'minimal' archives only temperature "
                            "and composition and recovers Stokes fields during "
                            "the reverse sweep. 'stokes cached' additionally "
                            "archives velocity and pressure. This is not the "
                            "ASPECT restart-checkpoint mechanism.");
        }
        prm.leave_subsection();
        prm.enter_subsection("Time dependent");
        {
          prm.declare_entry("Trajectory cleanup policy", "safe incremental",
                            Patterns::Selection("safe incremental|on success|keep"),
                            "Lifetime policy for the temporary time-dependent "
                            "adjoint trajectory. 'safe incremental' writes and "
                            "commits an independent reverse checkpoint before "
                            "reclaiming obsolete records. 'on success' retains "
                            "all temporary data until a successful reverse "
                            "sweep. 'keep' never removes it automatically.");
          prm.declare_entry("Trajectory records per segment", "4",
                            Patterns::Integer(1),
                            "Number of logical trajectory records in each shared "
                            "MPI-I/O data file. This archive-layout setting must "
                            "match when reopening an adjoint trajectory.");
          prm.declare_entry("Trajectory anchor spacing", "1",
                            Patterns::Integer(1),
                            "Fixed Phase-B sparse replay anchor spacing. A value of 1 "
                            "retains the dense full-archive behavior; larger values retain "
                            "only three-level anchor packages while preserving the dense "
                            "accepted-step schedule journal.");
        }
        prm.leave_subsection();
        prm.enter_subsection("Diagnostics");
        {
          // Diagnostics are deliberately parameter-separated from production
          // trajectory policy. Enabling one may add reference storage or
          // comparisons, but must not change the selected forward/reverse
          // discretization.
          prm.declare_entry("Run trajectory replay diagnostic", "false",
                            Patterns::Bool(),
                            "Phase-A validation hook: replay the committed full "
                            "trajectory with its recorded accepted-step schedule "
                            "and compare every produced state with the archived "
                            "reference. This does not enable sparse anchors or "
                            "change the reverse sweep.");
          prm.declare_entry("Trajectory replay tolerance", "1e-11",
                            Patterns::Double(0.0),
                            "Maximum relative state, velocity, and gauge-normalized "
                            "pressure difference accepted by the Phase-A full-archive "
                            "trajectory replay diagnostic.");
          prm.declare_entry("Run sparse trajectory replay validation", "false",
                            Patterns::Bool(),
                            "Retain a separate dense full-trajectory reference archive and "
                            "compare sparse interval replay states against it. This reference "
                            "is diagnostic-only and is not used to satisfy reverse dependencies.");
        }
        prm.leave_subsection();
        prm.enter_subsection("Time dependent");
        {
          prm.declare_entry("Adjoint checkpoint interval", "50",
                            Patterns::Integer(1),
                            "Number of completed reverse steps between durable "
                            "adjoint restart checkpoints. This is independent "
                            "of ASPECT's forward checkpoint interval.");
        }
        prm.leave_subsection();
        prm.enter_subsection("Diagnostics");
        {
          prm.declare_entry("Stop after adjoint checkpoint", "-1",
                            Patterns::Integer(-1),
                            "Testing hook: after committing a checkpoint whose "
                            "next reverse step equals this value, terminate "
                            "deliberately. A negative value disables the hook.");
          prm.declare_entry("Lifecycle failure injection", "none",
                            Patterns::Selection("none|before checkpoint commit|"
                                                "after checkpoint commit before cleanup|"
                                                "during cleanup|after cleanup"),
                            "R0 test hook that deliberately interrupts the reverse "
                            "lifecycle at a transaction boundary. Do not enable "
                            "this in a production run.");
          prm.declare_entry("Checkpoint transaction failure injection", "none",
                            Patterns::Selection(
                              "none|after temporary checkpoint directory creation|"
                              "after rank-local checkpoint payload writing|"
                              "after all payloads before metadata publication|"
                              "after checkpoint metadata publication|"
                              "after committed checkpoint directory before latest publication|"
                              "after latest publication"),
                            "Benchmark-only MPI checkpoint transaction failure "
                            "point. The selected rank reports a controlled "
                            "failure at this collective boundary; all ranks then "
                            "exit through the same quiet path.");
          prm.declare_entry("Checkpoint transaction failure rank", "-1",
                            Patterns::Integer(-1),
                            "MPI rank used for the benchmark-only checkpoint "
                            "failure injection. -1 selects rank zero.");
          prm.declare_entry("Checkpoint transaction failure next reverse step", "-1",
                            Patterns::Integer(-1),
                            "Reverse checkpoint frontier at which to inject the "
                            "benchmark-only transaction failure. A negative value "
                            "disables the injection.");
          prm.declare_entry("Trajectory cleanup failure injection", "none",
                            Patterns::Selection(
                              "none|before cleanup starts|"
                              "after rank-local cleanup payload deletion|"
                              "before cleanup collective success|"
                              "after rank-local cleanup success before manifest update|"
                              "after physical segment deletion before manifest update|"
                              "after cleanup manifest update"),
                            "Benchmark-only MPI cleanup transaction failure point. "
                            "The selected rank reports a controlled failure and "
                            "all ranks stop together. Sparse cleanup can safely "
                            "repeat an interrupted pre-manifest segment deletion "
                            "from its durable checkpoint frontier.");
          prm.declare_entry("Trajectory cleanup failure rank", "-1",
                            Patterns::Integer(-1),
                            "MPI rank used for the benchmark-only cleanup failure "
                            "injection. -1 selects rank zero.");
          prm.declare_entry("Trajectory cleanup failure next reverse step", "-1",
                            Patterns::Integer(-1),
                            "Checkpoint frontier at which to inject the benchmark-only "
                            "cleanup failure. A negative value disables injection.");
          prm.declare_entry("Trajectory append failure injection", "none",
                            Patterns::Selection(
                              "none|after local serialization before shared-file write|"
                              "after collective payload write before manifest|"
                              "after payload write before manifest|after manifest publication"),
                            "Benchmark-only shared trajectory append transaction "
                            "failure point. All ranks stop through QuietException; "
                            "do not enable in production.");
          prm.declare_entry("Trajectory append failure rank", "-1",
                            Patterns::Integer(-1),
                            "MPI rank used for the benchmark-only trajectory append "
                            "failure injection. -1 selects rank zero.");
          prm.declare_entry("Trajectory append failure step", "-1",
                            Patterns::Integer(-1),
                            "Forward trajectory record at which to inject the "
                            "benchmark-only append failure. A negative value disables it.");
          prm.declare_entry("Run trajectory Stokes recovery diagnostic", "false",
                            Patterns::Bool(),
                            "For stokes cached trajectories, recover Stokes "
                            "fields from archived temperature and composition "
                            "at representative accepted steps and compare them "
                            "with the archived forward velocity and pressure. "
                            "This validates the minimal-mode reconstruction "
                            "path without changing the adjoint equations.");
          prm.declare_entry("Run matrix-free cold rebuild diagnostic", "false",
                            Patterns::Bool(),
                            "Immediately before the first reverse Stokes solve, "
                            "discard the matrix-free/GMG runtime state and let "
                            "the ordinary adjoint-restart reconstruction path "
                            "rebuild it. This is a targeted A/B diagnostic and "
                            "does not alter adjoint algebra or checkpoint data.");
          prm.declare_entry("Output MPI adjoint memory diagnostics", "false",
                            Patterns::Bool(),
                            "Emit a rank-local RSS and distributed-vector checkpoint "
                            "around replay reconstruction and the first reverse "
                            "Stokes setup. This is an allocation-free diagnostic for "
                            "the experimental MPI reverse path.");
          prm.declare_entry("Trajectory Stokes recovery tolerance", "1e-8",
                            Patterns::Double(0.0),
                            "Maximum permitted relative velocity and "
                            "pressure error in the trajectory Stokes recovery "
                            "diagnostic after pressure-gauge normalization.");
        }
        prm.leave_subsection();
        prm.enter_subsection("Time dependent");
        {
          prm.declare_entry("Advection preconditioner", "transpose ilu",
                            Patterns::Selection("none|forward ilu experimental|transpose ilu"),
                            "Preconditioner for the time-dependent transposed "
                            "advection solve. This affects only Krylov "
                            "convergence, not the discrete adjoint equation. "
                            "The default rebuilds ILU from an explicit A^T "
                            "matrix for every restored snapshot.");
        }
        prm.leave_subsection();
        prm.enter_subsection("Diagnostics");
        {
          prm.declare_entry("Run advection preconditioner comparison", "false",
                            Patterns::Bool(),
                            "For the first temperature and composition transpose "
                            "solves, compare unpreconditioned GMRES with "
                            "ILU(A^T) using the same restored matrix and "
                            "right hand side. This diagnostic does not alter "
                            "the production adjoint solve.");
          prm.declare_entry("Run material model adjoint VJP test", "false",
                            Patterns::Bool(),
                            "Run a standalone finite-difference dot-product "
                            "diagnostic for the MaterialModelAdjoint VJP backend. "
                            "This v1 diagnostic is restricted to the simple "
                            "material model.");
          prm.declare_entry("Material model adjoint VJP test epsilon", "1e-7",
                            Patterns::Double(0.0),
                            "Finite-difference epsilon used by the standalone "
                            "MaterialModelAdjoint VJP diagnostic.");
          prm.declare_entry("Run C to Stokes RHS transpose test", "false",
                            Patterns::Bool(),
                            "Run a finite-difference dot-product diagnostic for "
                            "the composition dependence of the assembled Stokes "
                            "right hand side. This is intended for small "
                            "adjoint benchmarks only.");
          prm.declare_entry("C to Stokes RHS transpose test epsilon", "1e-7",
                            Patterns::Double(0.0),
                            "Finite-difference epsilon used by the C to Stokes "
                            "RHS transpose diagnostic.");
          prm.declare_entry("Run direct Stokes viscosity derivative test", "false",
                            Patterns::Bool(),
                            "Run a direct finite-difference dot-product diagnostic "
                            "for the Stokes residual derivative with respect to "
                            "independent quadrature-point viscosity values.");
          prm.declare_entry("Direct Stokes viscosity derivative test epsilon", "1e-7",
                            Patterns::Double(0.0),
                            "Finite-difference epsilon used by the direct Stokes "
                            "viscosity derivative diagnostic.");
          prm.declare_entry("Run transposed advection constraint test", "false",
                            Patterns::Bool(),
                            "Run a regression diagnostic that checks the "
                            "transposed advection solve enforces homogeneous "
                            "adjoint values on constrained advection DoFs.");
          prm.declare_entry("Run artificial viscosity coefficient seed test", "false",
                            Patterns::Bool(),
                            "Run an isolated finite-difference dot-product "
                            "diagnostic for the derivative of the advection "
                            "residual with respect to independent cellwise "
                            "artificial-viscosity coefficients.");
          prm.declare_entry("Artificial viscosity coefficient seed test epsilon", "1e-7",
                            Patterns::Double(0.0),
                            "Finite-difference epsilon used by the artificial "
                            "viscosity coefficient seed diagnostic.");
          prm.declare_entry("Run artificial viscosity estimator transpose test", "false",
                            Patterns::Bool(),
                            "Run a standalone finite-difference dot-product "
                            "diagnostic for the transpose of the entropy-viscosity "
                            "estimator from cellwise artificial-viscosity seed "
                            "to old/old-old field and velocity states.");
          prm.declare_entry("Artificial viscosity estimator transpose test epsilon", "1e-7",
                            Patterns::Double(0.0),
                            "Finite-difference epsilon used by the artificial "
                            "viscosity estimator transpose diagnostic.");
          prm.declare_entry("Run local entropy residual transpose test", "false",
                            Patterns::Bool(),
                            "Run a frozen-denominator diagnostic for the local "
                            "max-residual/entropy-deviation numerator of the "
                            "alpha=2 entropy-viscosity estimator.");
          prm.declare_entry("Local entropy residual transpose test epsilon", "1e-7",
                            Patterns::Double(0.0),
                            "Finite-difference epsilon used by the local entropy "
                            "residual transpose diagnostic.");
          prm.declare_entry("Run global entropy denominator scalar test", "false",
                            Patterns::Bool(),
                            "Run a scalar finite-difference diagnostic for the "
                            "global_entropy_variation denominator in the alpha=2 "
                            "entropy-viscosity estimator.");
          prm.declare_entry("Global entropy denominator scalar test epsilon", "1e-7",
                            Patterns::Double(0.0),
                            "Finite-difference epsilon used by the global entropy "
                            "denominator scalar diagnostic.");
          prm.declare_entry("Run global entropy variation transpose test", "false",
                            Patterns::Bool(),
                            "Run a standalone finite-difference diagnostic for "
                            "global_entropy_variation(field_history), freezing "
                            "the average_field supplied by the extrapolated "
                            "field range.");
          prm.declare_entry("Global entropy variation transpose test epsilon", "1e-7",
                            Patterns::Double(0.0),
                            "Finite-difference epsilon used by the global entropy "
                            "variation transpose diagnostic.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    Manager<dim>::initialize_simulator(Simulator<dim> &simulator_object)
    {
      // Manager is a facade over stateful subcomponents, all of which must
      // reference the same mutable Simulator instance. Initialization is done
      // once here so workflows never carry independently owned solver state.
      simulator = &simulator_object;

      // Initialize the read-only SimulatorAccess interface explicitly. Calling
      // this->initialize_simulator() here would recurse into this overload.
      this->SimulatorAccess<dim>::initialize_simulator(simulator_object);
      objective_manager.initialize_simulator(simulator_object);
      diagnostics_manager.initialize_simulator(simulator_object);
      initial_temperature_optimization.initialize_simulator(simulator_object);
      trajectory_manager.initialize_simulator(simulator_object);
    }



    template <int dim>
    void
    Manager<dim>::initialize_simulator(const Simulator<dim> &)
    {
      // This override exists only to reject initialization through the generic
      // SimulatorAccess interface, which cannot provide the mutable simulator
      // access required by the adjoint workflow.
      AssertThrow(false,
                  ExcMessage("The adjoint manager requires mutable simulator access. "
                             "Call initialize_simulator(Simulator<dim> &) instead."));
    }



    template <int dim>
    void
    Manager<dim>::parse_parameters(ParameterHandler &prm)
    {
      Assert(simulator != nullptr, ExcInternalError());

      // Parse capability-owned sections first. The remaining values below are
      // facade routing switches or typed test injections; no numerical
      // operator configuration is hidden inside an objective implementation.
      objective_manager.parse_parameters(prm);
      initial_temperature_optimization.parse_parameters(prm);
      trajectory_manager.parse_parameters(prm);

      prm.enter_subsection("Material model");
      {
        material_model_name = prm.get("Model name");
      }
      prm.leave_subsection();

      prm.enter_subsection("Adjoint");
      {
        prm.enter_subsection("Time dependent");
        {
          advection_preconditioner = prm.get("Advection preconditioner");
        }
        prm.leave_subsection();

        prm.enter_subsection("Diagnostics");
        {
          run_trajectory_replay_check = prm.get_bool("Run trajectory replay diagnostic");
          trajectory_replay_tolerance = prm.get_double("Trajectory replay tolerance");
          run_sparse_trajectory_replay_check =
            prm.get_bool("Run sparse trajectory replay validation");
          const int requested_stop_after_checkpoint =
            prm.get_integer("Stop after adjoint checkpoint");
          // ParameterHandler exposes the user-facing -1 sentinel as a signed
          // value. Convert once at the boundary so internal code uses the
          // deal.II invalid-index convention and unsigned step counters.
          stop_after_adjoint_checkpoint =
            requested_stop_after_checkpoint < 0
            ? numbers::invalid_unsigned_int
            : static_cast<unsigned int>(requested_stop_after_checkpoint);
          lifecycle_failure_injection = Diagnostics::parse_lifecycle_failure_point(
                                          prm.get("Lifecycle failure injection"));
          checkpoint_transaction_failure_injection =
            Diagnostics::parse_checkpoint_failure_point(
              prm.get("Checkpoint transaction failure injection"));
          checkpoint_transaction_failure_rank =
            prm.get_integer("Checkpoint transaction failure rank");
          const int requested_checkpoint_failure_step =
            prm.get_integer("Checkpoint transaction failure next reverse step");
          checkpoint_transaction_failure_next_reverse_step =
            requested_checkpoint_failure_step < 0
            ? numbers::invalid_unsigned_int
            : static_cast<unsigned int>(requested_checkpoint_failure_step);
          trajectory_cleanup_failure_injection =
            Diagnostics::parse_cleanup_failure_point(
              prm.get("Trajectory cleanup failure injection"));
          trajectory_cleanup_failure_rank =
            prm.get_integer("Trajectory cleanup failure rank");
          const int requested_cleanup_failure_step =
            prm.get_integer("Trajectory cleanup failure next reverse step");
          trajectory_cleanup_failure_next_reverse_step =
            requested_cleanup_failure_step < 0
            ? numbers::invalid_unsigned_int
            : static_cast<unsigned int>(requested_cleanup_failure_step);
          trajectory_append_failure_injection = Diagnostics::parse_append_failure_point(
                                                  prm.get("Trajectory append failure injection"));
          trajectory_append_failure_rank = prm.get_integer("Trajectory append failure rank");
          const int requested_append_failure_step =
            prm.get_integer("Trajectory append failure step");
          trajectory_append_failure_step =
            requested_append_failure_step < 0
            ? numbers::invalid_unsigned_int
            : static_cast<unsigned int>(requested_append_failure_step);
          run_trajectory_stokes_recovery_check =
            prm.get_bool("Run trajectory Stokes recovery diagnostic");
          run_matrix_free_cold_rebuild_diagnostic =
            prm.get_bool("Run matrix-free cold rebuild diagnostic");
          diagnostics_manager.set_output_memory_checkpoints(
            prm.get_bool("Output MPI adjoint memory diagnostics"));
          trajectory_stokes_recovery_tolerance =
            prm.get_double("Trajectory Stokes recovery tolerance");
          run_advection_preconditioner_comparison =
            prm.get_bool("Run advection preconditioner comparison");
          run_material_model_adjoint_vjp_diagnostic =
            prm.get_bool("Run material model adjoint VJP test");
          material_model_adjoint_vjp_diagnostic_epsilon =
            prm.get_double("Material model adjoint VJP test epsilon");
          run_composition_stokes_rhs_transpose_diagnostic =
            prm.get_bool("Run C to Stokes RHS transpose test");
          composition_stokes_rhs_transpose_diagnostic_epsilon =
            prm.get_double("C to Stokes RHS transpose test epsilon");
          run_direct_stokes_viscosity_derivative_diagnostic =
            prm.get_bool("Run direct Stokes viscosity derivative test");
          direct_stokes_viscosity_derivative_diagnostic_epsilon =
            prm.get_double("Direct Stokes viscosity derivative test epsilon");
          run_transposed_advection_constraint_diagnostic =
            prm.get_bool("Run transposed advection constraint test");
          run_artificial_viscosity_coefficient_seed_diagnostic =
            prm.get_bool("Run artificial viscosity coefficient seed test");
          artificial_viscosity_coefficient_seed_diagnostic_epsilon =
            prm.get_double("Artificial viscosity coefficient seed test epsilon");
          run_artificial_viscosity_estimator_transpose_diagnostic =
            prm.get_bool("Run artificial viscosity estimator transpose test");
          artificial_viscosity_estimator_transpose_diagnostic_epsilon =
            prm.get_double("Artificial viscosity estimator transpose test epsilon");
          run_local_entropy_residual_transpose_diagnostic =
            prm.get_bool("Run local entropy residual transpose test");
          local_entropy_residual_transpose_diagnostic_epsilon =
            prm.get_double("Local entropy residual transpose test epsilon");
          run_global_entropy_denominator_scalar_diagnostic =
            prm.get_bool("Run global entropy denominator scalar test");
          global_entropy_denominator_scalar_diagnostic_epsilon =
            prm.get_double("Global entropy denominator scalar test epsilon");
          run_global_entropy_variation_transpose_diagnostic =
            prm.get_bool("Run global entropy variation transpose test");
          global_entropy_variation_transpose_diagnostic_epsilon =
            prm.get_double("Global entropy variation transpose test epsilon");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    Manager<dim>::initialize()
    {}



    template <int dim>
    const KernelRepository<dim> &
    Manager<dim>::get_kernels() const
    {
      return kernels;
    }



    template <int dim>
    std::map<std::string, double>
    Manager<dim>::get_objective_values() const
    {
      // Expose values by stable plugin name and do not leak ObjectiveResult
      // ownership. Time-dependent running/terminal aggregation is finalized by
      // its workflow before populating this result list.
      std::map<std::string, double> values;

      for (const auto &objective_result : objective_results)
        values[objective_result->objective_name] = objective_result->value;

      return values;
    }

    template <int dim>
    const LinearAlgebra::BlockVector &
    Manager<dim>::get_initial_temperature_gradient() const
    {
      return time_dependent_workflow.initial_temperature_gradient;
    }



    template <int dim>
    const std::vector<LinearAlgebra::BlockVector> &
    Manager<dim>::get_initial_composition_gradients() const
    {
      return time_dependent_workflow.initial_composition_gradients;
    }

    template <int dim>
    bool
    Manager<dim>::time_dependent_temperature_workflow_enabled() const
    {
      using NonlinearSolver = typename Parameters<dim>::NonlinearSolver;

      // The specialized solver scheme is the integration point that records
      // accepted snapshots and invokes the reverse sweep. Merely selecting a
      // time-dependent objective does not retrofit these hooks into an
      // arbitrary nonlinear solver scheme.
      return this->get_parameters().nonlinear_solver
             == NonlinearSolver::single_Advection_single_Stokes_adjoint_temperature;
    }


    template <int dim>
    bool
    Manager<dim>::optimization_enabled() const
    {
      return initial_temperature_optimization.enabled();
    }


    template <int dim>
    void
    Manager<dim>::run_initial_temperature_optimization()
    {
      initial_temperature_optimization.run(*this);
    }



    template <int dim>
    void
    Manager<dim>::inject_lifecycle_failure(
      const Diagnostics::LifecycleFailurePoint point,
      const unsigned int next_reverse_step) const
    {
      // Injection points are typed transaction boundaries. They are checked
      // by the workflow only after it has established the stated durable
      // frontier, making failures reproducible without string comparisons in
      // production storage code.
      if (lifecycle_failure_injection == point)
        AssertThrow(false,
                    ExcMessage("Injected adjoint lifecycle interruption "
                               + Diagnostics::to_string(point) + " before reverse step "
                               + Utilities::int_to_string(next_reverse_step)
                               + ". Restart with 'Lifecycle failure injection = none' "
                               "to verify recovery."));
    }



    template <int dim>
    void
    Manager<dim>::finalize_time_dependent_temperature()
    {
      // All reverse-time implementation and ownership live in the workflow;
      // the facade is intentionally a one-line integration entry point for
      // Simulator's specialized nonlinear solver scheme.
      time_dependent_workflow.run(*this);
    }



    template <int dim>
    void
    Manager<dim>::solve_instantaneous_stokes()
    {
      Assert(simulator != nullptr, ExcInternalError());

      // The steady workflow owns forward postprocessor preparation, objective
      // RHS construction, physical transpose solves, and kernel aggregation.
      // Manager retains only the resulting public artifacts.
      InstantaneousStokesWorkflow<dim>(*simulator, objective_manager)
      .run(objective_results, adjoint_states, kernels);
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template class Adjoint::Manager<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
