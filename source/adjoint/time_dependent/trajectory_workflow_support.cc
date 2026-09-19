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
    bool
    Manager<dim>::has_pending_adjoint_restart() const
    {
      if (!time_dependent_temperature_workflow_enabled())
        return false;

      const std::string restart_manifest = this->get_output_directory()
                                           + "adjoint_trajectory/adjoint_restart/latest";
      return Utilities::fexists(restart_manifest, this->get_mpi_communicator());
    }



    template <int dim>
    void
    Manager<dim>::validate_time_dependent_temperature_setup() const
    {
      using NonlinearSolver = typename Parameters<dim>::NonlinearSolver;

      const std::string impermeable_boundary_message =
        "The time-dependent adjoint currently requires u.n = 0 on physical "
        "boundaries. Prescribed normal inflow/outflow and inflow-dependent "
        "compositional boundary conditions are not supported in v1.";

      AssertThrow(simulator != nullptr, ExcInternalError());
      AssertThrow(this->get_parameters().nonlinear_solver
                  == NonlinearSolver::single_Advection_single_Stokes_adjoint_temperature,
                  ExcMessage("The time-dependent temperature adjoint requires "
                             "Nonlinear solver scheme = single Advection, single Stokes, adjoint temperature."));
      AssertThrow(this->introspection().n_chemical_composition_fields()
                  == this->introspection().n_compositional_fields,
                  ExcMessage("The time-dependent temperature adjoint with compositional controls "
                             "currently supports only compositional fields of type chemical composition."));
      for (unsigned int c = 0; c < this->introspection().n_compositional_fields; ++c)
        {
          const AdvectionField composition_field = AdvectionField::composition(c);
          AssertThrow(composition_field.advection_method(this->introspection())
                      == Parameters<dim>::AdvectionFieldMethod::fem_field,
                      ExcMessage("The time-dependent temperature adjoint with compositional controls "
                                 "currently supports only composition fields with method 'field'."));
          AssertThrow(composition_field.is_discontinuous(this->introspection()) == false,
                      ExcMessage("The time-dependent temperature adjoint with compositional controls "
                                 "supports only continuous composition elements."));
        }
      AssertThrow(this->get_parameters().use_operator_splitting == false,
                  ExcMessage("The time-dependent temperature adjoint v1 does not support operator splitting."));
      AssertThrow(this->get_parameters().include_melt_transport == false,
                  ExcMessage("The time-dependent temperature adjoint v1 does not support melt transport."));
      AssertThrow(this->get_parameters().advection_stabilization_method
                  == Parameters<dim>::AdvectionStabilizationMethod::entropy_viscosity,
                  ExcMessage("The time-dependent temperature adjoint v1 does not support SUPG stabilization. "
                             "Use Stabilization method = entropy viscosity."));
      AssertThrow(this->get_parameters().use_artificial_viscosity_smoothing == false,
                  ExcMessage("The time-dependent temperature adjoint v1 does not support artificial-viscosity smoothing. "
                             "The neighbor maximum used by smoothing needs an explicit transpose before it can be enabled."));
      AssertThrow(simulator->mesh_deformation == nullptr,
                  ExcMessage("The time-dependent temperature adjoint v1 supports only fixed meshes."));
      AssertThrow(this->get_parameters().adaptive_refinement_interval == 0
                  && this->get_parameters().additional_refinement_times.empty(),
                  ExcMessage("The time-dependent temperature adjoint trajectory "
                             "archive supports only a fixed mesh. Disable "
                             "scheduled and periodic mesh refinement."));
      AssertThrow(AdvectionField::temperature().is_discontinuous(this->introspection()) == false,
                  ExcMessage("The time-dependent temperature adjoint v1 supports only continuous temperature elements."));
      AssertThrow(this->get_heating_model_manager().get_active_plugin_names().empty(),
                  ExcMessage("The time-dependent temperature adjoint v1 does not yet support heating model derivatives, "
                             "including latent heat, adiabatic heating, shear heating, or radiogenic heating."));

      // For the standard Box geometry, the boundary id directly identifies
      // the Cartesian normal direction. This is a configuration-level check:
      // it rejects an explicitly prescribed normal velocity component without
      // sampling a boundary function. Curved-geometry velocity plugins may
      // legitimately use a full Cartesian component mask for a tangential
      // field, so their u.n=0 contract cannot be inferred from the mask and
      // is documented as a v1 user requirement instead.
      if (Plugins::plugin_type_matches<const GeometryModel::Box<dim>>(this->get_geometry_model()))
        for (const types::boundary_id boundary_id :
             this->get_boundary_velocity_manager().get_prescribed_boundary_velocity_indicators())
          {
            const unsigned int normal_component = boundary_id / 2;
            const ComponentMask component_mask =
              this->get_boundary_velocity_manager().get_component_mask(boundary_id);
            AssertThrow(component_mask[this->introspection().component_indices.velocities[normal_component]] == false,
                        ExcMessage(impermeable_boundary_message));
          }

      const MaterialModel::NonlinearDependence::ModelDependence model_dependence =
        this->get_material_model().get_model_dependence();
      AssertThrow((static_cast<int>(model_dependence.viscosity)
                   & (static_cast<int>(MaterialModel::NonlinearDependence::pressure)
                      | static_cast<int>(MaterialModel::NonlinearDependence::strain_rate))) == 0,
                  ExcMessage("The time-dependent temperature adjoint v1 supports only temperature- and "
                             "composition-dependent viscosity. Pressure- and strain-rate-dependent "
                             "viscosity remain unsupported."));
      if ((static_cast<int>(model_dependence.viscosity)
           & static_cast<int>(MaterialModel::NonlinearDependence::any_variable)) != 0)
        AssertThrow(this->get_parameters().material_averaging
                    == MaterialModel::MaterialAveraging::none,
                    ExcMessage("The time-dependent temperature adjoint v1 supports eta(T,C) only when "
                               "Material model/Material averaging = none. The transpose of viscosity "
                               "averaging/projection is not part of this v1 layer."));
      AssertThrow((static_cast<int>(model_dependence.thermal_conductivity)
                   & static_cast<int>(MaterialModel::NonlinearDependence::any_variable)) == 0,
                  ExcMessage("The time-dependent temperature adjoint v1 does not yet support thermal-conductivity derivatives."));
      AssertThrow((static_cast<int>(model_dependence.specific_heat)
                   & (static_cast<int>(MaterialModel::NonlinearDependence::pressure)
                      | static_cast<int>(MaterialModel::NonlinearDependence::strain_rate)
                      | static_cast<int>(MaterialModel::NonlinearDependence::compositional_fields))) == 0,
                  ExcMessage("The time-dependent temperature adjoint v1 supports only temperature-dependent specific heat. "
                             "Pressure-, strain-rate-, and composition-dependent specific heat remain unsupported."));
    }



    template <int dim>
    void
    Manager<dim>::record_forward_snapshot(const double outgoing_time_step,
                                          const bool has_outgoing_step)
    {
      trajectory_manager.record_forward_snapshot(*this, outgoing_time_step,
                                                 has_outgoing_step);
    }



    template <int dim>
    void
    Manager<dim>::mark_forward_snapshot_terminal()
    {
      trajectory_manager.mark_forward_snapshot_terminal(*this);
    }



    template <int dim>
    void
    Manager<dim>::recover_stokes_state(const unsigned int record,
                                       LinearAlgebra::BlockVector &state)
    {
      AssertThrow(trajectory_manager.has_archive(), ExcInternalError());
      const ReplayStepMetadata &metadata = trajectory_manager.archive().metadata(record);

      // The v1 dependency audit permits this reconstruction: Stokes depends
      // on T,C and time-dependent boundary data, but not on archived matrices,
      // constraints, pressure, or strain rate. The old fields are not read by
      // the supported incompressible/Picard Stokes path.
      simulator->solution = state;
      simulator->old_solution = state;
      simulator->old_old_solution = state;
      simulator->current_linearization_point = state;
      simulator->time = metadata.time;
      simulator->time_step = metadata.time_step;
      simulator->old_time_step = metadata.old_time_step;
      simulator->timestep_number = metadata.step;
      simulator->last_pressure_normalization_adjustment = 0.0;
      simulator->compute_current_constraints();

      {
        TimerOutput::Scope recovery_timer(this->get_computing_timer(),
                                          "Adjoint trajectory Stokes recovery");
        simulator->rebuild_stokes_matrix = true;
        simulator->rebuild_stokes_preconditioner = true;
        simulator->assemble_and_solve_stokes();
      }

      state = simulator->solution;
      ++trajectory_manager.stokes_recovery_solves;
    }



    template <int dim>
    void
    Manager<dim>::run_trajectory_stokes_recovery_diagnostic()
    {
      AssertThrow(trajectory_manager.has_archive(), ExcInternalError());
      AssertThrow(trajectory_manager.archive().mode()
                  == TrajectoryStorageMode::stokes_cached,
                  ExcMessage("The trajectory Stokes recovery diagnostic needs "
                             "'Trajectory storage mode = stokes cached' so it "
                             "can compare with the forward Stokes state."));

      std::vector<unsigned int> sampled_records(1, 0);
      if (trajectory_manager.archive().size() > 2)
        sampled_records.push_back((trajectory_manager.archive().size() - 1) / 2);
      if (trajectory_manager.archive().size() > 1)
        sampled_records.push_back(trajectory_manager.archive().size() - 1);
      std::sort(sampled_records.begin(), sampled_records.end());
      sampled_records.erase(std::unique(sampled_records.begin(), sampled_records.end()),
                            sampled_records.end());

      const unsigned int velocity_block =
        this->introspection().block_indices.velocities;
      const unsigned int pressure_block =
        this->introspection().block_indices.pressure;

      for (const unsigned int record : sampled_records)
        {
          LinearAlgebra::BlockVector recovered;
          recovered.reinit(this->introspection().index_sets.system_partitioning,
                           this->introspection().index_sets.system_relevant_partitioning,
                           this->get_mpi_communicator());
          recovered = 0.0;
          {
            TimerOutput::Scope read_timer(this->get_computing_timer(),
                                          "Adjoint trajectory read");
            trajectory_manager.archive().load(record,
                                              recovered,
                                              this->introspection().block_indices.temperature,
                                              this->introspection().block_indices.compositional_fields,
                                              velocity_block,
                                              pressure_block);
          }
          const LinearAlgebra::BlockVector forward_stokes_state(recovered);
          recovered.block(velocity_block) = 0.0;
          recovered.block(pressure_block) = 0.0;
          recover_stokes_state(record, recovered);

          LinearAlgebra::Vector velocity_difference(recovered.block(velocity_block));
          velocity_difference.add(-1.0,
                                  forward_stokes_state.block(velocity_block));
          const double velocity_error = velocity_difference.l2_norm()
                                        / std::max(forward_stokes_state.block(velocity_block).l2_norm(),
                                                   1e-50);

          LinearAlgebra::Vector pressure_difference(recovered.block(pressure_block));
          pressure_difference.add(-1.0,
                                  forward_stokes_state.block(pressure_block));
          // A pressure nullspace may leave a constant offset even though the
          // recovered Stokes solution is physically equivalent.
          pressure_difference.add(-pressure_difference.mean_value());
          const double pressure_error = pressure_difference.l2_norm()
                                        / std::max(forward_stokes_state.block(pressure_block).l2_norm(),
                                                   1e-50);

          this->get_pcout() << "   Adjoint trajectory Stokes recovery, record "
                            << record << ": relative velocity error = "
                            << std::setprecision(16) << velocity_error
                            << ", gauge-normalized relative pressure error = "
                            << pressure_error << std::setprecision(6) << std::endl;
          AssertThrow(velocity_error <= trajectory_stokes_recovery_tolerance
                      && pressure_error <= trajectory_stokes_recovery_tolerance,
                      ExcMessage("Adjoint trajectory Stokes recovery did not "
                                 "match the cached forward state within the "
                                 "configured tolerance."));
        }
    }



    template <int dim>
    void
    Manager<dim>::validate_trajectory_replay_capability() const
    {
      AssertThrow(simulator != nullptr, ExcInternalError());
      // Phase A deliberately has an explicit, small capability envelope. It
      // is not inferred from Plugin::save() because an empty payload does not
      // establish that update(), signals, or external resources are replay
      // deterministic.
      AssertThrow(simulator->particle_managers.empty()
                  && !this->get_parameters().volume_of_fluid_tracking_enabled,
                  ExcMessage("Phase-A trajectory replay does not support particles or volume-of-fluid state."));
      AssertThrow(!this->get_parameters().run_postprocessors_on_nonlinear_iterations,
                  ExcMessage("Phase-A trajectory replay does not support postprocessors on nonlinear iterations."));
      AssertThrow(Plugins::plugin_type_matches<const GeometryModel::Box<dim>>(
                    this->get_geometry_model())
                  && Plugins::plugin_type_matches<const MaterialModel::Simple<dim>>(
                    this->get_material_model())
                  && Plugins::plugin_type_matches<const GravityModel::Vertical<dim>>(
                    this->get_gravity_model()),
                  ExcMessage("Phase-A trajectory replay is explicitly limited to the "
                             "stateless Box/Simple/Vertical benchmark capability set. "
                             "Other plugin combinations require a dedicated replay audit."));
      AssertThrow(this->get_boundary_velocity_manager().get_active_plugin_names().empty(),
                  ExcMessage("Phase-A trajectory replay does not yet support prescribed "
                             "velocity plugins, including time-dependent functions."));
    }



    template <int dim>
    void
    Manager<dim>::run_trajectory_replay_diagnostic()
    {
      AssertThrow(trajectory_manager.has_archive() && trajectory_manager.archive().size() > 1,
                  ExcMessage("The trajectory replay diagnostic requires at least one accepted transition."));
      validate_trajectory_replay_capability();
      this->get_pcout() << "ASPECT_ADJOINT_REPLAY diagnostic_begin records="
                        << trajectory_manager.archive().size() << std::endl;

      const LinearAlgebra::BlockVector saved_solution = simulator->solution;
      const LinearAlgebra::BlockVector saved_old_solution = simulator->old_solution;
      const LinearAlgebra::BlockVector saved_old_old_solution = simulator->old_old_solution;
      const LinearAlgebra::BlockVector saved_linearization_point = simulator->current_linearization_point;
      const LinearAlgebra::BlockVector saved_system_rhs = simulator->system_rhs;
      const double saved_time = simulator->time;
      const double saved_time_step = simulator->time_step;
      const double saved_old_time_step = simulator->old_time_step;
      const unsigned int saved_timestep_number = simulator->timestep_number;
      const double saved_pressure_normalization_adjustment =
        simulator->last_pressure_normalization_adjustment;
      const unsigned int saved_stokes_recovery_solves = trajectory_manager.stokes_recovery_solves;

      const auto load_reference_state = [this](const unsigned int record)
      {
        LinearAlgebra::BlockVector locally_owned;
        locally_owned.reinit(this->introspection().index_sets.system_partitioning,
                             this->get_mpi_communicator());
        locally_owned = 0.0;
        trajectory_manager.archive().load(record,
                                          locally_owned,
                                          this->introspection().block_indices.temperature,
                                          this->introspection().block_indices.compositional_fields,
                                          this->introspection().block_indices.velocities,
                                          this->introspection().block_indices.pressure);
        LinearAlgebra::BlockVector result;
        result.reinit(this->introspection().index_sets.system_partitioning,
                      this->introspection().index_sets.system_relevant_partitioning,
                      this->get_mpi_communicator());
        result = locally_owned;
        if (trajectory_manager.archive().mode() == TrajectoryStorageMode::minimal)
          recover_stokes_state(record, result);
        return result;
      };
      const auto relative_difference = [](const LinearAlgebra::Vector &actual,
                                          const LinearAlgebra::Vector &reference)
      {
        LinearAlgebra::Vector difference(actual);
        difference.add(-1.0, reference);
        return difference.l2_norm() / std::max(reference.l2_norm(), 1e-50);
      };
      const auto locally_owned_copy = [this](const LinearAlgebra::BlockVector &source)
      {
        LinearAlgebra::BlockVector result;
        result.reinit(this->introspection().index_sets.system_partitioning,
                      this->get_mpi_communicator());
        result = source;
        return result;
      };
      const auto equal_time = [](const double left, const double right)
      {
        return std::abs(left - right)
               <= 64.0 * std::numeric_limits<double>::epsilon()
               * std::max({1.0, std::abs(left), std::abs(right)});
      };

      try
        {
          const ReplayStepMetadata &initial_metadata = trajectory_manager.archive().metadata(0);
          simulator->solution = load_reference_state(0);
          simulator->old_solution = load_reference_state(0);
          simulator->old_old_solution = load_reference_state(0);
          simulator->current_linearization_point = simulator->solution;
          simulator->time = initial_metadata.time;
          simulator->time_step = initial_metadata.time_step;
          simulator->old_time_step = initial_metadata.old_time_step;
          simulator->timestep_number = initial_metadata.step;
          simulator->last_pressure_normalization_adjustment = 0.0;
          this->get_pcout() << "ASPECT_ADJOINT_REPLAY initial_state_loaded" << std::endl;

          for (unsigned int record = 0; record + 1 < trajectory_manager.archive().size(); ++record)
            {
              const ReplayStepMetadata &metadata = trajectory_manager.archive().metadata(record);
              const ReplayStepMetadata &next_metadata = trajectory_manager.archive().metadata(record + 1);
              AssertThrow(metadata.has_outgoing_step,
                          ExcMessage("A non-final archived state is missing its accepted outgoing time step."));
              AssertThrow(equal_time(metadata.time + metadata.outgoing_time_step,
                                     next_metadata.time),
                          ExcMessage("The accepted trajectory schedule has inconsistent time metadata."));

              this->get_pcout() << "ASPECT_ADJOINT_REPLAY replay_begin step="
                                << metadata.step << std::endl;
              simulator->replay_accepted_timestep(metadata);
              this->get_pcout() << "ASPECT_ADJOINT_REPLAY replay_complete step="
                                << metadata.step << std::endl;

              const LinearAlgebra::BlockVector replayed_solution = simulator->solution;
              const LinearAlgebra::BlockVector replayed_old_solution = simulator->old_solution;
              const LinearAlgebra::BlockVector replayed_old_old_solution = simulator->old_old_solution;
              const LinearAlgebra::BlockVector replayed_linearization_point =
                simulator->current_linearization_point;
              const LinearAlgebra::BlockVector replayed_system_rhs = simulator->system_rhs;
              const LinearAlgebra::BlockVector replayed_locally_owned =
                locally_owned_copy(replayed_solution);
              const double replayed_time = simulator->time;
              const double replayed_time_step = simulator->time_step;
              const double replayed_old_time_step = simulator->old_time_step;
              const unsigned int replayed_timestep_number = simulator->timestep_number;
              const double replayed_pressure_normalization_adjustment =
                simulator->last_pressure_normalization_adjustment;

              this->get_pcout() << "ASPECT_ADJOINT_REPLAY reference_load_begin step="
                                << metadata.step << std::endl;
              const LinearAlgebra::BlockVector reference = load_reference_state(record + 1);
              this->get_pcout() << "ASPECT_ADJOINT_REPLAY reference_load_complete step="
                                << metadata.step << std::endl;
              const LinearAlgebra::BlockVector reference_locally_owned =
                locally_owned_copy(reference);
              LinearAlgebra::BlockVector difference(replayed_locally_owned);
              difference.add(-1.0, reference_locally_owned);
              const double state_relative_difference = difference.l2_norm()
                                                       / std::max(reference_locally_owned.l2_norm(), 1e-50);
              this->get_pcout() << "ASPECT_ADJOINT_REPLAY state_difference_complete step="
                                << metadata.step << std::endl;
              const unsigned int velocity_block = this->introspection().block_indices.velocities;
              const unsigned int pressure_block = this->introspection().block_indices.pressure;
              const double velocity_relative_difference =
                relative_difference(replayed_locally_owned.block(velocity_block),
                                    reference_locally_owned.block(velocity_block));
              LinearAlgebra::Vector pressure_difference(replayed_locally_owned.block(pressure_block));
              pressure_difference.add(-1.0, reference_locally_owned.block(pressure_block));
              pressure_difference.add(-pressure_difference.mean_value());
              const double pressure_relative_difference =
                pressure_difference.l2_norm()
                / std::max(reference_locally_owned.block(pressure_block).l2_norm(), 1e-50);
              this->get_pcout() << "ASPECT_ADJOINT_REPLAY field_differences_complete step="
                                << metadata.step << std::endl;
              const bool metadata_matches =
                replayed_timestep_number == next_metadata.step
                && equal_time(replayed_time, next_metadata.time)
                && equal_time(replayed_time_step, next_metadata.time_step)
                && equal_time(replayed_old_time_step, next_metadata.old_time_step);

              this->get_pcout()
                  << "ASPECT_ADJOINT_REPLAY step=" << metadata.step
                  << " time=" << std::setprecision(16) << metadata.time
                  << " outgoing_dt=" << metadata.outgoing_time_step
                  << " resulting_time=" << replayed_time
                  << " state_l2=" << replayed_locally_owned.l2_norm()
                  << " relative_state_difference=" << state_relative_difference
                  << " relative_velocity_difference=" << velocity_relative_difference
                  << " gauge_relative_pressure_difference=" << pressure_relative_difference
                  << " nonlinear_iterations=" << simulator->nonlinear_iteration + 1
                  << " metadata_match=" << (metadata_matches ? 1 : 0)
                  << std::setprecision(6) << std::endl;
              AssertThrow(metadata_matches,
                          ExcMessage("Accepted-step replay produced inconsistent time-step metadata."));
              AssertThrow(state_relative_difference <= trajectory_replay_tolerance
                          && velocity_relative_difference <= trajectory_replay_tolerance
                          && pressure_relative_difference <= trajectory_replay_tolerance,
                          ExcMessage("Accepted-step replay did not reproduce the archived "
                                     "full-trajectory reference within the configured tolerance."));

              // Loading a minimal reference may have reconstructed a Stokes
              // state and therefore altered runtime matrices. Restore the
              // candidate exactly, then make the next replay step rebuild its
              // transient Stokes runtime state.
              simulator->solution = replayed_solution;
              simulator->old_solution = replayed_old_solution;
              simulator->old_old_solution = replayed_old_old_solution;
              simulator->current_linearization_point = replayed_linearization_point;
              simulator->system_rhs = replayed_system_rhs;
              simulator->time = replayed_time;
              simulator->time_step = replayed_time_step;
              simulator->old_time_step = replayed_old_time_step;
              simulator->timestep_number = replayed_timestep_number;
              simulator->last_pressure_normalization_adjustment =
                replayed_pressure_normalization_adjustment;
              simulator->rebuild_stokes_matrix = true;
              simulator->rebuild_stokes_preconditioner = true;
            }
        }
      catch (...)
        {
          simulator->solution = saved_solution;
          simulator->old_solution = saved_old_solution;
          simulator->old_old_solution = saved_old_old_solution;
          simulator->current_linearization_point = saved_linearization_point;
          simulator->system_rhs = saved_system_rhs;
          simulator->time = saved_time;
          simulator->time_step = saved_time_step;
          simulator->old_time_step = saved_old_time_step;
          simulator->timestep_number = saved_timestep_number;
          simulator->last_pressure_normalization_adjustment =
            saved_pressure_normalization_adjustment;
          trajectory_manager.stokes_recovery_solves = saved_stokes_recovery_solves;
          throw;
        }

      simulator->solution = saved_solution;
      simulator->old_solution = saved_old_solution;
      simulator->old_old_solution = saved_old_old_solution;
      simulator->current_linearization_point = saved_linearization_point;
      simulator->system_rhs = saved_system_rhs;
      simulator->time = saved_time;
      simulator->time_step = saved_time_step;
      simulator->old_time_step = saved_old_time_step;
      simulator->timestep_number = saved_timestep_number;
      simulator->last_pressure_normalization_adjustment =
        saved_pressure_normalization_adjustment;
      simulator->rebuild_stokes_matrix = true;
      simulator->rebuild_stokes_preconditioner = true;
      trajectory_manager.stokes_recovery_solves = saved_stokes_recovery_solves;
    }



    template <int dim>
    void
    Manager<dim>::compare_sparse_replay_state(const unsigned int record,
                                              const LinearAlgebra::BlockVector &state)
    {
      if (!trajectory_manager.has_reference_archive())
        return;

      LinearAlgebra::BlockVector reference_owned;
      reference_owned.reinit(this->introspection().index_sets.system_partitioning,
                             this->get_mpi_communicator());
      trajectory_manager.reference_archive().load(record, reference_owned,
                                                  this->introspection().block_indices.temperature,
                                                  this->introspection().block_indices.compositional_fields,
                                                  this->introspection().block_indices.velocities,
                                                  this->introspection().block_indices.pressure);
      LinearAlgebra::BlockVector state_owned;
      state_owned.reinit(this->introspection().index_sets.system_partitioning,
                         this->get_mpi_communicator());
      state_owned = state;
      LinearAlgebra::BlockVector difference(state_owned);
      difference.add(-1.0, reference_owned);
      const double state_error = difference.l2_norm()
                                 / std::max(reference_owned.l2_norm(), 1e-50);
      const unsigned int velocity_block = this->introspection().block_indices.velocities;
      const unsigned int pressure_block = this->introspection().block_indices.pressure;
      LinearAlgebra::Vector velocity_difference(state_owned.block(velocity_block));
      velocity_difference.add(-1.0, reference_owned.block(velocity_block));
      const double velocity_error = velocity_difference.l2_norm()
                                    / std::max(reference_owned.block(velocity_block).l2_norm(), 1e-50);
      LinearAlgebra::Vector pressure_difference(state_owned.block(pressure_block));
      pressure_difference.add(-1.0, reference_owned.block(pressure_block));
      pressure_difference.add(-pressure_difference.mean_value());
      const double pressure_error = pressure_difference.l2_norm()
                                    / std::max(reference_owned.block(pressure_block).l2_norm(), 1e-50);
      trajectory_manager.sparse_max_state_relative_difference = std::max(trajectory_manager.sparse_max_state_relative_difference, state_error);
      trajectory_manager.sparse_max_velocity_relative_difference = std::max(trajectory_manager.sparse_max_velocity_relative_difference, velocity_error);
      trajectory_manager.sparse_max_pressure_relative_difference = std::max(trajectory_manager.sparse_max_pressure_relative_difference, pressure_error);
      // The sparse reference stores the exact forward Stokes field, whereas
      // minimal sparse replay deterministically recomputes it. The velocity
      // comparison therefore has the documented Stokes-recovery tolerance;
      // the transported state and gauge-normalized pressure retain the
      // stricter trajectory-replay tolerance.
      if (!(state_error <= trajectory_replay_tolerance
            && velocity_error <= trajectory_stokes_recovery_tolerance
            && pressure_error <= trajectory_replay_tolerance))
        {
          // A Debug-mode AssertThrow formats a deep exception trace on every
          // rank. That formatter can itself consume unbounded memory after a
          // sparse replay mismatch. Emit the first numerical evidence once
          // and use the existing quiet collective exit path instead.
          this->get_pcout()
              << "ASPECT_ADJOINT_SPARSE_REPLAY_MISMATCH record=" << record
              << " state_relative_difference=" << state_error
              << " velocity_relative_difference=" << velocity_error
              << " gauge_pressure_relative_difference=" << pressure_error
              << " replay_tolerance=" << trajectory_replay_tolerance
              << " stokes_recovery_tolerance=" << trajectory_stokes_recovery_tolerance
              << std::endl;
          throw QuietException();
        }
    }



    template <int dim>
    void
    Manager<dim>::cleanup_trajectory_after_checkpoint(const AdjointRestartState &state)
    {
      AssertThrow(trajectory_manager.has_archive(), ExcInternalError());
      if (trajectory_manager.configuration().cleanup_policy != TrajectoryCleanupPolicy::safe_incremental)
        return;

      // Cleanup is allowed only after the checkpoint containing this reverse
      // history and next-step frontier has committed. Thus restart can always
      // resume without any payload reclaimed by the following transaction.
      TimerOutput::Scope cleanup_timer(this->get_computing_timer(),
                                       "Adjoint trajectory cleanup");
      const bool inject_cleanup_transaction =
        state.cleanup_frontier == trajectory_cleanup_failure_next_reverse_step;
      const Diagnostics::FailureInjection<Diagnostics::CleanupFailurePoint> injection =
      {
        inject_cleanup_transaction
        ? trajectory_cleanup_failure_injection
        : Diagnostics::CleanupFailurePoint::none,
        trajectory_cleanup_failure_rank
      };
      if (trajectory_manager.configuration().anchor_spacing == 1)
        {
          // In dense mode reverse steps decrease monotonically. Records at or
          // beyond cleanup_frontier have already been consumed and can be
          // removed while older records remain available one by one.
          const TrajectoryCleanupResult cleanup =
            trajectory_manager.archive().reclaim_records_after(state.cleanup_frontier,
                                                               injection);
          if (!cleanup.complete)
            {
              this->get_pcout() << "   Adjoint trajectory cleanup transaction failed collectively; "
                                << "the committed checkpoint remains authoritative." << std::endl;
              if (Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()) > 1)
                throw QuietException();
              AssertThrow(false, ExcMessage("Adjoint trajectory cleanup transaction failed."));
            }
          std::cout << "ASPECT_ADJOINT_MPI_CLEANUP rank="
                    << Utilities::MPI::this_mpi_process(this->get_mpi_communicator())
                    << " next_reverse_step=" << state.next_reverse_step
                    << " cleanup_frontier=" << state.cleanup_frontier
                    << " local_records_deleted=" << cleanup.records_deleted
                    << " local_payload_files_deleted=" << cleanup.payload_files_deleted
                    << " local_bytes_reclaimed=" << cleanup.reclaimed.total_bytes()
                    << std::endl;
          return;
        }

      // Sparse mode reclaims whole replay intervals rather than individual
      // history states. Retain the lower anchor package selected by the next
      // reverse edge; newer anchor payloads are no longer reachable.
      const unsigned int required_anchor = trajectory_manager.required_anchor_for_reverse_step(
                                             state.next_reverse_step);
      AssertThrow(state.required_sparse_anchor == required_anchor,
                  ExcMessage("Sparse cleanup must retain the restart checkpoint's lower anchor."));
      const auto anchor_list = [this]()
      {
        std::vector<unsigned int> result;
        for (unsigned int record = 0; record < trajectory_manager.archive().size(); ++record)
          if (trajectory_manager.archive().has_record(record)
              && (record % trajectory_manager.configuration().anchor_spacing == 0
                  || record + 1 == trajectory_manager.archive().size()))
            result.push_back(record);
        return result;
      };
      const std::vector<unsigned int> live_before = anchor_list();
      const std::size_t archive_bytes_before = trajectory_manager.archive().shared_archive_bytes();
      const unsigned int segments_before = trajectory_manager.archive().shared_segment_count();
      inject_lifecycle_failure(Diagnostics::LifecycleFailurePoint::during_cleanup,
                               state.next_reverse_step);
      const TrajectoryCleanupResult cleanup =
        trajectory_manager.archive().reclaim_sparse_payloads_after(required_anchor,
                                                                   injection);
      if (!cleanup.complete)
        {
          this->get_pcout() << "   Sparse anchor cleanup transaction failed collectively; "
                            << "the committed checkpoint remains authoritative." << std::endl;
          if (Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()) > 1)
            throw QuietException();
          AssertThrow(false, ExcMessage("Sparse anchor cleanup transaction failed."));
        }
      AssertThrow(trajectory_manager.archive().has_record(required_anchor),
                  ExcMessage("Sparse cleanup removed the durable restart anchor."));
      const std::vector<unsigned int> live_after = anchor_list();
      const std::size_t archive_bytes_after = trajectory_manager.archive().shared_archive_bytes();
      const unsigned int segments_after = trajectory_manager.archive().shared_segment_count();
      const auto format_anchors = [](const std::vector<unsigned int> &anchors)
      {
        std::string text;
        for (const unsigned int anchor : anchors)
          text += (text.empty() ? "" : ",") + Utilities::int_to_string(anchor);
        return text.empty() ? std::string("none") : text;
      };
      std::vector<unsigned int> reclaimed_anchors;
      for (const unsigned int anchor : live_before)
        if (std::find(live_after.begin(), live_after.end(), anchor) == live_after.end())
          reclaimed_anchors.push_back(anchor);
      this->get_pcout() << "ASPECT_ADJOINT_SPARSE_CLEANUP next_reverse_step="
                        << state.next_reverse_step
                        << " required_lower_anchor=" << required_anchor
                        << " live_anchors=" << format_anchors(live_after)
                        << " logically_reclaimed_anchors=" << format_anchors(reclaimed_anchors)
                        << " physically_retained_segments=" << segments_after
                        << " physically_deleted_segments=" << segments_before - segments_after
                        << " archive_bytes=" << archive_bytes_after
                        << " reclaimed_bytes=" << archive_bytes_before - archive_bytes_after
                        << std::endl;
    }



    template <int dim>
    void
    Manager<dim>::materialize_forward_replay_window(const unsigned int record)
    {
      AssertThrow(trajectory_manager.configuration().anchor_spacing > 1 && trajectory_manager.has_archive(),
                  ExcInternalError());
      // Materialize one interval [anchor,end] on demand. The cached window also
      // includes anchor-1 and anchor-2, because the first replayed accepted
      // step and a BDF2 reverse operator both require this preceding history.
      const auto interval = trajectory_manager.sparse_interval_for_record(record);
      const unsigned int anchor = interval.first;
      const unsigned int end = interval.second;
      for (const ForwardReplayWindow &window : trajectory_manager.forward_replay_windows)
        if (record >= window.anchor && record <= window.end)
          return;

      const auto load_payload = [this](const unsigned int state_index)
      {
        // TrajectoryStore loads owned entries. Promote them to a ghosted FE
        // read view, and in minimal mode deterministically recover the omitted
        // quasi-static velocity/pressure before replay uses the state.
        LinearAlgebra::BlockVector owned;
        owned.reinit(this->introspection().index_sets.system_partitioning,
                     this->get_mpi_communicator());
        trajectory_manager.archive().load(state_index, owned,
                                          this->introspection().block_indices.temperature,
                                          this->introspection().block_indices.compositional_fields,
                                          this->introspection().block_indices.velocities,
                                          this->introspection().block_indices.pressure);
        LinearAlgebra::BlockVector result;
        result.reinit(this->introspection().index_sets.system_partitioning,
                      this->introspection().index_sets.system_relevant_partitioning,
                      this->get_mpi_communicator());
        result = owned;
        if (trajectory_manager.archive().mode() == TrajectoryStorageMode::minimal)
          recover_stokes_state(state_index, result);
        return result;
      };

      ForwardReplayWindow window;
      window.anchor = anchor;
      window.end = end;
      const unsigned int history_begin = anchor > 1 ? anchor - 2 : 0;
      for (unsigned int state_index = history_begin; state_index <= anchor; ++state_index)
        {
          AssertThrow(trajectory_manager.archive().has_record(state_index),
                      ExcMessage("Sparse anchor package is missing required BDF history."));
          window.states.push_back({state_index, load_payload(state_index)});
        }
      const auto state_at = [&window](const unsigned int state_index)
                            -> const LinearAlgebra::BlockVector &
      {
        for (const ReconstructedTrajectoryState &state : window.states)
          if (state.record == state_index)
            return state.state;
        AssertThrow(false, ExcInternalError());
        return window.states.front().state;
      };

      const ReplayStepMetadata &anchor_metadata = trajectory_manager.archive().metadata(anchor);
      // Hydrate Simulator's three forward history slots at the anchor exactly
      // as they appeared during the original run. Runtime matrices are marked
      // stale because they are not archived and belong to another time level.
      simulator->solution = state_at(anchor);
      simulator->old_solution = state_at(anchor > 0 ? anchor - 1 : anchor);
      simulator->old_old_solution = state_at(anchor > 1 ? anchor - 2 : anchor);
      simulator->current_linearization_point = simulator->solution;
      simulator->time = anchor_metadata.time;
      simulator->time_step = anchor_metadata.time_step;
      simulator->old_time_step = anchor_metadata.old_time_step;
      simulator->timestep_number = anchor_metadata.step;
      simulator->last_pressure_normalization_adjustment = 0.0;
      simulator->rebuild_stokes_matrix = true;
      simulator->rebuild_stokes_preconditioner = true;
      ++trajectory_manager.sparse_anchor_load_count;
      ++trajectory_manager.sparse_interval_replay_count;
      report_mpi_adjoint_memory_checkpoint(
        "sparse replay: anchor state hydrated before accepted-step replay",
        &simulator->solution);

      for (unsigned int transition = anchor; transition < end; ++transition)
        {
          // replay_accepted_timestep follows the recorded accepted dt rather
          // than asking the time-step manager to choose a new one. Its output
          // state transition+1 is retained in memory and optionally compared
          // with the diagnostic dense archive.
          report_mpi_adjoint_memory_checkpoint(
            "sparse replay: before accepted-step replay "
            + Utilities::int_to_string(transition), &simulator->solution);
          simulator->replay_accepted_timestep(trajectory_manager.archive().metadata(transition));
          report_mpi_adjoint_memory_checkpoint(
            "sparse replay: after accepted-step replay "
            + Utilities::int_to_string(transition), &simulator->solution);
          window.states.push_back({transition + 1, simulator->solution});
          ++trajectory_manager.sparse_replay_transition_count;
          report_mpi_adjoint_memory_checkpoint(
            "sparse replay: before dense-reference comparison "
            + Utilities::int_to_string(transition + 1), &window.states.back().state);
          compare_sparse_replay_state(transition + 1, window.states.back().state);
          report_mpi_adjoint_memory_checkpoint(
            "sparse replay: after dense-reference comparison "
            + Utilities::int_to_string(transition + 1), &window.states.back().state);
        }
      trajectory_manager.sparse_peak_window_state_count = std::max(
                                                            trajectory_manager.sparse_peak_window_state_count,
                                                            static_cast<unsigned int>(window.states.size()));
      this->get_pcout() << "ASPECT_ADJOINT_SPARSE_REPLAY anchor=" << anchor
                        << " end=" << end
                        << " anchor_loads=" << trajectory_manager.sparse_anchor_load_count
                        << " replayed_transitions=" << (end - anchor)
                        << " max_state_relative_difference="
                        << trajectory_manager.sparse_max_state_relative_difference
                        << " max_velocity_relative_difference="
                        << trajectory_manager.sparse_max_velocity_relative_difference
                        << " max_gauge_pressure_relative_difference="
                        << trajectory_manager.sparse_max_pressure_relative_difference << std::endl;
      trajectory_manager.forward_replay_windows.emplace_back(std::move(window));
      // Reverse traversal can straddle one interval boundary and request
      // history from the previous window. Two windows are therefore sufficient
      // and bound memory independently of total trajectory length.
      while (trajectory_manager.forward_replay_windows.size() > 2)
        trajectory_manager.forward_replay_windows.pop_front();
    }



    template <int dim>
    const LinearAlgebra::BlockVector &
    Manager<dim>::load_reconstructed_state(const unsigned int record)
    {
      AssertThrow(trajectory_manager.has_archive(), ExcInternalError());
      if (trajectory_manager.configuration().anchor_spacing > 1)
        {
          // Sparse mode delegates ownership to replay windows; references stay
          // valid while the deque retains the containing window. The caller
          // obtains all three snapshot references before later traversal can
          // evict an older window.
          for (const ForwardReplayWindow &window : trajectory_manager.forward_replay_windows)
            for (const ReconstructedTrajectoryState &state : window.states)
              if (state.record == record)
                return state.state;
          materialize_forward_replay_window(record);
          for (const ForwardReplayWindow &window : trajectory_manager.forward_replay_windows)
            for (const ReconstructedTrajectoryState &state : window.states)
              if (state.record == record)
                return state.state;
          AssertThrow(false, ExcMessage("Sparse replay window did not produce the requested state."));
        }
      // Dense archives still cache the last three loaded states to avoid
      // repeated MPI-IO for q_n,q_{n-1},q_{n-2} across adjacent reverse steps.
      for (const auto &cached : trajectory_manager.reverse_state_cache)
        if (cached.record == record)
          {
            report_mpi_adjoint_memory_checkpoint(
              "replay record " + Utilities::int_to_string(record) + " cache hit",
              &cached.state);
            return cached.state;
          }

      ReconstructedTrajectoryState reconstructed;
      reconstructed.record = record;
      report_mpi_adjoint_memory_checkpoint(
        "before replay record " + Utilities::int_to_string(record) + " materialization");
      LinearAlgebra::BlockVector locally_owned_state;
      locally_owned_state.reinit(this->introspection().index_sets.system_partitioning,
                                 this->get_mpi_communicator());
      locally_owned_state = 0.0;
      report_mpi_adjoint_memory_checkpoint(
        "after replay record " + Utilities::int_to_string(record)
        + " owned-vector reinit", &locally_owned_state);

      {
        TimerOutput::Scope read_timer(this->get_computing_timer(),
                                      "Adjoint trajectory read");
        trajectory_manager.archive().load(record,
                                          locally_owned_state,
                                          this->introspection().block_indices.temperature,
                                          this->introspection().block_indices.compositional_fields,
                                          this->introspection().block_indices.velocities,
                                          this->introspection().block_indices.pressure);
      }
      report_mpi_adjoint_memory_checkpoint(
        "after replay record " + Utilities::int_to_string(record)
        + " owned payload load", &locally_owned_state);

      reconstructed.state.reinit(this->introspection().index_sets.system_partitioning,
                                 this->introspection().index_sets.system_relevant_partitioning,
                                 this->get_mpi_communicator());
      reconstructed.state = locally_owned_state;
      report_mpi_adjoint_memory_checkpoint(
        "after replay record " + Utilities::int_to_string(record)
        + " ghost-view materialization", &reconstructed.state);

      if (trajectory_manager.archive().mode() == TrajectoryStorageMode::minimal)
        recover_stokes_state(record, reconstructed.state);

      // The reverse operator needs at most q_n, q_(n-1), and q_(n-2). Keep
      // exactly those recently used reconstructed states; the trajectory
      // itself remains disk-backed and never embeds this history.
      if (trajectory_manager.reverse_state_cache.size() == 3)
        trajectory_manager.reverse_state_cache.erase(trajectory_manager.reverse_state_cache.begin());
      trajectory_manager.reverse_state_cache.emplace_back(std::move(reconstructed));
      report_mpi_adjoint_memory_checkpoint(
        "after replay record " + Utilities::int_to_string(record)
        + " reverse-cache insertion",
        &trajectory_manager.reverse_state_cache.back().state);
      return trajectory_manager.reverse_state_cache.back().state;
    }



    template <int dim>
    ForwardStateView<dim>
    Manager<dim>::reconstruct_forward_snapshot(const unsigned int record)
    {
      AssertThrow(trajectory_manager.has_archive(), ExcInternalError());
      AssertIndexRange(record, trajectory_manager.archive().size());

      TimerOutput::Scope reconstruction_timer(this->get_computing_timer(),
                                              "Adjoint trajectory state reconstruction");
      report_mpi_adjoint_memory_checkpoint(
        "before forward snapshot reconstruction at record "
        + Utilities::int_to_string(record));
      // ForwardStateView is a non-owning description of one discrete time
      // level. At the first two records, unavailable histories collapse onto
      // the oldest physical state exactly as the forward BDF startup did.
      const LinearAlgebra::BlockVector &solution = load_reconstructed_state(record);
      const LinearAlgebra::BlockVector &old_solution =
        load_reconstructed_state(record > 0 ? record - 1 : record);
      const LinearAlgebra::BlockVector &old_old_solution =
        load_reconstructed_state(record > 1 ? record - 2 : record);
      const ReplayStepMetadata &metadata = trajectory_manager.archive().metadata(record);
      ForwardStateView<dim> snapshot(solution,
                                     old_solution,
                                     old_old_solution,
                                     record > 0 ? old_solution : solution,
                                     solution,
                                     metadata.time,
                                     metadata.time_step,
                                     metadata.old_time_step,
                                     metadata.step);
      if (trajectory_manager.configuration().anchor_spacing > 1 && record > 1
          && record % trajectory_manager.configuration().anchor_spacing == 1)
        this->get_pcout() << "ASPECT_ADJOINT_SPARSE_BOUNDARY snapshot=" << record
                          << " history=" << record - 1 << ',' << record - 2
                          << " provider=forward_replay_window" << std::endl;
      report_mpi_adjoint_memory_checkpoint(
        "after replay records for forward snapshot " + Utilities::int_to_string(record),
        &snapshot.solution.get());

      // The advection operator was assembled from the prior physical state.
      // For the supported Picard Stokes model the Stokes material point is
      // determined by the current T,C state, so no separately archived
      // linearization vector is required.
      // Rebuild the primal graph from the fixed mesh, the recorded time, and
      // configured boundary managers. The temporary snapshot keeps a copy
      // only while a legacy adjoint helper needs to homogenize it; it is never
      // an archived constraint record.
      simulator->solution = snapshot.solution.get();
      simulator->old_solution = snapshot.old_solution.get();
      simulator->old_old_solution = snapshot.old_old_solution.get();
      simulator->current_linearization_point = snapshot.advection_linearization_point.get();
      simulator->time = snapshot.time;
      simulator->time_step = snapshot.time_step;
      simulator->old_time_step = snapshot.old_time_step;
      simulator->timestep_number = snapshot.timestep_number;
      simulator->last_pressure_normalization_adjustment = 0.0;
      report_mpi_adjoint_memory_checkpoint(
        "before compute_current_constraints for replay record "
        + Utilities::int_to_string(record),
        &snapshot.solution.get());
      simulator->compute_current_constraints();
      // Boundary constraints can depend on recorded time. Recompute their
      // primal graph and copy it into the snapshot so all subsequent transpose
      // scatters act on the same constrained discrete space as the forward
      // accepted step.
      snapshot.constraints.reinit(simulator->current_constraints.get_local_lines(),
                                  this->introspection().index_sets.system_relevant_set);
      snapshot.constraints.copy_from(simulator->current_constraints);
      report_mpi_adjoint_memory_checkpoint(
        "after compute_current_constraints for replay record "
        + Utilities::int_to_string(record),
        &snapshot.solution.get());

      return snapshot;
    }



    template <int dim>
    void
    Manager<dim>::report_trajectory_statistics(const std::string &stage) const
    {
      if (!trajectory_manager.has_archive())
        return;

      std::size_t local_cache_memory = 0;
      for (const auto &cached : trajectory_manager.reverse_state_cache)
        local_cache_memory += cached.state.memory_consumption();
      const std::size_t local_current_record_memory =
        (trajectory_manager.reverse_state_cache.empty() ? 0
         : trajectory_manager.reverse_state_cache.back().state.memory_consumption());

      diagnostics_manager.report_trajectory(
        trajectory_manager.archive(),
        stage,
        local_cache_memory,
        local_current_record_memory,
        trajectory_manager.stokes_recovery_solves,
        trajectory_manager.sparse_peak_window_state_count,
        trajectory_manager.sparse_replay_transition_count,
        trajectory_manager.sparse_restart_replay_transition_count);
    }



    template <int dim>
    void
    Manager<dim>::report_mpi_adjoint_memory_checkpoint(
      const std::string &stage,
      const LinearAlgebra::BlockVector *state) const
    {
      const LinearAlgebra::BlockVector &reported_state =
        (state != nullptr ? *state : simulator->solution);
      diagnostics_manager.report_memory_checkpoint(stage,
                                                   reported_state,
                                                   trajectory_manager.reverse_state_cache.size());
    }

  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::Manager<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
