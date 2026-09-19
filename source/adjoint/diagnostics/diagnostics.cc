/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/adjoint/diagnostics/diagnostics.h>

#include <aspect/adjoint/core/manager.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <iostream>
#include <sstream>

#if defined(__APPLE__)
#  include <mach/mach.h>
#endif

namespace aspect
{
  namespace Adjoint
  {
    namespace Diagnostics
    {
      template <int dim>
      InitialTemperatureOptimizationChecks<dim>::InitialTemperatureOptimizationChecks(
        ::aspect::Adjoint::Manager<dim> &adjoint_manager)
        :
        manager(adjoint_manager)
      {}



      template <int dim>
      void
      InitialTemperatureOptimizationChecks<dim>::reuse() const
      {
        manager.run_initial_temperature_control_reuse_diagnostic();
      }



      template <int dim>
      void
      InitialTemperatureOptimizationChecks<dim>::single_evaluation() const
      {
        manager.run_initial_temperature_control_single_evaluation_diagnostic();
      }



      template <int dim>
      void
      InitialTemperatureOptimizationChecks<dim>::taylor() const
      {
        manager.run_initial_temperature_control_taylor_diagnostic();
      }



      template <int dim>
      void
      InitialTemperatureOptimizationChecks<dim>::initialization_stage(
        const unsigned int stage) const
      {
        manager.run_initial_temperature_control_initialization_diagnostic(stage);
      }



      template <int dim>
      AdvectionOperatorChecks<dim>::AdvectionOperatorChecks(
        AdvectionAdjointCalculator<dim> &advection_calculator)
        :
        calculator(advection_calculator)
      {}



      template <int dim>
      void
      AdvectionOperatorChecks<dim>::velocity_transpose(
        const AdvectionField &advection_field,
        const ForwardStateView<dim> &snapshot,
        const LinearAlgebra::BlockVector &current_field_adjoint,
        const double epsilon) const
      {
        calculator.run_velocity_transpose_diagnostic(advection_field,
                                                     snapshot,
                                                     current_field_adjoint,
                                                     epsilon);
      }



      template <int dim>
      void
      AdvectionOperatorChecks<dim>::transposed_constraints(
        const AdvectionField &advection_field,
        const ForwardStateView<dim> &snapshot,
        const LinearAlgebra::BlockVector &current_field_adjoint) const
      {
        calculator.run_transposed_advection_constraint_diagnostic(advection_field,
                                                                  snapshot,
                                                                  current_field_adjoint);
      }



      template <int dim>
      void
      AdvectionOperatorChecks<dim>::artificial_viscosity_coefficient_seed(
        const AdvectionField &advection_field,
        const ForwardStateView<dim> &snapshot,
        const LinearAlgebra::BlockVector &current_field_adjoint,
        const double epsilon) const
      {
        calculator.run_artificial_viscosity_coefficient_seed_diagnostic(
          advection_field, snapshot, current_field_adjoint, epsilon);
      }



      template <int dim>
      void
      AdvectionOperatorChecks<dim>::artificial_viscosity_estimator_transpose(
        const AdvectionField &advection_field,
        const ForwardStateView<dim> &snapshot,
        const double epsilon) const
      {
        calculator.run_artificial_viscosity_estimator_transpose_diagnostic(
          advection_field, snapshot, epsilon);
      }



      template <int dim>
      void
      AdvectionOperatorChecks<dim>::local_entropy_residual_transpose(
        const AdvectionField &advection_field,
        const ForwardStateView<dim> &snapshot,
        const double epsilon) const
      {
        calculator.run_local_entropy_residual_transpose_diagnostic(advection_field,
                                                                   snapshot,
                                                                   epsilon);
      }



      template <int dim>
      void
      AdvectionOperatorChecks<dim>::global_entropy_denominator_scalar(
        const AdvectionField &advection_field,
        const ForwardStateView<dim> &snapshot,
        const double epsilon) const
      {
        calculator.run_global_entropy_denominator_scalar_diagnostic(advection_field,
                                                                    snapshot,
                                                                    epsilon);
      }



      template <int dim>
      void
      AdvectionOperatorChecks<dim>::global_entropy_variation_transpose(
        const AdvectionField &advection_field,
        const ForwardStateView<dim> &snapshot,
        const double epsilon) const
      {
        calculator.run_global_entropy_variation_transpose_diagnostic(advection_field,
                                                                     snapshot,
                                                                     epsilon);
      }



      template <int dim>
      void
      Manager<dim>::set_output_memory_checkpoints(const bool enabled)
      {
        output_memory_checkpoints = enabled;
      }



      template <int dim>
      void
      Manager<dim>::report_trajectory(
        const TrajectoryStore &trajectory,
        const std::string &stage,
        const std::size_t local_cache_memory,
        const std::size_t local_current_record_memory,
        const unsigned int stokes_recovery_solves,
        const unsigned int sparse_peak_window_state_count,
        const unsigned int sparse_replay_transition_count,
        const unsigned int sparse_restart_replay_transition_count) const
      {
        // Keep accounting layers distinct: TrajectoryStore reports local
        // durable payload bytes, while callers report transient reconstructed
        // vectors. Summing local payload categories measures aggregate MPI
        // storage; max(VmRSS/VmPeak) identifies the worst rank rather than
        // producing a physically meaningless sum of process high-water marks.
        const TrajectoryStorageStatistics local_storage =
          trajectory.storage_statistics();
        const auto sum_bytes = [this](const std::size_t value)
        {
          return Utilities::MPI::sum(static_cast<unsigned long long>(value),
                                     this->get_mpi_communicator());
        };

        // In minimal mode the currently reconstructed record contains u,p that
        // were absent on disk. Attribute its memory separately so diagnostics
        // expose the storage/recomputation tradeoff instead of hiding it in
        // the generic reverse cache total.
        const std::size_t local_recovered_stokes_memory =
          trajectory.mode() == TrajectoryStorageMode::minimal
          ? local_current_record_memory : 0;

        dealii::Utilities::System::MemoryStats process_memory {};
        dealii::Utilities::System::get_memory_stats(process_memory);
        const unsigned long int vm_peak =
          dealii::Utilities::MPI::max(process_memory.VmPeak,
                                      this->get_mpi_communicator());
        const unsigned long int vm_rss =
          dealii::Utilities::MPI::max(process_memory.VmRSS,
                                      this->get_mpi_communicator());

        const char *mode_name =
          trajectory.mode() == TrajectoryStorageMode::minimal
          ? "minimal (T,C)" : "stokes cached (T,C,u,p)";
        this->get_pcout() << "   Adjoint trajectory archive [" << stage << "]: mode = "
                          << mode_name
                          << ", records = " << trajectory.size()
                          << ", total bytes = " << sum_bytes(local_storage.total_bytes())
                          << ", mean bytes/record = "
                          << (trajectory.size() == 0 ? 0
                              : sum_bytes(local_storage.total_bytes()) / trajectory.size())
                          << " (metadata " << sum_bytes(local_storage.metadata_bytes)
                          << ", T " << sum_bytes(local_storage.temperature_bytes)
                          << ", C " << sum_bytes(local_storage.composition_bytes)
                          << ", u " << sum_bytes(local_storage.velocity_bytes)
                          << ", p " << sum_bytes(local_storage.pressure_bytes)
                          << ")." << std::endl;
        this->get_pcout() << "   Adjoint trajectory memory [" << stage
                          << "]: metadata object bytes = "
                          << sum_bytes(trajectory.memory_consumption())
                          << ", current loaded record bytes = "
                          << sum_bytes(local_current_record_memory)
                          << ", current recovered Stokes-state bytes = "
                          << sum_bytes(local_recovered_stokes_memory)
                          << ", reverse working-cache bytes = "
                          << sum_bytes(local_cache_memory)
                          << ", VmPeak = " << vm_peak << " kB, VmRSS = "
                          << vm_rss << " kB, Stokes recovery solves = "
                          << stokes_recovery_solves << "." << std::endl;
        this->get_pcout() << "ASPECT_ADJOINT_SPARSE_STORAGE stage=" << stage
                          << " dense_schedule_bytes=" << sum_bytes(local_storage.metadata_bytes)
                          << " anchor_payload_bytes="
                          << sum_bytes(local_storage.temperature_bytes
                                       + local_storage.composition_bytes
                                       + local_storage.velocity_bytes
                                       + local_storage.pressure_bytes)
                          << " shared_segment_count=" << trajectory.shared_segment_count()
                          << " archive_bytes=" << trajectory.shared_archive_bytes()
                          << " peak_replay_window_state_count="
                          << sparse_peak_window_state_count
                          << " total_replayed_transitions=" << sparse_replay_transition_count
                          << " restart_induced_extra_replay_transitions="
                          << sparse_restart_replay_transition_count
                          << std::endl;
      }



      template <int dim>
      void
      Manager<dim>::report_memory_checkpoint(
        const std::string &stage,
        const LinearAlgebra::BlockVector &state,
        const std::size_t reverse_cache_size) const
      {
        if (!output_memory_checkpoints)
          return;

        // Count ownership and relevance independently. Their difference is a
        // useful proxy for ghost/import pressure, while state.memory_consumption
        // would mix allocator overhead with the semantic vector layout.
        std::size_t locally_owned_dofs = 0;
        for (unsigned int block = 0; block < state.n_blocks(); ++block)
          locally_owned_dofs += state.block(block).locally_owned_elements().n_elements();

        std::size_t locally_relevant_dofs = 0;
        for (const IndexSet &index_set :
             this->introspection().index_sets.system_relevant_partitioning)
          locally_relevant_dofs += index_set.n_elements();

        // Normalize platform-specific process metrics to bytes. This method is
        // observational only and performs no distributed allocation, so it is
        // safe around the memory-sensitive sparse replay checkpoints.
        std::size_t rss_bytes = 0;
#if defined(__APPLE__)
        mach_task_basic_info_data_t task_info_data;
        mach_msg_type_number_t task_info_count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(),
                      MACH_TASK_BASIC_INFO,
                      reinterpret_cast<task_info_t>(&task_info_data),
                      &task_info_count) == KERN_SUCCESS)
          rss_bytes = task_info_data.resident_size;
#else
        dealii::Utilities::System::MemoryStats process_memory {};
        dealii::Utilities::System::get_memory_stats(process_memory);
        rss_bytes = process_memory.VmRSS * 1024ULL;
#endif

        // Emit one complete line per rank in a single write to reduce
        // interleaving in MPI stderr logs and make the record machine-parseable.
        std::ostringstream output;
        output << "ASPECT_ADJOINT_MPI_MEMORY rank="
               << Utilities::MPI::this_mpi_process(this->get_mpi_communicator())
               << " stage=\"" << stage << "\""
               << " rss_bytes=" << rss_bytes
               << " global_vector_size=" << state.size()
               << " locally_owned_dofs=" << locally_owned_dofs
               << " locally_relevant_dofs=" << locally_relevant_dofs
               << " reverse_cache_size=" << reverse_cache_size
               << '\n';
        std::cerr << output.str() << std::flush;
      }
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template class Adjoint::Diagnostics::Manager<dim>; \
  template class Adjoint::Diagnostics::AdvectionOperatorChecks<dim>; \
  template class Adjoint::Diagnostics::InitialTemperatureOptimizationChecks<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
