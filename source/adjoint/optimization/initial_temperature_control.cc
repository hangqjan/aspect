/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include <aspect/adjoint/optimization/initial_temperature_control.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/solver_cg.h>

#include <map>

namespace aspect
{
  namespace Adjoint
  {
    namespace
    {
      void finish_insert(LinearAlgebra::Vector &vector)
      {
        vector.compress(dealii::VectorOperation::insert);
      }
    }

    template <int dim>
    InitialTemperatureControl<dim>::InitialTemperatureControl(
      const dealii::DoFHandler<dim> &dof_handler,
      const dealii::ComponentMask &temperature_mask,
      const dealii::AffineConstraints<double> &constraints,
      const MPI_Comm communicator)
      : InitialTemperatureControl(dof_handler, temperature_mask, constraints,
                                  communicator, 0, dealii::IndexSet(), dealii::IndexSet())
    {}

    template <int dim>
    InitialTemperatureControl<dim>::InitialTemperatureControl(
      const dealii::DoFHandler<dim> &dof_handler,
      const dealii::ComponentMask &temperature_mask,
      const dealii::AffineConstraints<double> &constraints,
      const MPI_Comm communicator,
      const types::global_dof_index temperature_block_first_dof,
      const dealii::IndexSet &temperature_block_owned,
      const dealii::IndexSet &temperature_block_relevant)
      : dof_handler(dof_handler)
      , temperature_mask(temperature_mask)
      , constraints(constraints)
      , communicator(communicator)
      , temperature_block_first_dof(temperature_block_first_dof)
      , locally_owned_temperature_dofs(dof_handler.n_dofs())
      , locally_relevant_temperature_dofs(dof_handler.n_dofs())
      , locally_owned_temperature_block_dofs(temperature_block_owned)
      , locally_relevant_temperature_block_dofs(temperature_block_relevant)
      , locally_owned_controls(0)
      , locally_relevant_controls(0)
      , homogeneous_constraints(temperature_block_owned.size() == 0 ? dof_handler.locally_owned_dofs() : temperature_block_owned,
                                temperature_block_relevant.size() == 0 ? dealii::DoFTools::extract_locally_relevant_dofs(dof_handler) : temperature_block_relevant)
      , physical_constraints(temperature_block_owned.size() == 0 ? dof_handler.locally_owned_dofs() : temperature_block_owned,
                             temperature_block_relevant.size() == 0 ? dealii::DoFTools::extract_locally_relevant_dofs(dof_handler) : temperature_block_relevant)
    {
      if (locally_owned_temperature_block_dofs.size() == 0)
        locally_owned_temperature_block_dofs = dof_handler.locally_owned_dofs();
      if (locally_relevant_temperature_block_dofs.size() == 0)
        locally_relevant_temperature_block_dofs = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
      // DoFTools::extract_dofs() deliberately returns only locally owned
      // DoFs. Local mass-matrix assembly also visits hanging-node masters on
      // ghost cells, however, so a purely owned map would make P incomplete
      // on partition boundaries. Build the temperature subset from both
      // locally owned and ghost cells. This gives every locally assembled
      // temperature row access to its complete reduced-control expansion.
      locally_relevant_temperature_dofs.clear();
      std::vector<types::global_dof_index> temperature_dofs(
        dof_handler.get_fe().dofs_per_cell);
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() || cell->is_ghost())
          {
            cell->get_dof_indices(temperature_dofs);
            for (unsigned int i=0; i<temperature_dofs.size(); ++i)
              if (temperature_mask[dof_handler.get_fe().system_to_component_index(i).first])
                locally_relevant_temperature_dofs.add_index(temperature_dofs[i]);
          }
      locally_relevant_temperature_dofs.compress();
      locally_owned_temperature_dofs =
        locally_relevant_temperature_dofs & dof_handler.locally_owned_dofs();
      locally_owned_temperature_dofs.compress();
      const types::global_dof_index local_size =
        std::count_if(locally_owned_temperature_dofs.begin(),
                      locally_owned_temperature_dofs.end(),
                      [this](const auto index)
      {
        return !this->constraints.is_constrained(index);
      });
      // Each unconstrained temperature DoF owns one reduced coefficient. An
      // exclusive scan assigns globally contiguous reduced indices without a
      // root-side gather: rank r owns [first, first+local_size). Dirichlet
      // DoFs intentionally have no control coordinate because their values
      // are imposed by the physical boundary condition, not optimization.
      const unsigned long long local_size_mpi =
        static_cast<unsigned long long>(local_size);
      unsigned long long first_mpi = 0;
      unsigned long long global_size_mpi = 0;
      MPI_Exscan(&local_size_mpi, &first_mpi, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
      MPI_Allreduce(&local_size_mpi, &global_size_mpi, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                    communicator);
      if (dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
        first_mpi = 0;
      const types::global_dof_index first =
        static_cast<types::global_dof_index>(first_mpi);
      const types::global_dof_index global_size =
        static_cast<types::global_dof_index>(global_size_mpi);
      locally_owned_controls = dealii::IndexSet(global_size);
      locally_owned_controls.add_range(first, first + local_size);
      locally_owned_controls.compress();

      types::global_dof_index next = first;
      for (const auto index : locally_owned_temperature_dofs)
        if (!constraints.is_constrained(index))
          full_to_reduced.emplace(index, next++);

      // Free masters needed by a local constraint row may be owned by another
      // rank. Exchange the full-DoF -> reduced-DoF mapping once during setup;
      // subsequent expansion and mass assembly are then purely local lookups.
      const auto owners = dealii::Utilities::MPI::compute_index_owner(
                            dof_handler.locally_owned_dofs(),
                            locally_relevant_temperature_dofs,
                            communicator);
      AssertDimension(owners.size(), locally_relevant_temperature_dofs.n_elements());

      const unsigned int my_rank =
        dealii::Utilities::MPI::this_mpi_process(communicator);
      std::map<unsigned int, std::vector<types::global_dof_index>> requested_dofs;
      auto owner = owners.begin();
      for (const auto index : locally_relevant_temperature_dofs)
        {
          if (*owner != my_rank)
            requested_dofs[*owner].push_back(index);
          ++owner;
        }

      const auto received_requests =
        dealii::Utilities::MPI::some_to_some(communicator, requested_dofs);
      std::map<unsigned int, std::vector<types::global_dof_index>> responses;
      for (const auto &request : received_requests)
        {
          auto &response = responses[request.first];
          response.reserve(request.second.size());
          for (const auto index : request.second)
            {
              const auto entry = full_to_reduced.find(index);
              response.push_back(entry == full_to_reduced.end() ?
                                 dealii::numbers::invalid_dof_index : entry->second);
            }
        }

      const auto received_responses =
        dealii::Utilities::MPI::some_to_some(communicator, responses);
      for (const auto &request : requested_dofs)
        {
          const auto response = received_responses.find(request.first);
          AssertThrow(response != received_responses.end()
                      && response->second.size() == request.second.size(),
                      ExcMessage("The reduced temperature control map exchange failed."));
          for (unsigned int i=0; i<request.second.size(); ++i)
            if (response->second[i] != dealii::numbers::invalid_dof_index)
              full_to_reduced.emplace(request.second[i], response->second[i]);
        }

      locally_relevant_controls = dealii::IndexSet(global_size);
      // P maps reduced free coefficients c to a homogeneous conforming
      // temperature perturbation delta_T=P c. A constrained row can reference
      // remote free masters, so this relevant set is the union of all reduced
      // column indices appearing in locally needed rows of P.
      for (const auto index : locally_relevant_temperature_dofs)
        for (const auto &entry : expand(index))
          locally_relevant_controls.add_index(entry.first);
      locally_relevant_controls.compress();
      build_homogeneous_constraints();
    }

    template <int dim>
    const dealii::IndexSet &InitialTemperatureControl<dim>::locally_owned_control_dofs() const
    {
      return locally_owned_controls;
    }

    template <int dim>
    const dealii::IndexSet &InitialTemperatureControl<dim>::locally_relevant_control_dofs() const
    {
      return locally_relevant_controls;
    }

    template <int dim>
    const dealii::IndexSet &InitialTemperatureControl<dim>::temperature_block_owned_dofs() const
    {
      return locally_owned_temperature_block_dofs;
    }

    template <int dim>
    const dealii::IndexSet &InitialTemperatureControl<dim>::temperature_block_relevant_dofs() const
    {
      return locally_relevant_temperature_block_dofs;
    }

    template <int dim>
    void InitialTemperatureControl<dim>::assert_temperature_block_mapping() const
    {
      unsigned long long local_count = 0;
      for (const auto dof : locally_owned_temperature_dofs)
        {
          const auto block_index = temperature_block_index(dof);
          AssertThrow(locally_owned_temperature_block_dofs.is_element(block_index),
                      ExcMessage("A locally owned mixed temperature DoF does not map "
                                 "to a locally owned temperature-block entry."));
          ++local_count;
        }
      const unsigned long long global_count =
        dealii::Utilities::MPI::sum(local_count, communicator);
      AssertThrow(global_count == locally_owned_temperature_block_dofs.size(),
                  ExcMessage("The temperature-block mapping does not cover exactly "
                             "the global temperature DoFs."));
    }

    template <int dim>
    types::global_dof_index InitialTemperatureControl<dim>::n_control_dofs() const
    {
      return locally_owned_controls.size();
    }

    template <int dim>
    void InitialTemperatureControl<dim>::assert_control_layout(const LinearAlgebra::Vector &control) const
    {
      AssertThrow(control.size() == n_control_dofs()
                  && control.locally_owned_elements() == locally_owned_controls,
                  ExcMessage("Initial-temperature control has an incompatible reduced layout."));
    }

    template <int dim>
    void InitialTemperatureControl<dim>::assert_temperature_layout(
      const LinearAlgebra::Vector &temperature) const
    {
      AssertThrow(!temperature.has_ghost_elements()
                  && temperature.size() == locally_owned_temperature_block_dofs.size()
                  && temperature.locally_owned_elements() == locally_owned_temperature_block_dofs,
                  ExcMessage("Initial-temperature injection requires the owned-only "
                             "temperature-block vector layout."));
    }

    template <int dim>
    types::global_dof_index
    InitialTemperatureControl<dim>::temperature_block_index(const types::global_dof_index dof) const
    {
      AssertThrow(dof >= temperature_block_first_dof,
                  ExcMessage("A temperature DoF precedes its component-wise block."));
      const types::global_dof_index result = dof - temperature_block_first_dof;
      AssertThrow(result < locally_owned_temperature_block_dofs.size(),
                  ExcMessage("A temperature DoF maps outside the temperature block."));
      return result;
    }

    template <int dim>
    typename InitialTemperatureControl<dim>::Expansion
    InitialTemperatureControl<dim>::expand(const types::global_dof_index dof) const
    {
      AssertThrow(locally_relevant_temperature_dofs.is_element(dof),
                  ExcMessage("A temperature control expansion was requested for a non-relevant DoF."));
      if (!constraints.is_constrained(dof))
        {
          const auto entry = full_to_reduced.find(dof);
          AssertThrow(entry != full_to_reduced.end(),
                      ExcMessage("A free temperature DoF has no reduced control index."));
          return {{entry->second, 1.0}};
        }

      Expansion result;
      const auto *entries = constraints.get_constraint_entries(dof);
      if (entries == nullptr)
        return result; // Dirichlet row: no homogeneous control dependence.
      // Recursively substitute constraint rows until only free reduced
      // coordinates remain. This constructs one sparse row of P; multiplying
      // weights along nested edges is essential when a master is constrained.
      for (const auto &entry : *entries)
        for (const auto &master : expand(entry.first))
          result.emplace_back(master.first, entry.second * master.second);
      return result;
    }

    template <int dim>
    typename InitialTemperatureControl<dim>::Expansion
    InitialTemperatureControl<dim>::expansion(const types::global_dof_index dof) const
    {
      return expand(dof);
    }

    template <int dim>
    void InitialTemperatureControl<dim>::build_homogeneous_constraints()
    {
      // Maintain two block-indexed graphs. The homogeneous graph describes
      // admissible perturbations delta_T and hence P. The physical graph has
      // the same linear entries plus original inhomogeneities and reconstructs
      // a conforming physical initial temperature.
      for (const auto index : locally_relevant_temperature_dofs)
        if (constraints.is_constrained(index)
            && homogeneous_constraints.can_store_line(temperature_block_index(index)))
          {
            const auto mapped_index = temperature_block_index(index);
            homogeneous_constraints.add_line(mapped_index);
            physical_constraints.add_line(mapped_index);
            physical_constraints.set_inhomogeneity(mapped_index,
                                                   constraints.get_inhomogeneity(index));
            if (const auto *entries = constraints.get_constraint_entries(index))
              for (const auto &entry : *entries)
                {
                  const auto mapped_entry = temperature_block_index(entry.first);
                  homogeneous_constraints.add_entry(mapped_index, mapped_entry, entry.second);
                  physical_constraints.add_entry(mapped_index, mapped_entry, entry.second);
                }
          }
      homogeneous_constraints.close();
      physical_constraints.close();
    }

    template <int dim>
    void InitialTemperatureControl<dim>::inject_perturbation(
      const LinearAlgebra::Vector &control, LinearAlgebra::Vector &temperature) const
    {
      assert_control_layout(control);
      assert_temperature_layout(temperature);

      // AffineConstraints::distribute() creates a temporary Trilinos vector
      // with its import IndexSet as an ownership map. In deal.II 9.6 that is
      // unsafe when a hanging-node row needs a remote master: the temporary
      // map overlaps between ranks. Import just the needed reduced control
      // coefficients with the same explicit point-to-point exchange used to
      // build full_to_reduced, then evaluate the already-recursive P rows.
      // This is deliberately an owned-only output operation: the caller may
      // safely assign the result into ASPECT's ghosted solution block after
      // this routine returns, but never writes through a ghost ownership map.
      dealii::IndexSet remotely_needed_controls(n_control_dofs());
      for (const auto index : locally_owned_temperature_dofs)
        for (const auto &entry : expand(index))
          if (!locally_owned_controls.is_element(entry.first))
            remotely_needed_controls.add_index(entry.first);
      remotely_needed_controls.compress();
      const auto control_owners = dealii::Utilities::MPI::compute_index_owner(
                                    locally_owned_controls, remotely_needed_controls, communicator);
      AssertDimension(control_owners.size(), remotely_needed_controls.n_elements());
      const unsigned int my_rank =
        dealii::Utilities::MPI::this_mpi_process(communicator);
      std::map<unsigned int, std::vector<types::global_dof_index>> requests;
      auto owner = control_owners.begin();
      for (const auto index : remotely_needed_controls)
        {
          AssertThrow(*owner != my_rank,
                      ExcInternalError());
          requests[*owner].push_back(index);
          ++owner;
        }
      const auto received_requests =
        dealii::Utilities::MPI::some_to_some(communicator, requests);
      std::map<unsigned int, std::vector<double>> responses;
      for (const auto &request : received_requests)
        {
          auto &response = responses[request.first];
          response.reserve(request.second.size());
          for (const auto index : request.second)
            {
              AssertThrow(locally_owned_controls.is_element(index),
                          ExcInternalError());
              response.push_back(control[index]);
            }
        }
      const auto received_responses =
        dealii::Utilities::MPI::some_to_some(communicator, responses);
      std::map<types::global_dof_index, double> imported_control_values;
      for (const auto &request : requests)
        {
          const auto response = received_responses.find(request.first);
          AssertThrow(response != received_responses.end()
                      && response->second.size() == request.second.size(),
                      ExcMessage("The reduced temperature control value exchange failed."));
          for (unsigned int i = 0; i < request.second.size(); ++i)
            imported_control_values.emplace(request.second[i], response->second[i]);
        }

      // Evaluate delta_T_i=sum_j P_ij c_j for each owned temperature DoF. The
      // exchange above supplies exactly the remote c_j referenced by local
      // rows, avoiding an overlapping writable Trilinos vector.
      temperature = 0.0;
      for (const auto index : locally_owned_temperature_dofs)
        {
          double value = 0.0;
          for (const auto &entry : expand(index))
            if (locally_owned_controls.is_element(entry.first))
              value += entry.second * control[entry.first];
            else
              {
                const auto imported = imported_control_values.find(entry.first);
                AssertThrow(imported != imported_control_values.end(),
                            ExcInternalError());
                value += entry.second * imported->second;
              }
          temperature[temperature_block_index(index)] = value;
        }
      finish_insert(temperature);
    }

    template <int dim>
    void InitialTemperatureControl<dim>::inject_physical(
      const LinearAlgebra::Vector &control, const LinearAlgebra::Vector &fixed_temperature,
      LinearAlgebra::Vector &temperature) const
    {
      assert_control_layout(control);
      assert_temperature_layout(fixed_temperature);
      assert_temperature_layout(temperature);
      temperature = fixed_temperature;
      // T_fixed already satisfies the physical (possibly inhomogeneous)
      // constraints captured at the optimization baseline. Construct the
      // perturbation with the homogeneous P map and add it instead of
      // redistributing T_fixed through a second constraint object.
      LinearAlgebra::Vector perturbation(locally_owned_temperature_block_dofs,
                                         communicator);
      inject_perturbation(control, perturbation);
      temperature += perturbation;

      // The canonical ASPECT baseline is normally already conforming, but a
      // physical injection also has to be correct for an arbitrary owned
      // T_fixed. Reapply inhomogeneous rows without
      // AffineConstraints::distribute(): its temporary overlapping Trilinos
      // import map is not a writable owned-only map in this workflow.
      dealii::IndexSet remotely_needed_temperature_masters(dof_handler.n_dofs());
      for (const auto index : locally_owned_temperature_dofs)
        if (constraints.is_constrained(index))
          if (const auto *entries = constraints.get_constraint_entries(index))
            for (const auto &entry : *entries)
              if (!locally_owned_temperature_dofs.is_element(entry.first))
                remotely_needed_temperature_masters.add_index(entry.first);
      remotely_needed_temperature_masters.compress();
      const auto temperature_owners = dealii::Utilities::MPI::compute_index_owner(
                                        dof_handler.locally_owned_dofs(), remotely_needed_temperature_masters, communicator);
      AssertDimension(temperature_owners.size(),
                      remotely_needed_temperature_masters.n_elements());
      const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(communicator);
      std::map<unsigned int, std::vector<types::global_dof_index>> requests;
      auto owner = temperature_owners.begin();
      for (const auto index : remotely_needed_temperature_masters)
        {
          AssertThrow(*owner != my_rank, ExcInternalError());
          requests[*owner].push_back(index);
          ++owner;
        }
      const auto received_requests =
        dealii::Utilities::MPI::some_to_some(communicator, requests);
      std::map<unsigned int, std::vector<double>> responses;
      for (const auto &request : received_requests)
        {
          auto &response = responses[request.first];
          response.reserve(request.second.size());
          for (const auto index : request.second)
            {
              AssertThrow(locally_owned_temperature_dofs.is_element(index), ExcInternalError());
              response.push_back(temperature[temperature_block_index(index)]);
            }
        }
      const auto received_responses =
        dealii::Utilities::MPI::some_to_some(communicator, responses);
      std::map<types::global_dof_index, double> imported_temperature_values;
      for (const auto &request : requests)
        {
          const auto response = received_responses.find(request.first);
          AssertThrow(response != received_responses.end()
                      && response->second.size() == request.second.size(),
                      ExcMessage("The physical temperature value exchange failed."));
          for (unsigned int i = 0; i < request.second.size(); ++i)
            imported_temperature_values.emplace(request.second[i], response->second[i]);
        }
      for (const auto index : locally_owned_temperature_dofs)
        if (constraints.is_constrained(index))
          {
            double value = constraints.get_inhomogeneity(index);
            if (const auto *entries = constraints.get_constraint_entries(index))
              for (const auto &entry : *entries)
                if (locally_owned_temperature_dofs.is_element(entry.first))
                  value += entry.second * temperature[temperature_block_index(entry.first)];
                else
                  {
                    const auto imported = imported_temperature_values.find(entry.first);
                    AssertThrow(imported != imported_temperature_values.end(), ExcInternalError());
                    value += entry.second * imported->second;
                  }
            temperature[temperature_block_index(index)] = value;
          }
      finish_insert(temperature);
    }

    template <int dim>
    void InitialTemperatureControl<dim>::restrict_covector(
      const LinearAlgebra::Vector &raw_covector, LinearAlgebra::Vector &control_covector) const
    {
      assert_control_layout(control_covector);
      assert_temperature_layout(raw_covector);
      // Since delta_T=P delta_c, a full-space covector r_T satisfies
      //   delta_J=r_T^T P delta_c=(P^T r_T)^T delta_c.
      // Restriction is therefore P^T. compress(add) is required because a
      // reduced master may receive constrained rows owned by several ranks.
      control_covector = 0.0;
      for (const auto index : locally_owned_temperature_dofs)
        for (const auto &entry : expand(index))
          control_covector[entry.first] += raw_covector[temperature_block_index(index)] * entry.second;
      control_covector.compress(dealii::VectorOperation::add);
    }

    template <int dim>
    void InitialTemperatureControl<dim>::assemble_mass_matrix(
      const dealii::Mapping<dim> &mapping, const dealii::Quadrature<dim> &quadrature)
    {
      if (reduced_mass_matrix != nullptr)
        return;
      // The optimization metric is physical L2. With delta_T=P c and the full
      // temperature mass matrix M_T, the reduced Riesz map is
      //
      //                         M_c=P^T M_T P.
      //
      // Build its sparsity from reduced coordinates that co-occur in local
      // expansions of temperature basis functions.
      dealii::DynamicSparsityPattern dsp(locally_relevant_controls);
      const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
      std::vector<types::global_dof_index> dofs(dofs_per_cell);
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(dofs);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              if (temperature_mask[dof_handler.get_fe().system_to_component_index(i).first])
                for (const auto &row : expand(dofs[i]))
                  for (unsigned int j=0; j<dofs_per_cell; ++j)
                    if (temperature_mask[dof_handler.get_fe().system_to_component_index(j).first])
                      for (const auto &column : expand(dofs[j]))
                        dsp.add(row.first, column.first);
          }
      reduced_mass_matrix = std::make_shared<LinearAlgebra::SparseMatrix>();
      dealii::SparsityTools::distribute_sparsity_pattern(dsp,
                                                         locally_owned_controls,
                                                         communicator,
                                                         locally_relevant_controls);
      reduced_mass_matrix->reinit(locally_owned_controls, dsp, communicator);

      // Assemble the triple product at element level. Every full mass entry
      // integral phi_i phi_j is multiplied by one P coefficient from row i
      // and one from row j before being inserted into reduced coordinates.
      dealii::FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature,
                                      dealii::update_values | dealii::update_JxW_values);
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(dofs);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              if (temperature_mask[dof_handler.get_fe().system_to_component_index(i).first])
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  if (temperature_mask[dof_handler.get_fe().system_to_component_index(j).first])
                    {
                      double entry = 0.0;
                      for (unsigned int q=0; q<quadrature.size(); ++q)
                        entry += fe_values.shape_value(i,q) * fe_values.shape_value(j,q) * fe_values.JxW(q);
                      for (const auto &row : expand(dofs[i]))
                        for (const auto &column : expand(dofs[j]))
                          reduced_mass_matrix->add(row.first, column.first,
                                                   row.second * entry * column.second);
                    }
          }
      reduced_mass_matrix->compress(dealii::VectorOperation::add);
      mass_preconditioner = std::make_shared<LinearAlgebra::PreconditionJacobi>();
      typename LinearAlgebra::PreconditionJacobi::AdditionalData data;
      mass_preconditioner->initialize(*reduced_mass_matrix, data);
    }

    template <int dim>
    std::shared_ptr<const LinearAlgebra::SparseMatrix>
    InitialTemperatureControl<dim>::mass_matrix() const
    {
      AssertThrow(reduced_mass_matrix != nullptr, ExcMessage("The reduced temperature mass matrix is not assembled."));
      return reduced_mass_matrix;
    }

    template <int dim>
    void InitialTemperatureControl<dim>::solve_l2_riesz(
      const LinearAlgebra::Vector &rhs, LinearAlgebra::Vector &result,
      const double relative_tolerance) const
    {
      assert_control_layout(rhs);
      assert_control_layout(result);
      AssertThrow(reduced_mass_matrix != nullptr && mass_preconditioner != nullptr,
                  ExcMessage("The reduced temperature mass matrix is not assembled."));
      // Convert reduced covector r=P^T dJ/dT into its L2 gradient g through
      // M_c g=r, so dJ[c](s)=<g,s>_L2. M_c is SPD on independent free
      // controls, permitting Jacobi-preconditioned conjugate gradients.
      const double rhs_norm = rhs.l2_norm();
      result = 0.0;
      if (rhs_norm == 0.0)
        return;
      dealii::SolverControl control(5*n_control_dofs(), relative_tolerance * rhs_norm);
      dealii::SolverCG<LinearAlgebra::Vector> solver(control);
      solver.solve(*reduced_mass_matrix, result, rhs, *mass_preconditioner);
    }

    template class InitialTemperatureControl<2>;
    template class InitialTemperatureControl<3>;
  }
}
