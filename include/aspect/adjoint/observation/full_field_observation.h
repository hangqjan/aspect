/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.
*/

#ifndef _aspect_adjoint_observation_full_field_observation_h
#define _aspect_adjoint_observation_full_field_observation_h

#include <aspect/advection_field.h>
#include <aspect/linear_algebra_types.h>
#include <aspect/simulator_access.h>
#include <aspect/structured_data.h>
#include <aspect/utilities.h>

#include <deal.II/base/parsed_function.h>

#include <memory>

namespace aspect
{
  namespace Adjoint
  {
    namespace Observation
    {
      /**
       * Target data for a scalar full-field observation, d(x) for x in the
       * model volume. This class owns only the observation representation and
       * its data source. The objective selects the physical quantity and owns
       * the volume-L2 misfit and its adjoint seed.
       *
       * After initialize() the target is either evaluated analytically or is
       * a discrete vector in the current ASPECT finite-element space. File
       * input is materialized once, outside objective evaluation loops.
       */
      template <int dim>
      class FullFieldObservation : public SimulatorAccess<dim>
      {
        public:
          /** Backends currently supported for the full-field representation. */
          enum class Source
          {
            function,
            structured_file
          };

          FullFieldObservation();

          static void declare_parameters(ParameterHandler &prm);
          void parse_parameters(ParameterHandler &prm);

          /** Materialize a configured structured-file source in @p field. */
          void initialize(const AdvectionField &field);

          double value(const Point<dim> &position) const;

          bool is_discrete() const;
          const LinearAlgebra::BlockVector &discrete_state() const;

          /** Install an in-memory source in the current ASPECT layout. */
          void set_discrete_state(const LinearAlgebra::BlockVector &state);
          void clear_discrete_state();

        private:
          Source source;
          Functions::ParsedFunction<dim> target_function;
          Utilities::Coordinates::CoordinateSystem coordinate_system;
          std::unique_ptr<Utilities::AsciiDataInitial<dim>> structured_data;
          // A captured state takes precedence over a file-backed state, so a
          // synthetic twin can temporarily override a configured file without
          // discarding the materialized reference data.
          std::unique_ptr<LinearAlgebra::BlockVector> file_observation;
          std::unique_ptr<LinearAlgebra::BlockVector> in_memory_observation;

          /** Reject observations that became stale after a mesh/layout change. */
          void assert_current_layout(const LinearAlgebra::BlockVector &state) const;
      };
    }
  }
}

#endif
