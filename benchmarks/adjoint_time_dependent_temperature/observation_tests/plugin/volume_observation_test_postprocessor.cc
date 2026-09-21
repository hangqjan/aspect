/* Benchmark-local diagnostic plugin for terminal volume observations. */
#include <aspect/adjoint/core/manager.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/exceptions.h>

#include <cstdlib>
#include <exception>
#include <iostream>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VolumeObservationTestPostprocessor
      : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Volume observation test");
          prm.declare_entry("Mode", "none", Patterns::Selection("none|in memory|invalid layout"),
                            "Install a copied, perturbed terminal observation.");
          prm.declare_entry("Target kind", "copy", Patterns::Selection("copy|constant"),
                            "Use the forward field or a constant selected-field target.");
          prm.declare_entry("Field", "temperature", Patterns::Anything(),
                            "temperature or the name of one compositional field.");
          prm.declare_entry("Offset", "0", Patterns::Double(),
                            "Constant added to the copied selected field.");
          prm.declare_entry("Suppress exception stack trace", "false", Patterns::Bool(),
                            "Benchmark workaround for a platform stack-trace failure.");
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Volume observation test");
          mode = prm.get("Mode");
          target_kind = prm.get("Target kind");
          field_name = prm.get("Field");
          offset = prm.get_double("Offset");
          suppress_exception_stack_trace = prm.get_bool("Suppress exception stack trace");
          prm.leave_subsection();
          prm.leave_subsection();

          // The malformed-file case must exercise the production
          // AsciiDataInitial parser. This local switch only suppresses the
          // deal.II stack trace on macOS builds where rendering that trace
          // loops indefinitely; it does not suppress or alter the exception.
          if (suppress_exception_stack_trace)
            dealii::deal_II_exceptions::suppress_stacktrace_in_exceptions();
        }

        std::pair<std::string, std::string> execute(TableHandler &) override
        {
          if (mode == "none")
            return {};

          const AdvectionField field = field_name == "temperature"
                                       ? AdvectionField::temperature()
                                       : AdvectionField::composition(
                                         this->introspection().compositional_index_for_name(field_name));
          if (mode == "invalid layout")
            {
              LinearAlgebra::BlockVector invalid;
              install_or_report_expected_failure(
                field, invalid,
                "The terminal observation is incompatible with the current DoF layout");
              return {};
            }
          LinearAlgebra::BlockVector observation(
            this->introspection().index_sets.system_partitioning,
            this->get_mpi_communicator());
          observation = this->get_solution();
          auto &block = observation.block(field.block_index(this->introspection()));
          for (const auto index : block.locally_owned_elements())
            block(index) = (target_kind == "constant" ? 0. : block(index)) + offset;
          observation.compress(VectorOperation::insert);

          // The manager owns objective selection, copying, ghost updates and
          // layout validation. Repeating this at every accepted postprocess
          // leaves the final accepted state installed for reverse finalization.
          install_or_report_expected_failure(
            field, observation,
            "A compatible terminal scalar-field objective is required for a synthetic observation");
          return {"Volume observation test:", "installed in-memory terminal target"};
        }

      private:
        void install_or_report_expected_failure(
          const AdvectionField &field,
          const LinearAlgebra::BlockVector &observation,
          const std::string &expected_error) const
        {
          try
            {
              const_cast<Simulator<dim> &>(this->get_simulator())
              .get_adjoint_manager().set_terminal_observation(field, observation);
            }
          catch (const std::exception &)
            {
              // ASPECT's generic postprocessor exception path is collective.
              // This test plugin deliberately turns an expected local API
              // rejection into a deterministic nonzero test-process exit.
              // Do not enter static/MPI teardown after a deliberately
              // rejected observation: ASPECT's postprocessor error path is
              // collective and otherwise leaves this test process alive.
              // The stable context identifies the production check exercised
              // without asking deal.II to lazily materialize the nested
              // exception's stack trace (which is not safe on this error
              // path in the test runtime).
              std::cerr << "ASPECT_OBSERVATION_EXPECTED_FAILURE: "
                        << expected_error << std::endl;
              std::cerr.flush();
              std::_Exit(EXIT_FAILURE);
            }
        }

        std::string mode = "none";
        std::string target_kind = "copy";
        std::string field_name = "temperature";
        double offset = 0.;
        bool suppress_exception_stack_trace = false;
    };

    ASPECT_REGISTER_POSTPROCESSOR(VolumeObservationTestPostprocessor,
                                  "volume observation test",
                                  "Benchmark-only installer for a discrete terminal volume observation.")
  }
}
