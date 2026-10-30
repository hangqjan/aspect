/* Benchmark-local reader probe for ASPECT-native geoid surface observations. */
#include <aspect/adjoint/observation/surface_field_observation.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/exceptions.h>

#include <cmath>
#include <iomanip>
#include <iostream>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class GeoidSurfaceObservationTestPostprocessor
      : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Geoid surface observation test");
          prm.declare_entry("Mode", "none", Patterns::Selection("none|geoid"),
                            "Evaluate one ASPECT-native geoid surface observation.");
          prm.declare_entry("Longitude", "0", Patterns::Double(),
                            "Longitude of the test query in degrees.");
          prm.declare_entry("Latitude", "0", Patterns::Double(-90., 90.),
                            "Latitude of the test query in degrees.");
          prm.declare_entry("Suppress exception stack trace", "false", Patterns::Bool(),
                            "Benchmark-only macOS workaround for deal.II stack-trace rendering.");
          prm.enter_subsection("Observation");
          {
            Adjoint::Observation::SurfaceFieldObservation<dim>::declare_parameters(prm);
          }
          prm.leave_subsection();
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Geoid surface observation test");
          mode = prm.get("Mode");
          longitude = prm.get_double("Longitude");
          latitude = prm.get_double("Latitude");
          const bool suppress_stacktrace = prm.get_bool("Suppress exception stack trace");
          prm.enter_subsection("Observation");
          {
            observation.parse_parameters(prm);
          }
          prm.leave_subsection();
          prm.leave_subsection();
          prm.leave_subsection();

          if (suppress_stacktrace)
            // Test-only: avoids the known macOS deal.II stack-trace allocation
            // runaway. It does not suppress the actual production exception.
            dealii::deal_II_exceptions::suppress_stacktrace_in_exceptions();
        }

        std::pair<std::string, std::string> execute(TableHandler &) override
        {
          if (mode == "none")
            return {};

          observation.initialize_simulator(this->get_simulator());
          observation.set_native_quantity(
            Adjoint::Observation::SurfaceFieldObservation<dim>::NativeQuantity::geoid);
          observation.initialize();

          const double longitude_radians = longitude * constants::degree_to_radians;
          const double latitude_radians = latitude * constants::degree_to_radians;
          Point<dim> point;
          point[0] = std::cos(latitude_radians) * std::cos(longitude_radians);
          point[1] = std::cos(latitude_radians) * std::sin(longitude_radians);
          if constexpr (dim == 3)
            point[2] = std::sin(latitude_radians);

          std::cout << std::setprecision(17)
                    << "ASPECT_GEOID_SURFACE_OBSERVATION rank="
                    << Utilities::MPI::this_mpi_process(this->get_mpi_communicator())
                    << " value=" << observation.value(point) << std::endl;
          return {"Geoid surface observation test:", "evaluated native geoid observation"};
        }

      private:
        std::string mode = "none";
        double longitude = 0.;
        double latitude = 0.;
        Adjoint::Observation::SurfaceFieldObservation<dim> observation;
    };

    ASPECT_REGISTER_POSTPROCESSOR(GeoidSurfaceObservationTestPostprocessor,
                                  "geoid surface observation test",
                                  "Benchmark-only probe for native geoid surface observations.")
  }
}
