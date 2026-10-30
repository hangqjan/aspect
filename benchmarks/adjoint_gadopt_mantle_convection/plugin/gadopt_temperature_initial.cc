/* Exact analytic G-ADOPT temperature profile for runnable benchmark inputs. */

#include <aspect/initial_temperature/interface.h>

#include <cmath>

namespace aspect
{
  namespace InitialTemperature
  {
    template <int dim>
    class GadoptTemperatureInitial : public Interface<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Initial temperature model");
          prm.enter_subsection("Gadopt temperature initial");
          {
            prm.declare_entry("Include localized anomaly", "true", Patterns::Bool(),
                              "Whether to include the G-ADOPT 0.1-amplitude Gaussian anomaly.");
          }
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Initial temperature model");
          prm.enter_subsection("Gadopt temperature initial");
          {
            include_localized_anomaly = prm.get_bool("Include localized anomaly");
          }
          prm.leave_subsection();
          prm.leave_subsection();
        }

        double initial_temperature(const Point<dim> &position) const override
        {
          AssertThrow(dim == 2,
                      ExcMessage("The G-ADOPT temperature benchmark is two-dimensional."));
          const double x = position[0];
          const double y = position[1];
          const double background = 0.5 * (std::erf((1.0 - y) * 3.0)
                                            + std::erf(-3.0 * y) + 1.0);
          const double anomaly = 0.1 * std::exp(-0.5 * ((x - 0.5) * (x - 0.5)
                                                         + (y - 0.2) * (y - 0.2))
                                               / (0.1 * 0.1));
          return background + (include_localized_anomaly ? anomaly : 0.0);
        }

      private:
        bool include_localized_anomaly = true;
    };

    ASPECT_REGISTER_INITIAL_TEMPERATURE_MODEL(GadoptTemperatureInitial,
                                               "gadopt temperature initial",
                                               "The exact analytic initial field from the G-ADOPT mantle-convection "
                                               "benchmark, including an optional localized anomaly.")
  }
}
