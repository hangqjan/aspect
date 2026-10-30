/*
 * Exact G-ADOPT initial temperature profile, used here as a chemical
 * composition field. A plugin is required because ASPECT's parsed-function
 * backend does not provide erf().
 */

#include <aspect/initial_composition/interface.h>

#include <cmath>

namespace aspect
{
  namespace InitialComposition
  {
    template <int dim>
    class GadoptCompositionInitial : public Interface<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Initial composition model");
          prm.enter_subsection("Gadopt composition initial");
          {
            prm.declare_entry("Include localized anomaly", "true", Patterns::Bool(),
                              "Whether to include the G-ADOPT 0.1-amplitude Gaussian anomaly.");
          }
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Initial composition model");
          prm.enter_subsection("Gadopt composition initial");
          {
            include_localized_anomaly = prm.get_bool("Include localized anomaly");
          }
          prm.leave_subsection();
          prm.leave_subsection();
        }

        double initial_composition(const Point<dim> &position,
                                   const unsigned int composition) const override
        {
          AssertThrow(dim == 2,
                      ExcMessage("The G-ADOPT composition benchmark is two-dimensional."));
          AssertIndexRange(composition, 1);
          static_cast<void>(composition);
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

    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(GadoptCompositionInitial,
                                              "gadopt composition initial",
                                              "The exact analytic initial field from the G-ADOPT mantle-convection "
                                              "adjoint reference, supplied as chemical composition.")
  }
}
