/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#ifndef _aspect_postprocess_self_gravitating_geoid_h
#define _aspect_postprocess_self_gravitating_geoid_h

#include <aspect/postprocess/interface.h>
#include <aspect/postprocess/self_gravitating_geoid_operator.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Postprocess
  {
    /** Forward postprocessor exposing the coupled l>=2 solution. */
    template <int dim>
    class SelfGravitatingGeoid : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string, std::string> execute(TableHandler &statistics) override;
        std::list<std::string> required_other_postprocessors() const override;
        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;

        const SelfGravitatingGeoidCoefficients &get_result() const;

      private:
        unsigned int minimum_degree = 2;
        unsigned int maximum_degree = 20;
        SelfGravitatingGeoidCoefficients result;
    };
  }
}

#endif
