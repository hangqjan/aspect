/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.
*/

#include <aspect/adjoint/observation/spherical_harmonic_observation.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

namespace aspect
{
  namespace Adjoint
  {
    namespace Observation
    {
      template <int dim>
      SphericalHarmonicObservation<dim>::SphericalHarmonicObservation()
        : format(Format::generic), time_selection("final"), timestep(0)
      {}



      template <int dim>
      void SphericalHarmonicObservation<dim>::declare_parameters(ParameterHandler &prm)
      {
        prm.declare_entry("Observation format", "generic",
                          Patterns::Selection("generic|ASPECT native"),
                          "Coefficient-file format. ASPECT native uses the canonical "
                          "geoid degree/order/cosine/sine convention.");
        prm.declare_entry("Observation file", "", Patterns::Anything(),
                          "Explicit coefficient filename, overriding reference-directory discovery.");
        prm.enter_subsection("ASPECT reference");
        {
          prm.declare_entry("Output directory", "", Patterns::Anything(),
                            "Output directory of the standalone ASPECT reference run.");
          prm.declare_entry("Time selection", "final", Patterns::Selection("final|timestep"),
                            "Select the final available native geoid coefficient output or a timestep.");
          prm.declare_entry("Timestep", "0", Patterns::Integer(0),
                            "Timestep selected when Time selection is timestep.");
        }
        prm.leave_subsection();
      }



      template <int dim>
      void SphericalHarmonicObservation<dim>::parse_parameters(ParameterHandler &prm)
      {
        format = prm.get("Observation format") == "generic"
                 ? Format::generic : Format::aspect_native_geoid;
        observation_file = prm.get("Observation file");
        prm.enter_subsection("ASPECT reference");
        {
          reference_output_directory = prm.get("Output directory");
          time_selection = prm.get("Time selection");
          timestep = prm.get_integer("Timestep");
        }
        prm.leave_subsection();
        observation.reset();
      }



      template <int dim>
      std::string SphericalHarmonicObservation<dim>::discover_file() const
      {
        if (!observation_file.empty())
          return observation_file;
        AssertThrow(format == Format::aspect_native_geoid && !reference_output_directory.empty(),
                    ExcMessage("A coefficient observation requires Observation file or an ASPECT reference output directory."));
        const std::string directory = reference_output_directory
                                      + (reference_output_directory.back() == '/' ? "" : "/") + "geoid/";
        const std::string prefix = "geoid_anomaly_SH_coefficients.";
        const auto file_for_step = [&directory, &prefix](const unsigned int step)
        {
          return directory + prefix + Utilities::int_to_string(step, 5);
        };
        if (time_selection == "timestep")
          {
            const std::string file = file_for_step(timestep);
            AssertThrow(Utilities::fexists(file, this->get_mpi_communicator()),
                        ExcMessage("Requested native geoid coefficient file is missing: " + file));
            return file;
          }
        unsigned int final_step = numbers::invalid_unsigned_int;
        std::error_code error;
        for (const auto &entry : std::filesystem::directory_iterator(directory, error))
          if (!error && entry.is_regular_file())
            {
              const std::string name = entry.path().filename().string();
              if (name.compare(0, prefix.size(), prefix) == 0)
                {
                  const std::string suffix = name.substr(prefix.size());
                  if (suffix.size() == 5 && std::all_of(suffix.begin(), suffix.end(), ::isdigit))
                    final_step = std::max(final_step == numbers::invalid_unsigned_int ? 0U : final_step,
                                          static_cast<unsigned int>(std::stoul(suffix)));
                }
            }
        AssertThrow(!error && final_step != numbers::invalid_unsigned_int,
                    ExcMessage("No native geoid coefficient file was found in " + directory));
        return file_for_step(final_step);
      }



      template <int dim>
      void SphericalHarmonicObservation<dim>::initialize(const unsigned int minimum_degree,
                                                         const unsigned int maximum_degree) const
      {
        if (observation != nullptr)
          {
            AssertThrow(observation->minimum_degree() == minimum_degree
                        && observation->maximum_degree() == maximum_degree,
                        ExcMessage("The harmonic observation degree range changed after loading."));
            return;
          }
        const std::string filename = discover_file();
        std::ifstream input(filename);
        AssertThrow(input, ExcMessage("Unable to open coefficient observation file: " + filename));
        std::string header;
        std::getline(input, header);
        AssertThrow(header == "# degree order cosine_coefficient sine_coefficient",
                    ExcMessage("Coefficient observation must have the canonical degree/order/cosine/sine header."));
        auto loaded = std::make_unique<Postprocess::SphericalHarmonicCoefficients>(minimum_degree, maximum_degree);
        std::vector<bool> seen(loaded->size(), false);
        std::string line;
        unsigned int line_number = 1;
        while (std::getline(input, line))
          {
            ++line_number;
            if (line.empty() || line[0] == '#')
              continue;
            std::istringstream row(line);
            unsigned int degree, order;
            double cosine, sine;
            std::string extra;
            AssertThrow((row >> degree >> order >> cosine >> sine) && !(row >> extra),
                        ExcMessage("Malformed harmonic coefficient row " + std::to_string(line_number)
                                   + " in " + filename));
            AssertThrow(order <= degree, ExcMessage("Harmonic order exceeds degree in " + filename));
            AssertThrow(degree >= minimum_degree && degree <= maximum_degree,
                        ExcMessage("Harmonic coefficient degree is outside the configured observation range."));
            const unsigned int index = loaded->coefficient_index(degree, order);
            AssertThrow(!seen[index], ExcMessage("Duplicate harmonic coefficient (l,m) in " + filename));
            seen[index] = true;
            loaded->cosine(index) = cosine;
            loaded->sine(index) = sine;
          }
        AssertThrow(std::all_of(seen.begin(), seen.end(), [](const bool entry)
        {
          return entry;
        }),
        ExcMessage("Harmonic observation is missing one or more required (l,m) coefficients."));
        observation = std::move(loaded);
      }



      template <int dim>
      const Postprocess::SphericalHarmonicCoefficients &
      SphericalHarmonicObservation<dim>::coefficients() const
      {
        AssertThrow(observation != nullptr, ExcMessage("Harmonic observation was accessed before initialization."));
        return *observation;
      }
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::Observation::SphericalHarmonicObservation<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
