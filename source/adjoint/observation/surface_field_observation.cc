/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.
*/

#include <aspect/adjoint/observation/surface_field_observation.h>

#include <aspect/geometry_model/interface.h>
#include <aspect/geometry_model/spherical_shell.h>

#include <boost/geometry/index/predicates.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

namespace aspect
{
  namespace Adjoint
  {
    namespace Observation
    {
      namespace
      {
        std::vector<std::string> words(const std::string &line)
        {
          std::istringstream input(line);
          std::vector<std::string> result;
          std::string word;

          while (input >> word)
            result.push_back(word);

          return result;
        }



        bool starts_with(const std::string &value, const std::string &prefix)
        {
          return value.size() >= prefix.size()
                 && value.compare(0, prefix.size(), prefix) == 0;
        }
      }



      template <int dim>
      SurfaceFieldObservation<dim>::SurfaceFieldObservation()
        : source(Source::function),
          format(Format::generic),
          native_quantity(NativeQuantity::none),
          target_function(1),
          coordinate_system(Utilities::Coordinates::cartesian),
          time_selection("final"),
          timestep(0),
          surface_boundary("top"),
          interpolation_neighbors(8),
          maximum_distance_factor(4.0),
          coordinate_layout(CoordinateLayout::cartesian),
          initialized(false),
          characteristic_spacing(0.0)
      {}



      template <int dim>
      void SurfaceFieldObservation<dim>::declare_parameters(ParameterHandler &prm)
      {
        prm.declare_entry("Observation source", "function",
                          Patterns::Selection("function|file"),
                          "Source for this scalar surface observation.");
        prm.declare_entry("Observation format", "generic",
                          Patterns::Selection("generic|ASPECT native"),
                          "File format when Observation source is file. ASPECT native "
                          "selects the documented dynamic-topography or geoid "
                          "surface output required by the owning objective.");
        prm.declare_entry("Observation file", "", Patterns::Anything(),
                          "Explicit observation filename. If empty, ASPECT-native "
                          "input is discovered from the reference output directory.");
        prm.declare_entry("Interpolation neighbors", "8", Patterns::Integer(1),
                          "Number of cached scattered surface samples used by the "
                          "inverse-distance interpolation.");
        prm.declare_entry("Maximum interpolation distance factor", "4", Patterns::Double(0.),
                          "Reject a target point whose nearest observation is farther "
                          "than this multiple of the median nearest-sample spacing.");
        prm.enter_subsection("ASPECT reference");
        {
          prm.declare_entry("Output directory", "", Patterns::Anything(),
                            "Output directory of the standalone ASPECT reference run.");
          prm.declare_entry("Time selection", "final", Patterns::Selection("final|timestep"),
                            "Select the final available native output or an explicit timestep.");
          prm.declare_entry("Timestep", "0", Patterns::Integer(0),
                            "Timestep selected when Time selection is timestep.");
          prm.declare_entry("Surface boundary", "top", Patterns::Anything(),
                            "Symbolic boundary name for ASPECT-native dynamic topography.");
        }
        prm.leave_subsection();
        prm.enter_subsection("Target function");
        {
          prm.declare_entry("Coordinate system", "cartesian",
                            Patterns::Selection("cartesian|spherical|depth"),
                            "Coordinate system used by a function surface observation.");
          Functions::ParsedFunction<dim>::declare_parameters(prm, 1);
          prm.declare_entry("Function expression", "0", Patterns::Anything(),
                            "Target scalar surface observation.");
        }
        prm.leave_subsection();
      }



      template <int dim>
      void SurfaceFieldObservation<dim>::parse_parameters(ParameterHandler &prm)
      {
        source = prm.get("Observation source") == "function" ? Source::function : Source::file;
        format = prm.get("Observation format") == "generic" ? Format::generic : Format::aspect_native;
        observation_file = prm.get("Observation file");
        interpolation_neighbors = prm.get_integer("Interpolation neighbors");
        maximum_distance_factor = prm.get_double("Maximum interpolation distance factor");
        prm.enter_subsection("ASPECT reference");
        {
          reference_output_directory = prm.get("Output directory");
          time_selection = prm.get("Time selection");
          timestep = prm.get_integer("Timestep");
          surface_boundary = prm.get("Surface boundary");
        }
        prm.leave_subsection();
        prm.enter_subsection("Target function");
        {
          coordinate_system = Utilities::Coordinates::string_to_coordinate_system(
                                prm.get("Coordinate system"));
          target_function.parse_parameters(prm);
        }
        prm.leave_subsection();
        initialized = false;
        samples.clear();
        spatial_index.reset();
      }



      template <int dim>
      void SurfaceFieldObservation<dim>::set_native_quantity(const NativeQuantity quantity)
      {
        AssertThrow(native_quantity == NativeQuantity::none || native_quantity == quantity,
                    ExcMessage("One surface observation cannot be reused for different physical quantities."));
        native_quantity = quantity;
      }



      template <int dim>
      std::string SurfaceFieldObservation<dim>::discover_file() const
      {
        if (!observation_file.empty())
          return observation_file;
        AssertThrow(format == Format::aspect_native,
                    ExcMessage("A generic surface observation requires Observation file."));
        AssertThrow(!reference_output_directory.empty(),
                    ExcMessage("ASPECT-native surface observations require ASPECT reference/Output directory "
                               "or an explicit Observation file."));
        AssertThrow(native_quantity != NativeQuantity::none, ExcInternalError());

        const std::string directory = reference_output_directory
                                      + (reference_output_directory.back() == '/' ? "" : "/")
                                      + (native_quantity == NativeQuantity::dynamic_topography
                                         ? "dynamic_topography/" : "geoid/");
        const std::string prefix = native_quantity == NativeQuantity::dynamic_topography
                                   ? "dynamic_topography_" + surface_boundary + "."
                                   : "geoid_anomaly.";
        const auto filename_for_step = [&directory, &prefix](const unsigned int step)
        {
          return directory + prefix + Utilities::int_to_string(step, 5);
        };
        if (time_selection == "timestep")
          {
            const std::string filename = filename_for_step(timestep);
            AssertThrow(Utilities::fexists(filename, this->get_mpi_communicator()),
                        ExcMessage("Requested ASPECT-native surface observation timestep file is missing: " + filename));
            return filename;
          }

        unsigned int final_step = numbers::invalid_unsigned_int;
        std::error_code error;
        for (const auto &entry : std::filesystem::directory_iterator(directory, error))
          if (!error && entry.is_regular_file())
            {
              const std::string name = entry.path().filename().string();
              if (starts_with(name, prefix))
                {
                  const std::string suffix = name.substr(prefix.size());
                  if (suffix.size() == 5
                      && std::all_of(suffix.begin(), suffix.end(), ::isdigit))
                    final_step = std::max(final_step == numbers::invalid_unsigned_int ? 0U : final_step,
                                          static_cast<unsigned int>(std::stoul(suffix)));
                }
            }
        AssertThrow(!error && final_step != numbers::invalid_unsigned_int,
                    ExcMessage("No documented ASPECT-native surface output matching " + directory + prefix
                               + "NNNNN was found."));
        return filename_for_step(final_step);
      }



      template <int dim>
      Point<3> SurfaceFieldObservation<dim>::embedding_point_from_longitude_latitude(
        const double longitude, const double latitude) const
      {
        const double lon = longitude * constants::degree_to_radians;
        const double lat = latitude * constants::degree_to_radians;
        return Point<3>(std::cos(lat) * std::cos(lon),
                        std::cos(lat) * std::sin(lon), std::sin(lat));
      }



      template <int dim>
      bool SurfaceFieldObservation<dim>::uses_spherical_distance() const
      {
        return coordinate_layout == CoordinateLayout::longitude_latitude
               || this->get_geometry_model().natural_coordinate_system()
               == Utilities::Coordinates::spherical;
      }



      template <int dim>
      Point<3> SurfaceFieldObservation<dim>::embedding_point(const Point<dim> &position) const
      {
        Point<3> embedded;
        if (uses_spherical_distance())
          {
            const double radius = position.norm();
            AssertThrow(radius > std::numeric_limits<double>::min(),
                        ExcMessage("Surface observation cannot evaluate the origin in spherical geometry."));
            for (unsigned int d = 0; d < dim; ++d)
              embedded[d] = position[d] / radius;
          }
        else
          for (unsigned int d = 0; d < dim; ++d)
            embedded[d] = position[d];
        return embedded;
      }



      template <int dim>
      void SurfaceFieldObservation<dim>::validate_geometry() const
      {
        if (format == Format::aspect_native && native_quantity == NativeQuantity::geoid)
          {
            AssertThrow(dim == 3
                        && dynamic_cast<const GeometryModel::SphericalShell<dim> *>(
                          &this->get_geometry_model()) != nullptr,
                        ExcMessage("ASPECT-native geoid surface observations require a 3D spherical shell geometry."));
          }
      }



      template <int dim>
      void SurfaceFieldObservation<dim>::read_file(const std::string &filename)
      {
        std::ifstream input(filename);
        AssertThrow(input, ExcMessage("Unable to open surface observation file: " + filename));
        std::string header;
        std::getline(input, header);
        AssertThrow(!header.empty() && header[0] == '#',
                    ExcMessage("Surface observation file must start with a commented column header: " + filename));
        const std::vector<std::string> header_words = words(header.substr(1));
        AssertThrow(!header_words.empty(), ExcMessage("Surface observation file has an empty header: " + filename));

        if (format == Format::aspect_native)
          {
            AssertThrow(native_quantity != NativeQuantity::none, ExcInternalError());
            if (native_quantity == NativeQuantity::dynamic_topography)
              {
                const std::string expected = "dynamic_topography_" + surface_boundary;
                AssertThrow(header_words.back() == expected,
                            ExcMessage("ASPECT-native dynamic-topography observation expected final column '"
                                       + expected + "' but found '" + header_words.back() + "'."));
                AssertThrow(header_words.size() == dim + 1,
                            ExcMessage("ASPECT-native dynamic-topography file has an unsupported coordinate layout."));
                coordinate_layout = CoordinateLayout::cartesian;
              }
            else
              {
                AssertThrow(header_words.back() == "geoid_anomaly",
                            ExcMessage("ASPECT-native geoid observation expected final column 'geoid_anomaly'."));
                const bool xyz = header_words.size() == 4 && header_words[0] == "x"
                                 && header_words[1] == "y" && header_words[2] == "z";
                const bool lonlat = header_words.size() == 3 && header_words[0] == "longitude"
                                    && header_words[1] == "latitude";
                AssertThrow(xyz || lonlat,
                            ExcMessage("ASPECT-native geoid file must use x y z or longitude latitude coordinates."));
                coordinate_layout = lonlat ? CoordinateLayout::longitude_latitude
                                    : CoordinateLayout::cartesian;
              }
          }
        else
          {
            AssertThrow(header_words.size() == dim + 1,
                        ExcMessage("Generic surface observations require x y [z] value columns."));
            coordinate_layout = CoordinateLayout::cartesian;
          }

        std::string line;
        unsigned int line_number = 1;
        while (std::getline(input, line))
          {
            ++line_number;
            if (line.empty() || line[0] == '#')
              continue;
            std::istringstream row(line);
            std::vector<double> values;
            double value;
            while (row >> value)
              values.push_back(value);
            const unsigned int coordinate_count = coordinate_layout == CoordinateLayout::longitude_latitude ? 2 : dim;
            AssertThrow(values.size() == coordinate_count + 1,
                        ExcMessage("Malformed surface observation row " + std::to_string(line_number)
                                   + " in " + filename));
            Point<3> point;
            if (coordinate_layout == CoordinateLayout::longitude_latitude)
              point = embedding_point_from_longitude_latitude(values[0], values[1]);
            else
              {
                for (unsigned int d = 0; d < dim; ++d)
                  point[d] = values[d];
                if (uses_spherical_distance())
                  {
                    const double radius = point.norm();
                    AssertThrow(radius > std::numeric_limits<double>::min(),
                                ExcMessage("Surface observation contains an origin sample."));
                    point /= radius;
                  }
              }
            samples.emplace_back(point, values.back());
          }
        AssertThrow(!samples.empty(), ExcMessage("Surface observation file contains no samples: " + filename));
      }



      template <int dim>
      void SurfaceFieldObservation<dim>::initialize()
      {
        if (initialized)
          return;
        if (source == Source::function)
          {
            initialized = true;
            return;
          }
        validate_geometry();
        read_file(discover_file());
        std::vector<std::pair<Point<3>, unsigned int>> leaves;
        leaves.reserve(samples.size());
        for (unsigned int index = 0; index < samples.size(); ++index)
          leaves.emplace_back(samples[index].first, index);
        spatial_index = std::make_unique<RTree<std::pair<Point<3>, unsigned int>>>(
                          pack_rtree(leaves));

        std::vector<double> spacings;
        spacings.reserve(samples.size());
        namespace bgi = boost::geometry::index;
        for (const auto &sample : samples)
          {
            std::vector<std::pair<Point<3>, unsigned int>> nearest;
            spatial_index->query(bgi::nearest(sample.first,
                                              std::min<unsigned int>(8, samples.size())),
                                 std::back_inserter(nearest));
            for (const auto &candidate : nearest)
              {
                const double distance = sample.first.distance(candidate.first);
                if (distance > std::numeric_limits<double>::epsilon())
                  {
                    spacings.push_back(distance);
                    break;
                  }
              }
          }
        AssertThrow(!spacings.empty(),
                    ExcMessage("Surface observation requires at least two distinct samples."));
        std::nth_element(spacings.begin(), spacings.begin() + spacings.size() / 2, spacings.end());
        characteristic_spacing = spacings[spacings.size() / 2];
        AssertThrow(characteristic_spacing > std::numeric_limits<double>::min(),
                    ExcMessage("Surface observation samples are not spatially distinct."));
        initialized = true;
      }



      template <int dim>
      double SurfaceFieldObservation<dim>::value(const Point<dim> &position) const
      {
        AssertThrow(initialized, ExcMessage("Surface observation was evaluated before initialization."));
        if (source == Source::function)
          {
            const Utilities::NaturalCoordinate<dim> point =
              this->get_geometry_model().cartesian_to_other_coordinates(position, coordinate_system);
            return target_function.value(Utilities::convert_array_to_point<dim>(point.get_coordinates()));
          }

        const Point<3> query = embedding_point(position);
        namespace bgi = boost::geometry::index;
        std::vector<std::pair<Point<3>, unsigned int>> nearest;
        spatial_index->query(bgi::nearest(query, interpolation_neighbors), std::back_inserter(nearest));
        AssertThrow(!nearest.empty(), ExcInternalError());
        // The R-tree nearest predicate bounds the result set but does not make
        // its output order part of this observation's contract. Sort the small
        // set explicitly so exact samples and equal-distance ties are handled
        // deterministically before inverse-distance weights are formed.
        std::sort(nearest.begin(), nearest.end(), [&query](const auto &first, const auto &second)
        {
          const double first_distance = query.distance(first.first);
          const double second_distance = query.distance(second.first);
          return first_distance == second_distance ? first.second < second.second
                 : first_distance < second_distance;
        });
        const double nearest_distance = query.distance(nearest.front().first);
        AssertThrow(nearest_distance <= maximum_distance_factor * characteristic_spacing,
                    ExcMessage("Surface observation point is incompatible with the reference surface samples. "
                               "The nearest sample is farther than the configured geometry-validation limit."));
        if (nearest_distance <= std::numeric_limits<double>::epsilon())
          return samples[nearest.front().second].second;

        double weighted_value = 0.0;
        double total_weight = 0.0;
        for (const auto &entry : nearest)
          {
            const double chord = query.distance(entry.first);
            const double distance = uses_spherical_distance()
                                    ? 2.0 * std::asin(std::min(1.0, 0.5 * chord))
                                    : chord;
            const double weight = 1.0 / (distance * distance);
            weighted_value += weight * samples[entry.second].second;
            total_weight += weight;
          }
        return weighted_value / total_weight;
      }



      template <int dim>
      bool SurfaceFieldObservation<dim>::uses_file_source() const
      {
        return source == Source::file;
      }
    }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) template class Adjoint::Observation::SurfaceFieldObservation<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
