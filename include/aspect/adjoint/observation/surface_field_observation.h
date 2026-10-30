/*
  Copyright (C) 2026 by the ASPECT authors.

  This file is part of ASPECT.
*/

#ifndef _aspect_adjoint_observation_surface_field_observation_h
#define _aspect_adjoint_observation_surface_field_observation_h

#include <aspect/simulator_access.h>
#include <aspect/utilities.h>

#include <deal.II/base/parsed_function.h>
#include <deal.II/numerics/rtree.h>

#include <memory>

namespace aspect
{
  namespace Adjoint
  {
    namespace Observation
    {
      /** A scalar observable represented by samples on a model boundary. */
      template <int dim>
      class SurfaceFieldObservation : public SimulatorAccess<dim>
      {
        public:
          enum class NativeQuantity
          {
            none,
            dynamic_topography,
            geoid
          };

          SurfaceFieldObservation();

          static void declare_parameters(ParameterHandler &prm);
          void parse_parameters(ParameterHandler &prm);

          /** Set by the physical objective, never by a duplicate user parameter. */
          void set_native_quantity(const NativeQuantity quantity);

          /** Read and index file data once. Safe to call repeatedly. */
          void initialize();

          /** Evaluate the configured function or cached sampled observation. */
          double value(const Point<dim> &position) const;

          bool uses_file_source() const;

        private:
          enum class Source
          {
            function, file
          };
          enum class Format
          {
            generic, aspect_native
          };
          enum class CoordinateLayout
          {
            cartesian, longitude_latitude
          };

          Source source;
          Format format;
          NativeQuantity native_quantity;
          Functions::ParsedFunction<dim> target_function;
          Utilities::Coordinates::CoordinateSystem coordinate_system;
          std::string observation_file;
          std::string reference_output_directory;
          std::string time_selection;
          unsigned int timestep;
          std::string surface_boundary;
          unsigned int interpolation_neighbors;
          double maximum_distance_factor;
          CoordinateLayout coordinate_layout;
          bool initialized;
          double characteristic_spacing;
          std::vector<std::pair<Point<3>, double>> samples;
          std::unique_ptr<RTree<std::pair<Point<3>, unsigned int>>> spatial_index;

          std::string discover_file() const;
          void read_file(const std::string &filename);
          Point<3> embedding_point(const Point<dim> &position) const;
          Point<3> embedding_point_from_longitude_latitude(const double longitude,
                                                           const double latitude) const;
          bool uses_spherical_distance() const;
          void validate_geometry() const;
      };
    }
  }
}

#endif
