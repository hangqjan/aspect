/*
 Copyright (C) 2015 - 2024 by the authors of the ASPECT code.

 This file is part of ASPECT.

 ASPECT is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2, or (at your option)
 any later version.

 ASPECT is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with ASPECT; see the file LICENSE.  If not see
 <http://www.gnu.org/licenses/>.
 */


#include <aspect/simulator.h>
#include <aspect/mesh_deformation/free_surface.h>
#include <aspect/utilities.h>
#include <aspect/postprocess/geoid.h>
#include <aspect/postprocess/geoid_harmonics.h>
#include <aspect/postprocess/dynamic_topography.h>
#include <aspect/postprocess/boundary_densities.h>
#include <aspect/geometry_model/spherical_shell.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    SphericalHarmonicCoefficients
    Geoid<dim>::to_spherical_harmonic_coefficients(const std::vector<std::vector<double>> &spherical_function) const
    {
      std::vector<double> cosi(spherical_function.size(),0);
      std::vector<double> sini(spherical_function.size(),0);
      SphericalHarmonicCoefficients coefficients(minimum_degree, maximum_degree);
      std::vector<double> cosine(coefficients.size(), 0.0);
      std::vector<double> sine(coefficients.size(), 0.0);

      unsigned int coefficient_index = 0;
      for (unsigned int ideg = minimum_degree; ideg <= maximum_degree; ++ideg)
        {
          for (unsigned int iord = 0; iord <= ideg; ++iord, ++coefficient_index)
            {
              // Do the spherical harmonic expansion.
              for (unsigned int ds_num = 0; ds_num < spherical_function.size(); ++ds_num)
                {
                  // Normalization after Dahlen and Tromp (1986) Appendix B.6.
                  const std::pair<double,double> sph_harm_vals = aspect::Utilities::real_spherical_harmonic(ideg,iord,spherical_function.at(ds_num).at(0),spherical_function.at(ds_num).at(1));
                  const double cos_component = sph_harm_vals.first; // real / cos part
                  const double sin_component = sph_harm_vals.second; // imaginary / sin part

                  cosi.at(ds_num) = (spherical_function.at(ds_num).at(3) * cos_component);
                  sini.at(ds_num) = (spherical_function.at(ds_num).at(3) * sin_component);
                }
              // Integrate the contribution of each spherical infinitesimal.
              double cosii = 0;
              double sinii = 0;
              for (unsigned int ds_num = 0; ds_num < spherical_function.size(); ++ds_num)
                {
                  cosii += cosi.at(ds_num) * spherical_function.at(ds_num).at(2);
                  sinii += sini.at(ds_num) * spherical_function.at(ds_num).at(2);
                }
              cosine[coefficient_index] = cosii;
              sine[coefficient_index] = sinii;
            }
        }
      // Sum over each processor.
      dealii::Utilities::MPI::sum(cosine, this->get_mpi_communicator(), cosine);
      dealii::Utilities::MPI::sum(sine, this->get_mpi_communicator(), sine);

      coefficients.set_coefficients(cosine, sine);
      return coefficients;
    }

    template <int dim>
    typename Geoid<dim>::TopographyHarmonics
    Geoid<dim>::topography_contribution(const double &/*outer_radius*/,
                                        const double &/*inner_radius*/) const
    {
      Assert(false, ExcNotImplemented());
      return {};
    }

    template <>
    Geoid<3>::TopographyHarmonics
    Geoid<3>::topography_contribution(const double &outer_radius,
                                      const double &inner_radius) const
    {
      // Get a pointer to the boundary densities postprocessor.
      const Postprocess::BoundaryDensities<3> &boundary_densities =
        this->get_postprocess_manager().template get_matching_active_plugin<Postprocess::BoundaryDensities<3>>();

      const double top_layer_average_density = boundary_densities.density_at_top();
      const double bottom_layer_average_density = boundary_densities.density_at_bottom();

      const types::boundary_id top_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      const types::boundary_id bottom_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

      const unsigned int quadrature_degree = this->introspection().polynomial_degree.temperature;
      const QGauss<2> quadrature_formula_face(quadrature_degree);

      FEFaceValues<3> fe_face_values (this->get_mapping(),
                                      this->get_fe(),
                                      quadrature_formula_face,
                                      update_values |
                                      update_quadrature_points |
                                      update_JxW_values);

      // Vectors to store the location, infinitesimal area, and topography associated with each quadrature point of each surface and bottom cell respectively.
      std::vector<std::pair<Point<3>,std::pair<double,double>>> surface_stored_values;
      std::vector<std::pair<Point<3>,std::pair<double,double>>> cmb_stored_values;

      // Loop over all of the boundary cells and if one is at
      // surface or CMB, evaluate the topography vector there.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          {
            unsigned int face_idx = numbers::invalid_unsigned_int;
            bool at_upper_surface = false;
            {
              for (const unsigned int f : cell->face_indices())
                {
                  if (cell->at_boundary(f) && cell->face(f)->boundary_id() == top_boundary_id)
                    {
                      // If the cell is at the top boundary, assign face_idx.
                      face_idx = f;
                      at_upper_surface = true;
                      break;
                    }
                  else if (cell->at_boundary(f) && cell->face(f)->boundary_id() == bottom_boundary_id)
                    {
                      // If the cell is at the bottom boundary, assign face_idx.
                      face_idx = f;
                      at_upper_surface = false;
                      break;
                    }
                }
              // If the cell is not at the boundary, jump to the next cell.
              if (face_idx == numbers::invalid_unsigned_int)
                continue;
            }

            // Focus on the boundary cell's upper face if on the top boundary and lower face if on the bottom boundary.
            fe_face_values.reinit(cell, face_idx);

            // Topography is evaluated at each quadrature
            // point on every top/bottom cell's boundary face. The
            // reason to do this -- as opposed to using a single
            // value per boundary face -- is that later in the
            // spherical harmonic expansion, we will calculate
            // sin(theta)*d_theta*d_phi by
            // infinitesimal_area/radius^2. The accuracy of this
            // transfer gets better as infinitesimal_area gets
            // closer to zero, so using every boundary quadrature
            // point's associated area (in the form of
            // FEFaceValues::JxW) will lead to better accuracy in
            // spherical harmonic expansion compared to using just
            // one average value per face, especially in the coarse
            // meshes.

            // If the cell is at the top boundary, add its contributions to the topography surface storage vector.
            if (at_upper_surface)
              {
                if (include_surface_topography_contribution == true)
                  {
                    if (use_free_surface_topography == true)
                      {
                        for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                          {
                            const Point<3> current_position = fe_face_values.quadrature_point(q);
                            const double topography = this->get_geometry_model().height_above_reference_surface(current_position);
                            surface_stored_values.emplace_back (current_position, std::make_pair(fe_face_values.JxW(q), topography));
                          }
                      }
                    else
                      {
                        // Get a reference to the dynamic topography postprocessor.
                        const Postprocess::DynamicTopography<3> &dynamic_topography =
                          this->get_postprocess_manager().template get_matching_active_plugin<Postprocess::DynamicTopography<3>>();

                        // Get the already-computed dynamic topography solution.
                        const LinearAlgebra::BlockVector &topo_vector = dynamic_topography.topography_vector();

                        std::vector<double> topo_values(quadrature_formula_face.size());

                        fe_face_values[this->introspection().extractors.temperature].get_function_values(topo_vector, topo_values);

                        for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                          {
                            surface_stored_values.emplace_back (fe_face_values.quadrature_point(q), std::make_pair(fe_face_values.JxW(q), topo_values[q]));
                          }
                      }
                  }
                else
                  {
                    // If the cell is at the top boundary, but you do not include the surface topography in the geoid computation,
                    // add its zero contributions to the topography surface storage vector.
                    if (at_upper_surface)
                      {
                        for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                          {
                            surface_stored_values.emplace_back (fe_face_values.quadrature_point(q), std::make_pair(fe_face_values.JxW(q), 0.0));
                          }
                      }
                  }
              }

            // If the cell is at the bottom boundary, add its contributions to the topography CMB storage vector.
            if (at_upper_surface == false)
              {
                if (include_cmb_topography_contribution == true)
                  {
                    if (use_free_cmb_topography == true)
                      {
                        for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                          {
                            const Point<3> current_position = fe_face_values.quadrature_point(q);
                            const double topography = this->get_geometry_model().height_above_reference_surface(current_position) + (outer_radius - inner_radius);
                            cmb_stored_values.emplace_back (current_position, std::make_pair(fe_face_values.JxW(q), topography));
                          }
                      }
                    else
                      {
                        // Get a reference to the dynamic topography postprocessor.
                        const Postprocess::DynamicTopography<3> &dynamic_topography =
                          this->get_postprocess_manager().template get_matching_active_plugin<Postprocess::DynamicTopography<3>>();

                        // Get the already-computed dynamic topography solution.
                        const LinearAlgebra::BlockVector &topo_vector = dynamic_topography.topography_vector();

                        std::vector<double> topo_values(quadrature_formula_face.size());

                        fe_face_values[this->introspection().extractors.temperature].get_function_values(topo_vector, topo_values);

                        for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                          {
                            cmb_stored_values.emplace_back (fe_face_values.quadrature_point(q), std::make_pair(fe_face_values.JxW(q), topo_values[q]));
                          }
                      }
                  }
                else
                  {
                    // If the cell is at the bottom boundary, but you do not include the CMB topography in the geoid computation,
                    // add its zero contributions to the topography CMB storage vector.
                    if (at_upper_surface == false)
                      {
                        for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                          {
                            cmb_stored_values.emplace_back (fe_face_values.quadrature_point(q), std::make_pair(fe_face_values.JxW(q), 0.0));
                          }
                      }
                  }
              }
          }

      std::vector<std::vector<double>> surface_topo_spherical_function;
      std::vector<std::vector<double>> cmb_topography_spherical_function;

      for (const auto &surface_stored_value : surface_stored_values)
        {
          const std::array<double,3> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(surface_stored_value.first);

          // Calculate spherical infinitesimal sin(theta)*d_theta*d_phi by infinitesimal_area/radius^2
          const double infinitesimal = surface_stored_value.second.first/(outer_radius*outer_radius);

          // Theta, phi, spherical infinitesimal, and surface topography
          surface_topo_spherical_function.emplace_back(std::vector<double> {scoord[2],
                                                                            scoord[1],
                                                                            infinitesimal,
                                                                            surface_stored_value.second.second
                                                                           });
        }

      for (const auto &cmb_stored_value : cmb_stored_values)
        {
          const std::array<double,3> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(cmb_stored_value.first);

          // Calculate spherical infinitesimal sin(theta)*d_theta*d_phi by infinitesimal_area/radius^2
          const double infinitesimal = cmb_stored_value.second.first/(inner_radius*inner_radius);

          // Theta, phi, spherical infinitesimal, and CMB dynamic topography
          cmb_topography_spherical_function.emplace_back(std::vector<double> {scoord[2],
                                                                              scoord[1],
                                                                              infinitesimal,
                                                                              cmb_stored_value.second.second
                                                                             });
        }

      TopographyHarmonics result;
      result.surface.average_density = top_layer_average_density;
      result.surface.topography =
        to_spherical_harmonic_coefficients(surface_topo_spherical_function);
      result.cmb.average_density = bottom_layer_average_density;
      result.cmb.topography =
        to_spherical_harmonic_coefficients(cmb_topography_spherical_function);
      return result;
    }


    template <int dim>
    std::pair<std::string,std::string>
    Geoid<dim>::execute (TableHandler &)
    {
      // Current geoid code only works for spherical shell geometry.
      AssertThrow (Plugins::plugin_type_matches<const GeometryModel::SphericalShell<dim>>(this->get_geometry_model())
                   &&
                   dim == 3,
                   ExcMessage("The geoid postprocessor is currently only implemented for the 3d spherical shell geometry model."));

      Utilities::create_directory (this->get_output_directory() + "geoid/",
                                   this->get_mpi_communicator(),
                                   /* silent=*/true);

      const GeometryModel::SphericalShell<dim> &geometry_model =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>> (this->get_geometry_model());

      // Get the value of the outer radius and inner radius.
      const double outer_radius = geometry_model.outer_radius();
      const double inner_radius = geometry_model.inner_radius();

      const types::boundary_id top_boundary_id = geometry_model.translate_symbolic_boundary_name_to_id("top");

      // Get the value of the surface gravity acceleration from the gravity model.
      Point<dim> surface_point;
      surface_point[0] = outer_radius;
      const double surface_gravity = this->get_gravity_model().gravity_vector(surface_point).norm();

      // Get fully scaled spherical-harmonic geoid coefficients from the
      // mantle-density contribution.
      const SphericalHarmonicCoefficients density_geoid_coefficients =
        compute_density_geoid_harmonics(*this, minimum_degree, maximum_degree,
                                        outer_radius, surface_gravity);

      // Initialize the surface and CMB density contrasts with NaNs because they may be unused in case of no topography contribution.
      double surface_delta_rho = numbers::signaling_nan<double>();
      double cmb_density_contrast = numbers::signaling_nan<double>();

      // Get the spherical harmonic coefficients of the surface and CMB topography.
      const TopographyHarmonics topography_harmonics =
        topography_contribution(outer_radius, inner_radius);

      // Get the density contrast at the surface and CMB to replace the initialized NaN values.
      // The surface and CMB density contrasts will be used later to calculate geoid,
      // and the spherical harmonic output of the surface and CMB topography contribution to geoid.
      surface_delta_rho = topography_harmonics.surface.average_density - density_above;
      cmb_density_contrast = density_below - topography_harmonics.cmb.average_density;

      // The boundary projection and degree scaling are deliberately separate:
      // free-surface and CBF paths prepare the same coefficient-space input,
      // and the shared scaling routines are also used by adjoint objectives.
      const SphericalHarmonicCoefficients surface_topography_geoid_coefficients =
        scale_surface_topography_geoid_harmonics(
          topography_harmonics.surface.topography,
          outer_radius,
          surface_gravity,
          surface_delta_rho);
      const SphericalHarmonicCoefficients cmb_topography_geoid_coefficients =
        scale_cmb_topography_geoid_harmonics(
          topography_harmonics.cmb.topography,
          outer_radius,
          inner_radius,
          surface_gravity,
          cmb_density_contrast);

      geoid_coefficients = density_geoid_coefficients;
      for (unsigned int index = 0; index < geoid_coefficients.size(); ++index)
        {
          geoid_coefficients.cosine(index) +=
            surface_topography_geoid_coefficients.cosine(index)
            + cmb_topography_geoid_coefficients.cosine(index);
          geoid_coefficients.sine(index) +=
            surface_topography_geoid_coefficients.sine(index)
            + cmb_topography_geoid_coefficients.sine(index);
        }

      const QMidpoint<dim-1> quadrature_formula_face_center;
      Assert(quadrature_formula_face_center.size() == 1, ExcInternalError());
      FEFaceValues<dim> fe_face_center_values (this->get_mapping(),
                                               this->get_fe(),
                                               quadrature_formula_face_center,
                                               update_values |
                                               update_quadrature_points|
                                               update_JxW_values);

      // Define a vector to store the location of the cells along the surface.
      std::vector<Point<dim>> surface_cell_locations;

      // Loop over all the cells to get the locations of the surface cells to prepare for the geoid computation.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          {
            // If the cell is at the top boundary, store the cell's upper face midpoint location.
            for (const unsigned int f : cell->face_indices())
              if (cell->at_boundary(f) && cell->face(f)->boundary_id() == top_boundary_id)
                {
                  fe_face_center_values.reinit(cell,f);
                  const Point<dim> midpoint_at_top_face = fe_face_center_values.get_quadrature_points().at(0);
                  surface_cell_locations.push_back(midpoint_at_top_face);
                  break;
                }
          }

      // Transfer the geocentric coordinates of the surface cells to the surface spherical coordinates (theta,phi)
      std::vector<std::pair<double,double>> surface_cell_spherical_coordinates;
      for (unsigned int i=0; i<surface_cell_locations.size(); ++i)
        {
          const std::array<double,dim> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(surface_cell_locations.at(i));
          const double phi = scoord[1];
          const double theta = scoord[2];
          surface_cell_spherical_coordinates.emplace_back(theta,phi);
        }

      // Compute the grid geoid anomaly based on spherical harmonics.
      std::vector<double> geoid_anomaly;
      for (const auto &surface_cell_spherical_coordinate : surface_cell_spherical_coordinates)
        {
          int ind = 0;
          double geoid_value = 0;
          for (unsigned int ideg =  minimum_degree; ideg < maximum_degree+1; ++ideg)
            {
              for (unsigned int iord = 0; iord < ideg+1; ++iord)
                {
                  // Normalization after Dahlen and Tromp (1986) Appendix B.6.
                  const std::pair<double,double> sph_harm_vals = aspect::Utilities::real_spherical_harmonic(ideg,iord,surface_cell_spherical_coordinate.first,surface_cell_spherical_coordinate.second);
                  const double cos_component = sph_harm_vals.first; // real / cos part
                  const double sin_component = sph_harm_vals.second; // imaginary / sin part

                  geoid_value += geoid_coefficients.cosine(ind) * cos_component
                                 + geoid_coefficients.sine(ind) * sin_component;
                  ++ind;
                }
            }
          geoid_anomaly.push_back(geoid_value);
        }

      // The user can get the spherical harmonic coefficients of the density anomaly contribution if needed
      if (output_density_anomaly_contribution_coefficients == true)
        {
          // Have a stream into which we write the SH coefficients data from density anomaly contribution.
          // The text stream is then later sent to processor 0.
          std::ostringstream output_density_anomaly_contribution_coefficients;

          write_spherical_harmonic_coefficients(
            output_density_anomaly_contribution_coefficients,
            density_geoid_coefficients);

          const std::string density_anomaly_contribution_spherical_harmonic_filename = this->get_output_directory() +
                                                                                       "geoid/density_anomaly_contribution_SH_coefficients." +
                                                                                       dealii::Utilities::int_to_string(this->get_timestep_number(), 5);

          // Because each processor already held all the SH coefficients from density anomaly contribution, we only need to stop by the processor 0 to get the data.
          // On processor 0, collect all the data and put them into the output density anomaly contribution SH coefficients file.
          if (dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              std::ofstream density_anomaly_contribution_spherical_harmonic_file (density_anomaly_contribution_spherical_harmonic_filename);
              density_anomaly_contribution_spherical_harmonic_file << "# "
                                                                   << "degree order cosine_coefficient sine_coefficient"
                                                                   << std::endl;

              // Write out the data on processor 0.
              density_anomaly_contribution_spherical_harmonic_file << output_density_anomaly_contribution_coefficients.str();
            }
        }

      // The user can get the spherical harmonic coefficients of the surface topography contribution if needed
      if (output_surface_topography_contribution_coefficients == true)
        {
          // Have a stream into which we write the SH coefficients data from surface topography contribution.
          // The text stream is then later sent to processor 0.
          std::ostringstream output_surface_topography_contribution_coefficients;

          write_spherical_harmonic_coefficients(
            output_surface_topography_contribution_coefficients,
            surface_topography_geoid_coefficients);

          const std::string surface_topography_contribution_spherical_harmonic_filename = this->get_output_directory() +
                                                                                          "geoid/surface_topography_contribution_SH_coefficients." +
                                                                                          dealii::Utilities::int_to_string(this->get_timestep_number(), 5);

          // Because each processor already held all the SH coefficients from surface topography contribution,
          // we only need to stop by the processor 0 to get the data. On processor 0, collect all the data
          // and put them into the output surface topography contribution SH coefficients file.
          if (dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              std::ofstream surface_topography_contribution_spherical_harmonic_file (surface_topography_contribution_spherical_harmonic_filename);
              surface_topography_contribution_spherical_harmonic_file << "# "
                                                                      << "degree order cosine_coefficient sine_coefficient"
                                                                      << std::endl;
              std::ostringstream output_surface_delta_rho;
              output_surface_delta_rho << surface_delta_rho;
              surface_topography_contribution_spherical_harmonic_file << "surface density contrast(kg/m^3): "
                                                                      << output_surface_delta_rho.str()
                                                                      << std::endl;
              // Write out the data on processor 0
              surface_topography_contribution_spherical_harmonic_file << output_surface_topography_contribution_coefficients.str();
            }
        }

      // The user can get the spherical harmonic coefficients of the CMB topography contribution if needed.
      if (output_cmb_topography_contribution_coefficients == true)
        {
          // Have a stream into which we write the SH coefficients data from CMB topography contribution.
          // The text stream is then later sent to processor 0.
          std::ostringstream output_cmb_topography_contribution_coefficients;

          write_spherical_harmonic_coefficients(
            output_cmb_topography_contribution_coefficients,
            cmb_topography_geoid_coefficients);

          const std::string cmb_topography_contribution_spherical_harmonic_filename = this->get_output_directory() +
                                                                                      "geoid/CMB_topography_contribution_SH_coefficients." +
                                                                                      dealii::Utilities::int_to_string(this->get_timestep_number(), 5);

          // Because each processor already held all the SH coefficients from CMB topography contribution, we only need to stop by the processor 0
          // to get the data. On processor 0, collect all the data and put them into the output CMB topography contribution SH coefficients file.
          if (dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              std::ofstream cmb_topography_contribution_spherical_harmonic_file (cmb_topography_contribution_spherical_harmonic_filename);
              cmb_topography_contribution_spherical_harmonic_file << "# "
                                                                  << "degree order cosine_coefficient sine_coefficient"
                                                                  << std::endl;
              std::ostringstream output_cmb_density_contrast;
              output_cmb_density_contrast << cmb_density_contrast;
              cmb_topography_contribution_spherical_harmonic_file << "CMB density contrast(kg/m^3): "
                                                                  << output_cmb_density_contrast.str()
                                                                  << std::endl;
              // Write out the data on processor 0
              cmb_topography_contribution_spherical_harmonic_file << output_cmb_topography_contribution_coefficients.str();
            }
        }

      // The user can get the spherical harmonic coefficients of the geoid anomaly if needed.
      if (output_geoid_anomaly_coefficients == true)
        {
          // Have a stream into which we write the geoid anomaly SH coefficients data.
          // The text stream is then later sent to processor 0.
          std::ostringstream output_geoid_anomaly_coefficients;

          write_spherical_harmonic_coefficients(output_geoid_anomaly_coefficients,
                                                geoid_coefficients);

          const std::string geoid_anomaly_spherical_harmonic_filename = this->get_output_directory() +
                                                                        "geoid/geoid_anomaly_SH_coefficients." +
                                                                        dealii::Utilities::int_to_string(this->get_timestep_number(), 5);

          // Because each processor already held all the geoid anomaly SH coefficients, we only need to stop by the processor 0 to get the data.
          // On processor 0, collect all the data and put them into the output geoid anomaly SH coefficients file.
          if (dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              std::ofstream geoid_anomaly_spherical_harmonic_file (geoid_anomaly_spherical_harmonic_filename);
              geoid_anomaly_spherical_harmonic_file << "# "
                                                    << "degree order cosine_coefficient sine_coefficient"
                                                    << std::endl;

              // Write out the data on processor 0.
              geoid_anomaly_spherical_harmonic_file << output_geoid_anomaly_coefficients.str();
            }
        }

      // Have a stream into which we write the geoid height data. the text stream is then
      // later sent to processor 0.
      std::ostringstream output;

      // On processor 0, write the header lines
      if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
        {
          output << "# "
                 << ((output_in_lat_lon == true)? "longitude latitude" : "x y z")
                 << " geoid_anomaly" << std::endl;
        }

      // Prepare the output data.
      if (output_in_lat_lon == true)
        {
          double lon, lat;
          for (unsigned int i=0; i<surface_cell_spherical_coordinates.size(); ++i)
            {
              // Transfer the spherical coordinates to geographical coordinates.
              lat = 90. - surface_cell_spherical_coordinates.at(i).first * constants::radians_to_degree;
              lon = (surface_cell_spherical_coordinates.at(i).second <= numbers::PI
                     ?
                     surface_cell_spherical_coordinates.at(i).second * constants::radians_to_degree
                     :
                     surface_cell_spherical_coordinates.at(i).second * constants::radians_to_degree - 360.);

              // Write the solution to the stream output.
              output << lon
                     << ' '
                     << lat
                     << ' '
                     << geoid_anomaly.at(i)
                     << std::endl;
            }
        }
      else
        {
          for (unsigned int i=0; i<surface_cell_locations.size(); ++i)
            {
              // Write the solution to the stream output.
              output << surface_cell_locations.at(i)
                     << ' '
                     << geoid_anomaly.at(i)
                     << std::endl;
            }
        }

      const std::string filename = this->get_output_directory() +
                                   "geoid/geoid_anomaly." +
                                   dealii::Utilities::int_to_string(this->get_timestep_number(), 5);

      Utilities::collect_and_write_file_content(filename, output.str(), this->get_mpi_communicator());

      // Prepare the free-air gravity anomaly output.
      if (output_gravity_anomaly == true)
        {
          // Have a stream into which we write the gravity anomaly data. the text stream is then
          // later sent to processor 0.
          std::ostringstream output_gravity_anomaly;

          // On processor 0, write the header lines:
          if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              output_gravity_anomaly << "# "
                                     << ((output_in_lat_lon == true)? "longitude latitude" : "x y z")
                                     << " gravity_anomaly" << std::endl;
            }

          // Compute the grid gravity anomaly based on spherical harmonics.
          std::vector<double> gravity_anomaly;
          gravity_anomaly.reserve(surface_cell_spherical_coordinates.size());

          for (const auto &surface_cell_spherical_coordinate : surface_cell_spherical_coordinates)
            {
              int ind = 0;
              double gravity_value = 0;
              for (unsigned int ideg =  minimum_degree; ideg < maximum_degree+1; ++ideg)
                {
                  for (unsigned int iord = 0; iord < ideg+1; ++iord)
                    {
                      // Normalization after Dahlen and Tromp (1986) Appendix B.6.
                      const std::pair<double,double> sph_harm_vals = aspect::Utilities::real_spherical_harmonic(ideg,iord,surface_cell_spherical_coordinate.first,surface_cell_spherical_coordinate.second);
                      const double cos_component = sph_harm_vals.first; // real / cos part
                      const double sin_component = sph_harm_vals.second; // imaginary / sin part

                      // The conversion from geoid to gravity anomaly is given by gravity_anomaly = (l-1)*g/R_surface * geoid_anomaly
                      // based on Forte (2007) equation [97].
                      gravity_value += (geoid_coefficients.cosine(ind) * cos_component
                                        + geoid_coefficients.sine(ind) * sin_component)
                                       * (ideg - 1) * surface_gravity / outer_radius;
                      ++ind;
                    }
                }
              gravity_anomaly.push_back(gravity_value);
            }

          // Prepare the output data.
          if (output_in_lat_lon == true)
            {
              double lon, lat;
              for (unsigned int i=0; i<surface_cell_spherical_coordinates.size(); ++i)
                {
                  // Transfer the spherical coordinates to geographical coordinates.
                  lat = 90. - surface_cell_spherical_coordinates.at(i).first * constants::radians_to_degree;
                  lon = (surface_cell_spherical_coordinates.at(i).second <= numbers::PI
                         ?
                         surface_cell_spherical_coordinates.at(i).second * constants::radians_to_degree
                         :
                         surface_cell_spherical_coordinates.at(i).second * constants::radians_to_degree - 360.);

                  // Write the solution to the stream output.
                  output_gravity_anomaly << lon
                                         << ' '
                                         << lat
                                         << ' '
                                         << gravity_anomaly.at(i)
                                         << std::endl;
                }
            }
          else
            {
              for (unsigned int i=0; i<surface_cell_locations.size(); ++i)
                {
                  // Write the solution to the stream output.
                  output_gravity_anomaly << surface_cell_locations.at(i)
                                         << ' '
                                         << gravity_anomaly.at(i)
                                         << std::endl;
                }
            }

          const std::string filename = this->get_output_directory() +
                                       "geoid/gravity_anomaly." +
                                       dealii::Utilities::int_to_string(this->get_timestep_number(), 5);

          Utilities::collect_and_write_file_content(filename, output_gravity_anomaly.str(), this->get_mpi_communicator());
        }

      return std::pair<std::string,std::string>("Writing geoid anomaly:",
                                                filename);
    }

    template <int dim>
    std::list<std::string>
    Geoid<dim>::required_other_postprocessors() const
    {
      std::list<std::string> deps;

      if ( (include_surface_topography_contribution == true && use_free_surface_topography == false) || (include_cmb_topography_contribution == true && use_free_cmb_topography == false) )
        deps.emplace_back("dynamic topography");

      deps.emplace_back("boundary densities");

      return deps;
    }

    template <int dim>
    double
    Geoid<dim>::evaluate (const Point<dim> &/*p*/) const
    {
      Assert(false, ExcNotImplemented());
      return 0;
    }

    template <>
    double
    Geoid<3>::evaluate (const Point<3> &p) const
    {
      const std::array<double,3> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(p);
      const double theta = scoord[2];
      const double phi = scoord[1];
      return geoid_coefficients.evaluate(theta, phi);
    }

    template <int dim>
    void
    Geoid<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Geoid");
        {
          prm.declare_entry("Include surface topography contribution", "true",
                            Patterns::Bool(),
                            "Option to include the contribution from surface topography on geoid. The default is true.");
          prm.declare_entry("Include CMB topography contribution", "true",
                            Patterns::Bool(),
                            "Option to include the contribution from CMB topography on geoid. The default is true.");
          prm.declare_entry("Maximum degree","20",
                            Patterns::Integer (0),
                            "This parameter can be a random positive integer. However, the value normally should not exceed the maximum "
                            "degree of the initial perturbed temperature field. For example, if the initial temperature uses S40RTS, the "
                            "maximum degree should not be larger than 40.");
          prm.declare_entry("Minimum degree","2",
                            Patterns::Integer (0),
                            "This parameter normally is set to 2 since the perturbed gravitational potential at degree 1 always vanishes "
                            "in a reference frame with the planetary center of mass same as the center of figure.");
          prm.declare_entry("Output data in geographical coordinates", "false",
                            Patterns::Bool(),
                            "Option to output the geoid anomaly in geographical coordinates (latitude and longitude). "
                            "The default is false, so the postprocessor will output the data in geocentric coordinates (x,y,z) as normally.");
          prm.declare_entry("Density above","0.",
                            Patterns::Double (0.),
                            "The density value above the surface boundary.");
          prm.declare_entry("Density below","9900.",
                            Patterns::Double (0.),
                            "The density value below the CMB boundary.");
          prm.declare_entry("Output geoid anomaly coefficients", "false",
                            Patterns::Bool(),
                            "Option to output the spherical harmonic coefficients of the geoid anomaly up to the maximum degree. "
                            "The default is false, so the postprocessor will only output the geoid anomaly in grid format. ");
          prm.declare_entry("Output surface topography contribution coefficients", "false",
                            Patterns::Bool(),
                            "Option to output the spherical harmonic coefficients of the surface topography contribution "
                            "to the maximum degree. The default is false. ");
          prm.declare_entry("Output CMB topography contribution coefficients", "false",
                            Patterns::Bool(),
                            "Option to output the spherical harmonic coefficients of the CMB topography contribution "
                            "to the maximum degree. The default is false. ");
          prm.declare_entry("Output density anomaly contribution coefficients", "false",
                            Patterns::Bool(),
                            "Option to output the spherical harmonic coefficients of the density anomaly contribution to the "
                            "maximum degree. The default is false. ");
          prm.declare_entry("Output gravity anomaly", "false",
                            Patterns::Bool(),
                            "Option to output the free-air gravity anomaly up to the maximum degree. "
                            "The unit of the output is in SI, hence $m/s^2$ ($1mgal = 10^-5 m/s^2$). The default is false. ");

          prm.declare_alias("Output gravity anomaly","Also output the gravity anomaly");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

    template <int dim>
    void
    Geoid<dim>::parse_parameters (ParameterHandler &prm)
    {
      CitationInfo::add("geoid");

      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Geoid");
        {
          include_surface_topography_contribution = prm.get_bool ("Include surface topography contribution");
          include_cmb_topography_contribution = prm.get_bool ("Include CMB topography contribution");
          maximum_degree = prm.get_integer ("Maximum degree");
          minimum_degree = prm.get_integer ("Minimum degree");
          output_in_lat_lon = prm.get_bool ("Output data in geographical coordinates");
          density_above = prm.get_double ("Density above");
          density_below = prm.get_double ("Density below");
          output_geoid_anomaly_coefficients = prm.get_bool ("Output geoid anomaly coefficients");
          output_surface_topography_contribution_coefficients = prm.get_bool ("Output surface topography contribution coefficients");
          output_cmb_topography_contribution_coefficients = prm.get_bool ("Output CMB topography contribution coefficients");
          output_density_anomaly_contribution_coefficients = prm.get_bool ("Output density anomaly contribution coefficients");
          output_gravity_anomaly = prm.get_bool ("Output gravity anomaly");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

    template <int dim>
    void
    Geoid<dim>::initialize ()
    {
      // Find if the included boundaries are active free surfaces
      if (include_surface_topography_contribution == true || include_cmb_topography_contribution == true)
        {
          if (this->get_parameters().mesh_deformation_enabled == true)
            {
              const  types::boundary_id surface_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
              const  types::boundary_id bottom_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

              const std::set<types::boundary_id> mesh_deformation_boundaries = this->get_mesh_deformation_handler().get_active_mesh_deformation_boundary_indicators();

              use_free_surface_topography = mesh_deformation_boundaries.find(surface_id) != mesh_deformation_boundaries.end();
              use_free_cmb_topography = mesh_deformation_boundaries.find(bottom_id) != mesh_deformation_boundaries.end();
            }
        }
    }

  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(Geoid,
                                  "geoid",
                                  "A postprocessor that computes a representation of "
                                  "the geoid based on the density structure in the mantle, "
                                  "as well as the topography at the surface and "
                                  "core mantle boundary (CMB) if desired. The topography is based on the "
                                  "dynamic topography postprocessor in case of no free surface, "
                                  "and based on the real surface from the geometry model in case "
                                  "of a free surface. The geoid is computed "
                                  "from a spherical harmonic expansion, so the geometry "
                                  "of the domain must be a 3d spherical shell.")
  }
}
