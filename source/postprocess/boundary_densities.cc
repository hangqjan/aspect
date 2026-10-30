/*
  Copyright (C) 2011 - 2022 by the authors of the ASPECT code.

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


#include <aspect/postprocess/boundary_densities.h>
#include <aspect/geometry_model/interface.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <limits>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    BoundaryDensities<dim>::execute (TableHandler &statistics)
    {
      const Quadrature<dim-1> &quadrature_formula = this->introspection().face_quadratures.temperature;

      FEFaceValues<dim> fe_face_values (this->get_mapping(),
                                        this->get_fe(),
                                        quadrature_formula,
                                        update_values |
                                        update_gradients |
                                        update_quadrature_points |
                                        update_JxW_values);

      double local_top_density = 0.;
      double local_bottom_density = 0.;
      double local_top_area = 0.;
      double local_bottom_area = 0.;

      typename MaterialModel::Interface<dim>::MaterialModelInputs in(fe_face_values.n_quadrature_points, this->n_compositional_fields());
      typename MaterialModel::Interface<dim>::MaterialModelOutputs out(fe_face_values.n_quadrature_points, this->n_compositional_fields());
      in.requested_properties = MaterialModel::MaterialProperties::density;
      std::vector<std::vector<double>> composition_values(this->n_compositional_fields(), std::vector<double>(fe_face_values.n_quadrature_points));

      const types::boundary_id top_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      const types::boundary_id bottom_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

      // loop over all of the surface cells and if one less than h/3 away from
      // the top or bottom surface, evaluate the density on that face
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          for (const unsigned int f : cell->face_indices())
            {
              bool cell_at_top = false;
              bool cell_at_bottom = false;

              // Test for top or bottom surface cell faces
              if (cell->at_boundary(f) && cell->face(f)->boundary_id() == top_boundary_id)
                cell_at_top = true;
              if (cell->at_boundary(f) && cell->face(f)->boundary_id() == bottom_boundary_id)
                cell_at_bottom = true;

              if (cell_at_top || cell_at_bottom)
                {
                  // handle surface cells
                  fe_face_values.reinit (cell, f);
                  fe_face_values[this->introspection().extractors.temperature]
                  .get_function_values (this->get_solution(), in.temperature);
                  fe_face_values[this->introspection().extractors.pressure]
                  .get_function_values (this->get_solution(), in.pressure);
                  fe_face_values[this->introspection().extractors.velocities]
                  .get_function_symmetric_gradients (this->get_solution(), in.strain_rate);

                  in.position = fe_face_values.get_quadrature_points();

                  for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                    fe_face_values[this->introspection().extractors.compositional_fields[c]]
                    .get_function_values(this->get_solution(),
                                         composition_values[c]);
                  for (unsigned int i=0; i<fe_face_values.n_quadrature_points; ++i)
                    {
                      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                        in.composition[i][c] = composition_values[c][i];
                    }

                  this->get_material_model().evaluate(in, out);

                  // calculate the top/bottom properties
                  if (cell_at_top)
                    for ( unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
                      {
                        local_top_density += out.densities[q] * fe_face_values.JxW(q);
                        local_top_area += fe_face_values.JxW(q);
                      }
                  if (cell_at_bottom)
                    for ( unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
                      {
                        local_bottom_density += out.densities[q] * fe_face_values.JxW(q);
                        local_bottom_area += fe_face_values.JxW(q);
                      }
                }
            }

      // vector for packing local values before MPI summing them
      double values[4] = {local_bottom_area, local_top_area, local_bottom_density, local_top_density};

      Utilities::MPI::sum<double, 4>( values, this->get_mpi_communicator(), values );

      top_density = values[3] / values[1]; // density over area
      bottom_density = values[2] / values[0]; // density over area

      statistics.add_value ("Density at top (kg/m^3)",
                            top_density);
      statistics.add_value ("Density at bottom (kg/m^3)",
                            bottom_density);

      // also make sure that the other columns filled by this object
      // all show up with sufficient accuracy and in scientific notation
      {
        const char *columns[] = { "Density at top (kg/m^3)",
                                  "Density at bottom (kg/m^3)"
                                };
        for (auto &column : columns)
          {
            statistics.set_precision (column, 8);
            statistics.set_scientific (column, true);
          }
      }

      std::ostringstream output;
      output.precision(4);
      output << top_density << " kg/m^3, "
             << bottom_density << " kg/m^3";

      return std::pair<std::string, std::string> ("Density at top/bottom of domain:",
                                                  output.str());
    }

    template <int dim>
    double
    BoundaryDensities<dim>::density_at_top() const
    {
      return top_density;
    }

    template <int dim>
    double
    BoundaryDensities<dim>::density_at_bottom() const
    {
      return bottom_density;
    }



    template <int dim>
    Vector<double>
    compute_boundary_mean_density_transpose(
      const SimulatorAccess<dim> &access,
      const BoundaryMeanDensityBoundary boundary,
      const double scalar_seed)
    {
      const Quadrature<dim-1> &face_quadrature = access.introspection().face_quadratures.temperature;
      FEFaceValues<dim> face_values(access.get_mapping(), access.get_fe(), face_quadrature,
                                    update_JxW_values);
      const unsigned int volume_quadrature_degree =
        access.get_fe().base_element(access.introspection().base_elements.velocities).degree + 1;
      const QGauss<dim> volume_quadrature(volume_quadrature_degree);
      FEValues<dim> volume_values(access.get_mapping(), access.get_fe(), volume_quadrature,
                                  update_JxW_values);
      const types::boundary_id boundary_id = access.get_geometry_model().
                                             translate_symbolic_boundary_name_to_id(boundary == BoundaryMeanDensityBoundary::top
                                                                                    ? "top" : "bottom");

      double local_area = 0.0;
      for (const auto &cell : access.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          for (const unsigned int face : cell->face_indices())
            if (cell->at_boundary(face) && cell->face(face)->boundary_id() == boundary_id)
              {
                face_values.reinit(cell, face);
                for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                  local_area += face_values.JxW(q);
              }
      const double area = Utilities::MPI::sum(local_area, access.get_mpi_communicator());
      AssertThrow(area > std::numeric_limits<double>::min(),
                  ExcMessage("Boundary mean-density transpose found zero selected-boundary area."));

      Vector<double> result(access.get_triangulation().n_active_cells());
      for (const auto &cell : access.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            volume_values.reinit(cell);
            double cell_volume = 0.0;
            for (unsigned int q = 0; q < volume_quadrature.size(); ++q)
              cell_volume += volume_values.JxW(q);
            AssertThrow(cell_volume > std::numeric_limits<double>::min(),
                        ExcMessage("Boundary mean-density transpose encountered a locally owned cell with zero volume."));

            double face_integral = 0.0;
            if (cell->at_boundary())
              for (const unsigned int face : cell->face_indices())
                if (cell->at_boundary(face) && cell->face(face)->boundary_id() == boundary_id)
                  {
                    face_values.reinit(cell, face);
                    for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                      face_integral += face_values.JxW(q);
                  }
            result(cell->active_cell_index()) = scalar_seed * face_integral / area / cell_volume;
          }
      return result;
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    template Vector<double> compute_boundary_mean_density_transpose<2>(
      const SimulatorAccess<2> &, const BoundaryMeanDensityBoundary, const double);
    template Vector<double> compute_boundary_mean_density_transpose<3>(
      const SimulatorAccess<3> &, const BoundaryMeanDensityBoundary, const double);
    ASPECT_REGISTER_POSTPROCESSOR(BoundaryDensities,
                                  "boundary densities",
                                  "A postprocessor that computes the laterally averaged "
                                  "density at the top and bottom of the domain.")
  }
}
