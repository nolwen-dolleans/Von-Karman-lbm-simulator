#include <lbm/physics.hpp>

#include <cassert>
#include <cstdlib>

#include <omp.h>

// #include <lbm/communications.hpp>
#include <lbm/config.hpp>
#include <lbm/structures.hpp>

#if DIRECTIONS == 9 && DIMENSIONS == 2
/// Definition of the 9 base vectors used to discretize the directions on each mesh.
const Vector direction_matrix[DIRECTIONS] = {
  // clang-format off
  {+0.0, +0.0},
  {+1.0, +0.0}, {+0.0, +1.0}, {-1.0, +0.0}, {+0.0, -1.0},
  {+1.0, +1.0}, {-1.0, +1.0}, {-1.0, -1.0}, {+1.0, -1.0},
  // clang-format on
};
#else
#error Need to define adapted direction matrix.
#endif

#if DIRECTIONS == 9
/// Weigths used to compensate the differences in lenght of the 9 directional vectors.
const double equil_weight[DIRECTIONS] = {
  // clang-format off
  4.0 / 9.0,
  1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
  // clang-format on
};

/// Opposite directions for bounce back implementation
const int opposite_of[DIRECTIONS] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
#else
#error Need to define adapted equilibrium distribution function
#endif


double get_vect_norm_2(Vector const a, Vector const b) {
  double res = 0.0;
  for (size_t k = 0; k < DIMENSIONS; k++) {
    res += a[k] * b[k];
  }
  return res;
}

double get_cell_density(const lbm_mesh_cell_t cell) {
  assert(cell != NULL);
  double res = 0.0;
  for (size_t k = 0; k < DIRECTIONS; k++) {
    res += cell[k];
  }
  return res;
}

double total_time = 0.0;
long long call_count = 0;

void afficher_stats() {
	using namespace std::chrono;
	
	fprintf(stderr, "Appels : %d\n", call_count);
	fprintf(stderr, "Temps total : %f  s\n", total_time);
}

void get_cell_velocity(Vector v, const lbm_mesh_cell_t cell, double cell_density) {
  assert(v != NULL);
  assert(cell != NULL);
	
	const double inv_cell_density = 1.0 / cell_density;
	v[0] = 0.0, v[1] = 0.0;

	for (size_t k = 0; k < DIRECTIONS; k++) {
		v[0] += cell[k] * direction_matrix[k][0];
		v[1] += cell[k] * direction_matrix[k][1];
	}
	v[0] *= inv_cell_density;
	v[1] *= inv_cell_density;
}

double compute_equilibrium_profile(const Vector& velocity, double density, int direction, const double &v2) {
  //const double v2 = get_vect_norm_2(velocity, velocity);

  // Compute `e_i * v_i / c`
  const double p  = get_vect_norm_2(direction_matrix[direction], velocity);
  const double p2 = p * p;

  // Terms without density and direction weight
  double f_eq = (1.0 + (3.0 * p) + ((9.0 / 2.0) * p2) - ((3.0 / 2.0) * v2)) * equil_weight[direction] * density;

  return f_eq;
}

inline void compute_cell_collision(lbm_mesh_cell_t &cell_out, const lbm_mesh_cell_t &cell_in, const Vector& v, double v2, double density) {
  // Loop on microscopic directions
  double f_eq;
  for (size_t k = 0; k < DIRECTIONS; k++) {
    // Compute f at equilibrium
    f_eq = compute_equilibrium_profile(v, density, k, v2);
    // Compute f_out
    cell_out[k] = cell_in[k] - RELAX_PARAMETER * (cell_in[k] - f_eq);
  }
}

void compute_bounce_back(lbm_mesh_cell_t &cell) {
  double tmp[DIRECTIONS];
  for (size_t k = 0; k < DIRECTIONS; k++) {
    tmp[k] = cell[opposite_of[k]];
  }
  for (size_t k = 0; k < DIRECTIONS; k++) {
    cell[k] = tmp[k];
  }
}

double helper_compute_poiseuille(const size_t i, const size_t size) {
  const double y = (double)(i - 1);
  const double L = (double)(size - 1);
  return 4.0 * INFLOW_MAX_VELOCITY / (L * L) * (L * y - y * y);
}

void compute_inflow_zou_he_poiseuille_distr(const Mesh* mesh, lbm_mesh_cell_t cell, size_t id_y) {
#if DIRECTIONS != 9
#error Implemented only for 9 directions
#endif

  // Set macroscopic fluid info
  // Poiseuille distribution on X and null on Y
  // We just want the norm, so `v = v_x`
  const double v = helper_compute_poiseuille(id_y, mesh->height);

  // Compute rho from U and inner flow on surface
  const double rho = (cell[0] + cell[2] + cell[4] + 2 * (cell[3] + cell[6] + cell[7])) / (1.0 - v);

  // Now compute unknown microscopic values
  cell[1] = cell[3]; // + (2.0/3.0) * density * v_y <--- no velocity on Y so v_y = 0
  cell[5] = cell[7] - (1.0 / 2.0) * (cell[2] - cell[4])
            + (1.0 / 6.0) * (rho * v); // + (1.0/2.0) * rho * v_y    <--- no velocity on Y so v_y = 0
  cell[8] = cell[6] + (1.0 / 2.0) * (cell[2] - cell[4])
            + (1.0 / 6.0) * (rho * v); //- (1.0/2.0) * rho * v_y    <--- no velocity on Y so v_y = 0

  // No need to copy already known one as the value will be "loss" in the wall at propagatation time
}

void compute_outflow_zou_he_const_density(lbm_mesh_cell_t cell) {
#if DIRECTIONS != 9
#error Implemented only for 9 directions
#endif

  double const rho = 1.0;
  // Compute macroscopic velocity depending on inner flow going onto the wall
  const double v = -1.0 + (1.0 / rho) * (cell[0] + cell[2] + cell[4] + 2 * (cell[1] + cell[5] + cell[8]));

  // Now can compute unknown microscopic values
  cell[3] = cell[1] - (2.0 / 3.0) * rho * v;
  cell[7] = cell[5]
            + (1.0 / 2.0) * (cell[2] - cell[4])
            // - (1.0/2.0) * (rho * v_y)    <--- no velocity on Y so v_y = 0
            - (1.0 / 6.0) * (rho * v);
  cell[6] = cell[8]
            + (1.0 / 2.0) * (cell[4] - cell[2])
            // + (1.0/2.0) * (rho * v_y)    <--- no velocity on Y so v_y = 0
            - (1.0 / 6.0) * (rho * v);
}

void special_cells(Mesh* mesh, lbm_mesh_type_t* mesh_type, const lbm_comm_t* mesh_comm) {
  // Loop on all inner cells
  lbm_mesh_cell_t mesh_cell;
  for (size_t j = 1; j < mesh->height - 1; j++) {
    for (size_t i = 1; i < mesh->width - 1; i++) {
		mesh_cell = Mesh_get_cell(mesh, i, j);
      switch (*(lbm_cell_type_t_get_cell(mesh_type, i, j))) {
      case CELL_FUILD:
        break;
      case CELL_BOUNCE_BACK:
        compute_bounce_back(mesh_cell);
        break;
      case CELL_LEFT_IN:
        compute_inflow_zou_he_poiseuille_distr(mesh, mesh_cell, j + mesh_comm->y);
        break;
      case CELL_RIGHT_OUT:
        compute_outflow_zou_he_const_density(mesh_cell);
        break;
      }
    }
  }
}

void collision(Mesh* mesh_out, const Mesh* mesh_in) {
  // Loop on all inner cells
  lbm_mesh_cell_t mesh_cell_in, mesh_cell_out;
  Vector v;
  double density, v2;
  for (size_t j = 1; j < mesh_in->height - 1; j++) {
    for (size_t i = 1; i < mesh_in->width - 1; i++) {
      mesh_cell_in = Mesh_get_cell(mesh_in, i, j);

      density = get_cell_density(mesh_cell_in);
      get_cell_velocity(v, mesh_cell_in, density);
      v2 = get_vect_norm_2(v, v);

      mesh_cell_out = Mesh_get_cell(mesh_out, i, j);
      compute_cell_collision(mesh_cell_out, mesh_cell_in, v, v2, density);
    }
  }
}

void propagation(Mesh* mesh_out, const Mesh* mesh_in) {
  const int width = (int)mesh_out->width-1;
  const int height = (int)mesh_out->height-1;
  int ii, jj, i, j, k;
  lbm_mesh_cell_t cell_out;
  for (j = 1; j < height; ++j) {
    for (i = 1; i < width; ++i) {
      // For all direction
      cell_out = Mesh_get_cell(mesh_out, i, j);
      for (k = 0; k < DIRECTIONS; ++k) {
        // Compute destination point
        ii = i - (int)direction_matrix[k][0];
        jj = j - (int)direction_matrix[k][1];
        // Propagate to neighboor nodes
        cell_out[k] = Mesh_get_cell(mesh_in, ii, jj)[k];
      }
    }
  }
}