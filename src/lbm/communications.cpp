#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "mpi.h"


#include <lbm/communications.hpp>
// #include <lbm/tpl_loader.hpp>
#include <lbm/physics.hpp>

/// @brief Saves the result of one step of computation.
///
/// This function can be called multiple times when a MPI save on multiple
/// processes happens (e.g. saving them one at a time on each domain).
/// Writes only velocities and macroscopic densities in the form of single
/// precision floating-point numbers.
///
/// @param fp File descriptor to write to.
/// @param mesh Domain to save.
void save_frame(FILE* fp, const Mesh* mesh) {
  // Write buffer to write float instead of double
  lbm_file_entry_t buffer[WRITE_BUFFER_ENTRIES];
  // Loop on all values
  size_t cnt = 0;
  for (size_t i = 1; i < mesh->width - 1; i++) {
	for (size_t j = 1; j < mesh->height - 1; j++) {
	  // Compute macroscopic values
	  const double density = get_cell_density(Mesh_get_cell(mesh, i, j));
	  Vector v;
	  get_cell_velocity(v, Mesh_get_cell(mesh, i, j), density);
	  const double norm = std::sqrt(get_vect_norm_2(v, v));
	  // Fill buffer
	  buffer[cnt].rho = density;
	  buffer[cnt].v   = norm;
	  cnt++;
	  assert(cnt <= WRITE_BUFFER_ENTRIES);
	  // Flush buffer if full
	  if (cnt == WRITE_BUFFER_ENTRIES) {
		fwrite(buffer, sizeof(lbm_file_entry_t), cnt, fp);
		cnt = 0;
	  }
	}
  }
  // Final flush
  if (cnt != 0) {
	fwrite(buffer, sizeof(lbm_file_entry_t), cnt, fp);
  }
}
static int lbm_helper_pgcd(int a, int b) {
  int c;
  while (b != 0) {
	c = a % b;
	a = b;
	b = c;
  }
  return a;
}
// static int PMPI_Syncall_cb(MPI_Comm comm) {
//   static int (*__builtin_fence_ps)() = rt_tpl_sync(comm, __builtin_fence_ps, MPI_HINT_VTBL);
//   return __builtin_fence_ps();
// }
static int helper_get_rank_id(int nb_x, int nb_y, int rank_x, int rank_y) {
  if (rank_x < 0 || rank_x >= nb_x) {
	return -1;
  } else if (rank_y < 0 || rank_y >= nb_y) {
	return -1;
  } else {
	return (rank_x + rank_y * nb_x);
  }
}

void lbm_comm_print(const lbm_comm_t* mesh_comm) {
  int rank;
  MPI_Comm_rank(mesh_comm->cart_comm, &rank);

  static bool first_call = true;
  if (first_call && rank == RANK_MASTER) {
	first_call = false;
	fprintf(
	  stderr,
	  "%4s| %8s %8s %8s %8s | %12s %12s %12s %12s | %6s %6s | %6s %6s\n",
	  "RANK",
	  "TOP",
	  "BOTTOM",
	  "LEFT",
	  "RIGHT",
	  "TOP LEFT",
	  "TOP RIGHT",
	  "BOTTOM LEFT",
	  "BOTTOM RIGHT",
	  "POS X",
	  "POS Y",
	  "DIM X",
	  "DIM Y"
	);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  fprintf(
	stderr,
	"%4d| %7d  %7d  %7d  %7d  | %11d  %11d  %11d  %11d  | %5d  %5d  | %5d  %5d \n",
	rank,
	mesh_comm->top_id,
	mesh_comm->bottom_id,
	mesh_comm->left_id,
	mesh_comm->right_id,
	mesh_comm->corner_id[CORNER_TOP_LEFT],
	mesh_comm->corner_id[CORNER_TOP_RIGHT],
	mesh_comm->corner_id[CORNER_BOTTOM_LEFT],
	mesh_comm->corner_id[CORNER_BOTTOM_RIGHT],
	mesh_comm->x,
	mesh_comm->y,
	mesh_comm->width,
	mesh_comm->height
  );
}
// static MPI_Syncfunc_t* MPI_Syncall = PMPI_Syncall_cb;

void lbm_comm_init(lbm_comm_t* mesh_comm, int rank, int comm_size, uint32_t width, uint32_t height) {
  // Compute splitting
	const float target   = std::sqrt((float)comm_size * width / height);

	float best  = MAXFLOAT;
	int nb_x  = 1, nb_y = comm_size;

	for (int i = 1; i <= comm_size; ++i) {
		if (comm_size % i != 0)  continue;
		int j = comm_size / i;
		if (width  % i != 0)     continue;
		if (height % j != 0)     continue;

		float diff = std::fabs(j - target);
		if (diff < best) {
			best = diff;
			nb_x = i;
			nb_y = j;
		}
	}
	
	
	fprintf(stderr, "%d x %d\n", nb_x, nb_y);
	
	int dim[2];
	int coord[2];
	dim[0] = nb_x;
	dim[1] = nb_y;
	
	int periodical[2] = {0,0};			//0 because the board doesn't need to communicate between each others
	int reorder = 1;
	
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim, periodical, reorder, &mesh_comm->cart_comm);
	
	
	
	

  assert(nb_x * nb_y == comm_size);
  if (height % nb_y != 0) {
	fatal("Can't get a 2D cut for current problem size and number of processes.");
  }
	int r;
	MPI_Comm_rank(mesh_comm->cart_comm, &r);
	MPI_Cart_coords(mesh_comm->cart_comm, r, 2, coord);
  // Compute current rank position (ID)
  int rank_x = coord[0];
  int rank_y = coord[1];

  // Setup nb
  mesh_comm->nb_x = nb_x;
  mesh_comm->nb_y = nb_y;

  // Setup size (+2 for ghost cells on border)
  mesh_comm->width  = width / nb_x + 2;
  mesh_comm->height = height / nb_y + 2;

  // Setup position
  mesh_comm->x = rank_x * width / nb_x;
  mesh_comm->y = rank_y * height / nb_y;

  // Compute neighbour nodes id
	
	MPI_Cart_shift(mesh_comm->cart_comm, 0, 1, &mesh_comm->left_id, &mesh_comm->right_id);
	MPI_Cart_shift(mesh_comm->cart_comm, 1, 1, &mesh_comm->top_id,  &mesh_comm->bottom_id);
	
	
//  mesh_comm->corner_id[CORNER_TOP_LEFT]     = helper_get_rank_id(nb_x, nb_y, rank_x - 1, rank_y - //1);
//  mesh_comm->corner_id[CORNER_TOP_RIGHT]    = helper_get_rank_id(nb_x, nb_y, rank_x + 1, rank_y - //1);
//  mesh_comm->corner_id[CORNER_BOTTOM_LEFT]  = helper_get_rank_id(nb_x, nb_y, rank_x - 1, rank_y + //1);
//  mesh_comm->corner_id[CORNER_BOTTOM_RIGHT] = helper_get_rank_id(nb_x, nb_y, rank_x + 1, rank_y + 1);
	auto cart_neighbour = [&](int dx, int dy) -> int {
		int t[2] = { coord[0] + dx, coord[1] + dy };
		if (t[0] < 0 || t[0] >= nb_x) return MPI_PROC_NULL;
		if (t[1] < 0 || t[1] >= nb_y) return MPI_PROC_NULL;
		int r;
		MPI_Cart_rank(mesh_comm->cart_comm, t, &r);
		return r;
	};

	mesh_comm->corner_id[CORNER_TOP_LEFT]     = cart_neighbour(-1, -1);
	mesh_comm->corner_id[CORNER_TOP_RIGHT]    = cart_neighbour(+1, -1);
	mesh_comm->corner_id[CORNER_BOTTOM_LEFT]  = cart_neighbour(-1, +1);
	mesh_comm->corner_id[CORNER_BOTTOM_RIGHT] = cart_neighbour(+1, +1);
  // If more than 1 on y, need transmission buffer
  if (nb_y > 1) {
	mesh_comm->buffer = static_cast<double*>(malloc(sizeof(double) * DIRECTIONS * width / nb_x));
  } else {
	mesh_comm->buffer = NULL;
  }
	
	
	MPI_Type_vector(
		mesh_comm->height - 2,
		DIRECTIONS,
		mesh_comm->width * DIRECTIONS,
		MPI_DOUBLE,
		&mesh_comm->col_type
	);
	MPI_Type_commit(&mesh_comm->col_type);
	

  lbm_comm_print(mesh_comm);
}

void lbm_comm_release(lbm_comm_t* mesh_comm) {
  mesh_comm->x        = 0;
  mesh_comm->y        = 0;
  mesh_comm->width    = 0;
  mesh_comm->height   = 0;
  mesh_comm->right_id = -1;
  mesh_comm->left_id  = -1;
  if (mesh_comm->buffer != NULL) {
	free(mesh_comm->buffer);
  }
}

/// @brief Start of the horizontal asynchronous communications.
/// @param mesh_comm Mesh communicator to use.
/// @param mesh_to_process Mesh to use when exchanging phantom meshes.
/// @param target_rank Rank to communicate with.
/// @param x X coordinate to use.
static void lbm_comm_sync_ghosts_horizontal(
	lbm_comm_t* mesh, Mesh* mesh_to_process,
	lbm_comm_type_t comm_type, int target_rank,
	uint32_t x, MPI_Request* request)
{
	if (target_rank == MPI_PROC_NULL || target_rank == -1) {
		*request = MPI_REQUEST_NULL;
		return;
	}
	


	MPI_Status status;
	double* col = Mesh_get_cell(mesh_to_process, x, 1);
	//MPI_Request request;
	switch (comm_type) {
		case COMM_SEND:
			MPI_Isend(col, 1, mesh->col_type, target_rank, 0,
					  mesh->cart_comm, request);
			break;
		case COMM_RECV:
			MPI_Irecv(col, 1, mesh->col_type, target_rank, 0,
					  mesh->cart_comm, request);
			break;
		default:
			fatal("unknown type of communication");
		}
	
}

/// @brief Start of the diagonal asynchronous communications.
/// @param mesh_comm Mesh communicator to use.
/// @param mesh_to_process Mesh to use when exchanging phantom meshes.
/// @param target_rank Rank to communicate with.
/// @param x X coordinate to use.
/// @param y Y coordinate to use.
static void lbm_comm_sync_ghosts_diagonal(
  lbm_comm_t* mesh_comm,
  Mesh* mesh_to_process,
  lbm_comm_type_t comm_type,
  int target_rank,
  uint32_t x,
  uint32_t y,
  MPI_Request *request
) {
  // If target is -1, no comm
  if (target_rank == MPI_PROC_NULL) {
	  *request = MPI_REQUEST_NULL;
	return;
  }

  switch (comm_type) {
  case COMM_SEND:
	MPI_Isend(Mesh_get_cell(mesh_to_process, x, y), DIRECTIONS, MPI_DOUBLE, target_rank, 0, mesh_comm->cart_comm, request);
	break;
  case COMM_RECV:
	MPI_Irecv(Mesh_get_cell(mesh_to_process, x, y), DIRECTIONS, MPI_DOUBLE, target_rank, 0, mesh_comm->cart_comm, request);
	break;
  default:
	fatal("unknown type of communication");
  }
}

/// @brief Start of the vertical asynchronous communications.
/// @param mesh_comm Mesh communicator to use.
/// @param mesh_to_process Mesh to use when exchanging phantom meshes.
/// @param target_rank Rank to communicate with.
/// @param y Y coordinate to use.
static void
lbm_comm_sync_ghosts_vertical(lbm_comm_t* mesh_comm, Mesh* mesh_to_process, lbm_comm_type_t comm_type, int target_rank, uint32_t y, MPI_Request *request) {
  // if target is -1, no comm
  if (target_rank == MPI_PROC_NULL) {
	  *request = MPI_REQUEST_NULL;
	return;
  }
	
	lbm_mesh_cell_t row = Mesh_get_cell(mesh_to_process, 1, y);
  MPI_Status status;
  //MPI_Request request;
  switch (comm_type) {
  case COMM_SEND:
		MPI_Isend(row,  (mesh_to_process->width - 2) * DIRECTIONS, MPI_DOUBLE, target_rank, 42, mesh_comm->cart_comm, request);
	break;
  case COMM_RECV:
		MPI_Irecv(
		  row,
		  (mesh_to_process->width - 2) * DIRECTIONS,
		  MPI_DOUBLE,
		  target_rank,
		  42,
				  mesh_comm->cart_comm,
		  request
		);
	break;
  default:
	fatal("unknown type of communication");
  }
}

void lbm_comm_halo_exchange(lbm_comm_t* mesh, Mesh* mesh_to_process, MPI_Request* reqs_v) {
  int rank;
  MPI_Comm_rank(mesh->cart_comm, &rank);

  // Left to right phase
  lbm_comm_sync_ghosts_horizontal(mesh, mesh_to_process, COMM_SEND, mesh->right_id, mesh->width - 2, &reqs_v[0]);
  lbm_comm_sync_ghosts_horizontal(mesh, mesh_to_process, COMM_RECV, mesh->left_id, 0, &reqs_v[1]);
  // Prevent comm mixing to avoid bugs
  //MPI_Barrier(mesh->cart_comm);

  // Right to left phase
  lbm_comm_sync_ghosts_horizontal(mesh, mesh_to_process, COMM_SEND, mesh->left_id, 1, &reqs_v[2]);
  lbm_comm_sync_ghosts_horizontal(mesh, mesh_to_process, COMM_RECV, mesh->right_id, mesh->width - 1, &reqs_v[3]);
  // Prevent comm mixing to avoid bugs
  //MPI_Barrier(mesh->cart_comm);

  // Top to bottom phase
  lbm_comm_sync_ghosts_vertical(mesh, mesh_to_process, COMM_SEND, mesh->bottom_id, mesh->height - 2, &reqs_v[4]);
  lbm_comm_sync_ghosts_vertical(mesh, mesh_to_process, COMM_RECV, mesh->top_id, 0, &reqs_v[5]);
  // Prevent comm mixing to avoid bugs
  //MPI_Barrier(mesh->cart_comm);

  // Bottom to top phase
  lbm_comm_sync_ghosts_vertical(mesh, mesh_to_process, COMM_SEND, mesh->top_id, 1, &reqs_v[6]);
  lbm_comm_sync_ghosts_vertical(mesh, mesh_to_process, COMM_RECV, mesh->bottom_id, mesh->height - 1, &reqs_v[7]);
  // Prevent comm mixing to avoid bugs
  //MPI_Barrier(mesh->cart_comm);

  // Top left phase
  lbm_comm_sync_ghosts_diagonal(mesh, mesh_to_process, COMM_SEND, mesh->corner_id[CORNER_TOP_LEFT], 1, 1, &reqs_v[8]);
  lbm_comm_sync_ghosts_diagonal(mesh,
	mesh_to_process,
	COMM_RECV,
	mesh->corner_id[CORNER_BOTTOM_RIGHT],
	mesh->width - 1,
	mesh->height - 1,
	&reqs_v[9]
  );
  // Prevent comm mixing to avoid bugs
  //MPI_Barrier(mesh->cart_comm);

  // Bottom left phase
  lbm_comm_sync_ghosts_diagonal(mesh, mesh_to_process, COMM_SEND, mesh->corner_id[CORNER_BOTTOM_LEFT], 1, mesh->height - 2, &reqs_v[10]);
  lbm_comm_sync_ghosts_diagonal(mesh, mesh_to_process, COMM_RECV, mesh->corner_id[CORNER_TOP_RIGHT], mesh->width - 1, 0, &reqs_v[11]);
  // Prevent comm mixing to avoid bugs
  //MPI_Barrier(mesh->cart_comm);

  // Top right phase
  lbm_comm_sync_ghosts_diagonal(mesh, mesh_to_process, COMM_SEND, mesh->corner_id[CORNER_TOP_RIGHT], mesh->width - 2, 1, &reqs_v[12]);
  lbm_comm_sync_ghosts_diagonal(mesh, mesh_to_process, COMM_RECV, mesh->corner_id[CORNER_BOTTOM_LEFT], 0, mesh->height - 1, &reqs_v[13]);
  // Prevent comm mixing to avoid bugs
  //MPI_Barrier(mesh->cart_comm);

  // Bottom right phase
  lbm_comm_sync_ghosts_diagonal(mesh,
	mesh_to_process,
	COMM_SEND,
	mesh->corner_id[CORNER_BOTTOM_RIGHT],
	mesh->width - 2,
	mesh->height - 2,
	&reqs_v[14]
  );
  lbm_comm_sync_ghosts_diagonal(mesh, mesh_to_process, COMM_RECV, mesh->corner_id[CORNER_TOP_LEFT], 0, 0, &reqs_v[15]);
  // Prevent comm mixing to avoid bugs
  //MPI_Barrier(mesh->cart_comm);


  // Synchronize all remaining in-flight communications before exiting
  //MPI_Barrier(mesh->cart_comm);
	
}

void save_frame_all_domain(lbm_comm_t * mesh, FILE* fp, Mesh* source_mesh, Mesh* temp) {
  int comm_size, rank;
  MPI_Comm_size(mesh->cart_comm, &comm_size);
  MPI_Comm_rank(mesh->cart_comm, &rank);

  // If we have more than one process
  if (1 < comm_size) {
	if (rank == RANK_MASTER) {
	  // Rank 0 renders its local Mesh
	  save_frame(fp, source_mesh);
	  // Rank 0 receives & render other processes meshes
	  for (ssize_t i = 1; i < comm_size; i++) {
		MPI_Status status;
		MPI_Recv(
		  temp->cells,
		  source_mesh->width * source_mesh->height * DIRECTIONS,
		  MPI_DOUBLE,
		  i,
		  0,
		  mesh->cart_comm,
		  &status
		);
		save_frame(fp, temp);
	  }
	} else {
	  // All other ranks send their local mesh
	  MPI_Send(
		source_mesh->cells,
		source_mesh->width * source_mesh->height * DIRECTIONS,
		MPI_DOUBLE,
		RANK_MASTER,
		0,
		mesh->cart_comm
	  );
	}
  } else {
	// Only 0 renders its local mesh
	save_frame(fp, source_mesh);
  }
}
