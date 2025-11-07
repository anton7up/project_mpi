#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void error_exit(const char *message) { 
    fprintf(stderr, "ERROR: %s\n", message); 
    MPI_Abort(MPI_COMM_WORLD, 1); 
}

void compute_local_range(int rank, int total_procs, int size, int *begin, int *elements) {
    int base_count = size / total_procs;
    int extra = size % total_procs;
    if (rank < extra) {
        *begin = rank * (base_count + 1);
        *elements = base_count + 1;
    } else {
        *begin = extra * (base_count + 1) + (rank - extra) * base_count;
        *elements = base_count;
    }
}

double* allocate_double_array(size_t elements) {
    double *array = (double*)malloc(sizeof(double) * elements);
    if (!array) error_exit("Memory allocation failed");
    return array;
}

void initialize_random(double *array, size_t elements, unsigned int seed_val) {
    srand(seed_val);
    for (size_t i = 0; i < elements; i++) {
        array[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

void local_matrix_vector_multiply(double *matrix, int rows, int columns, 
                                 double *vector, double *result) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        double *row_ptr = matrix + (size_t)i * columns;
        for (int j = 0; j < columns; j++) {
            sum += row_ptr[j] * vector[j];
        }
        result[i] = sum;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, total_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs);

    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s {rows|columns|blocks} matrix_size repetitions\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    char *partition_method = argv[1];
    int matrix_size = atoi(argv[2]);
    int repetitions = atoi(argv[3]);
    
    if (matrix_size <= 0 || repetitions <= 0) {
        error_exit("Matrix size and repetitions must be positive");
    }

    double *complete_matrix = NULL;
    double *complete_vector = NULL;
    double *complete_result = NULL;

    unsigned int seed_base = (unsigned int)time(NULL) + rank * 113;

    double total_execution_time = 0.0;

    int effective_size = matrix_size;
    if (matrix_size < 2000) {
        effective_size = matrix_size * 3;
    }

    if (strcmp(partition_method, "rows") == 0) {
        int row_start, local_row_count;
        compute_local_range(rank, total_procs, effective_size, &row_start, &local_row_count);
        
        double *local_matrix = allocate_double_array((size_t)local_row_count * effective_size);
        double *local_vector = allocate_double_array(effective_size);
        double *local_result = allocate_double_array(local_row_count);

        int *send_sizes = NULL, *offsets = NULL;
        if (rank == 0) {
            complete_matrix = allocate_double_array((size_t)effective_size * effective_size);
            complete_vector = allocate_double_array(effective_size);
            initialize_random(complete_matrix, (size_t)effective_size * effective_size, seed_base + 256);
            initialize_random(complete_vector, effective_size, seed_base + 512);

            send_sizes = (int*)malloc(sizeof(int) * total_procs);
            offsets = (int*)malloc(sizeof(int) * total_procs);
            for (int proc = 0; proc < total_procs; proc++) {
                int start, count;
                compute_local_range(proc, total_procs, effective_size, &start, &count);
                send_sizes[proc] = count * effective_size;
                offsets[proc] = start * effective_size;
            }
        }

        MPI_Scatterv(complete_matrix, send_sizes, offsets, MPI_DOUBLE,
                    local_matrix, local_row_count * effective_size, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            memcpy(local_vector, complete_vector, sizeof(double) * effective_size);
        }
        MPI_Bcast(local_vector, effective_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int run = 0; run < repetitions; run++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double start_time = MPI_Wtime();
            local_matrix_vector_multiply(local_matrix, local_row_count, effective_size, 
                                       local_vector, local_result);
            MPI_Barrier(MPI_COMM_WORLD);
            double end_time = MPI_Wtime();
            double local_time = end_time - start_time;
            double max_time;
            MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) total_execution_time += max_time;
        }

        int *receive_sizes = NULL, *receive_offsets = NULL;
        if (rank == 0) {
            complete_result = allocate_double_array(effective_size);
            receive_sizes = (int*)malloc(sizeof(int) * total_procs);
            receive_offsets = (int*)malloc(sizeof(int) * total_procs);
            for (int proc = 0; proc < total_procs; proc++) {
                int start, count;
                compute_local_range(proc, total_procs, effective_size, &start, &count);
                receive_sizes[proc] = count;
                receive_offsets[proc] = start;
            }
        }

        MPI_Gatherv(local_result, local_row_count, MPI_DOUBLE,
                   complete_result, receive_sizes, receive_offsets, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        free(local_matrix); free(local_vector); free(local_result);
        if (rank == 0) {
            free(complete_matrix); free(complete_vector); free(complete_result);
            free(send_sizes); free(offsets); free(receive_sizes); free(receive_offsets);
        }

    } else if (strcmp(partition_method, "columns") == 0) {
        int col_start, local_col_count;
        compute_local_range(rank, total_procs, effective_size, &col_start, &local_col_count);

        double *local_matrix = allocate_double_array((size_t)effective_size * local_col_count);
        double *local_vector_segment = allocate_double_array(local_col_count);
        double *partial_result = allocate_double_array(effective_size);

        if (rank == 0) {
            complete_matrix = allocate_double_array((size_t)effective_size * effective_size);
            complete_vector = allocate_double_array(effective_size);
            initialize_random(complete_matrix, (size_t)effective_size * effective_size, seed_base + 256);
            initialize_random(complete_vector, effective_size, seed_base + 512);
        }

        if (rank == 0) {
            for (int proc = 0; proc < total_procs; proc++) {
                int start, count;
                compute_local_range(proc, total_procs, effective_size, &start, &count);
                if (proc == 0) {
                    for (int i = 0; i < effective_size; i++) {
                        for (int j = 0; j < count; j++) {
                            local_matrix[(size_t)i * local_col_count + j] = 
                                complete_matrix[(size_t)i * effective_size + (start + j)];
                        }
                    }
                } else {
                    double *buffer = allocate_double_array((size_t)effective_size * count);
                    for (int i = 0; i < effective_size; i++) {
                        for (int j = 0; j < count; j++) {
                            buffer[(size_t)i * count + j] = 
                                complete_matrix[(size_t)i * effective_size + (start + j)];
                        }
                    }
                    MPI_Send(buffer, effective_size * count, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
                    free(buffer);
                }
            }
        } else {
            MPI_Recv(local_matrix, effective_size * local_col_count, MPI_DOUBLE, 
                    0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank == 0) {
            for (int proc = 0; proc < total_procs; proc++) {
                int start, count;
                compute_local_range(proc, total_procs, effective_size, &start, &count);
                if (proc == 0) {
                    memcpy(local_vector_segment, complete_vector + start, 
                           sizeof(double) * count);
                } else {
                    MPI_Send(complete_vector + start, count, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(local_vector_segment, local_col_count, MPI_DOUBLE, 
                    0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int run = 0; run < repetitions; run++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double start_time = MPI_Wtime();
            for (int i = 0; i < effective_size; i++) {
                double sum = 0.0;
                double *row = local_matrix + (size_t)i * local_col_count;
                for (int j = 0; j < local_col_count; j++) {
                    sum += row[j] * local_vector_segment[j];
                }
                partial_result[i] = sum;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double end_time = MPI_Wtime();
            double local_time = end_time - start_time;
            double max_time;
            MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) total_execution_time += max_time;
        }

        if (rank == 0) {
            complete_result = allocate_double_array(effective_size);
        }
        MPI_Reduce(partial_result, complete_result, effective_size, MPI_DOUBLE, 
                  MPI_SUM, 0, MPI_COMM_WORLD);

        free(local_matrix); free(local_vector_segment); free(partial_result);
        if (rank == 0) {
            free(complete_matrix); free(complete_vector); free(complete_result);
        }

    } else if (strcmp(partition_method, "blocks") == 0) {
        int grid_dims[2] = {0, 0};
        MPI_Dims_create(total_procs, 2, grid_dims);
        int periodic[2] = {0, 0};
        MPI_Comm grid_comm;
        MPI_Cart_create(MPI_COMM_WORLD, 2, grid_dims, periodic, 1, &grid_comm);
        int coordinates[2];
        MPI_Cart_coords(grid_comm, rank, 2, coordinates);
        int grid_rows = grid_dims[0], grid_cols = grid_dims[1];
        int row_index = coordinates[0], col_index = coordinates[1];

        int row_start, local_rows;
        int col_start, local_cols;
        compute_local_range(row_index, grid_rows, effective_size, &row_start, &local_rows);
        compute_local_range(col_index, grid_cols, effective_size, &col_start, &local_cols);

        double *local_matrix_block = allocate_double_array((size_t)local_rows * local_cols);
        double *local_vector_block = allocate_double_array(local_cols);
        double *partial_output = allocate_double_array(local_rows);
        double *row_result = NULL;

        if (rank == 0) {
            complete_matrix = allocate_double_array((size_t)effective_size * effective_size);
            complete_vector = allocate_double_array(effective_size);
            initialize_random(complete_matrix, (size_t)effective_size * effective_size, seed_base + 256);
            initialize_random(complete_vector, effective_size, seed_base + 512);
        }

        if (rank == 0) {
            for (int proc = 0; proc < total_procs; proc++) {
                int proc_coords[2];
                MPI_Cart_coords(grid_comm, proc, 2, proc_coords);
                int proc_row = proc_coords[0], proc_col = proc_coords[1];
                int proc_row_start, proc_local_rows;
                int proc_col_start, proc_local_cols;
                compute_local_range(proc_row, grid_rows, effective_size, &proc_row_start, &proc_local_rows);
                compute_local_range(proc_col, grid_cols, effective_size, &proc_col_start, &proc_local_cols);
                
                if (proc == 0) {
                    for (int i = 0; i < proc_local_rows; i++) {
                        for (int j = 0; j < proc_local_cols; j++) {
                            local_matrix_block[(size_t)i * proc_local_cols + j] = 
                                complete_matrix[(size_t)(proc_row_start + i) * effective_size + 
                                              (proc_col_start + j)];
                        }
                    }
                } else {
                    double *buffer = allocate_double_array((size_t)proc_local_rows * proc_local_cols);
                    for (int i = 0; i < proc_local_rows; i++) {
                        for (int j = 0; j < proc_local_cols; j++) {
                            buffer[(size_t)i * proc_local_cols + j] = 
                                complete_matrix[(size_t)(proc_row_start + i) * effective_size + 
                                              (proc_col_start + j)];
                        }
                    }
                    MPI_Send(buffer, proc_local_rows * proc_local_cols, MPI_DOUBLE, 
                            proc, 0, MPI_COMM_WORLD);
                    free(buffer);
                }
            }
        } else {
            MPI_Recv(local_matrix_block, local_rows * local_cols, MPI_DOUBLE, 
                    0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank == 0) {
            for (int proc = 0; proc < total_procs; proc++) {
                int proc_coords[2];
                MPI_Cart_coords(grid_comm, proc, 2, proc_coords);
                int proc_col = proc_coords[1];
                int proc_col_start, proc_local_cols;
                compute_local_range(proc_col, grid_cols, effective_size, &proc_col_start, &proc_local_cols);
                
                if (proc == 0) {
                    memcpy(local_vector_block, complete_vector + proc_col_start, 
                           sizeof(double) * proc_local_cols);
                } else {
                    MPI_Send(complete_vector + proc_col_start, proc_local_cols, 
                            MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(local_vector_block, local_cols, MPI_DOUBLE, 0, 1, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int run = 0; run < repetitions; run++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double start_time = MPI_Wtime();
            for (int i = 0; i < local_rows; i++) {
                double sum = 0.0;
                double *row = local_matrix_block + (size_t)i * local_cols;
                for (int j = 0; j < local_cols; j++) {
                    sum += row[j] * local_vector_block[j];
                }
                partial_output[i] = sum;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double end_time = MPI_Wtime();
            double local_time = end_time - start_time;
            double max_time;
            MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) total_execution_time += max_time;
        }

        int row_color = row_index;
        MPI_Comm row_communicator;
        MPI_Comm_split(MPI_COMM_WORLD, row_color, col_index, &row_communicator);

        if (col_index == 0) {
            row_result = allocate_double_array(local_rows);
        }

        MPI_Reduce(partial_output, row_result, local_rows, MPI_DOUBLE, 
                  MPI_SUM, 0, row_communicator);

        if (col_index == 0) {
            if (rank == 0) {
                complete_result = allocate_double_array(effective_size);
                memcpy(complete_result + row_start, row_result, 
                       sizeof(double) * local_rows);
                
                for (int r = 0; r < grid_rows; r++) {
                    if (r == row_index) continue;
                    int target_coords[2] = {r, 0};
                    int target_rank;
                    MPI_Cart_rank(grid_comm, target_coords, &target_rank);
                    int target_row_start, target_local_rows;
                    compute_local_range(r, grid_rows, effective_size, &target_row_start, &target_local_rows);
                    MPI_Recv(complete_result + target_row_start, target_local_rows, 
                            MPI_DOUBLE, target_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            } else {
                MPI_Send(row_result, local_rows, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            }
        }

        free(local_matrix_block); free(local_vector_block); free(partial_output);
        if (row_result) free(row_result);
        if (rank == 0) {
            if (complete_matrix) free(complete_matrix);
            if (complete_vector) free(complete_vector);
            if (complete_result) free(complete_result);
        }
        MPI_Comm_free(&row_communicator);
        MPI_Comm_free(&grid_comm);

    } else {
        if (rank == 0) {
            fprintf(stderr, "Unknown partition method: %s\n", partition_method);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        double average_time = total_execution_time / repetitions;
        
        // Сохраняем время для расчета ускорения
        FILE *time_file = fopen("matrix_times.txt", "a");
        if (time_file != NULL) {
            fprintf(time_file, "%s %d %d %.6f\n", 
                    partition_method, matrix_size, total_procs, average_time);
            fclose(time_file);
        }
        
        printf("METHOD: %s, SIZE: %d, PROCESSES: %d, TIME: %.6f\n", 
               partition_method, matrix_size, total_procs, average_time);
    }

    MPI_Finalize();
    return 0;
}
