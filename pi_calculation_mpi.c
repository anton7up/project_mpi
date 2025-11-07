#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#ifndef PI_CONST
#define PI_CONST 3.14159265358979323846
#endif

long long calculate_circle_hits(long long iterations_per_proc, int proc_id) {
    unsigned int rand_seed = (unsigned int)time(NULL) * (proc_id + 1);
    long long hits_in_circle = 0;
    
    for (long long i = 0; i < iterations_per_proc; i++) {
        double coord_x = (double)rand_r(&rand_seed) / RAND_MAX;
        double coord_y = (double)rand_r(&rand_seed) / RAND_MAX;
        
        if (coord_x * coord_x + coord_y * coord_y <= 1.0) {
            hits_in_circle++;
        }
    }
    
    return hits_in_circle;
}

int main(int argc, char *argv[]) {
    int proc_id, total_procs;
    long long total_iterations;
    int distribution_type = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs);
    
    if (argc < 3) {
        if (proc_id == 0) {
            printf("Usage: %s <total_iterations> <distribution_type>\n", argv[0]);
            printf("Distribution types: 0 - balanced, 1 - fixed\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    total_iterations = atoll(argv[1]);
    distribution_type = atoi(argv[2]);
    
    if (total_iterations <= 0) {
        if (proc_id == 0) {
            printf("Error: iterations count must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    long long local_iterations;
    if (distribution_type == 0) {
        local_iterations = total_iterations / total_procs;
        int remainder = total_iterations % total_procs;
        if (proc_id < remainder) {
            local_iterations++;
        }
    } else {
        local_iterations = total_iterations / total_procs;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double computation_start = MPI_Wtime();
    
    long long local_hits = calculate_circle_hits(local_iterations, proc_id);
    long long global_hits = 0;
    
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double computation_end = MPI_Wtime();
    
    if (proc_id == 0) {
        double computed_pi = 4.0 * (double)global_hits / (double)total_iterations;
        double difference = fabs(computed_pi - PI_CONST);
        double time_taken = computation_end - computation_start;
        
        FILE *time_file = fopen("pi_times.txt", "a");
        if (time_file != NULL) {
            fprintf(time_file, "%d %lld %d %.6f\n", 
                    total_procs, total_iterations, distribution_type, time_taken);
            fclose(time_file);
        }
        
        printf("=== PI CALCULATION RESULTS ===\n");
        printf("Method: Monte Carlo\n");
        printf("Distribution: %s\n", (distribution_type == 0) ? "balanced" : "fixed");
        printf("Processes: %d\n", total_procs);
        printf("Total iterations: %lld\n", total_iterations);
        printf("Computed PI: %.10f\n", computed_pi);
        printf("Actual PI: %.10f\n", PI_CONST);
        printf("Difference: %.10f\n", difference);
        printf("Time: %.6f seconds\n", time_taken);
        printf("Iterations per second: %.0f\n", total_iterations / time_taken);
    }
    
    MPI_Finalize();
    return 0;
}
