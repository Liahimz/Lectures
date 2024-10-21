#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  // Get the rank (identifier) of the current process

    const int N = 1000;  // Array size
    int chunk_size = N / world_size;  // Size of the array chunk for each process
    std::vector<int> data(N);

    // Initialize the array only in the main process
    if (world_rank == 0) {
        for (int i = 0; i < N; ++i) {
            data[i] = i + 1;
        }
    }

    // Distribute data among processes
    std::vector<int> local_data(chunk_size);
    MPI_Scatter(data.data(), chunk_size, MPI_INT, local_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Local summing in each process
    int local_sum = 0;
    for (int i = 0; i < chunk_size; ++i) {
        local_sum += local_data[i];
    }

    // Collect sums from each process in the main process
    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // The main process calculates the average value
    if (world_rank == 0) {
        double average = static_cast<double>(global_sum) / N;
        std::cout << "Final sum: " << global_sum << std::endl;
        std::cout << "Average: " << average << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
