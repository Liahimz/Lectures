#include <omp.h>
#include <iostream>
#include <vector>
int main() {
    const int N = 1000;
    std::vector<int> data(N);
    // Initialize the array with values
    for (int i = 0; i < N; ++i) {
        data[i] = i + 1;
    }
    int sum = 0;
    int max_val = 0;
    // Parallel section
    #pragma omp parallel
    {
        // Parallelize the loop for summing the elements
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            #pragma omp critical
            {
                sum += data[i];
            }
        }
        // Synchronize threads before the next operations
        #pragma omp barrier
        // Split tasks into sections: finding the maximum and output the result
        #pragma omp sections
        {
            // Section 1: Find the maximum value in the array
            #pragma omp section
            {
                int local_max = 0;
                for (int i = 0; i < N; ++i) {
                    if (data[i] > local_max) {
                        local_max = data[i];
                    }
                }
                #pragma omp critical
                {
                    if (local_max > max_val) {
                        max_val = local_max;
                    }
                }
            }
            // Section 2: Output partial results
            #pragma omp section
            {
                std::cout << "Partial sum (after sum): " << sum << std::endl;
            }
        }
        // Final thread synchronization
        #pragma omp barrier
        // Compute and output the result after all threads have completed
        #pragma omp single
        {
            double average = static_cast<double>(sum) / N;
            std::cout << "Final sum: " << sum << std::endl;
            std::cout << "Average: " << average << std::endl;
            std::cout << "Max value: " << max_val << std::endl;
        }
    }
    return 0;
}