#include <tbb/tbb.h>
#include <iostream>
#include <vector>

class SumReducer {
public:
    int sum;
    
    SumReducer() : sum(0) {}
    SumReducer(SumReducer& s, tbb::split) { sum = 0; }

    void operator()(const tbb::blocked_range<int>& range) {
        int local_sum = sum;
        for (int i = range.begin(); i < range.end(); ++i) {
            local_sum += i + 1;
        }
        sum = local_sum;
    }

    void join(const SumReducer& rhs) {
        sum += rhs.sum;
    }
};

int main() {
    const int N = 1000;  // Array size
    std::vector<int> data(N);

    // Initialize the array
    for (int i = 0; i < N; ++i) {
        data[i] = i + 1;
    }

    // Use tbb::parallel_reduce for parallel sum calculation
    SumReducer reducer;
    tbb::parallel_reduce(tbb::blocked_range<int>(0, N), reducer);

    // Calculate the average value
    double average = static_cast<double>(reducer.sum) / N;
    std::cout << "Final sum: " << reducer.sum << std::endl;
    std::cout << "Average: " << average << std::endl;

    return 0;
}
