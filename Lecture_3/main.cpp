#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <random>

enum class Operation
{
    Add,
    Mul,
    MaxAbs
};

constexpr int MODULE = 1000 * 1000 * 1000 + 9;

// Templated class ReductorTpl
template <Operation Op>
class ReductorTpl
{
public:
    void load(const std::vector<int> &data) { this->data = data; }

    int reduce() const
    {
        int result = (Op == Operation::Mul) ? 1 : 0;
        for (int x : data)
        {
            result = operator_(result, x);
        }
        return result;
    }

private:
    int operator_(int r, int x) const;
    std::vector<int> data;
};

// Specialization for Operation::Add
template <>
int ReductorTpl<Operation::Add>::operator_(int r, int x) const
{
    return r + x;
}

// Specialization for Operation::Mul
template <>
int ReductorTpl<Operation::Mul>::operator_(int r, int x) const
{
    return (r * x) % MODULE;
}

// Specialization for Operation::MaxAbs
template <>
int ReductorTpl<Operation::MaxAbs>::operator_(int r, int x) const
{
    return std::max(r, std::abs(x));
}

// Virtual base class for reductor
class ReductorVirtual
{
public:
    void load(const std::vector<int> &data) { data_ = data; }
    int reduce() const
    {
        int r = init_();
        for (int x : data_)
        {
            r = operator_(r, x);
        }
        return r;
    }
    virtual ~ReductorVirtual() = default;

private:
    virtual int init_() const { return 0; }
    virtual int operator_(int r, int x) const = 0;
    std::vector<int> data_;
};

// Derived classes implementing specific operations
class ReductorVirtualAdd : public ReductorVirtual
{
public:
    int operator_(int r, int x) const override { return r + x; }
};

class ReductorVirtualMul : public ReductorVirtual
{
public:
    int init_() const override { return 1; }
    int operator_(int r, int x) const override { return (r * x) % MODULE; }
};

class ReductorVirtualMaxAbs : public ReductorVirtual
{
public:
    int operator_(int r, int x) const override
    {
        return std::max(r, std::abs(x));
    }
};

// Templated function to measure time for template-based reduction
template <Operation op>
std::pair<double, int> measure_reduce_time_tpl_run(const std::vector<int> &data)
{
    ReductorTpl<op> reductor;
    reductor.load(data);
    auto start = std::chrono::high_resolution_clock::now();
    int r = reductor.reduce();
    auto end = std::chrono::high_resolution_clock::now();
    return {1e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), r};
}

// Wrapper function to call templated reductor
std::pair<double, int> measure_reduce_time_tpl(const std::vector<int> &data, Operation op)
{
    if (Operation::Add == op)
    {
        return measure_reduce_time_tpl_run<Operation::Add>(data);
    }
    else if (Operation::Mul == op)
    {
        return measure_reduce_time_tpl_run<Operation::Mul>(data);
    }
    else if (Operation::MaxAbs == op)
    {
        return measure_reduce_time_tpl_run<Operation::MaxAbs>(data);
    }
    throw std::runtime_error("incorrect operator");
}

// Function to measure time for virtual-based reduction
std::pair<double, int> measure_reduce_time_virt(const std::vector<int> &data, Operation op)
{
    std::unique_ptr<ReductorVirtual> reductor;
    if (Operation::Add == op)
    {
        reductor = std::unique_ptr<ReductorVirtual>(new ReductorVirtualAdd);
    }
    else if (Operation::Mul == op)
    {
        reductor = std::unique_ptr<ReductorVirtual>(new ReductorVirtualMul);
    }
    else if (Operation::MaxAbs == op)
    {
        reductor = std::unique_ptr<ReductorVirtual>(new ReductorVirtualMaxAbs);
    }
    else
    {
        throw std::runtime_error("incorrect operator");
    }

    reductor->load(data);
    auto start = std::chrono::high_resolution_clock::now();
    int r = reductor->reduce();
    auto end = std::chrono::high_resolution_clock::now();
    return {1e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), r};
}



std::vector<int> generate_n(int n) {
    //seting up vector with random variables
    std::vector<int> data(n);
    static std::random_device rd;  // Random number seed
    static std::mt19937 gen(rd());  // Mersenne Twister, standard generator
    std::uniform_int_distribution<int> dis(-100, 100);  // Range [-100, 100]
    std::generate(data.begin(), data.end(), [&]() { return dis(gen); });
    return data;
}


// Main function to demonstrate the usage
int main()
{
    // std::vector<int> data = {1, -2, 3, 4, -5, 6};
    auto data = generate_n(10000000);
   // Template-based reductions
    std::pair<double, int> result_add_tpl = measure_reduce_time_tpl(data, Operation::Add);
    std::pair<double, int> result_mul_tpl = measure_reduce_time_tpl(data, Operation::Mul);
    std::pair<double, int> result_maxabs_tpl = measure_reduce_time_tpl(data, Operation::MaxAbs);

    std::cout << "Template Add: Time = " << result_add_tpl.first << " ms, Result = " << result_add_tpl.second << std::endl;
    std::cout << "Template Mul: Time = " << result_mul_tpl.first << " ms, Result = " << result_mul_tpl.second << std::endl;
    std::cout << "Template MaxAbs: Time = " << result_maxabs_tpl.first << " ms, Result = " << result_maxabs_tpl.second << std::endl;

    // Virtual-based reductions
    std::pair<double, int> result_add_virt = measure_reduce_time_virt(data, Operation::Add);
    std::pair<double, int> result_mul_virt = measure_reduce_time_virt(data, Operation::Mul);
    std::pair<double, int> result_maxabs_virt = measure_reduce_time_virt(data, Operation::MaxAbs);

    std::cout << "Virtual Add: Time = " << result_add_virt.first << " ms, Result = " << result_add_virt.second << std::endl;
    std::cout << "Virtual Mul: Time = " << result_mul_virt.first << " ms, Result = " << result_mul_virt.second << std::endl;
    std::cout << "Virtual MaxAbs: Time = " << result_maxabs_virt.first << " ms, Result = " << result_maxabs_virt.second << std::endl;

    return 0;
}
