#pragma once
#include <vector>

class OptResult
{
private:
    unsigned int n_iter;
    std::vector<double> min_point;
    double min_value;
public:
    OptResult();
    OptResult(const std::vector<double>& min_point_, double min_value_, unsigned int n_iter_);
    OptResult(const OptResult& other);
    OptResult(OptResult&& other) noexcept;

    void swap(OptResult& other) noexcept;
    OptResult& operator=(OptResult other) noexcept;
    unsigned int get_n_iter() const;
    std::vector<double> get_min_point() const;
    double get_min_value() const;
};

