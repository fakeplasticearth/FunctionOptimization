#pragma once
#include <vector>

struct dimensional_limits {
    double lower = -1, upper = 1;
    dimensional_limits() {}
    dimensional_limits(double low, double up) : lower(low), upper(up) {}
};

class BoxArea
{
protected:
    unsigned int n_dim = 0; // колво измерений
    std::vector<dimensional_limits> limits; // ограничения

public:
    BoxArea();
    BoxArea(unsigned int n_dim_, const std::vector<dimensional_limits>& lims);
    BoxArea(const BoxArea& other);
    BoxArea(BoxArea&& other) noexcept;

    unsigned int get_n_dim() const;
    void swap(BoxArea& other) noexcept;
    BoxArea& operator=(BoxArea other) noexcept;
    dimensional_limits get_limits(unsigned int i) const;
    bool is_in(const std::vector<double>& point) const;
    ~BoxArea();
};

