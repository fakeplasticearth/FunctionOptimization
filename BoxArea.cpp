#include "BoxArea.h"

BoxArea::BoxArea() {

}

BoxArea::BoxArea(unsigned int n_dim_, const std::vector<dimensional_limits>& lims) : n_dim(n_dim_), limits(lims) {
    
}

BoxArea::BoxArea(const BoxArea& other) : n_dim(other.n_dim), limits(other.limits) {
}

BoxArea::BoxArea(BoxArea&& other) noexcept: n_dim(other.n_dim), limits(other.limits) {
    other.n_dim = 0;
    other.limits.resize(0);
}

bool BoxArea::is_in(const std::vector<double>& point) const {
    for (int i = 0; i < point.size(); ++i)
        if (point[i] < limits[i].lower || point[i] > limits[i].upper) 
            return false;
    return true;
}

void BoxArea::swap(BoxArea& other) noexcept {
    std::swap(n_dim, other.n_dim);
    std::swap(limits, other.limits);
}

BoxArea& BoxArea::operator=(BoxArea other) noexcept {
    this -> swap(other);
    return *this;
}

unsigned int BoxArea::get_n_dim() const {
    return limits.size();
}

dimensional_limits BoxArea::get_limits(unsigned int i) const {
    return limits[i];
}

BoxArea::~BoxArea() {
    limits.clear();
}