#include "OptInfo.h"

OptInfo::OptInfo(): n_iter(0) {

}

OptInfo::OptInfo(const OptInfo& other) : n_iter(other.n_iter) {

}

OptInfo::OptInfo(OptInfo&& other) noexcept : n_iter(other.n_iter) {
    other.n_iter = 0;
}

OptInfo::OptInfo(unsigned int n_iter_) : n_iter(n_iter_) {

}

void OptInfo::swap(OptInfo& other) noexcept {
    std::swap(this -> n_iter, other.n_iter);
}


OptInfo& OptInfo::operator=(OptInfo other) {
    this -> swap(other);
    return *this;
}

unsigned int OptInfo::get_n_iter() const {
    return n_iter;
}