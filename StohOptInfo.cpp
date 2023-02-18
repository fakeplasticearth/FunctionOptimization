#include "StohOptInfo.h"

StohOptInfo::StohOptInfo() : OptInfo(), n_iter_after_imp(100), last_imp_norm(1e-4) {

}

StohOptInfo::StohOptInfo(unsigned int n_iter_, unsigned int n_iter_after_imp_, double last_imp_norm_) : OptInfo(n_iter_),
n_iter_after_imp(n_iter_after_imp_), last_imp_norm(last_imp_norm_) {

}

StohOptInfo::StohOptInfo(const StohOptInfo& other) : OptInfo(other), n_iter_after_imp(other.n_iter_after_imp), last_imp_norm(other.last_imp_norm) {

}

StohOptInfo::StohOptInfo(StohOptInfo&& other) noexcept : OptInfo(other), n_iter_after_imp(other.n_iter_after_imp), last_imp_norm(other.last_imp_norm) {

}

void StohOptInfo::swap(StohOptInfo& other) noexcept {
    std::swap(n_iter_after_imp, other.n_iter_after_imp);
    OptInfo::swap(other);
}

StohOptInfo& StohOptInfo::operator=(StohOptInfo other) noexcept {
    this -> swap(other);
    return *this;
}

unsigned int StohOptInfo::get_n_iter_after_imp() const {
    return n_iter_after_imp;
}

double StohOptInfo::get_last_imp_norm() const {
    return last_imp_norm;
}