#include "NormDiffSSC.h"

NormDiffSSC::NormDiffSSC(): min_last_imp_norm(1e-4) {

}

NormDiffSSC::NormDiffSSC(unsigned int max_iter, double min_last_imp_norm_) : StopCriterion(max_iter), min_last_imp_norm(min_last_imp_norm_) {

}

NormDiffSSC::NormDiffSSC(const NormDiffSSC& other) : StopCriterion(other), min_last_imp_norm(other.min_last_imp_norm) {

}

NormDiffSSC::NormDiffSSC(NormDiffSSC&& other) noexcept: StopCriterion(other), min_last_imp_norm(other.min_last_imp_norm) {
    other.min_last_imp_norm = 0.;
}

NormDiffSSC::~NormDiffSSC() {

}

void NormDiffSSC::swap(NormDiffSSC& other) noexcept {
    StopCriterion::swap(other);
    std::swap(min_last_imp_norm, other.min_last_imp_norm);
}

NormDiffSSC& NormDiffSSC::operator=(NormDiffSSC other) noexcept {
    this -> swap(other);
    return *this;
}

bool NormDiffSSC::criterion(OptInfo* opt_info) {
    StohOptInfo* stoh_info = static_cast<StohOptInfo*>(opt_info);
    if (stoh_info -> get_n_iter() >= max_n_iter || stoh_info -> get_last_imp_norm() < min_last_imp_norm)
        return 1;
    return 0;
}

double NormDiffSSC::get_min_last_imp_norm() const {
    return min_last_imp_norm;
}