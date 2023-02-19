#include "MinRelImpNormGDSC.h"

MinRelImpNormGDSC::MinRelImpNormGDSC() {

}

MinRelImpNormGDSC::MinRelImpNormGDSC(unsigned int max_iter, double min_rel_imp_norm_) : StopCriterion(max_iter),
min_rel_imp_norm(min_rel_imp_norm_) {

}

MinRelImpNormGDSC::MinRelImpNormGDSC(const MinRelImpNormGDSC& other) : StopCriterion(other),
min_rel_imp_norm(other.min_rel_imp_norm) {

}

MinRelImpNormGDSC::MinRelImpNormGDSC(MinRelImpNormGDSC&& other) noexcept : StopCriterion(other), min_rel_imp_norm(other.min_rel_imp_norm) {

}

MinRelImpNormGDSC::~MinRelImpNormGDSC() {

}

void MinRelImpNormGDSC::swap(MinRelImpNormGDSC& other) noexcept {
    StopCriterion::swap(other);
    std::swap(min_rel_imp_norm, other.min_rel_imp_norm);
}

MinRelImpNormGDSC& MinRelImpNormGDSC::operator=(MinRelImpNormGDSC other) noexcept {
    this->swap(other);
    return *this;
}

bool MinRelImpNormGDSC::criterion(OptInfo* opt_info) {
    GDOptInfo* gd_info = static_cast<GDOptInfo*>(opt_info);
    if (gd_info->get_n_iter() >= max_n_iter || gd_info->get_last_step_norm() < min_rel_imp_norm)
        return 1;
    return 0;
}

double MinRelImpNormGDSC::get_min_rel_imp_norm() const {
    return min_rel_imp_norm;
}