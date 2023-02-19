#include "MinStepNormGDSC.h"

MinStepNormGDSC::MinStepNormGDSC() {

}

MinStepNormGDSC::MinStepNormGDSC(unsigned int max_iter, double min_step_norm_) : StopCriterion(max_iter),
min_step_norm(min_step_norm_) {

}

MinStepNormGDSC::MinStepNormGDSC(const MinStepNormGDSC& other) : StopCriterion(other),
min_step_norm(other.min_step_norm) {

}

MinStepNormGDSC::MinStepNormGDSC(MinStepNormGDSC&& other) noexcept : StopCriterion(other), min_step_norm(other.min_step_norm) {

}

MinStepNormGDSC::~MinStepNormGDSC() {

}

void MinStepNormGDSC::swap(MinStepNormGDSC& other) noexcept {
    StopCriterion::swap(other);
    std::swap(min_step_norm, other.min_step_norm);
}

MinStepNormGDSC& MinStepNormGDSC::operator=(MinStepNormGDSC other) noexcept {
    this->swap(other);
    return *this;
}

bool MinStepNormGDSC::criterion(OptInfo* opt_info) {
    GDOptInfo* gd_info = static_cast<GDOptInfo*>(opt_info);
    if (gd_info->get_n_iter() >= max_n_iter || gd_info->get_last_step_norm() < min_step_norm)
        return 1;
    return 0;
}

double MinStepNormGDSC::get_min_step_norm() const {
    return min_step_norm;
}