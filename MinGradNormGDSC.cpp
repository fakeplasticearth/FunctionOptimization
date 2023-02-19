#include "MinGradNormGDSC.h"

MinGradNormGDSC::MinGradNormGDSC() {

}

MinGradNormGDSC::MinGradNormGDSC(unsigned int max_iter, double min_grad_norm_) : StopCriterion(max_iter),
    min_grad_norm(min_grad_norm_) {

}

MinGradNormGDSC::MinGradNormGDSC(const MinGradNormGDSC& other) : StopCriterion(other), 
    min_grad_norm(other.min_grad_norm) {

}

MinGradNormGDSC::MinGradNormGDSC(MinGradNormGDSC&& other) noexcept : StopCriterion(other), min_grad_norm(other.min_grad_norm) {
    
}

MinGradNormGDSC::~MinGradNormGDSC() {

}

void MinGradNormGDSC::swap(MinGradNormGDSC& other) noexcept {
    StopCriterion::swap(other);
    std::swap(min_grad_norm, other.min_grad_norm);
}

MinGradNormGDSC& MinGradNormGDSC::operator=(MinGradNormGDSC other) noexcept {
    this->swap(other);
    return *this;
}

bool MinGradNormGDSC::criterion(OptInfo* opt_info) {
    GDOptInfo* gd_info = static_cast<GDOptInfo*>(opt_info);
    if (gd_info->get_n_iter() >= max_n_iter || gd_info->get_grad_norm() < min_grad_norm)
        return 1;
    return 0;
}

double MinGradNormGDSC::get_min_grad_norm() const {
    return min_grad_norm;
}