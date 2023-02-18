#include "IterAfterImpSSC.h"

IterAfterImpSSC::IterAfterImpSSC(): max_n_iter_after_imp(500) {

}

IterAfterImpSSC::IterAfterImpSSC(unsigned int max_iter, unsigned int max_after_improve): StopCriterion(max_iter), 
max_n_iter_after_imp(max_after_improve) {

}

IterAfterImpSSC::IterAfterImpSSC(const IterAfterImpSSC& other): StopCriterion(other), max_n_iter_after_imp(other.max_n_iter_after_imp) {

}

IterAfterImpSSC::IterAfterImpSSC(IterAfterImpSSC&& other) noexcept: StopCriterion(other), max_n_iter_after_imp(other.max_n_iter_after_imp) {
    other.max_n_iter_after_imp = 0;
}

IterAfterImpSSC::~IterAfterImpSSC() {

}

void IterAfterImpSSC::swap(IterAfterImpSSC& other) noexcept {
    StopCriterion::swap(other);
    std::swap(max_n_iter_after_imp, other.max_n_iter_after_imp);
}

IterAfterImpSSC& IterAfterImpSSC::operator=(IterAfterImpSSC other) noexcept {
    this -> swap(other);
    return *this;
}

bool IterAfterImpSSC::criterion(OptInfo* opt_info) {
    StohOptInfo* stoh_info = static_cast<StohOptInfo*>(opt_info);
    if (stoh_info -> get_n_iter() >= max_n_iter || stoh_info -> get_n_iter_after_imp() >= max_n_iter_after_imp)
        return 1;
    return 0;
}

unsigned int IterAfterImpSSC::get_max_n_iter_after_imp() const {
    return max_n_iter_after_imp;
}