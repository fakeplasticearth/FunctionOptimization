#include "IterOnlySSC.h"

IterOnlySSC::IterOnlySSC() {

}

IterOnlySSC::IterOnlySSC(unsigned int max_iter) : StopCriterion(max_iter) {

}

IterOnlySSC::IterOnlySSC(const IterOnlySSC& other) : StopCriterion(other) {

}

IterOnlySSC::IterOnlySSC(IterOnlySSC&& other) noexcept : StopCriterion(other) {

}

IterOnlySSC::~IterOnlySSC() {

}

void IterOnlySSC::swap(IterOnlySSC& other) noexcept {
    StopCriterion::swap(other);
}

IterOnlySSC& IterOnlySSC::operator=(IterOnlySSC other) noexcept {
    this -> swap(other);
    return *this;
}

bool IterOnlySSC::criterion(OptInfo* opt_info) {
    if (opt_info -> get_n_iter() >= max_n_iter) 
        return 1;
    return 0;
}