#include "StopCriterion.h"

StopCriterion::StopCriterion(): max_n_iter(5000) {

}

StopCriterion::StopCriterion(unsigned int max_n_iter_) : max_n_iter(max_n_iter_) {

}

StopCriterion::StopCriterion(const StopCriterion& other) : max_n_iter(other.max_n_iter) {

}

StopCriterion::StopCriterion(StopCriterion&& other) noexcept : max_n_iter(std::move(other.max_n_iter)) {
}

void StopCriterion::swap(StopCriterion& other) noexcept {
    std::swap(max_n_iter, other.max_n_iter);
}

bool StopCriterion::operator()(OptInfo* opt_info) {
    return criterion(opt_info);
}