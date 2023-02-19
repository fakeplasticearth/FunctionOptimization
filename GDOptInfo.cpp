#include "GDOptInfo.h"

GDOptInfo::GDOptInfo() : OptInfo() {

}

GDOptInfo::GDOptInfo(unsigned int n_iter_, double grad_norm_, double last_step_norm_, double rel_imp_norm_) : OptInfo(n_iter_),
    grad_norm(grad_norm_), last_step_norm(last_step_norm_), rel_imp_norm(rel_imp_norm_) {

}

GDOptInfo::GDOptInfo(const GDOptInfo& other) : OptInfo(other), grad_norm(other.grad_norm), last_step_norm(other.last_step_norm),
   rel_imp_norm(other.rel_imp_norm) {

}

GDOptInfo::GDOptInfo(GDOptInfo&& other) noexcept : OptInfo(other), grad_norm(other.grad_norm), last_step_norm(other.last_step_norm),
    rel_imp_norm(other.rel_imp_norm) {

}

void GDOptInfo::swap(GDOptInfo& other) noexcept {
    OptInfo::swap(other);
    std::swap(grad_norm, other.grad_norm);
    std::swap(last_step_norm, other.last_step_norm);
    std::swap(rel_imp_norm, other.rel_imp_norm);
}

GDOptInfo& GDOptInfo::operator=(GDOptInfo other) noexcept {
    this->swap(other);
    return *this;
}

double GDOptInfo::get_grad_norm() const {
    return grad_norm;
}

double GDOptInfo::get_last_step_norm() const {
    return last_step_norm;
}

double GDOptInfo::get_rel_imp_norm() const {
    return rel_imp_norm;
}

GDOptInfo::~GDOptInfo() {
    
}