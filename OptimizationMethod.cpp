#include "OptimizationMethod.h"

OptimizationMethod::OptimizationMethod() {

}

OptimizationMethod::OptimizationMethod(Function* func_, BoxArea* box_area_, StopCriterion* stop_criterion_) : func(func_),
    box_area(box_area_), stop_criterion(stop_criterion_) {
    n_dim = box_area->get_n_dim();
}

OptimizationMethod::OptimizationMethod(const OptimizationMethod& other) : func(other.func), box_area(other.box_area),
stop_criterion(other.stop_criterion) {
    
}

OptimizationMethod::OptimizationMethod(OptimizationMethod&& other) noexcept: func(std::move(other.func)), box_area(std::move(other.box_area)),
    stop_criterion(std::move(other.stop_criterion)), n_dim(other.n_dim), n_iter(other.n_iter)
{
    other.n_dim = 0;
    other.n_iter = 0;
}

void OptimizationMethod::swap(OptimizationMethod& other) noexcept {
    std::swap(other.func, func);
    std::swap(box_area, other.box_area);
    std::swap(stop_criterion, other.stop_criterion);
    std::swap(n_iter, other.n_iter);
    std::swap(n_dim, other.n_dim);
}

unsigned int OptimizationMethod::get_n_dim() const {
    return n_dim;
}

unsigned int OptimizationMethod::get_n_iter() const {
    return n_iter;
}

OptimizationMethod::~OptimizationMethod() {
    point_history.resize(0);
    box_area = nullptr;
    func = nullptr;
    stop_criterion = nullptr;
}