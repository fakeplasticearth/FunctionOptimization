#include "StohasticOptimization.h"

StohasticOptimization::StohasticOptimization() : OptimizationMethod(), delta(1), p(0.5),
    n_iter_after_improvment(0), seed(std::chrono::steady_clock::now().time_since_epoch().count()),
    generator(seed), distribution(0., 1.) {
    
}

StohasticOptimization::StohasticOptimization(Function* func_, BoxArea* box_area_, StopCriterion* stop_criterion_, 
    const std::vector<double>& first_point_, double delta_ = 1, double p_ = 0.5) : OptimizationMethod(func_, box_area_, stop_criterion_), 
    first_point(first_point_), delta(delta_), p(p_), n_iter_after_improvment(0), seed (std::chrono::steady_clock::now().time_since_epoch().count()), generator(seed),
    distribution(0., 1.), curr_min_value((*func_)(first_point_)) {

}

StohasticOptimization::StohasticOptimization(const StohasticOptimization& other) : OptimizationMethod(other), first_point(other.first_point),
    delta(other.delta), p(other.p), curr_min_value(other.curr_min_value), n_iter_after_improvment(other.n_iter_after_improvment),
    generator(other.generator), distribution(other.distribution) {
    
}

StohasticOptimization::StohasticOptimization(StohasticOptimization&& other) : OptimizationMethod(other), first_point(std::move(other.first_point)),
    delta(other.delta), p(other.p), curr_min_value(other.curr_min_value), n_iter_after_improvment(other.n_iter_after_improvment),
    generator(other.generator), distribution(other.distribution) {
    other.delta = 0;
    other.p = 0;
    other.curr_min_value = DBL_MAX;
    other.n_iter_after_improvment = 0;
}

void StohasticOptimization::swap(StohasticOptimization& other) noexcept {
    OptimizationMethod::swap(other);
    std::swap(delta, other.delta);
    std::swap(p, other.p);
    std::swap(curr_min_value, other.curr_min_value);
    std::swap(first_point, other.first_point);
    std::swap(n_iter_after_improvment, other.n_iter_after_improvment);
    std::swap(generator, other.generator);
    std::swap(distribution, other.distribution);
}

StohasticOptimization& StohasticOptimization::operator=(StohasticOptimization other) noexcept {
    this -> swap(other);
    return *this;
}

void StohasticOptimization::set_first_point(const std::vector<double>& point) {
    first_point = point;
}

double StohasticOptimization::unif(double lower, double upper) {
    return distribution(generator) * (upper - lower) + lower;
}

OptInfo* StohasticOptimization::get_opt_info() const {
    return new StohOptInfo(n_iter, n_iter_after_improvment, last_imp_norm);
}

void StohasticOptimization::make_step() {
    std::vector<double> curr_point = point_history.back();
    std::vector<double> next_point(n_dim);

    if (p < unif(0, 1)) {
        for (int i = 0; i < n_dim; ++i)
            next_point[i] = unif(box_area->get_limits(i).lower, box_area->get_limits(i).upper);
    }
    else {
        for (int i = 0; i < n_dim; ++i) {
            next_point[i] = unif(std::max(box_area->get_limits(i).lower, curr_point[i] - delta), std::min(
                curr_point[i] + delta, box_area->get_limits(i).upper));
        }
    }

    double next_point_value = (*func)(next_point);
    if (curr_min_value > next_point_value) {
        last_imp_norm = curr_min_value - next_point_value;
        curr_min_value = next_point_value;
        point_history.push_back(next_point);
    }
    else
        ++n_iter_after_improvment;
    return;
}

OptResult StohasticOptimization::optimize() {
    point_history.push_back(first_point);
    curr_min_value = (*func)(point_history.back());
    while (!stop_criterion -> criterion(get_opt_info())) {
        make_step();
        ++n_iter;
    }

    return OptResult(point_history.back(), curr_min_value, n_iter);
}

StohasticOptimization::~StohasticOptimization() {
    first_point.resize(0);
}