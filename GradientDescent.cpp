#include "GradientDescent.h"

double l2_norm(std::vector<double> arg) {
    double ss = 0.;
    for (int i = 0; i < arg.size(); ++i)
        ss += arg[i] * arg[i];
    return sqrt(ss);
}

double l1_norm(std::vector<double> arg) {
    double ms = 0.;
    for (int i = 0; i < arg.size(); ++i)
        ms += abs(arg[i]);
    return ms;
}

std::vector<double> operator*(double scalar, const std::vector<double>& vec) {
    std::vector<double> res(vec.size());
    for (int i = 0; i < vec.size(); ++i) {
        res[i] = scalar * vec[i];
    }
    return res;
}

std::vector<double> operator+(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    std::vector<double> res(vec1.size());
    for (int i = 0; i < vec1.size(); ++i) {
        res[i] = vec1[i] + vec2[i];
    }
    return res;
}

std::vector<double> operator-(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    std::vector<double> res(vec1.size());
    for (int i = 0; i < vec1.size(); ++i) {
        res[i] = vec1[i] - vec2[i];
    }
    return res;
}

GradientDescent::GradientDescent() : OptimizationMethod() {

}

GradientDescent::GradientDescent(Function* func_, MDFunction* grad_, BoxArea* box_area_, StopCriterion* stop_criterion_,
    const std::vector<double>& first_point_, std::string norm_name_ = "l2") : OptimizationMethod(func_, box_area_, stop_criterion_),
    grad(grad_), first_point(first_point_), curr_min_value((*func_)(first_point_)), lr(3e-4), norm_name(norm_name_),
    last_step_norm(DBL_MAX), rel_imp_norm(DBL_MAX) {
    if (norm_name == "l2") {
        grad_norm = l2_norm((*grad)(first_point));
    }
    else if (norm_name == "l1") {
        grad_norm = l1_norm((*grad)(first_point));
    }
    else {
        throw std::invalid_argument("Norm name should be l2 or l1.");
    }
}

GradientDescent::GradientDescent(const GradientDescent& other) : OptimizationMethod(other), first_point(other.first_point),
    curr_min_value(other.curr_min_value), lr(other.lr), grad_norm(other.grad_norm), last_step_norm(other.last_step_norm),
    rel_imp_norm(other.rel_imp_norm), value_history(other.value_history) {

}

GradientDescent::GradientDescent(GradientDescent&& other) noexcept: OptimizationMethod(other), first_point(std::move(other.first_point)),
    curr_min_value(other.curr_min_value), lr(other.lr), grad_norm(other.grad_norm), last_step_norm(other.last_step_norm),
    rel_imp_norm(other.rel_imp_norm), value_history(other.value_history) {
    other.curr_min_value = DBL_MAX;
    other.grad = nullptr;
    other.value_history.resize(0);
}

void GradientDescent::swap(GradientDescent& other) noexcept {
    OptimizationMethod::swap(other);
    std::swap(curr_min_value, other.curr_min_value);
    std::swap(first_point, other.first_point);
    std::swap(lr, other.lr);
    std::swap(grad_norm, other.grad_norm);
    std::swap(last_step_norm, other.last_step_norm);
    std::swap(rel_imp_norm, other.rel_imp_norm);
    std::swap(value_history, other.value_history);
}

GradientDescent& GradientDescent::operator=(GradientDescent other) noexcept {
    this->swap(other);
    return *this;
}

void GradientDescent::set_first_point(const std::vector<double>& point) {
    first_point = point;
}

OptInfo* GradientDescent::get_opt_info() const {
    return new GDOptInfo(n_iter, grad_norm, last_step_norm, rel_imp_norm);
}

void GradientDescent::make_step() {
    std::vector<double> curr_point = point_history.back();
    double curr_value = value_history.back();
    std::vector<double> next_point(n_dim);
    std::vector<double> grad_value = (*grad)(curr_point);
    double tmp_lr = lr;

    if (!box_area->is_in(next_point)) {
        for (int i = 0; i < func->get_n_dim(); ++i) {
            dimensional_limits limit = box_area->get_limits(i);
            if (curr_point[i] - tmp_lr * grad_value[i] > limit.upper)
                tmp_lr = (curr_point[i] - limit.upper) / grad_value[i];
            if (curr_point[i] - tmp_lr * grad_value[i] < limit.lower)
                tmp_lr = (curr_point[i] - limit.lower) / grad_value[i];
        }
    }

    next_point = curr_point - tmp_lr * grad_value;
    point_history.push_back(next_point);
    value_history.push_back((*func)(next_point));
    if (norm_name == "l2") {
        grad_norm = l2_norm(grad_value);
        last_step_norm = l2_norm(next_point - curr_point);
    }
    else {
        grad_norm = l1_norm(grad_value);
        last_step_norm = l1_norm(next_point - curr_point);
    }
    if (value_history.size() > 1) {
        double next_value = (*func)(next_point);
        rel_imp_norm = abs((next_value - curr_value) / next_value);
    }

    return;
}

OptResult GradientDescent::optimize() {
    point_history.push_back(first_point);
    value_history.push_back((*func)(first_point));
    curr_min_value = value_history.back();
    while (!stop_criterion->criterion(get_opt_info())) {
        make_step();
        ++n_iter;
    }

    return OptResult(point_history.back(), curr_min_value, n_iter);
}

GradientDescent::~GradientDescent() {
    first_point.resize(0);
}