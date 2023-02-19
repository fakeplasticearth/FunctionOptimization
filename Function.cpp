#include "Function.h"

Function::Function() {

}

Function::Function(double (*func_)(std::vector<double>), unsigned int n_dim_) : func(func_), n_dim(n_dim_) {

}

Function::Function(const Function& other) : func(other.func), n_dim(other.n_dim) {

}

Function::Function(Function&& other) noexcept : n_dim(other.n_dim), func(other.func) {
    other.func = nullptr;
    other.n_dim = 0;
}

double Function::operator()(const std::vector<double>& args) const {
    return func(args);
}

Function& Function::operator=(Function other) {
    this->swap(other);
    return *this;
}

void Function::swap(Function& other) noexcept {
    std::swap(n_dim, other.n_dim);
    std::swap(func, other.func);
}

unsigned int Function::get_n_dim() const {
    return n_dim;
}

void Function::set_func(double (*new_func)(std::vector<double>), unsigned int new_n_dim) {
    n_dim = new_n_dim;
    func = new_func;
}

Function::~Function() {
    func = nullptr;
    n_dim = 0;
}