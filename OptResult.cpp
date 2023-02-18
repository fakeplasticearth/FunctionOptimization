#include "OptResult.h"

OptResult::OptResult(): min_value(DBL_MAX), n_iter(0) {

}

OptResult::OptResult(const std::vector<double>& min_point_, double min_value_, unsigned int n_iter_): min_point(min_point_),
	min_value(min_value_), n_iter(n_iter_) {

}

OptResult::OptResult(const OptResult& other): min_point(other.min_point), min_value(other.min_value), n_iter(other.n_iter) {

}

OptResult::OptResult(OptResult&& other) noexcept: min_point(other.min_point), min_value(other.min_value), n_iter(other.n_iter) {
	other.n_iter = 0;
	other.min_point.resize(0);
}

void OptResult::swap(OptResult& other) noexcept {
	std::swap(min_point, other.min_point);
	std::swap(min_value, other.min_value);
	std::swap(n_iter, other.n_iter);
}

OptResult& OptResult::operator=(OptResult other) noexcept {
	this -> swap(other);
	return *this;
}

unsigned int OptResult::get_n_iter() const {
	return n_iter;
}

std::vector<double> OptResult::get_min_point() const {
	return min_point;
}

double OptResult::get_min_value() const {
	return min_value;
}