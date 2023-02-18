#pragma once
#include <vector>

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