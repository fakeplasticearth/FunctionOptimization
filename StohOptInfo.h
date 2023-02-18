#pragma once
#include "OptInfo.h"

class StohOptInfo : public OptInfo {
private:
    unsigned int n_iter_after_imp;
    double last_imp_norm;
public:
    StohOptInfo();
    StohOptInfo(unsigned int n_iter_, unsigned int n_iter_after_imp, double last_imp_norm_);
    StohOptInfo(const StohOptInfo& other);
    StohOptInfo(StohOptInfo&& other) noexcept;

    void swap(StohOptInfo& other) noexcept;
    StohOptInfo& operator=(StohOptInfo other) noexcept;
    unsigned int get_n_iter_after_imp() const;
    double get_last_imp_norm() const;
    ~StohOptInfo() {};
};

