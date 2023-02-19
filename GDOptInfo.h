#pragma once
#include "OptInfo.h"

class GDOptInfo : public OptInfo {
private:
    double grad_norm;
    double last_step_norm;
    double rel_imp_norm;
public:
    GDOptInfo();
    GDOptInfo(unsigned int n_iter_, double grad_norm_, double last_step_norm_, double rel_imp_norm_);
    GDOptInfo(const GDOptInfo& other);
    GDOptInfo(GDOptInfo&& other) noexcept;

    void swap(GDOptInfo& other) noexcept;
    GDOptInfo& operator=(GDOptInfo other) noexcept;
    double get_grad_norm() const;
    double get_last_step_norm() const;
    double get_rel_imp_norm() const;
    ~GDOptInfo();
};

