#pragma once
#include "StopCriterion.h"
#include "GDOptInfo.h"

class MinGradNormGDSC : public StopCriterion {
private:
    double min_grad_norm;
public:
    MinGradNormGDSC();
    MinGradNormGDSC(unsigned int max_n_iter_, double min_grad_norm_);
    MinGradNormGDSC(const MinGradNormGDSC& other);
    MinGradNormGDSC(MinGradNormGDSC&& other) noexcept;

    void swap(MinGradNormGDSC& other) noexcept;
    MinGradNormGDSC& operator=(MinGradNormGDSC other) noexcept;
    bool criterion(OptInfo* status);
    double get_min_grad_norm() const;
    ~MinGradNormGDSC();
};

