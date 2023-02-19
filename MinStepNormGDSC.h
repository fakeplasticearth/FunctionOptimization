#pragma once
#include "StopCriterion.h"
#include "GDOptInfo.h"

class MinStepNormGDSC : public StopCriterion {
private:
    double min_step_norm;
public:
    MinStepNormGDSC();
    MinStepNormGDSC(unsigned int max_n_iter_, double min_step_norm_);
    MinStepNormGDSC(const MinStepNormGDSC& other);
    MinStepNormGDSC(MinStepNormGDSC&& other) noexcept;

    void swap(MinStepNormGDSC& other) noexcept;
    MinStepNormGDSC& operator=(MinStepNormGDSC other) noexcept;
    bool criterion(OptInfo* status);
    double get_min_step_norm() const;
    ~MinStepNormGDSC();
};

