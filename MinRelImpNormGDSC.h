#pragma once
#include "StopCriterion.h"
#include "GDOptInfo.h"

class MinRelImpNormGDSC : public StopCriterion {
private:
    double min_rel_imp_norm;
public:
    MinRelImpNormGDSC();
    MinRelImpNormGDSC(unsigned int max_n_iter_, double min_rel_imp_norm_);
    MinRelImpNormGDSC(const MinRelImpNormGDSC& other);
    MinRelImpNormGDSC(MinRelImpNormGDSC&& other) noexcept;

    void swap(MinRelImpNormGDSC& other) noexcept;
    MinRelImpNormGDSC& operator=(MinRelImpNormGDSC other) noexcept;
    bool criterion(OptInfo* status);
    double get_min_rel_imp_norm() const;
    ~MinRelImpNormGDSC();
};

