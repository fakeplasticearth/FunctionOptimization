#pragma once
#include "StopCriterion.h"
#include "StohOptInfo.h"

class NormDiffSSC : public StopCriterion {
private:
    double min_last_imp_norm;
public:
    NormDiffSSC();
    NormDiffSSC(unsigned int max_n_iter_, double min_last_imp_norm_);
    NormDiffSSC(const NormDiffSSC& other);
    NormDiffSSC(NormDiffSSC&& other) noexcept;

    void swap(NormDiffSSC& other) noexcept;
    NormDiffSSC& operator=(NormDiffSSC other) noexcept;
    bool criterion(OptInfo* opt_info);
    double get_min_last_imp_norm() const;
    ~NormDiffSSC();
};

